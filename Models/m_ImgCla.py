
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional
import torch.nn as nn
import torch
from Models.base_vit import ViT
import clip


class Standard_m(PreTrainedModel):
    def __init__(self, model_args, data_args, training_args):
        backbone = model_args.model_name_or_path if model_args.model_name_img is None else model_args.model_name_img
        pretrained = model_args.flag_pretrain
        num_labels = len(data_args.labels)
        image_size = data_args.img_size
        input_channel = data_args.input_channel
        cache_dir = model_args.cache_dir_model

        config = PretrainedConfig(name_or_path = backbone)
        super().__init__(config)

        torch.hub.set_dir(cache_dir)
        if backbone == "resnet18":
            self.model_base = models.resnet18(weights= None if not pretrained else models.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == "resnet34":
            self.model_base = models.resnet34(weights= None if not pretrained else models.ResNet34_Weights.IMAGENET1K_V1)
        elif backbone == "resnet50":
            self.model_base = models.resnet50(weights= None if not pretrained else models.ResNet50_Weights.IMAGENET1K_V1)
        elif backbone == "resnext101_32x8d":
            self.model_base = models.quantization.resnext101_32x8d(weights= None if not pretrained else models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
        elif backbone == "vgg11":
            self.model_base = models.vgg11(weights= None if not pretrained else models.VGG11_Weights.IMAGENET1K_V1)
        elif backbone == "densenet121":
            self.model_base = models.densenet121(weights= None if not pretrained else models.DenseNet121_Weights.IMAGENET1K_V1)
        elif backbone[:4] == "vit_":
            name = backbone[4:]
            self.model_base = ViT(name = name, pretrained = pretrained, image_size = image_size)
        elif backbone[:5] == "CLIP_":
            self.model_base = CLIP_inference(model_args, data_args)
            if training_args.resume_from_checkpoint is not None:
                from safetensors.torch import load_file
                checkpoint = load_file(training_args.resume_from_checkpoint)
                self.model_base.load_state_dict(checkpoint)
                print("Load state dict success! path = %s"%(training_args.resume_from_checkpoint))

        self.num_labels = num_labels
        if "CLIP_" not in backbone:
            if "resn" in backbone:
                self.feature_dim = self.model_base.fc.in_features
                if num_labels != 1000:
                    self.model_base.conv1 = nn.Conv2d(input_channel, 64, 3, 1, 1, bias=False)
                    self.model_base.maxpool = nn.Identity()
                    self.model_base.fc = nn.Linear(self.feature_dim, self.num_labels)
            elif "vit_" in backbone:# and isinstance(self.model_base, models.VisionTransformer):
                self.feature_dim = self.model_base.fc.in_features
                if num_labels != 1000:
                    self.model_base.fc = nn.Linear(self.feature_dim, self.num_labels)
            elif "vgg" in backbone:
                self.model_base.classifier._modules["6"] = nn.Linear(self.model_base.classifier._modules["6"].in_features, self.num_labels)
            elif "densenet" in backbone:
                self.model_base.classifier = nn.Linear(self.model_base.classifier.in_features, self.num_labels)

        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ImageClassifierOutputWithNoAttention:

        logits = self.model_base(pixel_values)
        loss = F.cross_entropy(logits, labels)

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=None)



from Models.m_CL import CLIP
from transformers import AutoTokenizer


class CLIP_inference(CLIP):
    def __init__(self, model_args, data_args, **kwargs):
        super(CLIP_inference, self).__init__(model_args, data_args, **kwargs)
        self.max_seq_length = data_args.max_seq_length
        self.num_labels = len(data_args.labels)
        self.labels = data_args.labels
        self.prompts_format = ["A photo of a", None]
        if self.flag_openai_clip:
            self.tokenizer = clip.tokenize
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_text,
                                                        cache_dir=model_args.cache_dir_model,
                                                        revision=model_args.model_revision,
                                                        trust_remote_code=model_args.trust_remote_code)
        self.text_features = self.init_fixed_text_features()
        
    def init_fixed_text_features(self):
        prompts_text = [" ".join([n if n is not None else c for n in self.prompts_format]) for c in self.labels]
        # print("prompts_text = ", prompts_text)
        if self.flag_openai_clip:
            encoded_captions = self.tokenizer(prompts_text, context_length=self.max_seq_length, truncate=True)
            input_ids = encoded_captions.to(self.clip_model.token_embedding.weight.device)
            with torch.no_grad():
                text_features, _ = self.extract_text_features(input_ids, None)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            encoded_captions = self.tokenizer(prompts_text, max_length=self.max_seq_length, padding="max_length", truncation=True)
            input_ids, attention_mask = torch.tensor(encoded_captions["input_ids"]), torch.tensor(encoded_captions["attention_mask"])
            input_ids, attention_mask = input_ids.to(self.t_prime.device), attention_mask.to(self.t_prime.device)
            with torch.no_grad():
                text_features, _ = self.extract_text_features(input_ids, attention_mask)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        del self.tokenizer
        del self.max_seq_length
        del self.prompts_format
        del self.labels
        return text_features

    def forward(self,x):
        return self.forward_logits(x)
    
    def forward_logits(self, x):
        text_features = self.text_features
        image_features, _ = self.extract_image_features(x)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.t().to(x.device)
        similarity = torch.reshape(similarity, (similarity.shape[0], self.num_labels))
        # similarity, _ = torch.max(similarity, dim = -1)
        return similarity