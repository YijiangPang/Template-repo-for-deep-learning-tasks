import torch.nn as nn
from torchvision import models
from transformers import AutoConfig, AutoTokenizer, DistilBertModel, BertModel
import torch
import numpy as np
import torch.nn.functional as F

from transformers.configuration_utils import PretrainedConfig
from transformers.models.clip.modeling_clip import CLIPOutput
from typing import Optional, Tuple, Union
from Models.base_vit import ViT
import clip


class ImageEncoder(nn.Module):
    def __init__(self, model_args, data_args):
        super(ImageEncoder, self).__init__()
        backbone, img_size, pretrained = model_args.model_name_img, data_args.img_size, model_args.flag_pretrain
        if backbone == "resnet18":
            model_base = models.resnet18(weights= None if not pretrained else models.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == "resnet34":
            model_base = models.resnet34(weights= None if not pretrained else models.ResNet34_Weights.IMAGENET1K_V1)
        elif backbone == "resnet50":
            model_base = models.resnet50(weights= None if not pretrained else models.ResNet50_Weights.IMAGENET1K_V1)
        elif backbone[:4] == "vit_":
            name = backbone[4:]
            model_base = ViT(name = name, pretrained = pretrained, image_size = img_size)

        if "resn" in backbone:
            feature_dim = model_base.fc.in_features
            model_base.fc = nn.Identity()
        elif "vit_" in backbone:
            feature_dim = model_base.fc.in_features
            model_base.fc = nn.Identity()
        self.model_base, self.feature_dim = model_base, feature_dim

    def forward(self, x):
        return self.model_base(x)

#TextEncoder
class TextEncoder(nn.Module):

    def __init__(self, model_args):
        super(TextEncoder, self).__init__()
        model_name, pretrained = model_args.model_name_text, model_args.flag_pretrain
        if pretrained:
            if model_name == "bert-base-uncased":
                self.model = BertModel.from_pretrained(model_name)
            elif model_name == "distilbert-base-uncased":
                self.model = DistilBertModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(
                                model_name,
                                cache_dir=model_args.cache_dir_model,
                                revision=model_args.model_revision,
                                token=model_args.token,
                                trust_remote_code=model_args.trust_remote_code
                                )
            if model_name == "bert-base-uncased":
                self.model = BertModel(config)
            elif model_name == "distilbert-base-uncased":
                self.model = DistilBertModel(config)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state.mean(dim=1)


class CLIP(torch.nn.Module):

    def __init__(self,
                 model_args, data_args,
                 text_mlp_dim=768,
                 proj_dim=256,
                 init_tau=np.log(1.0),
                 init_b=0):
        super(CLIP, self).__init__()
        self.flag_openai_clip = True if model_args.model_name_img[:17] == "CLIP_openai_clip_" else False
        self.config = PretrainedConfig(name_or_path =  model_args.model_name_img)
        if self.flag_openai_clip:
            self.clip_model, _ = clip.load(model_args.model_name_img[17:])
            self.clip_model = self.clip_model.to(torch.float)
            self.t_prime = self.clip_model.logit_scale
            self.b = nn.Parameter(torch.ones([]) * init_b)
            #RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px
        else:
            # model_args.model_name_img = model_args.model_name_img
            self.image_encoder = ImageEncoder(model_args, data_args)
            img_feature_dim = self.image_encoder.feature_dim
            self.text_encoder = TextEncoder(model_args)

            self.image_projection = torch.nn.Sequential(
                torch.nn.Linear(img_feature_dim, img_feature_dim, bias=False),
                torch.nn.ReLU(),
                torch.nn.Linear(img_feature_dim, proj_dim, bias=False))
            
            self.text_projection = torch.nn.Sequential(
                torch.nn.Linear(text_mlp_dim, text_mlp_dim, bias=False),
                torch.nn.ReLU(),
                torch.nn.Linear(text_mlp_dim, proj_dim, bias=False))
            
            self.t_prime = nn.Parameter(torch.ones([]) * init_tau)
            self.b = nn.Parameter(torch.ones([]) * init_b)

    def extract_image_features(self, images):
        if self.flag_openai_clip:
            image_embeds = self.clip_model.encode_image(images)
            return image_embeds, None
        else:
            image_outputs = self.image_encoder(images)
            return self.image_projection(image_outputs), image_outputs

    def extract_text_features(self, input_ids, attention_mask):
        if self.flag_openai_clip:
            text_embeds = self.clip_model.encode_text(input_ids)
            return text_embeds, None
        else:
            text_outputs = self.text_encoder(input_ids, attention_mask)
            return self.text_projection(text_outputs), text_outputs
    
    def contrastive_loss(self, image_embeds, text_embeds, flag_loss_avg = True):
        logits =  image_embeds @ text_embeds.t() * self.t_prime.exp() + self.b
        targets = torch.arange(logits.size(0)).to(logits.device)
        loss_images = F.cross_entropy(logits, targets) if flag_loss_avg else F.cross_entropy(logits, targets, reduction = "none")
        loss_texts = F.cross_entropy(logits.t(), targets) if flag_loss_avg else F.cross_entropy(logits.t(), targets, reduction = "none")
        return (loss_images + loss_texts) / 2
    
    def similarity_score(self, image_embeds, text_embeds):
        score = torch.sum(image_embeds*text_embeds, dim=-1)
        return score
    
    # @replace_return_docstrings(output_type=CLIPOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CLIPOutput]:
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        image_embeds, image_outputs = self.extract_image_features(pixel_values)
        text_embeds, text_outputs = self.extract_text_features(input_ids, attention_mask)

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * self.t_prime.exp()
        logits_per_image = logits_per_text.T

        loss = self.contrastive_loss(image_embeds, text_embeds) if return_loss else None

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=image_outputs,
            vision_model_output=text_outputs,
        )


def pre_model(model_args, data_args):
    # model = VisionTextDualEncoderModel.from_vision_text_pretrained(model_name_img, model_name_text, 
    #                                                                 cache_dir=model_args.cache_dir_model,
    #                                                                 revision=model_args.model_revision,
    #                                                                 trust_remote_code=model_args.trust_remote_code)
    # image_processor = AutoImageProcessor.from_pretrained(model_name_img,
    #                                                         cache_dir=model_args.cache_dir_model,
    #                                                         revision=model_args.model_revision,
    #                                                         trust_remote_code=model_args.trust_remote_code)

    flag_openai_clip = True if model_args.model_name_img[:17] == "CLIP_openai_clip_" else False
    model = CLIP(model_args, data_args)
    if flag_openai_clip:
        tokenizer = clip.tokenize
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_text,
                                                cache_dir=model_args.cache_dir_model,
                                                revision=model_args.model_revision,
                                                trust_remote_code=model_args.trust_remote_code)
    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_vision_model:
        _freeze_params(model.vision_model)

    if model_args.freeze_text_model:
        _freeze_params(model.text_model)

    return model, tokenizer