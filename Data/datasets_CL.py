import torch
from PIL import Image
from io import BytesIO
from base64 import b64decode, b64encode
import io
from datasets import load_dataset
from tqdm.auto import tqdm
from torchvision import transforms as T
from torch.utils.data import random_split


class COCOCAPTIONS(torch.utils.data.Dataset):
    image_key, text_key = "image_base64_str", "outputs"
    instance_valid_name = ["input_ids", "attention_mask"]
    def __init__(self, data_args, model_args, tokenizer_func, flag_split):
        assert flag_split in ["train", "validation", "test"]
        flag_openai_clip = True if model_args.model_name_img[:17] == "CLIP_openai_clip_" else False
        data = self.load_dataset(data_args, flag_split)

        _, data = random_split(data, [0.98, 0.02])

        #deal with captions
        captions, id_invalid_caption = [], []
        for idx, row in enumerate(tqdm(data, desc="extract captions")):
            caption = row[self.text_key]
            caption = caption[0] if isinstance(caption, list) else caption
            caption = self.func_caption_operation(caption)
            if caption is not None:
                captions.append(caption)
            else:
                caption = " "
                captions.append(caption)
                id_invalid_caption.append(idx)
        if flag_openai_clip:
            encoded_captions = tokenizer_func(captions, context_length=data_args.max_seq_length, truncate=True)
        else:
            encoded_captions = tokenizer_func(captions, max_length=data_args.max_seq_length, padding="max_length", truncation=True)

        #create each instance
        self.data = []
        for idx, row in enumerate(tqdm(data, desc="create data")):
            if idx in id_invalid_caption: continue  #ignore img with invalid caption
            img = row[self.image_key]
            img = img[0] if isinstance(img, list) else img
            if img.format is None: continue
            if isinstance(img, Image.Image):
                buffer = io.BytesIO()
                img.save(buffer, format=img.format)  # Use the appropriate format (e.g., JPEG, PNG)
                buffer.seek(0)  # Move the pointer to the start of the buffer
                img = b64encode(buffer.getvalue()).decode("utf-8")
            input_ids = encoded_captions[idx] if flag_openai_clip else torch.tensor(encoded_captions["input_ids"][idx])
            attention_mask = torch.tensor(0.0) if flag_openai_clip else torch.tensor(encoded_captions["attention_mask"][idx])
            self.data.append([img,  {"input_ids": input_ids, "attention_mask": attention_mask}])
        self.transforms = self.get_image_tranforms(image_size = (data_args.img_size, data_args.img_size))

    @staticmethod
    def func_caption_operation(caption):
        return caption

    @staticmethod
    def func_get_caption(row):
        row, text_key = row
        caption = row[text_key]
        caption = caption[0] if isinstance(caption, list) else caption
        return caption
    
    @staticmethod
    def func_append_instance(row):
        row, input_ids, attention_mask, image_key = row
        img = row[image_key]
        img = img[0] if isinstance(img, list) else img
        d = [img,  {"input_ids": input_ids, "attention_mask": attention_mask if attention_mask is not None else torch.tensor(0.0)}]
        return d

    def load_dataset(self, data_args, flag_split):
        return load_dataset("MMInstruction/M3IT", data_args.dataset_name, cache_dir = data_args.cache_dir_data, trust_remote_code=True)[flag_split]

    def __getitem__(self, idx):
        image, encoded_text = self.data[idx]
        image = self.load_img(image)
        image = self.transforms(image)
        instance = {key: value.clone().detach() for key, value in encoded_text.items() if key in self.instance_valid_name}
        instance["pixel_values"] = image
        return instance
    
    @staticmethod
    def load_img(img):
        return Image.open(BytesIO(b64decode(img)))

    def __len__(self):
        return len(self.data)
    
    def get_image_tranforms(self, image_size):
        return T.Compose([
            T.Resize(image_size),
            T.Lambda(self._grayscale_to_rgb),
            T.ToTensor()
        ])
    
    @staticmethod
    def _grayscale_to_rgb(img):
        if img.mode != "RGB":
            return img.convert("RGB")
        return img
    

class FLICKR30K(COCOCAPTIONS):
    image_key, text_key = "image", "caption"
    def load_dataset(self, data_args, flag_split):
        if flag_split == "validation": flag_split = "val"
        data = load_dataset(data_args.dataset_name, cache_dir = data_args.cache_dir_data, trust_remote_code=True)["test"]
        data_split = []
        for idx, row in enumerate(tqdm(data, desc="split flickr30k")):
            if row["split"] == flag_split: data_split.append(row)
        assert len(data_split) > 0
        return data_split
    
    @staticmethod
    def func_single(d):
        (d, flag_split) = d
        if d["split"] == flag_split:
            return d
        else:
            return None   
    

class MIMIC_CHEST_XRAY_V1_TRAIN(COCOCAPTIONS):
    image_key, text_key = "image", "report"
    def load_dataset(self, data_args, flag_split):
        return load_dataset(data_args.dataset_name, cache_dir = data_args.cache_dir_data, trust_remote_code=True)[flag_split]
    
    
class MMIMDB(COCOCAPTIONS):
    image_key, text_key = "images", "messages"
    def load_dataset(self, data_args, flag_split):
        return load_dataset(data_args.dataset_name, cache_dir = data_args.cache_dir_data, trust_remote_code=True)[flag_split]
    @staticmethod
    def func_caption_operation(caption_raw):
        caption = caption_raw["content"]
        caption = caption.split("\nPlot: ")[1].split("\nNote")[0]
        if len(caption) == 0:
            caption = None
            # raise Exception("invalid caption = %s"%(caption_raw))
        return caption


def init_data(data_args, model_args, tokenizer_func):

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        input_ids =  torch.stack([example["input_ids"] for example in examples]) #torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
        attention_mask = torch.stack([example["attention_mask"] for example in examples]) #torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_loss": True,
        }

    if data_args.dataset_name == "coco":
        return COCOCAPTIONS(data_args, model_args, tokenizer_func, "train"), \
                COCOCAPTIONS(data_args, model_args, tokenizer_func, "validation"), \
                COCOCAPTIONS(data_args, model_args, tokenizer_func, "test"), \
                collate_fn
    elif data_args.dataset_name == "nlphuji/flickr30k":
        return FLICKR30K(data_args, model_args, tokenizer_func, "train"), \
                FLICKR30K(data_args, model_args, tokenizer_func, "validation"), \
                FLICKR30K(data_args, model_args, tokenizer_func, "test"), \
                collate_fn
    elif data_args.dataset_name == "hongrui/mimic_chest_xray_v_1":
        dataset_local = MIMIC_CHEST_XRAY_V1_TRAIN(data_args, model_args, tokenizer_func, "train")
        dataset_train, dataset_valid, dataset_test = random_split(dataset_local, data_args.train_valid_test_split)
        print("Warning: training dataset is split into three parts: %s"%(data_args.train_valid_test_split))
        return dataset_train, dataset_valid, dataset_test, collate_fn
    elif data_args.dataset_name == "sxj1215/mmimdb":
        dataset_local = MMIMDB(data_args, model_args, tokenizer_func, "train")
        dataset_train, dataset_valid, dataset_test = random_split(dataset_local, data_args.train_valid_test_split)
        print("Warning: training dataset is split into three parts: %s"%(data_args.train_valid_test_split))
        return dataset_train, dataset_valid, dataset_test, collate_fn
    else:
        raise Exception("Invalid dataset_name: %s"%(data_args.dataset_name))
    
