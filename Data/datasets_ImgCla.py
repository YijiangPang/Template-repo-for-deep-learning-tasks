import os
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from Data.datasets_Base import datasets_Base


class datasets_ImgCla(datasets_Base):
    def __init__(self, model_args, data_args, training_args):
        datasets_Base.__init__(self, model_args, data_args, training_args)

    def get_dataset(self):

        dataset_column_names = self.dataset["train"].column_names if "train" in self.dataset else self.dataset["validation"].column_names
        if self.data_args.image_column_name not in dataset_column_names:
            raise ValueError(
                f"--image_column_name {self.data_args.image_column_name} not found in dataset '{self.data_args.dataset_name}'. "
                "Make sure to set `--image_column_name` to the correct audio column - one of "
                f"{', '.join(dataset_column_names)}."
            )
        if self.data_args.label_column_name not in dataset_column_names:
            raise ValueError(
                f"--label_column_name {self.data_args.label_column_name} not found in dataset '{self.data_args.dataset_name}'. "
                "Make sure to set `--label_column_name` to the correct text column - one of "
                f"{', '.join(dataset_column_names)}."
            )

        label_column_name = self.data_args.label_column_name
        self.data_args.labels = self.dataset["train"].features[label_column_name].names
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return dict(accuracy=accuracy_score(predictions, labels))

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            labels = torch.tensor([example[label_column_name] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

        return  self.dataset, compute_metrics, collate_fn, self.data_args

    @staticmethod
    def data_preprocess(dataset, data_args, training_args):

        # Define torchvision transforms to be applied to each image.
        size = data_args.img_size

        train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
            ]
        )
        val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
            ]
        )

        def preprocess_train(example_batch):
            """Apply _train_transforms across a batch."""
            example_batch["pixel_values"] = [
                train_transforms(image.convert("RGB")) for image in example_batch[data_args.image_column_name]
            ]
            return example_batch

        def preprocess_val(example_batch):
            """Apply _val_transforms across a batch."""
            example_batch["pixel_values"] = [
                val_transforms(image.convert("RGB")) for image in example_batch[data_args.image_column_name]
            ]
            return example_batch

        if training_args.do_train:
            if data_args.max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=data_args.shuffle_seed).select(range(data_args.max_train_samples))
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess_train)
        else:
            train_dataset = None

        if training_args.do_eval:
            if data_args.max_eval_samples is not None:
                dataset["validation"] = dataset["validation"].shuffle(seed=data_args.shuffle_seed).select(range(data_args.max_eval_samples))
            # Set the validation transforms
            eval_dataset = dataset["validation"].with_transform(preprocess_val)
        else:
            eval_dataset = None

        if training_args.do_predict:
            if data_args.max_predict_samples is not None:
                dataset["test"] = dataset["test"].shuffle(seed=data_args.shuffle_seed).select(range(data_args.max_predict_samples))
            test_dataset = dataset["test"].with_transform(preprocess_val)
        else:
            test_dataset = None

        return train_dataset, eval_dataset, test_dataset


