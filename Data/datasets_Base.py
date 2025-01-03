import os
from datasets import load_dataset


class datasets_Base:
    def __init__(self, model_args, data_args, training_args):
        self.load_dataset(model_args, data_args, training_args)

    def load_dataset(self, model_args, data_args, training_args):
        if data_args.dataset_name is not None:
            dataset = load_dataset(
                data_args.dataset_name,
                cache_dir=data_args.cache_dir_data,
                trust_remote_code=model_args.trust_remote_code,
            )
        else:
            data_files = {}
            if data_args.train_dir is not None:
                data_files["train"] = os.path.join(data_args.train_dir, "**")
            if data_args.validation_dir is not None:
                data_files["validation"] = os.path.join(data_args.validation_dir, "**")
            if data_args.test_file is not None:
                data_files["test"] =  os.path.join(data_args.test_file, "**")
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=data_args.cache_dir_data,
            )

        # If we don't have a validation split, split off a percentage of train as validation.
        data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
        if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
            split = dataset["train"].train_test_split(data_args.train_val_split)
            dataset["train"] = split["train"]
            dataset["validation"] = split["test"]

        self.dataset = dataset
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

    def get_dataset(self):

        def compute_metrics(eval_pred):
            pass

        def collate_fn(examples):
            pass

        return self.dataset, compute_metrics, collate_fn
    
    @staticmethod
    def data_preprocess(dataset, data_args, training_args):
        pass

