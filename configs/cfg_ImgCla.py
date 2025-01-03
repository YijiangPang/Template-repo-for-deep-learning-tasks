from dataclasses import dataclass, field
from configs.cfg_Base import DataArguments_base, ModelArguments_base, ProjArguments_base, cfg_Base


@dataclass
class DataArguments(DataArguments_base):
    image_column_name: str = field(default="img",metadata={"help": "The name of the dataset column containing the image data. Defaults to 'image'."})
    label_column_name: str = field(default="label",metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'."})
    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )
    labels: str  = field(default=None)
    img_size: int = field(default = 32)
    input_channel: int = field(default = 3)
    

@dataclass
class ModelArguments(ModelArguments_base):
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})


@dataclass
class ProjArguments(ProjArguments_base):
    proj_name: str = field(default = "ImgCla")
    solver: str = field(default = "solver_ImgCla" )


class cfg_ImgCla(cfg_Base):
    def __init__(self):
        cfg_Base.__init__(self, ModelArguments, DataArguments, ProjArguments)

    def get_cfg(self):
        self.training_args.remove_unused_columns = False
        self.training_args.metric_for_best_model = "accuracy"
        return self.logger, self.model_args, self.data_args, self.training_args, self.proj_args