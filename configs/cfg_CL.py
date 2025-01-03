from dataclasses import dataclass, field
from typing import Optional
from configs.cfg_Base import DataArguments_base, ModelArguments_base, ProjArguments_base, cfg_Base


@dataclass
class DataArguments(DataArguments_base):
    img_size: int = field(default = 224)
    shuffle_captions: bool = field(default=True)
    max_seq_length: int = field(default=77)
    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
    

@dataclass
class ModelArguments(ModelArguments_base):
    model_name_img: str = field(default=None)
    model_name_text: str = field(default=None)
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    image_processor_name: Optional[str] = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_fast_tokenizer: bool = field(default=True,metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},)
    freeze_vision_model: bool = field(default=False, metadata={"help": "Whether to freeze the vision model parameters or not."})
    freeze_text_model: bool = field(default=False, metadata={"help": "Whether to freeze the text model parameters or not."})

@dataclass
class ProjArguments(ProjArguments_base):
    proj_name: str = field(default = "CL")
    solver: str = field(default = "solver_CL" )


class cfg_CL(cfg_Base):
    def __init__(self):
        cfg_Base.__init__(self, ModelArguments, DataArguments, ProjArguments)

    def init_cfg(self):
        self.training_args.remove_unused_columns = False
        self.training_args.metric_for_best_model = "accuracy"
        return self.logger, self.model_args, self.data_args, self.training_args, self.proj_args