from dataclasses import dataclass, field
from typing import Optional
from configs.cfg_Base import DataArguments_base, ModelArguments_base, ProjArguments_base, cfg_Base


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


@dataclass
class DataArguments(DataArguments_base):
    task_name: Optional[str] = field(default=None,metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},)
    max_seq_length: int = field(default=128,metadata={"help": ("The maximum total input sequence length after tokenization. Sequences longer ""than this will be truncated, sequences shorter will be padded.")},)
    pad_to_max_length: bool = field(default=True, metadata={"help": ("Whether to pad all samples to `max_seq_length`. ""If False, will pad the samples dynamically when batching to the maximum length in the batch.")},)
    train_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the training data."})
    validation_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the validation data."})
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."
    num_labels: int = field(default=None)
    is_regression: bool = field(default=None)


@dataclass
class ModelArguments(ModelArguments_base):
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    use_fast_tokenizer: bool = field(default=True,metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},)


@dataclass
class ProjArguments(ProjArguments_base):
    proj_name: str = field(default = "LanCla")
    solver: str = field(default = "solver_LanCla" )


class cfg_LanCla(cfg_Base):
    def __init__(self):
        cfg_Base.__init__(self, ModelArguments, DataArguments, ProjArguments)