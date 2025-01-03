from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from dataclasses import dataclass, field
from typing import Optional, List
import logging
import os
import sys
import datasets
import transformers
import numpy as np
from time import localtime, strftime
import json


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class DataArguments_base:
    dataset_name: Optional[str] = field(default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."})
    shuffle_train_dataset: bool = field(default=True, metadata={"help": "Whether to shuffle the train dataset or not."})
    shuffle_seed: int = field(default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."})
    train_dir: Optional[str] = field(default=None)
    validation_dir: Optional[str] = field(default=None)
    train_val_split: Optional[float] = field(default=0.15)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_predict_samples: Optional[int] = field(default=None)
    cache_dir_data: Optional[str] = field(default=None)
    preprocessing_num_workers: int = field(default = 4)

    max_seq_length: int = field(default=None)
    train_valid_test_split: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1], metadata={"help": "when valid&test is not None, this will be used."})
    

@dataclass
class ModelArguments_base:
    model_name_or_path: str = field(default="microsoft/resnet-18")
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    cache_dir_model: Optional[str] = field(default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"})
    model_revision: str = field(default="main",metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},)
    token: str = field(default=None,metadata={"help": ("The token to use as HTTP bearer authorization for remote files. If not specified, will use the token ""generated when running `huggingface-cli login` (stored in `~/.huggingface`).")},)
    trust_remote_code: bool = field(default=True,)
    ignore_mismatched_sizes: bool = field(default=True,metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},)
    flag_pretrain: bool = field(default=False)
    force_download: bool = field(default=False)

    model_name_img: Optional[str] = field(default=None)
    model_name_text: Optional[str] = field(default=None)


@dataclass
class ProjArguments_base:
    proj_dir: str = field(default = None )
    flag_time: str = field(default = strftime("%Y-%m-%d_%H-%M-%S", localtime()))
    save_strategy_: str = field(default = "final" )#"no", "final"
    num_random: int = field(default = np.random.randint(9999))
    num_run: int = field(default = 3)
    proj_seed: int = field(default = np.random.randint(9999999))
    opt_name: str = field(default = "AdamW" )
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    

class cfg_Base:
    def __init__(self, ModelArguments, DataArguments, ProjArguments):
        logger = logging.getLogger(__name__)
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ProjArguments))
        model_args, data_args, training_args, proj_args = parser.parse_args_into_dataclasses()

        #init training_args
        training_args.logging_first_step = True
        training_args.dataloader_num_workers = 4

        #folders
        try:
            from pygit2 import Repository
            branch_name = Repository('.').head.shorthand
            log_sub_folder = "%s/%s_branch/log_%s_%s_%s_%04d"%(training_args.output_dir, branch_name, proj_args.proj_name, 
                                                               proj_args.opt_name, proj_args.flag_time, proj_args.num_random)
        except:
            log_sub_folder = "%s/log_%s_%s_%s_%04d"%(training_args.output_dir, proj_args.proj_name, 
                                                     proj_args.opt_name, proj_args.flag_time, proj_args.num_random)
        proj_args.proj_dir = log_sub_folder

        model_args.cache_dir_model = "/localscratch2/PublicModels"
        data_args.cache_dir_data = "/localscratch2/PublicDatasets"


        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        if training_args.should_log:
            # The default of training_args.log_level is passive, so we set log level at info here to have that default.
            transformers.utils.logging.set_verbosity_info()

        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
        )

        os.makedirs(log_sub_folder, exist_ok=True)
        with open(os.path.join(log_sub_folder, 'training_args.json'), 'w') as fout:
            fout.write(training_args.to_json_string())
        with open(os.path.join(log_sub_folder, 'data_args.json'), 'w', encoding='utf-8') as fout:
            json.dump(data_args.__dict__, fout, ensure_ascii=False, indent=4)
        with open(os.path.join(log_sub_folder, 'model_args.json'), 'w', encoding='utf-8') as fout:
            json.dump(model_args.__dict__, fout, ensure_ascii=False, indent=4)
        with open(os.path.join(log_sub_folder, 'proj_args.json'), 'w', encoding='utf-8') as fout:
            json.dump(proj_args.__dict__, fout, ensure_ascii=False, indent=4)

        self.logger = logger
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.proj_args = proj_args

    def get_cfg(self):
        return self.logger, self.model_args, self.data_args, self.training_args, self.proj_args