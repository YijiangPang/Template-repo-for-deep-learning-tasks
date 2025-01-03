from transformers import set_seed
import os
import json


class solver_Base():
    
    def __init__(self, logger, model_args, data_args, training_args, proj_args, seed_c):
        self.logger = logger
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.proj_args = proj_args
        self.set_seed(seed_c)
        
    def run(self):
        if self.training_args.do_train: self.do_train()
        if self.training_args.do_eval: self.do_eval()
        if self.training_args.do_predict: self.do_predict()
        self.finalize()

    def set_seed(self, seed):
        self.training_args.seed, self.data_args.shuffle_seed = seed, seed
        set_seed(seed)

    def finalize(self):
        #for recording
        self.proj_args.dataset_name = self.data_args.dataset_name

        self.proj_args.model_name_or_path = self.model_args.model_name_or_path
        self.proj_args.flag_pretrain = self.model_args.flag_pretrain

        self.proj_args.lr_scheduler_type = self.training_args.lr_scheduler_type
        self.proj_args.seed_c = self.training_args.seed
        self.proj_args.weight_decay = self.training_args.weight_decay
        self.proj_args.learning_rate = self.training_args.learning_rate

        with open(os.path.join(self.training_args.output_dir, 'proj_args.json'), 'w', encoding='utf-8') as fout:
            json.dump(self.proj_args.__dict__, fout, ensure_ascii=False, indent=4)