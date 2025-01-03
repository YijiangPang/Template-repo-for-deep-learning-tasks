from Optimizers.Optimizer_loader import optimizer_loader, lr_scheduler_loader
from Data.datasets_ImgCla import datasets_ImgCla
from Solvers.solver_Base import solver_Base
from Models.m_ImgCla import Standard_m
import torch
import math
from transformers import Trainer
# from Solvers.Trainer import TrainerCustomized as Trainer


class solver_ImgCla(solver_Base):
    
    def __init__(self, logger, model_args, data_args, training_args, proj_args, seed_c):
        self.logger = logger
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.proj_args = proj_args
        self.set_seed(seed_c)
        
        dataset_c = datasets_ImgCla(self.model_args, self.data_args, self.training_args)
        raw_datasets, compute_metrics, collate_fn, self.data_args = dataset_c.get_dataset()
        model = Standard_m(model_args = self.model_args, data_args = self.data_args, training_args = self.training_args)
        self.train_dataset, self.eval_dataset, self.test_dataset = dataset_c.data_preprocess(raw_datasets, self.data_args, self.training_args)

        total_steps = math.ceil(len(self.train_dataset)/(self.training_args.per_device_train_batch_size*torch.cuda.device_count())) * self.training_args.num_train_epochs
        optimizer = optimizer_loader(self.proj_args.opt_name, model, self.training_args.learning_rate, 
                                     self.proj_args.betas, weight_decay = self.training_args.weight_decay)
        lr_scheduler = lr_scheduler_loader(self.training_args.lr_scheduler_type, optimizer, T_max = total_steps)
        optimizers = (optimizer, lr_scheduler)

        self.trainer = Trainer(model=model,
                                args=self.training_args,
                                train_dataset=self.train_dataset if self.training_args.do_train else None,
                                eval_dataset=self.eval_dataset if self.training_args.do_eval else None,
                                data_collator = collate_fn,
                                optimizers = optimizers,
                                compute_metrics=compute_metrics,
                        )

    def do_train(self):
        train_result = self.trainer.train()
        if self.proj_args.save_strategy_ == "final": self.trainer.save_model()
        self.trainer.log_metrics("train", train_result.metrics)
        self.trainer.save_metrics("train", train_result.metrics)
        self.trainer.save_state()

    def do_eval(self):
        metrics = self.trainer.evaluate()
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

    def do_predict(self):
        self.logger.info("*** Test ***")
        metrics = self.trainer.evaluate(self.test_dataset, metric_key_prefix = "test")
        self.trainer.log_metrics("test", metrics)
        self.trainer.save_metrics("test", metrics)