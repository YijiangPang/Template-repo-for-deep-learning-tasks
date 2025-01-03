from Optimizers.Optimizer_loader import optimizer_loader, lr_scheduler_loader
from Data.datasets_CL import init_data
from Solvers.solver_Base import solver_Base
from Models.m_CL import pre_model
import torch
import math
from transformers import Trainer
# from Solvers.Trainer import TrainerCustomized as Trainer


class TrainerCL(Trainer):
    def __init__(self, **kwargs):
        Trainer.__init__(self, **kwargs)
    def handle_batch_data(self, examples):
        img, text, mask, return_loss = examples["pixel_values"], examples["input_ids"], examples["attention_mask"], examples["return_loss"]
        img, text, mask = img.to(self.device), text.to(self.device), mask.to(self.device)
        outputs = self.model(pixel_values = img, input_ids = text, attention_mask = mask, return_loss = return_loss)
        loss = outputs.loss
        return loss, None, None
    

class solver_CL(solver_Base):
    
    def __init__(self, logger, model_args, data_args, training_args, proj_args, seed_c):
        self.logger = logger
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.proj_args = proj_args
        self.set_seed(seed_c)
        
        model, tokenizer = pre_model(self.model_args, self.data_args)
        # self.data_args.img_size = model.config.vision_config.image_size
        self.train_dataset, self.eval_dataset, self.test_dataset, collate_fn = init_data(self.data_args, self.model_args, tokenizer)
        
        total_steps = math.ceil(len(self.train_dataset)/(self.training_args.per_device_train_batch_size*torch.cuda.device_count())) * self.training_args.num_train_epochs
        optimizer = optimizer_loader(self.proj_args.opt_name, model, self.training_args.learning_rate, 
                                     betas = self.proj_args.betas, weight_decay = self.training_args.weight_decay)
        lr_scheduler = lr_scheduler_loader(self.training_args.lr_scheduler_type, optimizer, T_max = total_steps)
        optimizers = (optimizer, lr_scheduler)

        self.trainer = TrainerCL(model=model,
                                args=self.training_args,
                                train_dataset=self.train_dataset if self.training_args.do_train else None,
                                eval_dataset=self.eval_dataset if self.training_args.do_eval else None,
                                data_collator = collate_fn,
                                optimizers = optimizers,
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