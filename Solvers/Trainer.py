from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import torch
import numpy as np
import os
import torch.nn as nn
from Utils.utils import dotdict
from transformers import Trainer
from transformers.utils import WEIGHTS_NAME
import math


class TrainerCustomized:

    from transformers.trainer_pt_utils import log_metrics, metrics_format, save_metrics
    _save = Trainer._save   #save model
    _load_from_checkpoint = Trainer._load_from_checkpoint
    _issue_warnings_after_load = Trainer._issue_warnings_after_load
    
    loss_func = torch.nn.functional.cross_entropy
    is_world_process_zero = lambda self : True
    is_fsdp_enabled = False

    def __init__(self, model, args, train_dataset, eval_dataset,
                        data_collator, optimizers, compute_metrics = None):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        (self.optimizer, self.lr_scheduler) = optimizers
        self.compute_metrics = compute_metrics
        self.init_env()

    def init_env(self):
        total_steps = math.ceil(len(self.train_dataset)/(self.args.per_device_train_batch_size*torch.cuda.device_count())) * self.args.num_train_epochs
        if self.train_dataset is not None:
            if self.optimizer is None:
                self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad == True], self.args.learning_rate)
            if self.lr_scheduler is None:
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = self.optimizer, T_max = total_steps)

            self.dataloader_train = DataLoader(dataset=self.train_dataset, batch_size=self.args.per_device_train_batch_size, num_workers = self.args.dataloader_num_workers, 
                                            collate_fn = self.data_collator, drop_last=True, shuffle=True, pin_memory=True, worker_init_fn = np.random.seed(self.args.seed))
        self.dataloader_eval = DataLoader(dataset=self.eval_dataset, batch_size=self.args.per_device_eval_batch_size, num_workers = self.args.dataloader_num_workers, 
                                          collate_fn = self.data_collator, drop_last=False, shuffle=False, pin_memory=True, worker_init_fn = np.random.seed(self.args.seed))

        self.state = {"epoch": self.args.num_train_epochs, "global_step": total_steps, "log_history":[]} if self.train_dataset is not None else {"epoch": self.args.num_train_epochs, "log_history":[]}

        self.model = self.model_parallel(self.model)
        self.train_result = dotdict()
        os.makedirs(self.args.output_dir, exist_ok=True)

        if self.args.resume_from_checkpoint is not None: 
            self._load_from_checkpoint(self.args.resume_from_checkpoint)

    def train(self):
        pbar = tqdm(initial = 0, total = len(self.dataloader_train)*self.args.num_train_epochs)
        iter, loss_all = 0, []
        for epoch in range(0, int(self.args.num_train_epochs), 1):
            self.model.train()       #model.train() doesn’t change param.requires_grad
            num_batch = len(self.dataloader_train)
            for batch_idx, examples in enumerate(self.dataloader_train):
                loss, _, _ = self.handle_batch_data(examples)
                self.optimizer.zero_grad()
                loss.backward()  
                self.optimizer.step()      
                self.lr_scheduler.step()

                loss_all.append(loss.item())
                pbar.update(1)
                pbar.set_description("loss=%.3f"%(loss.item()))
                self.func_batch_iter_called(loss, epoch, iter, flag_new_epoch = (batch_idx + 1) >= num_batch)
                iter = iter + 1
        pbar.close()
        self.train_result.metrics= {"train_loss":np.mean(loss_all)}
        return self.train_result
    
    def handle_batch_data(self, examples):
        x, y = examples["pixel_values"], examples["labels"]
        x = x.to(self.device).float()
        y = y.to(self.device).long()
        outputs = self.model(x, y) 
        loss, logits = outputs.loss, outputs.logits
        return loss, logits, y
    
    def func_batch_iter_called(self, loss, epoch, iter, flag_new_epoch):

        if self.args.eval_strategy == "steps" and iter%self.args.eval_steps == 0:
            metrics = {"step": iter} | self.evaluate()
            print(", ".join(["%s: %f"%(k, v) for k, v in metrics.items()]))
            self.state["log_history"].append(metrics)
        elif self.args.eval_strategy == "epoch" and flag_new_epoch:
            metrics = {"epoch": epoch} | self.evaluate()
            print(", ".join(["%s: %f"%(k, v) for k, v in metrics.items()]))
            self.state["log_history"].append(metrics)

        if self.args.logging_strategy == "steps" and iter%self.args.logging_steps == 0:
            metrics = {"step": iter, "loss":loss.item(), "learning_rate": self.lr_scheduler.get_lr()[-1]}
            print(", ".join(["%s: %f"%(k, v) for k, v in metrics.items()]))
            self.state["log_history"].append(metrics)

        if self.args.logging_strategy == "epoch" and flag_new_epoch:
            metrics = {"epoch": epoch, "loss":loss.item(), "learning_rate": self.lr_scheduler.get_lr()[-1]}
            print(", ".join(["%s: %f"%(k, v) for k, v in metrics.items()]))
            self.state["log_history"].append(metrics)

    def evaluate(self, eval_dataset = None, metric_key_prefix = "eval"):
        if eval_dataset is not None:
            dataloader_eval = DataLoader(dataset=eval_dataset, batch_size=self.args.per_device_eval_batch_size, num_workers = self.args.dataloader_num_workers, 
                                          collate_fn = self.data_collator, drop_last=False, shuffle=False, pin_memory=True, worker_init_fn = np.random.seed(self.args.seed))
        else:
            dataloader_eval = self.dataloader_eval
        self.model.eval()
        with torch.no_grad():
            y_all, yhat_all, loss_all = [], [], []
            for id_batch, examples in enumerate(dataloader_eval):
                loss, logits, y = self.handle_batch_data(examples)
                loss_all.append(loss.item())
                if logits is not None: yhat_all.append(logits.data.cpu().numpy())
                if y is not None: y_all.append(y.data.cpu().numpy())

        if len(y_all) > 0: y_all = np.concatenate(y_all, axis=0)
        if len(yhat_all) > 0: yhat_all = np.concatenate(yhat_all, axis=0)

        metrics_loss = {"loss" if metric_key_prefix is None else "%s_loss"%(metric_key_prefix): np.mean(loss_all)}
        if self.compute_metrics is not None:
            m = self.compute_metrics((yhat_all, y_all))
            metrics_defined = {"%s"%(k) if metric_key_prefix is None else "%s_%s"%(metric_key_prefix, k):v for k,v in m.items()}
            metrics =  metrics_defined | metrics_loss
        else:
            metrics = metrics_loss
        return metrics

    def save_state(self):
        with open(os.path.join(self.args.output_dir, 'trainer_state.json'), 'w', encoding='utf-8') as fout:
            json.dump(self.state, fout, ensure_ascii=False, indent=4)

    def model_parallel(self, model):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            model = model.to(self.device)
        return model.to(self.device)
    
    def save_model(self):
        state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        torch.save(state_dict, os.path.join(self.args.output_dir, WEIGHTS_NAME))
        print("state_dict is saved as %s"%(os.path.join(self.args.output_dir, WEIGHTS_NAME)))