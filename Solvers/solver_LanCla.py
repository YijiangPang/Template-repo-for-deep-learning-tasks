from Optimizers.Optimizer_loader import optimizer_loader, lr_scheduler_loader
from Data.datasets_glue import init_data, data_preprocess
from Solvers.solver_Base import solver_Base
from Models.m_LanCla import pre_model
import torch
import math
import numpy as np
import os
from transformers import Trainer


class solver_LanCla(solver_Base):
    
    def __init__(self, logger, model_args, data_args, training_args, proj_args, seed_c):
        self.logger = logger
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.proj_args = proj_args
        self.set_seed(seed_c)
        
        raw_datasets, label_list = init_data(self.model_args, self.data_args, self.training_args)
        model, tokenizer, config = pre_model(self.model_args, self.data_args)
        raw_datasets, train_dataset, eval_dataset, predict_dataset, compute_metrics, data_collator = data_preprocess(model, tokenizer, config, logger, self.model_args, self.data_args, self.training_args, raw_datasets, label_list)
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.predict_dataset = predict_dataset
        self.raw_datasets = raw_datasets
        self.label_list = label_list

        total_steps = math.ceil(len(self.train_dataset)/(self.training_args.per_device_train_batch_size*torch.cuda.device_count())) * self.training_args.num_train_epochs
        optimizer = optimizer_loader(self.proj_args.opt_name, model, self.training_args.learning_rate, 
                                     self.proj_args.betas, weight_decay = self.training_args.weight_decay)
        lr_scheduler = lr_scheduler_loader(self.training_args.lr_scheduler_type, optimizer, T_max = total_steps)
        optimizers = (optimizer, lr_scheduler)

        self.trainer = Trainer(model, 
                            self.training_args,
                            train_dataset=train_dataset if self.training_args.do_train else None,
                            eval_dataset=eval_dataset if self.training_args.do_eval else None,
                            tokenizer=tokenizer,
                            compute_metrics=compute_metrics,
                            optimizers = optimizers
                        )

    def do_train(self):
        train_result = self.trainer.train()
        metrics = train_result.metrics
        max_train_samples = (self.data_args.max_train_samples if self.data_args.max_train_samples is not None else len(self.train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))
        if self.proj_args.save_strategy_ == "final": self.trainer.save_model()
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

    def do_eval(self):
        self.logger.info("*** Evaluate ***")
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [self.data_args.task_name]
        eval_datasets = [self.eval_dataset]
        if self.data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = self.raw_datasets["validation_mismatched"]
            if self.data_args.max_eval_samples is not None:
                max_eval_samples = min(len(valid_mm_dataset), self.data_args.max_eval_samples)
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = self.trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (self.data_args.max_eval_samples if self.data_args.max_eval_samples is not None else len(eval_dataset))
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            self.trainer.log_metrics("eval", metrics)
            self.trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    def do_predict(self):
        self.logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [self.data_args.task_name]
        predict_datasets = [self.predict_dataset]
        if self.data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(self.raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = self.trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if self.data_args.is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(self.training_args.output_dir, f"predict_results_{task}.txt")
            if self.trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    self.logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if self.data_args.is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = self.label_list[item]
                            writer.write(f"{index}\t{item}\n")


    def finalize(self):
        self.proj_args.task_name = self.data_args.task_name
        solver_Base.finalize(self)
