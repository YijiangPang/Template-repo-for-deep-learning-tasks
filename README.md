# Template for deep learning project
This repo is built upon [huggingface-transformers-examples-pytorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch) and serves as a starting point for various deep learning tasks with following highlighted features:

* debuggable almost everywhere supported by customized ```Solver\Trainer.py``` and optimizer
* easy switch between integrated and customized configuration & model & dataset & optimizer & trainer & solver (new method)
* the framework incooperates popular image classification tasks, language classification tasks, and contrastive learning tasks
* for each run, the results are saved in checkpoint/current_time_folder. 

Note:
* customized trainer is enabled by uncomment `from Solvers.Trainer import TrainerCustomized as Trainer' in scuh as ```Solvers\solver_ImgCla.py```
* ```main.py``` demonstrates how to switch between single-setting & multi-settings run. 

# Image classification tasks
```
CUDA_VISIBLE_DEVICES=0 python main.py ImgCla --do_train --do_eval --do_predict --output_dir checkpoints --save_strategy no --save_strategy_ no --report_to none --proj_seed 123 --dataloader_num_workers 4 --dataset_name uoft-cs/cifar10 --img_size 32 --model_name_or_path resnet18 --flag_pretrain True  --eval_strategy steps --eval_steps 10 --logging_strategy steps --logging_steps 10 --per_device_train_batch_size 1024 --per_device_eval_batch_size 1024 --num_train_epochs 20 --num_run 3 --opt_name AdamW --lr_scheduler_type cosine --learning_rate 1e-3
```
where
* '--model_name_or_path' supports {densenet121, resnet18, vgg11, vit_B_16, openai_clip_RN50 (RN50 --> RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px)}
* '--dataset_name' supports [huggingface-datasets](https://huggingface.co/datasets) such as {uoft-cs/cifar10, ylecun/mnist}


# Language classification tasks
```
CUDA_VISIBLE_DEVICES=0 python main.py LanCla --do_train --do_eval --do_predict --save_strategy no --save_strategy_ no --report_to none --output_dir checkpoints --proj_seed 123 --model_name_or_path bert-base-uncased --flag_pretrain True --eval_strategy steps --eval_steps 5 --logging_strategy steps --logging_steps 5 --max_seq_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --num_train_epochs 5 --num_run 3 --opt_name AdamW --lr_scheduler_type cosine --learning_rate 5e-5 --task_name mrpc
```
where 
* '--model_name_or_path' supports {bert-base-uncased, distilbert-base-uncased, gpt2, llama3}
* '--task_name' supports {mrpc, wnli, cola, stsb, rte, sst2, qqp, mnli, qnli}


# Contrastive learning tasks
```
CUDA_VISIBLE_DEVICES=3 python main.py CL --do_train --do_eval --do_predict --output_dir checkpoints --save_strategy no --save_strategy_ no --report_to none --proj_seed 123 --dataloader_num_workers 4 --dataset_name nlphuji/flickr30k --img_size 224 --model_name_img resnet18 --model_name_text bert-base-uncased --flag_pretrain True --max_seq_length 77 --eval_strategy epoch --logging_strategy epoch --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --num_train_epochs 30 --num_run 3 --opt_name AdamW --lr_scheduler_type cosine --learning_rate 1e-4
```
where 
* '--dataset_name' support {coco, nlphuji/flickr30k, hongrui/mimic_chest_xray_v_1, sxj1215/mmimdb}
* '--model_name_img' supports {resnet18, resnet50, vit_B_16, CLIP_openai_clip_RN50 (RN50 --> RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px)}
* '--model_name_text' supports {bert-base-uncased, distilbert-base-uncased}. Note: invalid when choosing CLIP_openai_clip_?
    
Note:
* '--train_valid_test_split' is applied when choosing datasets hongrui/mimic_chest_xray_v_1 and sxj1215/mmimdb