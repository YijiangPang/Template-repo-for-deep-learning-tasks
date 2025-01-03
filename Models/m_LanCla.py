from transformers import GPT2ForSequenceClassification, LlamaForSequenceClassification, AutoModelForSequenceClassification, GPT2Tokenizer, LlamaTokenizer
from transformers import AutoConfig,AutoTokenizer


def pre_model(model_args, data_args):
    if model_args.model_name_or_path == "gpt2":
        model_id = "gpt2"
    elif model_args.model_name_or_path == "llama3":
        model_id = "meta-llama/Meta-Llama-3-8B"
    else:
        model_id = model_args.model_name_or_path

    config = AutoConfig.from_pretrained(
                    model_id,
                    num_labels=data_args.num_labels,
                    finetuning_task=data_args.task_name,
                    cache_dir=model_args.cache_dir_model,
                    revision=model_args.model_revision,
                    token=model_args.token,
                    trust_remote_code=model_args.trust_remote_code
                    )

    if model_id == "gpt2":
        if model_args.flag_pretrain:
            model = GPT2ForSequenceClassification.from_pretrained("gpt2", config=config)
        else:
            model = GPT2ForSequenceClassification(config=config)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", config=config)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    elif model_id == "meta-llama/Meta-Llama-3-8B":
        if model_args.flag_pretrain:
            model = LlamaForSequenceClassification.from_pretrained(model_id, config=config)
        else:
            model = LlamaForSequenceClassification(config=config)
        tokenizer = LlamaTokenizer.from_pretrained("model_id", config=config)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        if model_args.flag_pretrain:
            model = AutoModelForSequenceClassification.from_pretrained(model_id, config=config)
        else:
            model = AutoModelForSequenceClassification.from_config(config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name or model_id, config=config)
    # Define a padding token
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model.config.pad_token_id = tokenizer.pad_token_id

    # if not model_args.flag_pretrain:
    #     for _, m in model.named_modules():
    #         if hasattr(m, 'reset_parameters'):
    #             m.reset_parameters()

    return model, tokenizer, config