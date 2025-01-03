import torch


def optimizer_loader(name, m, lr, betas, weight_decay):
    p_train = [p for p in m.parameters() if p.requires_grad == True]
    if name == "Adam":
        opt = torch.optim.Adam(p_train, lr, betas = betas, weight_decay = weight_decay)
        # from Optimizers.Adam import Adam
        # opt = Adam(p_train, lr, betas = betas, weight_decay = weight_decay)
    elif name == "AdamW":
        opt = torch.optim.AdamW(p_train, lr, betas = betas, weight_decay = weight_decay)
        # from Optimizers.Adam import AdamW
        # opt = AdamW(p_train, lr, betas = betas, weight_decay = weight_decay)
    return opt


def lr_scheduler_loader(name, optimizer, T_max):
    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = T_max)
    elif name == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer = optimizer, factor=1.0, total_iters = T_max)
    return scheduler
