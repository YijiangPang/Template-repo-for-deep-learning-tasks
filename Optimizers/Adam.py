import math
import torch
import torch.nn.functional as F

class Adam(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    self.state_init(group, state, p)

                state['step'] += 1
                self.params_update(p, grad, group, state)
        return loss

    def params_update(self, param, grad, group, state):
        exp_avg, exp_avg_sq, step = state['exp_avg'], state['exp_avg_sq'], state['step']
        (beta1, beta2), lr, eps, gamma = group['betas'], group['lr'], group['eps'], group['weight_decay']

        if gamma != 0: grad = grad.add(gamma, param.data)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg = exp_avg/(1 - beta1 ** step)

        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)
        denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** step)).add_(eps)

        param.data.addcdiv_(exp_avg, denom, value = -lr)

    def state_init(self, group, state, p):
        state['step'] = 0
        state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
        state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

class AdamW(Adam):
    def __init__(self, params, lr=1.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        Adam.__init__(self, params, lr, betas, eps, weight_decay)

    def params_update(self, param, grad, group, state):
        exp_avg, exp_avg_sq, step = state['exp_avg'], state['exp_avg_sq'], state['step']
        (beta1, beta2), lr, eps, weight_decay = group['betas'], group['lr'], group['eps'], group['weight_decay']

        # Perform stepweight decay
        param.data.mul_(1 - lr * weight_decay)

        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg = exp_avg/(1 - beta1 ** step)

        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)
        denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** step)).add_(eps)

        param.data.addcdiv_(exp_avg, denom, value = -lr)