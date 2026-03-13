import torch
from torch import optim


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def split_weights(model, joint_optimizer_lrs):
    """Define parameter groups for PHEModel."""
    return [
        {'params': model.features.parameters(),
         'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
        {'params': model.add_on_layers.parameters(),
         'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
        {'params': model.hash_head.parameters(),
         'lr': joint_optimizer_lrs['hash_head'], 'weight_decay': 1e-3},
        {'params': model.prototypes,
         'lr': joint_optimizer_lrs['prototypes'], 'weight_decay': 1e-3},
    ]


def create_optimizer(args, model, joint_optimizer_lrs=None):
    opt_lower = args.opt.lower() # optimizer name
    weight_decay = args.weight_decay # weight decay coefficient
    parameters = split_weights(model, joint_optimizer_lrs) # per-module learning-rate table
    opt_args = dict(weight_decay=weight_decay) # build optimizer kwargs, e.g. {'weight_decay': 0.05}
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:  # Adam/AdamW epsilon
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None: # Adam/AdamW betas
        opt_args['betas'] = args.opt_betas


    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"


    return optimizer
