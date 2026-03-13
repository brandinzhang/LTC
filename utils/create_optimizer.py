import torch
from torch import optim

def split_weights(model, joint_optimizer_lrs):
    """
    Define parameter groups for LTCModel (minimal-change version, consistent with the original repo style).
    Expects joint_optimizer_lrs = {
        'text_mapper': ...,
        'fusion': ...,
        'proto': ...
    }
    """
    param_groups = []

    # 1) CrossAttentionMapper (generate pseudo text)
    if hasattr(model, 'text_mapper') and model.text_mapper is not None:
        param_groups.append({
            'params': model.text_mapper.parameters(),
            'lr': joint_optimizer_lrs['text_mapper'],
            'weight_decay': 1e-3,   # consistent with the original repo style
        })

    # 2) fusion head
    if hasattr(model, 'fusion') and model.fusion is not None:
        param_groups.append({
            'params': model.fusion.parameters(),
            'lr': joint_optimizer_lrs['fusion'],
            'weight_decay': 1e-3,
        })

    # 3) prototype layer
    if hasattr(model, 'proto') and model.proto is not None:
        param_groups.append({
            'params': model.proto.parameters(),   # parameters() includes the prototypes
            'lr': joint_optimizer_lrs['proto'],
            'weight_decay': 1e-3,
        })
    clip_vis_trainables = [p for p in model.clip.visual.parameters() if p.requires_grad]
    if len(clip_vis_trainables) > 0:
        param_groups.append({
            'params': clip_vis_trainables,
            'lr': joint_optimizer_lrs.get('clip_visual_last', 1e-5),  # recommended to be small
            'weight_decay': 1e-3
        })

    # 4) fallback: trainable parameters not covered by the groups above
    covered = set()
    for g in param_groups:
        for p in g['params']:
            covered.add(id(p))
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in covered]
    if len(rest) > 0:
        # use fusion lr as fallback (could also use text_mapper lr)
        param_groups.append({
            'params': rest,
            'lr': joint_optimizer_lrs.get('fusion', 1e-3),
            'weight_decay': 1e-3,
        })

    return param_groups


def create_optimizer(args, model, joint_optimizer_lrs=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    # === Same as the original: build parameter groups first (each group already has lr/weight_decay) ===
    parameters = split_weights(model, joint_optimizer_lrs)

    # remaining hyperparameters
    opt_args = dict(weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    # build optimizer
    if opt_lower in ('sgd', 'nesterov'):
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        raise ValueError(f"Invalid optimizer: {args.opt}")

    return optimizer
