import os
import time
import math
import torch
import random
import argparse
import datetime
import logging
import numpy as np
from pathlib import Path
from copy import deepcopy
import clip  # Used to load CLIP weights (visual encoder and text encoder)

from data.augmentations import get_transform
from data.datasets import build_dataset
from LTCmodel import LTCModel, ContrastiveLearningViewGenerator
from train_eval import train_one_epoch, evaluate, build_ncm_prototypes
from utils import create_optimizer, create_scheduler, get_logger
from config import (pretrain_path, oxford_pet_root, cub_root, car_root,
                    food_101_root, inaturalist_root, cifar_root, imagenet100m_root)


def get_args_parser():
    parser = argparse.ArgumentParser('LTC training', add_help=False)

    # ========= training parameters =========
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warm_epoch', type=int, default=100)
    parser.add_argument('--topK', type=int, default=100)

    # ========= model parameters =========
    parser.add_argument('--v_dim', type=int, default=512, help='Visual feature dimension')
    parser.add_argument('--t_dim', type=int, default=512, help='Text feature dimension')
    parser.add_argument('--m_dim', type=int, default=512, help='Multimodal fusion feature dimension')
    parser.add_argument('--fusion_hidden', type=int, default=1024, help='Hidden dimension in fusion head')
    parser.add_argument('--fusion_type', type=str, default='concat', choices=['mean', 'concat', 'attn'], help='Fusion method type')

    # Whether the prototype layer includes an explicit "unknown" class (this repo mainly uses threshold-based growing, so usually False)
    parser.add_argument('--include_unknown', action='store_true', default=False)
    parser.add_argument('--n_kv', type=int, default=8, help='Number of key/value tokens in cross-attention')
    parser.add_argument('--n_queries', type=int, default=16, help='Number of query tokens in cross-attention')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--ff_ratio', type=int, default=4, help='Feed-forward ratio in transformer')

    # ========= loss balancing / regularization =========
    parser.add_argument('--alpha', type=float, default=0.3)   # SupCon weight
    parser.add_argument('--beta', type=float, default=0.3)    # reserved
    parser.add_argument('--lamda', type=float, default=1.0)   # reserved

    # ========= Max-Margin + adaptive tau (aligned with train_eval) =========
    parser.add_argument('--gamma_mm', type=float, default=0.05, help='max-margin loss weight')
    parser.add_argument('--q_pos', type=float, default=0.20, help='quantile for known s_max')
    parser.add_argument('--q_neg', type=float, default=0.80, help='quantile for pseudo-unknown s_max')
    parser.add_argument('--m_pos', type=float, default=0.05, help='positive side margin')
    parser.add_argument('--m_neg', type=float, default=0.05, help='negative side margin')
    parser.add_argument('--tau_beta', type=float, default=0.010, help='EMA smoothing for tau')
    parser.add_argument('--tau_train', type=float, default=0.7, help='tau init at training')
    parser.add_argument('--tau_eval', type=float, default=0.7, help='fallback tau at eval if tau_ema missing')

    # ========= MKEE control =========
    parser.add_argument('--p_mkee', type=float, default=0.30, help='probability to inject MKEE batch')
    parser.add_argument('--warmup_epochs_mkee', type=int, default=3, help='epochs to warmup before MKEE starts')
    parser.add_argument('--mkee_alpha', type=float, default=0.4)
    parser.add_argument('--mkee_eps', type=float, default=0.05)
    parser.add_argument('--mkee_sigma', type=float, default=0.5)
    parser.add_argument('--mkee_lam1', type=float, default=0.1)
    parser.add_argument('--mkee_sigma_mult', type=float, default=1.0)

    # ========= augmentation parameters =========
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--interpolation', type=int, default=3)  # PIL.BICUBIC
    parser.add_argument('--crop_pct', type=float, default=0.875)

    # ========= Optimizer & LR schedule =========
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
    parser.add_argument('--opt-eps', type=float, default=1e-8, metavar='EPSILON')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV')
    parser.add_argument('--warmup-lr', type=float, default=1e-4, metavar='LR')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR')
    parser.add_argument('--decay-epochs', type=float, default=10, metavar='N')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE')

    # ========= Dataset parameters =========
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--mask_theta', type=float, default=0.1)
    parser.add_argument('--labeled_nums', type=int, default=0)
    parser.add_argument('--unlabeled_nums', type=int, default=0)
    parser.add_argument('--data_set', default='cub',
                        choices=['cub', 'scars', 'food', 'pets',
                                 'Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida',
                                 'Aves', 'Chromista', 'Fungi', 'Insecta', 'Mammalia',
                                 'Mollusca', 'Plantae', 'Protozoa', 'Reptilia',
                                 'cifar10', 'cifar100', 'imagenet100m'],
                        type=str, help='dataset name')

    parser.add_argument('--output_dir', default='exp/', help='path to save logs/checkpoints')
    parser.add_argument('--device', default='cuda', help='device for train/test')
    parser.add_argument('--seed', type=int, default=1028)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N')
    parser.add_argument('--eval', action='store_true', help='evaluation only')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--pin-mem', action='store_true')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


def get_outlog(args):
    if args.eval:
        logfile_dir = os.path.join(args.output_dir, "eval-logs")
    else:
        logfile_dir = os.path.join(args.output_dir, "train-logs")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    tb_dir = os.path.join(args.output_dir, "tf-logs")
    tb_log_dir = os.path.join(tb_dir, args.data_set)
    os.makedirs(logfile_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    log_name = (
        f"k+merge{args.data_set}_"
        f"a{args.alpha}_"
        f"gmm{getattr(args,'gamma_mm',0.0)}_"
        f"tb{getattr(args,'tau_train',0.7):.2f}_"
        f"pt{getattr(args,'p_mkee',0.3):.2f}_"
        f"wm{getattr(args,'warmup_epochs_mkee',3)}"
    )
    logger = get_logger(
        level=logging.INFO,
        mode="w",
        name=None,
        logger_fp=os.path.join(logfile_dir, log_name)
    )
    return logger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    # ============ seed & logger ============
    set_seed(args.seed)
    logger = get_outlog(args)
    logger.info("Start running with args: \n{}".format(args))
    device = torch.device(args.device)

    # ============ dataset ============
    dataset_train, dataset_val, test_dataset_unlabelled = build_dataset(args=args)
    args.known_num_classes = len(args.known_classes)  # known_classes is set by build_dataset

    # Two-view training augmentation (CLIP-style) + test augmentation
    train_tf, test_tf = get_transform('clip', image_size=args.image_size, args=args)
    dataset_train.transform = ContrastiveLearningViewGenerator(train_tf, n_views=2)
    dataset_val.transform = test_tf
    test_dataset_unlabelled.transform = test_tf

    logger.info("train {} | val {} | test_unlabeled {}".format(
        len(dataset_train), len(dataset_val), len(test_dataset_unlabelled)))

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    test_loader_unlabelled = torch.utils.data.DataLoader(
        test_dataset_unlabelled,
        num_workers=8,
        batch_size=256,
        shuffle=False,
        pin_memory=False
    )

    # === Single-view training loader (uses test_tf) for building class-mean prototypes ===
    centroid_ds = deepcopy(dataset_train)
    centroid_ds.transform = test_tf  # single-view, consistent with evaluation
    centroid_loader = torch.utils.data.DataLoader(
        centroid_ds,
        num_workers=args.num_workers,
        batch_size=256,
        shuffle=False,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # ============ model ============
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, _ = clip.load("ViT-B/16", device=device)
    clip_model = clip_model.float()  # use FP32 to avoid dtype mismatches
    model = LTCModel(args, clip_model).to(device)

    # ============ optimizer & scheduler ============
    joint_optimizer_lrs = {
        'text_mapper':     1e-2,
        'fusion':          1e-2,
        'proto':           1e-2,
        'clip_visual_last':1e-2,
    }
    optimizer = create_optimizer(args, model, joint_optimizer_lrs=joint_optimizer_lrs)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: {}'.format(n_parameters))
    for i, pg in enumerate(optimizer.param_groups):
        cfg = {k: v for k, v in pg.items() if k != 'params'}
        print(f"[Optimizer] Param Group {i} cfg: {cfg}")

    lr_scheduler, _ = create_scheduler(args, optimizer)

    # ============ resume ============
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if (not args.eval) and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        # Restore tau_ema for subsequent evaluation / continued training
        if 'tau_ema' in checkpoint:
            args.tau_ema = float(checkpoint['tau_ema'])
            logger.info(f"[Resume] loaded tau_ema={args.tau_ema:.4f} from checkpoint")

    # ============ train loop ============
    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(args.start_epoch, args.epochs):
        # Train one epoch (CE + SupCon + MaxMargin + quantile/EMA tau + MKEE warmup & sampling)
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_norm=args.clip_grad,
            args=args,
            set_training_mode=True
        )

        # Persist the epoch-updated tau_ema into args; log tau evolution and quantiles
        if 'tau_ema' in train_stats:
            args.tau_ema = float(train_stats['tau_ema'])
        tau_print   = train_stats.get('tau_ema', np.nan)
        tau_target  = train_stats.get('tau_target', np.nan)
        u_pos_print = train_stats.get('u_pos', np.nan)
        u_neg_print = train_stats.get('u_neg', np.nan)
        lpos_print  = train_stats.get('loss_mm_pos', np.nan)
        lneg_print  = train_stats.get('loss_mm_neg', np.nan)

        logger.info(
            f"[Epoch {epoch}] tau_ema={tau_print:.4f} | tau_target={tau_target:.4f} | "
            f"u_pos={u_pos_print:.4f} | u_neg={u_neg_print:.4f} | "
            f"L_pos={lpos_print:.4f} | L_neg={lneg_print:.4f}"
        )
        logger.info("Averaged stats:")
        logger.info(train_stats)

        lr_scheduler.step(epoch)

        # Save a lightweight checkpoint (includes tau_ema)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'tau_ema': float(getattr(args, "tau_ema", args.tau_eval)),
        }, os.path.join(ckpt_dir, f'last.pth'))

        # === Rebuild NCM prototypes using the current model + single-view training set ===
        ncm_protos = build_ncm_prototypes(
            train_loader_singleview=centroid_loader,
            model=model,
            device=device,
            known_classes=args.known_classes,
            save_path=None
        )

        # === Evaluate: use the trained tau_ema (fallback to tau_eval if missing) ===
        tau_eval_use = float(getattr(args, "tau_ema", getattr(args, "tau_eval", 0.7)))
        eval_stats = evaluate(
            test_loader=test_loader_unlabelled,
            model=model,
            device=device,
            known_classes=args.known_classes,
            tau=tau_eval_use,
            print_logits=False,               # set True when debugging samples
            protos_override=ncm_protos,
            topK=args.topK
        )
        logger.info(f"[Epoch {epoch}] Eval stats: {eval_stats['pretty']}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LTC training', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Dataset root configuration
    valid_super_categories = [
        'Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida',
        'Aves', 'Chromista', 'Fungi', 'Insecta', 'Mammalia',
        'Mollusca', 'Plantae', 'Protozoa', 'Reptilia'
    ]
    if args.data_set == 'cub':
        args.data_root = cub_root
    elif args.data_set == 'scars':
        args.data_root = car_root
    elif args.data_set == 'food':
        args.data_root = food_101_root
    elif args.data_set == 'pets':
        args.data_root = oxford_pet_root
    elif args.data_set == 'cifar10' or args.data_set == 'cifar100':
        args.data_root = cifar_root
    elif args.data_set == 'imagenet100m':
        args.data_root = imagenet100m_root
    elif args.data_set in valid_super_categories:
        args.data_root = inaturalist_root

    args.pretrain_path = pretrain_path

    main(args)
