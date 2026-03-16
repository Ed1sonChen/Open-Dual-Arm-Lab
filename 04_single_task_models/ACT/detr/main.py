import argparse
import torch

from .models import build_ACT_model, build_CNNMLP_model


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # optimizer
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    # backbone / pos embedding
    parser.add_argument('--backbone', default='resnet18', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--camera_names', default=[], nargs='*')
    parser.add_argument('--masks', action='store_true')

    # transformer
    parser.add_argument('--enc_layers', default=4, type=int)
    parser.add_argument('--dec_layers', default=7, type=int)
    parser.add_argument('--dim_feedforward', default=3200, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--pre_norm', action='store_true')

    # ACT
    parser.add_argument('--kl_weight', default=10, type=float)

    # IMPORTANT: single-arm xArm dims can override these
    parser.add_argument('--state_dim', default=14, type=int)
    parser.add_argument('--action_dim', default=14, type=int)

    return parser


def build_ACT_model_and_optimizer(args_override):
    parser = get_args_parser()
    args = parser.parse_args([])

    for k, v in args_override.items():
        setattr(args, k, v)

    print(f"[build_ACT_model_and_optimizer] state_dim={args.state_dim}, action_dim={args.action_dim}")

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    parser = get_args_parser()
    args = parser.parse_args([])

    for k, v in args_override.items():
        setattr(args, k, v)

    print(f"[build_CNNMLP_model_and_optimizer] state_dim={args.state_dim}, action_dim={args.action_dim}")

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer

