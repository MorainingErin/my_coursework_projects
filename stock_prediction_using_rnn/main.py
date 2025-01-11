# -*- coding: utf-8 -*-

# - Package imports - #
import argparse
import json
# import datetime
from pathlib import Path
import torch

from dataset import data_clean, create_datasets
from utils import set_random_seed
from worker import Worker


# - Coding Part - #
def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',
                        help='Train or Test',
                        type=str,
                        default='test',
                        choices=['train', 'test'])
    parser.add_argument('--seed',
                        help='Random seed',
                        type=int,
                        default=1901276)
    parser.add_argument('--res_dir',
                        help='Output folder',
                        type=str,
                        default='./res')
    parser.add_argument('--epoch_num',
                        help='Training epoch num',
                        type=int,
                        default=500)
    parser.add_argument('--batch_size',
                        help='Batch size',
                        default=8,
                        type=int)
    parser.add_argument('--lr',
                        help='Learning rate',
                        default=1e-2,
                        type=float)
    parser.add_argument('--lr_step',
                        help='Learning-rate decay step. 0 for ReduceLROnPleateau. '
                             'When negative, lr will increase for lr-searching.',
                        default=0,
                        type=int)
    parser.add_argument('--optimizer',
                        help='Optimizer selection',
                        default='adam',
                        choices=['adam', 'nadam', 'rmsprop'])
    parser.add_argument('--network',
                        help='Network type',
                        default='rnn',
                        choices=['rnn', 'lstm'])
    
    #
    # For experiments
    #
    parser.add_argument('--dataset',
                        help='Dataset choice',
                        type=str,
                        default='fix',
                        choices=['raw', 'fix', 'part'])
    parser.add_argument('--normalize',
                        help='Data normalization method',
                        default='minmax',
                        type=str,
                        choices=['log', 'minmax', 'winminmax', 'returns'])
    parser.add_argument('--seqlen',
                        help='Sequence length for training',
                        default=8,
                        type=int)

    args = parser.parse_args()
    return args


def post_process(args):
    # 1. Set run_tag and res_dir
    args.run_tag = '-'.join([
        f'{args.network}',
        f'{args.lr}({args.lr_step})',
        f'B{args.batch_size}',
        f'{args.dataset}',
        f'{args.normalize}',
        f'{args.seqlen}',
    ])

    # 2. Write json file to out_dir
    args.res_dir = str(Path(args.res_dir) / args.run_tag)
    output_json = Path(args.res_dir) / 'params.json'
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_json), 'w+') as file:
        json.dump(vars(args), file, indent=2)
    
    # 3. Set device
    args.device = (
        'cuda'
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    return args


def main():

    args = get_args()
    args = post_process(args)

    # Set random seed
    set_random_seed(args.seed)

    # Clean dataset or not
    data_folder = Path('./data')
    data_clean(data_folder / 'Google_Stock_Price_Train.csv')
    data_clean(data_folder / 'Google_Stock_Price_Test.csv')

    # Create dataset
    train_dataset, test_dataset = None, None
    if args.mode == 'train':
        eval_split = 0.2
        seqlen = args.seqlen
    elif args.mode == 'test':
        eval_split = None
        seqlen = -1
    else:
        raise NotImplementedError(f'Unknown mode: {args.mode}')
    if args.dataset == 'raw':
        train_dataset, test_dataset = create_datasets(
            data_folder / f'Google_Stock_Price_{args.mode.capitalize()}.csv',
            column_names=['Open', 'High', 'Low', 'Close', 'Volume'],
            device=args.device,
            seq_len=seqlen,
            normalize=args.normalize,
            eval_split=eval_split
        )
    elif args.dataset == 'fix':
        train_dataset, test_dataset = create_datasets(
            data_folder / f'Google_Stock_Price_{args.mode.capitalize()}_Clean.csv',
            column_names=['Open', 'High', 'Low', 'Close', 'Volume'],
            device=args.device,
            seq_len=seqlen,
            normalize=args.normalize,
            eval_split=eval_split
        )
    elif args.dataset == 'part':
        train_dataset, test_dataset = create_datasets(
            data_folder / f'Google_Stock_Price_{args.mode.capitalize()}.csv',
            column_names=['Open', 'High', 'Low', 'Volume'],
            device=args.device,
            seq_len=seqlen,
            normalize=args.normalize,
            eval_split=eval_split
        )
    else:
        raise NotImplementedError(f'Unknown dataset option: {args.dataset}')

    worker = Worker(args, train_dataset, test_dataset)
    worker.init()
    worker.run()
    pass


if __name__ == '__main__':
    main()