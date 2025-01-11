# -*- coding: utf-8 -*-

# - Package imports - #
import argparse
import json
# import datetime
from pathlib import Path
import torch

from dataset import CIFARDataset
from worker import Worker


# - Coding Part - #
def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',
                        help='Train or Test',
                        type=str,
                        default='test',
                        choices=['train', 'test'])
    parser.add_argument('--dataset',
                        help='Split or Full',
                        type=str,
                        default='split',
                        choices=['full', 'split'])
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
                        default=128,
                        type=int)
    parser.add_argument('--lr',
                        help='Learning rate',
                        default=1e-2,
                        type=float)
    parser.add_argument('--lr_step',
                        help='Learning-rate decay step. 0 for no decay. '
                             'When negative, lr will increase for lr-searching.',
                        default=0,
                        type=int)
    parser.add_argument('--optimizer',
                        help='Optimizer selection',
                        default='adam',
                        choices=['adam', 'nadam', 'rmsprop'])
    parser.add_argument('--network',
                        help='Network type',
                        default='lenet',
                        choices=['vgg', 'lenet', 'resnet'])
    parser.add_argument('--dropout',
                        help='dropout rate',
                        default=0.0,
                        type=float)
    parser.add_argument('--data_aug',
                        help='Data augmentation',
                        default='none',
                        type=str,
                        choices=['none', 'geo', 'color', 'all'])
    args = parser.parse_args()
    return args


def post_process(args):
    # 1. Set run_tag and res_dir
    args.run_tag = f'{args.network}-{args.lr}({args.lr_step})B{args.batch_size}-{args.dropout}-{args.data_aug}'

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

    # Create dataset
    data_folder = Path('./cifar-10-batches-py')
    meta_file = data_folder / 'batches.meta'
    train_dataset, test_dataset = None, None
    if args.dataset == 'split':   # Validation
        train_dataset = CIFARDataset([data_folder / f'data_batch_{i}' for i in [1, 2, 3, 4]],
                                     device=args.device, data_aug=args.data_aug, meta_file_name=meta_file)
        test_dataset = CIFARDataset([data_folder / 'data_batch_5'],
                                     device=args.device, meta_file_name=meta_file)
    elif args.dataset == 'full':  # Test
        train_dataset = CIFARDataset([data_folder / f'data_batch_{i}' for i in [1, 2, 3, 4, 5]],
                                     device=args.device, data_aug=args.data_aug, meta_file_name=meta_file)
        test_dataset = CIFARDataset([data_folder / 'test_batch'],
                                     device=args.device, meta_file_name=meta_file)
    
    worker = Worker(args, train_dataset, test_dataset)
    worker.init()
    worker.run()
    pass


if __name__ == '__main__':
    main()