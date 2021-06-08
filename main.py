import argparse
import logging
import sys
import os
from pathlib import Path
from misc import utils
import numpy as np

from constants import SEARCH_SPACE_LIST

parser = argparse.ArgumentParser("Block Search")
parser.add_argument('-s', '--space', type=str, default="OFAPred", choices=SEARCH_SPACE_LIST, help="Search space")
parser.add_argument('-n', '--num_archs', type=int, default=100, help="Number of archs to sample per specification")
parser.add_argument('-b', '--blocks', type=str, default=None, help="Blocks to test")
parser.add_argument('-a', '--all', action='store_true', default=False, help="Evaluate ALL architectures")
parser.add_argument('-d', '--device', type=str, default='cuda:0', help="CPU or CUDA Device")
parser.add_argument('-m', '--metrics', nargs="+", type=int, default=None, help="Pick a subset of metrics")
parser.add_argument('--data', type=str, default="/data/ImageNet", help="Location of ImageNet")
parser.add_argument('--fast', action='store_true', default=False, help="Use fast validation dataloader")
parser.add_argument('--save', type=str, default='EXP', help="experiment name")
parser.add_argument('--no-log', action='store_true', default=False, help="No logging")

quantiles = [0.01, 0.05, 0.95, 0.99]


def print_archs_summary(search_space, arch_dict, archs, msg="Random Architectures, "):
    if not args.no_log:
        with open(os.path.join(args.save, "archs.txt"), "a") as arch_file:
            print(msg, file=arch_file)

    for metric in search_space.metrics:
        mu = np.mean(arch_dict[metric])
        sig = np.std(arch_dict[metric])
        msg += "%2.5f, %2.5f, " % (mu, sig)
        for quant in quantiles:
            field_quant = np.quantile(arch_dict[metric], quant)
            msg += "%2.5f, " % field_quant
        field_min = np.min(arch_dict[metric])
        field_max = np.max(arch_dict[metric])
        msg += "%2.5f, %2.5f, " % (field_min, field_max)

        if not args.no_log:
            with open(os.path.join(args.save, "archs.txt"), "a") as arch_file:
                best_arch_for_field = archs[arch_dict[metric].index(field_max)]
                worst_arch_for_field= archs[arch_dict[metric].index(field_min)]
                print("Highest %s arch: %s" % (metric, str(best_arch_for_field)), file=arch_file)
                print("Lowest %s arch: %s" % (metric, str(worst_arch_for_field)), file=arch_file)

    return msg


def make_header(search_space, header_msg="Blocks, "):
    for metric in search_space.metrics:
        header_msg += "%s mu, %s sig, %s 1%%, %s 5%%, %s 95%%, %s 99%%, %s min, %s max, " % \
                      (metric, metric, metric, metric, metric, metric, metric, metric)
    return header_msg


def main(args, logging):

    if args.space == 'OFAPred':
        from search_spaces.OFAPredictorSpace import OFAPredictorSpace as SearchModel
        logging("Running on Once-For-All Accuracy predictor")
        search_space = SearchModel(logger=logging, metrics=args.metrics, device=args.device,
                                   #resolutions=(160, 176, 192, 208, 224,),
                                   resolutions=(224,),
                                   depths=[2, 3, 4])

    elif args.space == 'OFASupernet' or args.space == "ProxylessSupernet":
        if args.space == "OFASupernet":
            from search_spaces.OFASupernet import OFASupernet as SearchModel
            logging("Running on Once-For-All Supernet")
        else:
            from search_spaces.ProxylessSupernet import ProxylessSupernet as SearchModel
            logging("Running on ProxylessNAS Supernet")
        search_space = SearchModel(logger=logging, metrics=args.metrics,
                                   imagenet_path=args.data,
                                   device=args.device,
                                   resolution=224,
                                   depths=[2, 3, 4],
                                   width=2,
                                   batch_size=100,
                                   fast=args.fast)

    elif args.space == "ResNet50Supernet":
        from search_spaces.ResNet50Supernet import ResNet50Supernet as SearchModel
        logging("Running on ResNet50 Supernet")
        search_space = SearchModel(logger=logging, metrics=args.metrics,
                                   imagenet_path=args.data,
                                   device=args.device,
                                   batch_size=25,
                                   fast=args.fast)

    else:
        raise NotImplementedError

    if args.all:
        blocks = search_space.all_blocks()
        header_msg = "Fixed block, "
        logging(make_header(search_space, header_msg=header_msg))

        # First evaluate random architectures
        random_archs = search_space.random_sample(n=args.num_archs)
        random_dict, _ = search_space.fully_evaluate_block(archs=random_archs)
        logging(print_archs_summary(search_space, random_dict, random_archs))

        for block in blocks:
            result_dict, archs = search_space.fully_evaluate_block(n=args.num_archs, block_list=block)
            msg = search_space.block_meaning(block_list=block)
            logging(print_archs_summary(search_space, result_dict, archs, msg=msg))

    else:
        if args.blocks is not None:
            logging("Specification:")
            result_dict, archs = search_space.fully_evaluate_block(n=args.num_archs, block_list=args.blocks)
            logging(print_archs_summary(search_space, result_dict, archs,
                                        msg=search_space.block_meaning(block_list=args.blocks)))
        else:
            logging("Sampling %d random architectures" % args.num_archs)
            archs = search_space.random_sample(n=args.num_archs)
            result_dict, _ = search_space.fully_evaluate_block(archs=archs)
            logging(print_archs_summary(search_space, result_dict, archs))


if __name__ == '__main__':
    args = parser.parse_args()

    if args.no_log:
        main(args, print)
        exit(0)

    # Folder where log, weights, and copy of script is stored
    args.save = Path('logs/{}/{}/'.format(args.space, args.save))
    utils.create_exp_dir(args.save)

    # Logging information, filehandler, etc.
    log_format = '%(asctime)s, %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if args.blocks is not None:
        args.blocks = eval(args.blocks)

    logging.info("args = %s", args)

    main(args, logging.info)
