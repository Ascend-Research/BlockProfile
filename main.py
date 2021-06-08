import argparse
import logging
import sys
import os
from pathlib import Path
from misc import utils
import numpy as np

from comparators import OFAComparator
from constants import SEARCH_SPACE_LIST
from search.EvolutionFinder import EvolutionFinder
from search import arch_constraints

# TODO BIG, split file into "profile.py" and "search.py"

parser = argparse.ArgumentParser("Block Search")
parser.add_argument('-s', '--space', type=str, default="OFAPred", choices=SEARCH_SPACE_LIST, help="Search space")
parser.add_argument('-n', '--num_archs', type=int, default=100, help="Number of archs to sample per specification")
parser.add_argument('-b', '--blocks', type=str, default=None, help="Blocks to test")
parser.add_argument('-f', '--filename', type=str, default=None, help="File where desired blocks may be found")
parser.add_argument('-a', '--all', action='store_true', default=False, help="Evaluate ALL architectures")
parser.add_argument('-d', '--device', type=str, default='cuda:0', help="CPU or CUDA Device")
parser.add_argument('-c', '--comparison', action='store_true', default=False,
                    help="Perform a sweeping comparison (only if args.all is not selected)")
parser.add_argument('-e', '--evolution', action='store_true', default=False, help="Run an evolutionary search")
parser.add_argument('-m', '--metrics', nargs="+", type=int, default=None, help="Pick a subset of metrics")
parser.add_argument('--constraint_type', type=str, default="FLOPS", choices=["FLOPS", "latency"],
                    help="Evolutionary constraint type")
parser.add_argument('--constraint', type=float, default=500, help="Evolutionary constraint magnitude")
parser.add_argument('--constraint_space', type=str, default="OFA_DEFAULT",
                    help="Constrained search space for search")
parser.add_argument('--save', type=str, default='EXP', help="experiment name")
parser.add_argument('--fast', action='store_true', default=False, help="Use fast validation dataloader")
parser.add_argument('--no-log', action='store_true', default=False, help="No logging")
parser.add_argument('--data', type=str, default="~/FastData/ImageNet/", help="Location of ImageNet data")

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

    if args.space == 'OFAPred' or args.space == 'ProxylessPred':
        from search_spaces.OFAPredictorSpace import OFAPredictorSpace as SearchModel
        from search.managers import ProxylessManager as Manager
        if args.space == 'OFAPred':
            logging("Running on Once-For-All Accuracy predictor")
        else:
            from search_spaces.ProxylessPredictorSpace import ProxylessPredictorSpace as SearchModel
            logging("Running on ProxylessNASPredictor Accuracy predictor")
        search_space = SearchModel(logger=logging, metrics=args.metrics, device=args.device,
                                   #resolutions=(160, 176, 192, 208, 224,),
                                   resolutions=(192, 208, 224,),
                                   #resolutions=(224,),
                                   depths=[2, 3, 4])

        space_string = eval("arch_constraints.%s" % args.constraint_space)
        search_manager = Manager(space_string)

    elif args.space == 'OFASupernet' or args.space == "ProxylessSupernet":
        from search_spaces.OFAPredictorSpace import OFAPredictorSpace as SearchModel
        from search.managers import ProxylessManager as Manager
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
                                   depths=[2,3,4],
                                   width=0,
                                   workers=8,
                                   batch_size=250,
                                   fast=args.fast)

        space_string = eval("arch_constraints.%s" % args.constraint_space)
        search_manager = Manager(space_string)

    elif args.space == "ResNet50Supernet":
        # TODO ResNet manager, integration
        from search_spaces.ResNet50Supernet import ResNet50Supernet as SearchModel
        logging("Running on ResNet50 Supernet")
        search_space = SearchModel(logger=logging, metrics=args.metrics,
                                   imagenet_path=args.data,
                                   device=args.device,
                                   batch_size=25,
                                   depth=[0, 1, 2],
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

    elif args.comparison:
        comparator = OFAComparator(search_space, n=args.num_archs)
        if type(args.blocks[0]) is list:
            for block in args.blocks:
                comparator.execute(block)
        else:
            comparator.execute(args.blocks)

    elif args.evolution:
        logging("Running evolutionary search with constraint %s, magnitude %2.4f" %
                     (args.constraint_type, args.constraint))
        evo_finder = EvolutionFinder(search_space, search_manager, constraint_type=args.constraint_type, logger=logging)
        _, best_info = evo_finder.run_evolution_search(constraint=args.constraint)

        logging("Best info:\n{}".format(best_info))

    else:
        if args.filename is not None:
            # TODO implement reading specifications from file
            raise NotImplementedError
        elif args.blocks is not None:
            logging("Specification:")
            result_dict, archs = search_space.fully_evaluate_block(n=args.num_archs, block_list=args.blocks)
            logging(print_archs_summary(search_space, result_dict, archs,
                                             msg=search_space.block_meaning(block_list=args.blocks)))
        else:
            logging("Sampling %d random architectures", args.num_archs)
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
