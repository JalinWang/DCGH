import argparse
from datetime import datetime
import os
import random
import torch
from loguru import logger

from main import train
from data.data_loader import load_data

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results

print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

def run():
    args = load_config()

    logdir = f"logs/DCGH-{args.dataset}-{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}"

    message = f"manualSeed={manualSeed}, alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}"


    # Config dataset here for convenience. For automated scripts, comment following codes out and modify `load_config()`.
    args.dataset = 'cifar-10'
    # args.dataset = 'nus-wide-tc21'
    # args.dataset = 'flickr25k'

    # Following args will be changed according to the dataset
    # args.num_samples = 10500
    # args.num_samples = 5000
    # args.max_iter = 50 
    # args.topk = 5000

    # dataset_root_path = "./dataset"
    dataset_root_path = "/data/wjn/dataset"
    if args.dataset == 'cifar-10':
        args.root = dataset_root_path
        args.topk = -1
        args.class_num = 10
        args.num_samples = 5000
        args.max_iter = 1
    elif args.dataset == 'nus-wide-tc21':
        args.root = os.path.join(dataset_root_path, "NUS-WIDE")
        args.topk = 5000
        args.class_num = 21
        args.num_samples = 10500
        args.max_iter = 150 # Flickr & NUS-WIDE
    elif args.dataset == 'flickr25k':
        args.root = os.path.join(dataset_root_path, "Flickr")
        args.topk = 5000
        args.class_num = 38
        args.num_query = 1000
        args.num_samples = 5000
        args.max_iter = 150 # Flickr & NUS-WIDE

    # End of dataset configuration

    os.makedirs(logdir, exist_ok=True)

    logger.add(f'{logdir}/_res.log', rotation='500 MB', level='INFO')
    logger.info(f"DCGH: {message}")
    logger.info(args)

    torch.backends.cudnn.benchmark = True

    # Load dataset
    query_dataloader, _, retrieval_dataloader = load_data(
        args.dataset,
        args.root,
        args.num_query,
        args.num_samples,
        args.batch_size,
        args.num_workers,
    )

    for code_length in args.code_length:
        mAP = train(
            query_dataloader,
            retrieval_dataloader,
            code_length,
            logdir,
            args
        )
        logger.info('[code_length:{}][best-mAP:{:.4f}]'.format(code_length, mAP))


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='DCGH Pytorch')

    # ***** The following dataset-related arguments are configurated above *****
    parser.add_argument('--dataset', default="cifar-10", type=str,
                        choices=["cifar-10", "nus-wide-tc21", "flickr25k"],
                        help='Dataset name.')
    parser.add_argument('--root', default="data/", type=str,
                        help='Path of dataset')
    parser.add_argument('--class-num', default=10, type=int,
                        help='Class number.(default: 10)')
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('--num-samples', default=2000, type=int,
                        help='Number of sampling data points.(default: 2000)')
    parser.add_argument('--topk', default=-1, type=int,
                    help='Calculate map of top k.(default: all)')
    parser.add_argument('--max-iter', default=50, type=int,
                    help='Number of iterations.(default: 50)')
    # ***** END *****


    parser.add_argument('--code-length', default='12,24,32,48', type=str,
                    help='Binary hash code length.(default: 12,24,32,48)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size.(default: 64)')
    parser.add_argument('--lr', default=4e-5, type=float,
                        help='Learning rate.(default: 4e-5)')
    parser.add_argument('--max-epoch', default=3, type=int,
                        help='Number of epochs.(default: 3)')
    parser.add_argument('--eval-iter', default=10, type=int,
                        help='Number of epoches between evaluation.(default: 10)')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='Number of loading data threads.(default: 0)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='Used gpu number.(default: 0)')


    parser.add_argument('--embedding-size', default=128, type=int,
                        help='Hyper-parameter.(default: 128)')
    parser.add_argument('--hidden-size', default=1024, type=int,
                        help='Hidden vector size in VAE&GCN.(default: 1024)')
    parser.add_argument('--gcn-dropout', default=0.5, type=float,
                        help='GCN Dropout layera.(default: 0.5)')


    parser.add_argument('--alpha', default=0.5, type=float,
                        help='alpha in loss.(default: 2)')
    parser.add_argument('--beta', default=0.5, type=float,
                        help='beta in loss.(default: 0.1)')
    parser.add_argument('--gamma', default=200, type=float,
                        help='Hyper-parameter.(default: 200)')

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))

    return args


if __name__ == '__main__':
    run()
