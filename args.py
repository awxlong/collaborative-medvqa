import argparse
# import datetime
import os
import random 
import numpy as np
import torch
import ast
import pdb

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dict_type(string):
    try:
        value = ast.literal_eval(string)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid dict value: {string}")
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(f"Value {value} is not a dict")
    for key, val in value.items():
        if not isinstance(val, int):
            raise argparse.ArgumentTypeError(f"Value {val} for key {key} in the dictionary is not an integer")
    return value

def dict_type_mil(string):
    try:
        value = ast.literal_eval(string)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid dict value: {string}")
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(f"Value {value} is not a dict")
    for key, val in value.items():
        # if not isinstance(key, int):
        #     raise argparse.ArgumentTypeError(f"Key {key} in the dictionary is not a int")
        if not isinstance(val, int):
            raise argparse.ArgumentTypeError(f"Value {val} for key {key} in the dictionary is not an integer")
    return value

def str_to_int_list(string):
    return [int(num) for num in string.split(',')]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type", type=str, default="gpt2-xl", choices=("gpt2-xl", "microsoft/biogpt","stanford-crfm/BioMedLM"))
    parser.add_argument(
        "--setting", type=str, default="frozen", choices=("lora", "frozen",'prefixtuning',"p_tuning","prompttuning", "unfrozen"))
    # parser.add_argument(
    #     "--ablation", type=str, default="none", choices=("remove_question", "remove_visual",'replace_visual',"swap"))
    parser.add_argument(
        "--mapping_type", type=str, default="MLP")
    parser.add_argument(
        "--prefix_length", type=int, default=8)
    parser.add_argument(
        "--dataset_path", type=str, default="../vqa_datasets/"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1)
    parser.add_argument(
        "--epochs", type=int, default=30)
    # parser.add_argument(
    #     "--dataset", type=str, default='pathvqa', choices=('pathvqa', 'ovqa', 'slake'))
    parser.add_argument(
        "--lr", type=float, default=1e-4)
    parser.add_argument(
        "--warmup_steps", type=int, default=600)
    parser.add_argument(
        "--grad_acc_steps", type=int, default=4, help="Amount of steps to simulate mini-batch gradient descent, default is 4")
    parser.add_argument(
        "--seed", type=int, default=42)
    parser.add_argument(
        "--iters_to_accumulate", type=int, default=4)
    parser.add_argument(
        "--validation_step", type=int, default=1000)
    parser.add_argument(
        "--out_dir", default="./checkpoints")
    parser.add_argument(
        "--checkpoint", type=str)
    parser.add_argument(
        "--eval", dest="eval", action="store_true")
    parser.add_argument(
        "--split", default="train", choices=('train', 'valid', 'test'))
    parser.add_argument(
        "--verbose", dest="verbose", action="store_true")
    
    args = parser.parse_args()

    # pdb.set_trace()

    seed_everything(args.seed)

    return args