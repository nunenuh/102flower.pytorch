import os
import json
import torch
import shutil


# get category to flower name
def get_all_flower_names():
    path = os.path.join('dataset','cat_to_name.json')
    with open(path, 'r') as f:
            cat_to_name = json.load(f)
    return cat_to_name


def flower_name(val, array_index=False):
    labels = get_all_flower_names()
    if array_index:
        val = val + 1
    return labels[str(val)]


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    path_normal = os.path.join('weights', filename)
    path_best = os.path.join('weights', 'model_best.pth')
    torch.save(state, path_normal)
    if is_best:
        shutil.copyfile(path_normal, path_best)
