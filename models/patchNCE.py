import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class UnpairedFromFolder(data.dataset):
    def __init__(self, opt):
        dataset_dir = "datasets"
        mode = "train"

        self.dir_A = os.path.join(dataset_dir, mode + 'A')
        self.dir_B = os.path.join(dataset_dir, mode + 'B')

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)



if __name__ == "__main__":
    train()