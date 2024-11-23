
import torch
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torchvision.transforms as T

def get_val_data_loader(annotations_file, images_dir, batch_size=4, shuffle=False):
    dataset = CocoDetection(images_dir, annotations_file, transform=T.ToTensor())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: tuple(zip(*x)))
    return data_loader