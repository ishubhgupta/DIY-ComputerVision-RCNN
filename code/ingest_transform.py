import torchvision.transforms as T
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

def get_data_loader(annotations_file, images_dir, batch_size=4, shuffle=True):
    # Create dataset
    dataset = CocoDetection(images_dir, annotations_file, transform=T.ToTensor())
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: tuple(zip(*x)))
    return data_loader


