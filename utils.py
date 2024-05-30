import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_train_val_COCO():

    transform = transforms.Compose([
    transforms.ToTensor()])

    coco_train_dataset = datasets.CocoDetection(root='/datasets/COCO-2017/train2017', 
                                      annFile='/datasets/COCO-2017/anno2017/instances_train2017.json',
                                      transform=transform)

    coco_val_dataset = datasets.CocoDetection(root='/datasets/COCO-2017/val2017', 
                                      annFile='/datasets/COCO-2017/anno2017/instances_val2017.json',
                                      transform=transform)

    return coco_train_dataset, coco_val_dataset


