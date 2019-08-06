import torch.utils.data
import torch.nn as nn
import torchvision as tv
from torchvision import transforms
import torch.nn.functional as F

def train(model, optimizer, criteria, train_loader):
    pass


def get_model():
    pass


def get_datasets():
    transfrom = tv.transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = tv.datasets.ImageFolder(
        root=train_data_path,
        transforms=transfrom
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    test_dataset = tv.datasets.ImageFolder(
        root=test_data_path,
        transfroms=transfrom
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0
    )

    return train_loader, test_loader


def main():
    train_loader, test_loader = get_datasets()

    model = get_model()
    optimizer = torch.optim.Adam(lr=learning_rate)
    criteria = F.cross_entropy

    train(model, optimizer, criteria, train_loader)


if __name__ == "__main__":
    train_data_path = "../../data/cat_and_dog/train"
    test_data_path = "../../data/cat_and_dog/validation"

    batch_size = 50
    epochs = 30
    learning_rate = 5e-4

    main()