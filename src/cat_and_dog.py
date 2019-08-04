import torch.utils.data
import torch.nn as nn
import torchvision as tv
from torchvision import transforms
import torch.nn.functional as F


class CatAndDog(nn.Module):
    def __init__(self, ic, channels=(32, 64, 32, 8), kernels=(3, 3, 3, 3)):
        super(CatAndDog, self).__init__()

        self.conv_layers = nn.ModuleList()

        in_ch = ic
        for c, k in zip(channels, kernels):
            self.conv_layers.append(nn.Conv2d(in_ch, c, k))
            self.conv_layers.append(nn.BatchNorm2d(c))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(2))
            in_ch = c

        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(nn.Linear(8*14*14, 100))
        self.linear_layers.append(nn.ReLU())
        self.linear_layers.append(nn.Linear(100, 2))
        # self.linear_layers.append(nn.Softmax())

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.reshape(x.size(0), -1)

        for layer in self.linear_layers:
            x = layer(x)
        return x


def main():
    train_loader, test_loader = get_dataset()

    module = CatAndDog(3).cuda()
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)

    train(module, optimizer, train_loader)

    torch.cuda.empty_cache()

    test(module, test_loader)

    torch.save(module.state_dict(), "../model/cat_and_dog_module")


def get_dataset():
    train_data_path = "../data/cat_and_dog/train"
    test_data_path = "../data/cat_and_dog/validation"

    transform_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = tv.datasets.ImageFolder(
        root=train_data_path,
        transform=transform_img
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    test_dataset = tv.datasets.ImageFolder(
        root=test_data_path,
        transform=transform_img
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    return train_loader, test_loader


def train(module, optimizer, train_loader):
    loss_sum = torch.tensor(0, dtype=torch.float32)
    for epoc in range(epochs):
        print("{} epoch".format(epoc))
        for idx, (x, y) in enumerate(train_loader):
            output = module(x.cuda())
            loss = F.cross_entropy(output, y.cuda())

            loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("\t{}".format(loss_sum.mean()))
        loss_sum = torch.tensor(0, dtype=torch.float32)


def test(module, test_loader):
    total = correct = 0
    for x, y in test_loader:
        output = module(x.cuda())

        val, pred_idx = output.max(1)
        correct += (pred_idx == y.cuda()).sum().item()
        total += val.shape[0]

    print ("correct : {} // total : {}".format(correct, total))
    print ("accuracy: {}".format(correct / total))


if __name__ == "__main__":
    batch_size = 50
    epochs = 30
    learning_rate = 5e-4

    main()

# https://github.com/ardamavi/Dog-Cat-Classifier/blob/master/get_model.py