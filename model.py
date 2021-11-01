import torch
from torch import nn


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


if __name__ == '__main__':
    myModule = MyModule()
    Input = torch.ones(64, 3, 32, 32)
    Output = myModule(Input)
    print(Output.shape)

