import torchvision.models as models
import torch
from torch import nn
from torch.nn import functional as F

from lib.gem import GeM


class Normalize(nn.Module):

    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        return F.normalize(x)


def build_model(pretrained):

    model_main = models.resnet34(pretrained=pretrained)
    model_main.avgpool = GeM()
    model_main.fc = nn.Linear(512, 512)

    model = nn.Sequential(model_main, Normalize())

    model.cuda()
    return model


def main():
    model = build_model(True)
    print(model)

    sample_tensor = torch.Tensor(1, 3, 288, 288).float().cuda()
    output = model.forward(sample_tensor)
    print(output.shape)


if __name__ == '__main__':
    main()
