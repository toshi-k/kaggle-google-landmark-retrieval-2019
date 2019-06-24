import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

print(torch.version.cuda)
print(torch.backends.cudnn.version())


class Normalize(nn.Module):

    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        return F.normalize(x)


class ModelMain(nn.Module):

    def __init__(self):
        super(ModelMain, self).__init__()
        self.l1 = nn.Linear(41, 128)
        self.l2 = nn.Linear(128, 256)
        self.f1 = nn.Linear(512, 512)
        self.f2 = nn.Linear(512, 512)
        self.f3 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(512)

    def forward(self, x):

        batch_size = len(x)

        h0 = x.transpose(1, 2).contiguous().view((-1, 41))

        h1 = self.bn1(F.leaky_relu(self.l1(h0)))
        h2 = self.bn2(F.leaky_relu(self.l2(h1)))

        h = h2.view((batch_size, -1, 256))
        h_sum = h.sum(dim=1)
        h_max = h.max(dim=1)[0]

        h = torch.cat((h_sum, h_max), 1)

        j1 = self.bn3(F.leaky_relu(self.f1(h)))
        j2 = self.bn4(F.leaky_relu(self.f2(j1)))
        j3 = self.f3(j2)

        return j3


def build_model(pretrained):

    model = nn.Sequential(ModelMain(), Normalize())

    model.cuda()
    return model


def debug():

    model = build_model(True)

    input_tensor = torch.FloatTensor(2, 41, 1000).zero_().cuda()
    output_tensor = model.forward(Variable(input_tensor))
    print(output_tensor.shape)
    # print(output_tensor)


if __name__ == '__main__':
    debug()
