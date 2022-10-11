import torch
from .memory import Memory
import torch.nn as nn


def Basic(intInput, intOutput, truncated=False):
    layers = (
        nn.Conv2d(
            in_channels=intInput,
            out_channels=intOutput,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(intOutput),
        nn.ReLU(inplace=False),
        nn.Conv2d(
            in_channels=intOutput,
            out_channels=intOutput,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(intOutput),
        nn.ReLU(inplace=False),
    )
    if truncated:
        return nn.Sequential(*layers[:-2])
    return nn.Sequential(*layers)


def Gen(intInput, intOutput, nc):
    return nn.Sequential(
        nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(nc),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(nc),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
        nn.Tanh(),
    )


def Upsample(nc, intOutput):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=nc,
            out_channels=intOutput,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        ),
        nn.BatchNorm2d(intOutput),
        nn.ReLU(inplace=False),
    )


class Encoder(nn.Module):
    def __init__(self, t_length=5, n_channel=3, task="prediction"):
        super().__init__()
        self.task = task
        self.moduleConv1 = Basic(n_channel * (t_length - 1), 64)
        self.modulePool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.moduleConv4 = Basic(256, 512, truncated=True)
        self.moduleBatchNorm = nn.BatchNorm2d(512)
        self.moduleReLU = nn.ReLU(inplace=False)

    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)
        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)
        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)
        tensorConv4 = self.moduleConv4(tensorPool3)
        if self.task == "prediction":
            return tensorConv4, tensorConv1, tensorConv2, tensorConv3
        # task == "reconstruction"
        return tensorConv4


class Decoder(nn.Module):
    def __init__(self, n_channel=3, task="prediction"):
        super().__init__()
        self.task = task
        upsample_out_divider = 1 if task == "reconstruction" else 2
        self.moduleConv = Basic(1024, 512)
        self.moduleUpsample4 = Upsample(512, 512 // upsample_out_divider)
        self.moduleDeconv3 = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 256 // upsample_out_divider)
        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 128 // upsample_out_divider)
        self.moduleDeconv1 = Gen(128, n_channel, 64)

    def forward(self, x, skip1=None, skip2=None, skip3=None):
        """Skips are None for Reconstruction decoder"""
        tensorConv = self.moduleConv(x)
        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        if self.task == "prediction":
            tensorUpsample4 = torch.cat((skip3, tensorUpsample4), dim=1)
        tensorDeconv3 = self.moduleDeconv3(tensorUpsample4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        if self.task == "prediction":
            tensorUpsample3 = torch.cat((skip2, tensorUpsample3), dim=1)
        tensorDeconv2 = self.moduleDeconv2(tensorUpsample3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        if self.task == "prediction":
            tensorUpsample2 = torch.cat((skip1, tensorUpsample2), dim=1)
        output = self.moduleDeconv1(tensorUpsample2)
        return output


class ConvAE(nn.Module):
    def __init__(
        self,
        n_channel=3,
        t_length=2,
        task="prediction",
    ):
        super().__init__()
        self.task = task
        self.encoder = Encoder(t_length, n_channel, task=task)
        self.memory = Memory()
        self.decoder = Decoder(n_channel, task=task)

    def forward(self, x, m_items, train=True):
        if self.task == "prediction":
            features, skip1, skip2, skip3 = self.encoder(x)
            skips = skip1, skip2, skip3
        else:  # self.task == "reconstruction"
            features = self.encoder(x)
            skips = tuple()  # no skip when task is reconstruction

        (
            updated_fea,
            m_items,
            softmax_score_query,
            softmax_score_memory,
            compactness,
            separateness,
        ) = self.memory(features, m_items, train)
        output = self.decoder(updated_fea, *skips)
        return (
            output,
            features,
            updated_fea,
            m_items,
            softmax_score_query,
            softmax_score_memory,
            compactness,
            separateness,
        )
