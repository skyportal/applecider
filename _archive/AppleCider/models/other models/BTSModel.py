import torch.nn as nn


class BTSModel(nn.Module):
    """
    Image model from BTSbot for science, reference, difference images 
    Paper: https://doi.org/10.3847/1538-4357/ad5666
    """ ""

    def __init__(self, config):
        super(BTSModel, self).__init__()

        self.classification = True if config["mode"] == "image" else False

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=config["input_channels"],
                out_channels=config["conv1_channels"],
                kernel_size=config["conv_kernel"],
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config["conv1_channels"],
                out_channels=config["conv1_channels"],
                kernel_size=config["conv_kernel"],
                padding="same",
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(config["i_dropout1"]),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=config["conv1_channels"],
                out_channels=config["conv2_channels"],
                kernel_size=config["conv_kernel"],
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config["conv2_channels"],
                out_channels=config["conv2_channels"],
                kernel_size=config["conv_kernel"],
                padding="same",
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Dropout(config["i_dropout2"]),
        )

        if self.classification:
            self.fc = nn.Linear(784, config["num_classes"])

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.shape[0], -1)

        if self.classification:
            x = self.fc(x)

        return x
