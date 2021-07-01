import torch
import torch.nn as nn
# import itertools as it
from typing import Sequence

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):

    def __init__(
            self,
            in_size,
            out_classes: int,
            channels: Sequence[int],
            pool_every: int,
            hidden_dims: Sequence[int],
            conv_params: dict = {},
            activation_type: str = "relu",
            activation_params: dict = {},
            pooling_type: str = "max",
            pooling_params: dict = {},
    ):

        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []

        if 'kernel_size' not in self.conv_params:
            self.conv_params = dict(kernel_size=3, stride=1, padding=1)
        if 'kernel_size' not in self.pooling_params:
            self.pooling_params = dict(kernel_size=2)

        pool_count = 0
        layers.append(torch.nn.Conv2d(in_channels, self.channels[0], kernel_size=self.conv_params['kernel_size'],
                                      stride=self.conv_params['stride'], padding=self.conv_params['padding']))
        layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))
        if len(layers) % (self.pool_every * 2) == 0:
            layers.append(torch.nn.Dropout2d(p=0.2, inplace=False))
            layers.append(POOLINGS[self.pooling_type](self.pooling_params['kernel_size']))
            pool_count += 1
        for i in range(len(self.channels) - 1):
            layers.append(
                torch.nn.Conv2d(self.channels[i], self.channels[i + 1], kernel_size=self.conv_params['kernel_size'],
                                stride=self.conv_params['stride'], padding=self.conv_params['padding']))
            layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))
            if (len(layers) - pool_count) % (self.pool_every * 2) == 0:
                layers.append(torch.nn.Dropout2d(p=0.2, inplace=False))
                layers.append(POOLINGS[self.pooling_type](self.pooling_params['kernel_size']))
                pool_count += 1
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:

        rng_state = torch.get_rng_state()
        try:
            with torch.no_grad():
                torch.random.set_rng_state(torch.manual_seed(42).get_state())
                x = torch.randn(self.in_size).unsqueeze(0)
                seq = self._make_feature_extractor()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                out = seq.to(device)(x.to(device))
                return out.view(-1, 1).size(0)
        finally:
            torch.set_rng_state(rng_state)

    def _make_classifier(self):
        layers = []

        # Discover the number of features after the CNN part.
        n_features = self._n_features()
        layers.append(torch.nn.Linear(n_features, self.hidden_dims[0]))
        layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))
        for i in range(len(self.hidden_dims) - 1):
            layers.append(torch.nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))
        layers.append(torch.nn.Linear(self.hidden_dims[-1], self.out_classes))
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.classifier(out.view(out.size(0), -1))
        return out


class ResidualBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            channels: Sequence[int],
            kernel_sizes: Sequence[int],
            batchnorm: bool = False,
            dropout: float = 0.0,
            activation_type: str = "relu",
            activation_params: dict = {},
            **kwargs,
    ):
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # main
        main_path = []
        main_path.append(torch.nn.Conv2d(in_channels, channels[0], kernel_size=kernel_sizes[0],
                                         padding=int((kernel_sizes[0] - 1) / 2)))
        main_path.append(torch.nn.Dropout2d(p=dropout, inplace=False))
        if batchnorm:
            main_path.append(torch.nn.BatchNorm2d(channels[0]))
        main_path.append(ACTIVATIONS[activation_type](**activation_params))
        for i in range(len(channels) - 2):
            main_path.append(torch.nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_sizes[i + 1],
                                             padding=int((kernel_sizes[i + 1] - 1) / 2)))
            main_path.append(torch.nn.Dropout2d(p=dropout, inplace=False))
            if batchnorm:
                main_path.append(torch.nn.BatchNorm2d(channels[i + 1]))
            main_path.append(ACTIVATIONS[activation_type](**activation_params))
        if len(channels) > 1:
            main_path.append(torch.nn.Conv2d(channels[-2], channels[-1], kernel_size=kernel_sizes[-1],
                                             padding=int((kernel_sizes[-1] - 1) / 2)))

        shortcut_path = []
        if in_channels != channels[-1]:
            shortcut_path.append(torch.nn.Conv2d(in_channels, channels[-1], kernel_size=1, bias=False))
        else:
            shortcut_path.append(torch.nn.Identity())

        self.main_path = nn.Sequential(*main_path)
        self.shortcut_path = nn.Sequential(*shortcut_path)


    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out



class ResNetClassifier(ConvClassifier):
    def __init__(
            self,
            in_size,
            out_classes,
            channels,
            pool_every,
            hidden_dims,
            batchnorm=False,
            dropout=0.0,
            **kwargs,
    ):

        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        if 'kernel_size' not in self.conv_params:
            self.conv_params = dict(kernel_size=3, stride=1, padding=1)
        if 'kernel_size' not in self.pooling_params:
            self.pooling_params = dict(kernel_size=2)

        for i in range(0, len(self.channels), self.pool_every):
            in_channels = self.in_size[0] if i == 0 else self.channels[i - 1]
            channels = self.channels[i:i + self.pool_every]
            if i + self.pool_every <= len(self.channels):
                layers.append(ResidualBlock(
                    in_channels=in_channels, channels=channels, kernel_sizes=[3] * self.pool_every,
                    batchnorm=self.batchnorm, dropout=self.dropout))
                layers.append(POOLINGS[self.pooling_type](self.pooling_params['kernel_size']))
            else:
                layers.append(ResidualBlock(
                    in_channels=in_channels, channels=channels, kernel_sizes=[3] * len(channels),
                    batchnorm=self.batchnorm, dropout=self.dropout))

        seq = nn.Sequential(*layers)
        return seq