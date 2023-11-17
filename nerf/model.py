import torch
import torch.nn as nn
import collections


class MLP(nn.Module):
    """
    Define the MLP architecture of NeRF following the paper
    """


    def __init__(self):
        super(MLP, self).__init__()

        # input dimensions
        # spatial coordinate positional embedding 3 x (L = 10 x 2(sin,cos))
        self.num_spatial_input = 60
        # views coordinate positional embedding 3 x (L = 4 x 2(sin,cos))
        self.num_camera_view_input = 24

        # features_vector dimension
        self.num_features = 256

        # output dimensions
        self.num_rgb = 3
        self.num_sigma = 1

        self.features_vector = self.features_vector_init();
        self.sigma = nn.Linear(in_features=self.num_features, out_features=self.num_sigma)
        self.rgb = self.rgb_init()

    # x -> fc(256) -> relu x 8
    def features_vector_init(self):
        layers = []
        in_features = self.num_spatial_input  # (x,y,z)
        for i in range(8):
            layers.append(("fc_{}".format(i), nn.Linear(in_features=in_features, out_features=self.num_features)))
            layers.append(("relu_".format(i), nn.ReLU()))
            in_features = self.num_features
        return nn.Sequential(collections.OrderedDict(layers))

    # x -> fc(128) -> relu -> fc(3) -> rgb
    def rgb_init(self):
        layers = []
        layers.append(("fc_10", nn.Linear(in_features=self.num_features + self.num_camera_view_input,
                                          out_features=self.num_features // 2)))
        layers.append(("relu_10", nn.ReLU()))
        layers.append(("fc_11", nn.Linear(in_features=self.num_features // 2, out_features=self.num_rgb)))
        return nn.Sequential(collections.OrderedDict(layers))

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        """

        :param x: spatial coordinates(x,y,z)
        :param d: camera direction
        :return: (r,g,b,volume_density)
        """

        features_vector = self.features_vector(x)
        sigma = self.sigma(features_vector)
        features_camera_view = torch.cat([features_vector, d], dim=-1)
        rgb = self.rgb(features_camera_view)

        output = torch.cat([rgb,sigma],dim=-1)

        return output

















