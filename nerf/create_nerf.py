import torch
from nerf.model import MLP
from nerf.encoding import PositionalEncoder


def init_nerf():
    """
    :return: The coarse and fine Multi Layer Perceptrons and the Positional encodings for the spatial coordinates and the views.
    """


    coarse_mlp = MLP()
    fine_mlp = MLP()

    embedding_pts = PositionalEncoder(L=10)
    embedding_views = PositionalEncoder(L=4)

    return coarse_mlp,fine_mlp,embedding_pts,embedding_views







