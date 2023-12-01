from .model_base import SSGmodelBase, SSGmodelBase_2_heads
from .models import SSGmodelTriplane, SSGmodelConv3D, SSGmodelTriplane_2_heads


def get_model(config):
    if config.G_struct == "triplane":
        return SSGmodelTriplane(config)
    elif config.G_struct == "conv3d":
        return SSGmodelConv3D(config)
    elif config.G_struct == 'triplane_2_heads':
        return SSGmodelTriplane_2_heads(config)
    else:
        raise NotImplementedError
