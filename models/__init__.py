

from karel_model import KarelLGRLModel


def get_model(args):
    MODEL_TYPES = {
        'karel-lgrl': KarelLGRLModel,
    }
    return MODEL_TYPES[args.model_type](args)
