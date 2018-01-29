from karel_model import KarelLGRLModel, KarelLGRLRefineModel


def get_model(args):
    MODEL_TYPES = {
        'karel-lgrl': KarelLGRLModel,
        'karel-lgrl-ref': KarelLGRLRefineModel,
    }
    return MODEL_TYPES[args.model_type](args)
