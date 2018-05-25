from program_synthesis.karel.models.karel_model import KarelLGRLModel
from program_synthesis.karel.models.karel_model import KarelLGRLRefineModel
from program_synthesis.karel.models.karel_trace_model import TracePredictionModel
from program_synthesis.karel.models.karel_trace_model import CodeFromTracesModel
from program_synthesis.karel.models.karel_edit_model import KarelStepEditModel


def get_model(args):
    MODEL_TYPES = {
        'karel-lgrl': KarelLGRLModel,
        'karel-lgrl-ref': KarelLGRLRefineModel,
        'karel-trace-pred': TracePredictionModel,
        'karel-code-trace': CodeFromTracesModel,
        'karel-edit': KarelStepEditModel,
    }
    return MODEL_TYPES[args.model_type](args)
