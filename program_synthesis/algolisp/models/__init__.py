
from program_synthesis.algolisp.models import seq2seq_model


def get_model(args):
    MODEL_TYPES = {
        'seq2seq': seq2seq_model.Seq2SeqModel
    }
    return MODEL_TYPES[args.model_type](args)
