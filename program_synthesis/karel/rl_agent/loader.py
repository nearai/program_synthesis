from program_synthesis.karel import arguments
from program_synthesis.karel import dataset
from program_synthesis.karel import models

from program_synthesis.karel.dataset import executor
from program_synthesis.tools import saver


def load_model(model_dir, model_type, step=None):
    args = saver.ArgsDict(model_dir=model_dir, model_type=model_type, step=step, cuda=False, restore_map_to_cpu=True)
    saver.restore_args(args)
    arguments.backport_default_args(args)
    dataset.set_vocab(args)
    m = models.get_model(args)
    eval_dataset = dataset.get_eval_dataset(args, m)
    m.model.eval()
    the_executor = executor.get_executor(args)()
    return m, eval_dataset, the_executor


baseline_model, baseline_eval_dataset, baseline_executor = load_model(
    'baseline-msr', 'karel-lgrl', 250100
)
