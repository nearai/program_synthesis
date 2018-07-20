import argparse


def get_arg_parser(title, mode):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--no-cuda', action='store_true', default=False)

    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--dataset_filter_code_length', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    # Variable shuffling in UAST.
    parser.add_argument('--variable_shuffle', action='store_true', default=False)
    parser.add_argument('--num_placeholders', type=int, default=20)
    parser.add_argument('--trainB_weight', type=float, default=1)
    parser.add_argument('--num_steps', type=int, default=25000)

    if mode == 'train':
        train_group = parser.add_argument_group('train')
        train_group.add_argument('--save_every_n', type=int, default=100)
        train_group.add_argument('--keep_every_n', type=int, default=10000000)
        train_group.add_argument('--log_interval', type=int, default=20)
        train_group.add_argument('--optimizer', type=str, default='adam')
        train_group.add_argument('--optimizer_weight_decay', type=float, default=0.0)
        train_group.add_argument('--lr', type=float, default=.001)
        train_group.add_argument('--lr_decay_steps', type=int)
        train_group.add_argument('--lr_decay_rate', type=float)
        train_group.add_argument('--clip', type=float)
        train_group.add_argument('--num_units', type=int, default=500)
        train_group.add_argument('--num_encoder_layers', type=int, default=1)
        train_group.add_argument('--num_decoder_layers', type=int, default=1)
        train_group.add_argument('--encoder_dropout', type=float, default=0.2)
        train_group.add_argument('--decoder_dropout', type=float, default=0.2)
        train_group.add_argument('--num_att_heads', type=int, default=8)
        train_group.add_argument('--bidirectional', action='store_true', default=False)

    elif mode == 'eval':
        eval_group = parser.add_argument_group('eval')
        eval_group.add_argument('--report_path')

    infer_group = parser.add_argument_group('infer')
    infer_group.add_argument('--max_decoder_length', type=int, default=100)
    infer_group.add_argument('--max_beam_trees', type=int, default=1)
    infer_group.add_argument('--max_eval_trials', type=int, default=1)

    runtime_group = parser.add_argument_group('runtime')
    runtime_group.add_argument(
        '--restore_map_to_cpu', action='store_true', default=False)

    return parser


def backport_default_args(args):
    """Backport default args."""
    backport = {
        "restore_map_to_cpu": False,
        "keep_every_n": 10000000,
        "optimizer": "adam",
        "dataset_filter_code_length": 0,
        "num_decoder_layers": 1,
        "decoder_dropout": 0.2,
        "encoder_dropout": 0.0,
    }
    for key, value in backport.items():
        if not hasattr(args, key):
            setattr(args, key, value)
