import argparse


def get_arg_parser(title, mode):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--model_type', type=str, default='seq2seq')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='metagen')
    parser.add_argument('--dataset_max_size', type=int, default=0)
    parser.add_argument('--dataset_max_code_length', type=int, default=0)
    parser.add_argument('--dataset_filter_code_length', type=int, default=0)
    parser.add_argument('--dataset_bucketing', action='store_true', default=False)
    # The larger this value the more aggressive the bucketing is, but the content of the buckets is less randomized.
    parser.add_argument('--dataset_macrobucket_size', type=int, default=100)
    parser.add_argument('--vocab_min_freq', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    # If true will keep the entire dataset in memory.
    parser.add_argument('--dataset_load_in_ram', action='store_true', default=False)
    parser.add_argument('--num_placeholders', type=int, default=20)

    if mode == 'train':
        train_group = parser.add_argument_group('train')
        train_group.add_argument('--save_every_n', type=int, default=100)
        train_group.add_argument('--keep_every_n', type=int, default=10000000)
        train_group.add_argument('--debug_every_n', type=int, default=20)
        train_group.add_argument('--eval_every_n', type=int, default=1000)
        train_group.add_argument('--eval_n_examples', type=int, default=10000)
        train_group.add_argument('--log_interval', type=int, default=20)
        train_group.add_argument('--optimizer', type=str, default='adam')
        train_group.add_argument('--optimizer_weight_decay', type=float, default=0.0)
        train_group.add_argument('--lr', type=float, default=.001)
        train_group.add_argument('--lr_decay_steps', type=int)
        train_group.add_argument('--lr_decay_rate', type=float)
        train_group.add_argument('--clip', type=float)
        train_group.add_argument('--n_warmup_steps', type=int, default=4000)
        train_group.add_argument('--num_epochs', type=int, default=10)
        train_group.add_argument('--num_units', type=int, default=500)
        train_group.add_argument('--num_encoder_layers', type=int, default=1)
        train_group.add_argument('--num_decoder_layers', type=int, default=1)
        train_group.add_argument('--encoder_dropout', type=float, default=0.2)
        train_group.add_argument('--decoder_dropout', type=float, default=0.2)
        train_group.add_argument('--num_att_heads', type=int, default=1)
        train_group.add_argument('--num_transformer_layers', type=int, default=2)
        train_group.add_argument(
            '--seq2seq_decoder', type=str, default='attn_decoder'
        )
        train_group.add_argument(
            '--bidirectional', action='store_true', default=False)
        # Either --read-text or --skip-text
        train_group.add_argument('--read-text', dest='read_text',
                            action='store_true', default=True)
        train_group.add_argument('--skip-text', dest='read_text', action='store_false')
        # Either --read-io or --skip-io
        train_group.add_argument('--read-io', dest='read_io',
                            action='store_true', default=False)
        train_group.add_argument('--skip-io', dest='read_io', action='store_false')
        train_group.add_argument('--io-count', type=int, default=3)

    elif mode == 'eval':
        eval_group = parser.add_argument_group('eval')
        eval_group.add_argument('--tag', type=str, default='')
        eval_group.add_argument('--example-id', type=int, default=None)
        eval_group.add_argument('--step', type=int, default=None)
        eval_group.add_argument('--eval-train', action='store_true', default=False)
        eval_group.add_argument('--hide-example-info', action='store_true', default=False)
        eval_group.add_argument('--report-path')

    infer_group = parser.add_argument_group('infer')
    infer_group.add_argument('--max_decoder_length', type=int, default=100)
    infer_group.add_argument('--max_beam_trees', type=int, default=1)
    infer_group.add_argument('--max_beam_iter', type=int, default=1000)
    infer_group.add_argument('--max_eval_trials', type=int, default=1)
    infer_group.add_argument('--min_prob_threshold', type=float, default=1e-5)
    infer_group.add_argument('--search-bfs', action='store_true', default=True)

    runtime_group = parser.add_argument_group('runtime')
    runtime_group.add_argument(
        '--restore-map-to-cpu', action='store_true', default=False)

    return parser


def backport_default_args(args):
    """Backport default args."""
    backport = {
        "restore_map_to_cpu": False,
        "keep_every_n": 10000000,
        "read_text": True,
        "read_io": False,
        "io_count": 3,
        "read_code": False,
        "optimizer": "adam",
        "dataset_filter_code_length": 0,
        "train_word_embeddings": True,
        "num_decoder_layers": 1,
        "decoder_dropout": 0.2,
        "encoder_dropout": 0.0,
    }
    for key, value in backport.items():
        if not hasattr(args, key):
            setattr(args, key, value)
