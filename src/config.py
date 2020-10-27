import argparse
from pathlib import Path


def load_config():
    parser = argparse.ArgumentParser(description="span based rst parser")
    parser.set_defaults(subcommand="None")
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('train', help="see train -h")
    parser_train.set_defaults(subcommand="train")

    def train_config(parser):
        train_settings = parser.add_argument_group('train settings')
        train_settings.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
        train_settings.add_argument('--lr', type=float, default=0.001)
        train_settings.add_argument('--weight-decay', type=float, default=1e-4)
        train_settings.add_argument('--dropout', type=float, default=0.4)
        train_settings.add_argument('--lr-decay', type=float, default=0.99)
        train_settings.add_argument('--grad-clipping', type=float, default=5.0)
        train_settings.add_argument('--metric', default='relation')
        train_settings.add_argument('--maximize-metric', action='store_true')
        train_settings.add_argument('--batch-size', type=int, default=10)
        train_settings.add_argument('--elmo-batch-size', type=int, default=128)
        train_settings.add_argument('--epochs', type=int, default=50)
        train_settings.add_argument('--cpu', action='store_true')
        train_settings.add_argument('--disable-tqdm', action='store_true')

        model_settings = parser.add_argument_group('model settings')
        model_settings.add_argument('--hierarchical-type',
                                    choices=['d2e', 'd2s', 'd2p', 'p2e', 'p2s', 's2e'], required=True)
        model_settings.add_argument('--label-type', choices=['skelton', 'ns', 'full'], default='full')
        model_settings.add_argument('--hidden', type=int, default=250)
        model_settings.add_argument('--margin', type=float, default=1.0)
        model_settings.add_argument('--elmo-embed', action='store_true')
        model_settings.add_argument('--gate-embed', action='store_true')
        model_settings.add_argument('--parent-label-embed', action='store_true')

        path_settings = parser.add_argument_group('path settings')
        path_settings.add_argument('--data-root', default='./data', type=Path)
        path_settings.add_argument('--train-file', required=True)
        path_settings.add_argument('--valid-file', required=True)
        path_settings.add_argument('--test-file', default=None)
        path_settings.add_argument('--hdf-file', default=None)
        path_settings.add_argument('--ns-vocab', default='ns.vocab')
        path_settings.add_argument('--relation-vocab', default='relation.vocab')
        path_settings.add_argument('--serialization-dir', default='models/', type=Path)
        path_settings.add_argument('--keep-all-serialized-models', action='store_true')
        path_settings.add_argument('--log-file', default='training.log')
        path_settings.add_argument('--model-name', default='model')

    train_config(parser_train)

    parser_finetune = subparsers.add_parser('finetune', help="see finetune -h")
    parser_finetune.set_defaults(subcommand="finetune")

    def finetune_config(parser):
        finetune_settings = parser.add_argument_group('finetune settings')
        finetune_settings.add_argument('--optimizer', choices=['adam', 'sgd'], default='sgd')
        finetune_settings.add_argument('--lr', type=float, default=0.01)
        finetune_settings.add_argument('--weight-decay', type=float, default=1e-4)
        finetune_settings.add_argument('--dropout', type=float, default=0.4)
        finetune_settings.add_argument('--lr-decay', type=float, default=0.99)
        finetune_settings.add_argument('--grad-clipping', type=float, default=5.0)
        finetune_settings.add_argument('--metric', default='relation')
        finetune_settings.add_argument('--maximize-metric', action='store_true')
        finetune_settings.add_argument('--batch-size', type=int, default=10)
        finetune_settings.add_argument('--elmo-batch-size', type=int, default=128)
        finetune_settings.add_argument('--epochs', type=int, default=10)
        finetune_settings.add_argument('--cpu', action='store_true')

        model_settings = parser.add_argument_group('model settings')
        model_settings.add_argument('--hierarchical-type',
                                    choices=['d2e', 'd2s', 'd2p', 'p2e', 'p2s', 's2e'], required=True)
        # model_settings.add_argument('--label-type', choices=['skelton', 'ns', 'full'], default='full')
        # model_settings.add_argument('--hidden', type=int, default=250)
        # model_settings.add_argument('--margin', type=float, default=1.0)
        # model_settings.add_argument('--elmo-embed', action='store_true')
        # model_settings.add_argument('--gate-embed', action='store_true')
        # model_settings.add_argument('--parent-label-embed', action='store_true')
        model_settings.add_argument('--freeze', action='store_true')

        path_settings = parser.add_argument_group('path settings')
        path_settings.add_argument('--model-path', required=True)
        path_settings.add_argument('--data-root', default='./data', type=Path)
        path_settings.add_argument('--train-file', required=True)
        path_settings.add_argument('--valid-file', required=True)
        path_settings.add_argument('--test-file', default=None)
        path_settings.add_argument('--hdf-file', default=None)
        path_settings.add_argument('--ns-vocab', default='ns.vocab')
        path_settings.add_argument('--relation-vocab', default='relation.vocab')
        path_settings.add_argument('--serialization-dir', default='models/', type=Path)
        path_settings.add_argument('--keep-all-serialized-models', action='store_true')
        path_settings.add_argument('--log-file', default='finetune.log')
        path_settings.add_argument('--model-name', default='model_finetune')

    finetune_config(parser_finetune)

    parser_test = subparsers.add_parser('test', help="see test -h")
    parser_test.set_defaults(subcommand="test")

    def test_config(parser):
        test_settings = parser.add_argument_group('test settings')
        test_settings.add_argument('--cpu', action='store_true')
        test_settings.add_argument('--hierarchical-type', choices=['d2e', 'd2s2e', 'd2p2s2e'], required=True)
        test_settings.add_argument('--use-hard-boundary', action='store_true')
        path_settings = parser.add_argument_group('path settings')
        path_settings.add_argument('--data-root', default='./data', type=Path)
        path_settings.add_argument('--test-file', required=True)
        path_settings.add_argument('--hdf-file', default=None)
        path_settings.add_argument('--model-path', required=True, nargs='+')
        path_settings.add_argument('--vocab-file', default=None, type=Path)
        path_settings.add_argument('--vocab-format', default='pickle', choices=['pickle', 'tsv', 'text'])
        path_settings.add_argument('--output-dir', default='output', type=Path)

    test_config(parser_test)

    parser_parse = subparsers.add_parser('parse', help="see parse -h")
    parser_parse.set_defaults(subcommand="parse")

    def parse_config(parser):
        parse_settings = parser.add_argument_group('parse settings')
        parse_settings.add_argument('--cpu', action='store_true')
        parse_settings.add_argument('--hierarchical-type', choices=['d2e', 'd2s2e', 'd2p2s2e'], required=True)
        parse_settings.add_argument('--use-hard-boundary', action='store_true')
        path_settings = parser.add_argument_group('path settings')
        path_settings.add_argument('--model-path', required=True, nargs='+')
        path_settings.add_argument('--hdf-file', default=None)
        path_settings.add_argument('--input-doc', required=True, nargs='+', type=Path)
        path_settings.add_argument('--vocab-file', default=None, type=Path)
        path_settings.add_argument('--vocab-format', default='pickle', choices=['pickle', 'tsv', 'text'])
        path_settings.add_argument('--output-dir', default='output', type=Path)

    parse_config(parser_parse)

    return parser.parse_args()
