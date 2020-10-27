import torch
import torch.optim as optim
import sys

from config import load_config
from dataset.fields import rstdt_fields
from dataset.data_loader import Dataset, load_vocab
from networks.parser import SpanBasedParser
from networks.ensemble import EnsembleParser
from networks.hierarchical import HierarchicalParser
from trainer.trainer import Trainer
from dataset.merge_file import Doc
from nltk import Tree


def main():
    config = load_config()
    if config.subcommand == 'train':
        train(config)
    elif config.subcommand == 'finetune':
        finetune(config)
    elif config.subcommand == 'test':
        test(config)
    elif config.subcommand == 'parse':
        parse(config)
    else:
        print('train / finetune / config / parse')
        return -1

    return 0


def train(config):
    fields = rstdt_fields()
    dataset = Dataset(config, fields)
    train_iter, valid_iter = dataset.load_train_valid()

    model = SpanBasedParser.build_model(config, fields)
    print(model)
    optimizer = {'adam': optim.Adam, 'sgd': optim.SGD}[config.optimizer](
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)
    trainer = Trainer(config, model, optimizer, scheduler, train_iter, valid_iter, fields)
    trainer.run()

    return


def finetune(config):
    fields = rstdt_fields()
    dataset = Dataset(config, fields)
    train_iter, valid_iter = dataset.load_train_valid()

    model = SpanBasedParser.load_pretrained_model(config.model_path, config, fields)
    print(model)
    optimizer = {'adam': optim.Adam, 'sgd': optim.SGD}[config.optimizer](
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)
    trainer = Trainer(config, model, optimizer, scheduler, train_iter, valid_iter, fields)
    trainer.run()

    return


def test(config):
    fields = rstdt_fields()
    dataset = Dataset(config, fields)
    test_iter = dataset.load_test()
    # model = SpanBasedParser.load_model(config.model_path[0], config, fields)
    # model = EnsembleParser.load_model(config.model_path, config, fields)
    model = HierarchicalParser.load_model(config.model_path, config, fields)
    # scores = Trainer.valid(model, test_iter)
    # print(scores)

    doc_ids = []
    pred_trees = []
    for batch in test_iter:
        batch.tree = None
        with torch.no_grad():
            output_dict = model(batch)

        doc_ids.extend(batch.doc_id)
        pred_trees.extend(output_dict['tree'])

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pred_trees = [Tree.fromstring(tree) for tree in pred_trees]
    for doc_id, tree in zip(doc_ids, pred_trees):
        tree_path = config.output_dir / '{}.tree'.format(doc_id)
        with open(tree_path, 'w') as f:
            print(tree, file=f)

    return


def parse(config):
    fields = rstdt_fields()
    glove_vocab = load_vocab(config.vocab_file, glove=True,
                             specials=['<unk>', '<pad>'], fmt=config.vocab_format)
    # override the vocab
    fields['tokenized_strings'][1].vocab = glove_vocab
    fields['tokenized_strings'][1].nesting_field.vocab = glove_vocab
    model = HierarchicalParser.load_model(config.model_path, config, fields)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    filelist = []
    if len(config.input_doc) == 1:
        # 単一のファイル or ディレクトリ
        input_doc = config.input_doc[0]
        if input_doc.is_dir():
            filelist = input_doc.iterdir()
        else:
            filelist = [input_doc]
    else:
        # 複数のファイル
        filelist = config.input_doc

    for doc_path in filelist:
        # doc_path: file path of merge format document
        tree_path = (config.output_dir / doc_path.name).with_suffix('.rst_tree')
        if tree_path.exists():
            print('already exists: {}'.format(tree_path), file=sys.stderr)
            continue

        print('processing: {}'.format(doc_path), file=sys.stderr)
        try:
            with torch.no_grad():
                doc = Doc.from_merge_file(doc_path)
                doc.set_fields(fields)
                tree = model.parse(doc)  # tree: NLTK.Tree
        except:
            continue

        with open(tree_path, 'w') as f:
            print(tree.pformat(10000000), file=f)

    return


if __name__ == '__main__':
    main()
