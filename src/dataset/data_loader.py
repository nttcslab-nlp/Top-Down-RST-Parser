from torchtext import data
from torchtext.vocab import Vocab
import torch
import pickle
from collections import Counter


class Dataset():
    def __init__(self, config, fields):
        self._config = config
        self._fields = fields
        self._device = torch.device('cpu') if config.cpu else torch.device('cuda:0')
        self._data_root = getattr(config, 'data_root', None)
        self._train_file = getattr(config, 'train_file', None)
        self._valid_file = getattr(config, 'valid_file', None)
        self._test_file = getattr(config, 'test_file', None)
        self._batch_size = getattr(config, 'batch_size', 5)

    def load_train_valid(self):
        # load dataset
        train, valid = data.TabularDataset.splits(
            path=self._data_root,
            train=self._train_file,
            validation=self._valid_file,
            format='json', fields=self._fields)

        # build vocab
        field_iter = self._fields.items()
        for json_key, field in field_iter:
            if not hasattr(field[1], 'use_vocab'):
                continue
            if field[1].use_vocab and not hasattr(field[1], 'vocab'):
                vectors = None
                if hasattr(field[1], 'glove') and field[1].glove:
                    vectors = 'glove.840B.300d'
                field[1].build_vocab(train, valid, vectors=vectors)

        # make iterator
        train_iter, valid_iter = data.Iterator.splits(
            (train, valid), sort_key=lambda x: len(x.word),
            batch_sizes=(self._batch_size, self._batch_size),
            device=self._device)

        return train_iter, valid_iter

    def load_test(self):
        test = data.TabularDataset(
            path=self._data_root / self._test_file, format='json', fields=self._fields)

        # build vocab
        field_iter = self._fields.items()
        for json_key, field in field_iter:
            if not hasattr(field[1], 'use_vocab'):
                continue
            if field[1].use_vocab and not hasattr(field[1], 'vocab'):
                vectors = None
                if hasattr(field[1], 'glove') and field[1].glove:
                    vectors = 'glove.840B.300d'
                field[1].build_vocab(test, vectors=vectors)

        # make iterator
        test_iter = data.Iterator(
            test, train=False,
            batch_size=1,
            device=self._device, sort=False)

        return test_iter


def load_vocab(file_path, glove=False, specials=[], fmt='pickle'):
    counter = None
    if fmt == 'pickle':
        with open(file_path, 'rb') as f:
            counter = pickle.load(f)

    elif fmt == 'tsv':
        counter = Counter()
        with open(file_path, 'r') as f:
            for line in f:
                elms = line.strip().split('\t')
                assert len(elms) == 2, 'vocab "{}" is invalid format'.format(file_path)
                word, freq = elms[0], int(elms[1])
                counter[word] = freq

    elif fmt == 'raw_text':
        # maybe a large vocab size will cause a memory overflow problem.
        with open(file_path, 'r') as f:
            for line in f:
                words = line.strip().split()
                counter[word] += 1

    vectors = "glove.840B.300d" if glove else None
    vocab = Vocab(counter, vectors=vectors, specials=specials)
    return vocab
