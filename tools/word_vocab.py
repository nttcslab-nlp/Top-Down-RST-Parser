import argparse
import json
import pickle
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=None)
    parser.add_argument('--valid', default=None)
    parser.add_argument('--test', default=None)
    parser.add_argument('--vocab', required=True)
    args = parser.parse_args()

    dataset = load([args.train, args.valid, args.test])
    word_counter = count(dataset)
    pickle_counter(word_counter, args.vocab)
    print('Vocab size: {}'.format(len(word_counter)))
    print('Samples:')
    print(word_counter.most_common()[:10])

    return 0


def load(fnames):
    dataset = []
    for fname in fnames:
        if fname is None:
            continue
        with open(fname) as f:
            for line in f:
                dataset.append(json.loads(line))

    return dataset


def count(dataset):
    word_counter = Counter()

    for data in dataset:
        words = sum([edu.split() for edu in data['tokenized_strings']], [])
        word_counter.update(words)

    return word_counter


def pickle_counter(counter, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(counter, f)
    return


if __name__ == '__main__':
    main()
