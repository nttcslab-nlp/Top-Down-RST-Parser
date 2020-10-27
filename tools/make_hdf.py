import argparse
import torch
import json
from hdf import HDF
from embedder import WordEmbedder
from data_loader import load_vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-file', required=True, nargs='+')
    parser.add_argument('--hdf-file', required=True)
    parser.add_argument('--vocab-file', required=True)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    dataset = load(args.json_file)
    device = torch.device('cpu') if args.cpu else torch.device('cuda:0')
    vocab = load_vocab(args.vocab_file, glove=True, specials=['<unk>', '<pad>'])
    embedder = WordEmbedder(vocab, 300, 0, use_elmo=True, device=device)
    embedder.to(device)

    vectors = {}
    for data in dataset:
        doc_id = data['doc_id']
        edu_strings = data['tokenized_strings']
        edu_starts_sentence = data['starts_sentence']
        edu_ends_sentence = edu_starts_sentence[1:] + [True]

        sentence = []
        sentences = []
        for edu, is_end in zip(edu_strings, edu_ends_sentence):
            sentence.extend(edu.split())
            if is_end:
                sentences.append(sentence)
                sentence = []

        with torch.no_grad():
            sentence_embeddings = embedder.embed_for_sentences(sentences)
        assert len(sentences) == sentence_embeddings.size(0)
        flatten_embeddings = torch.cat([embeddings[:len(sentence)] for embeddings, sentence in zip(sentence_embeddings, sentences)])
        assert sum([len(sentence) for sentence in sentences]) == flatten_embeddings.size(0)
        vectors[doc_id] = flatten_embeddings

    HDF.save_hdf(args.hdf_file, vectors)
    return


def load(files_path):
    if not isinstance(files_path, list):
        files_path = [files_path]
    dataset = []
    for file_path in files_path:
        with open(file_path) as f:
            for line in f:
                dataset.append(json.loads(line))

    return dataset


if __name__ == '__main__':
    main()
