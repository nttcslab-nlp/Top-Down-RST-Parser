import functools
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import dataset.rst_tree as rsttree
from nltk import Tree
from networks.parser import SpanBasedParser


class EnsembleParser(nn.Module):
    def __init__(self, parsers, device):
        super(EnsembleParser, self).__init__()
        self.parsers = parsers
        self.device = device

    @classmethod
    def load_model(cls, model_paths, config, fields):
        models = []
        for model_path in model_paths:
            models.append(SpanBasedParser.load_model(model_path, config, fields))

        device = torch.device('cpu') if config.cpu else torch.device('cuda:0')
        return cls(models, device)

    def parse(self, doc):
        batch = doc.to_batch(self.device)
        output = self.__call__(batch)
        tree = output['tree'][0]
        tree = Tree.fromstring(tree)
        return tree

    def forward(self, batch):
        # run embedder
        ensemble_rnn_outputs = []
        for parser in self.parsers:
            ensemble_rnn_outputs.append(parser.embed(batch))

        # run top-down parser
        pred_trees = []
        for i in range(len(batch)):
            tree, _ = self.ensemble_greedy_tree(
                [rnn_outputs[i] for rnn_outputs in ensemble_rnn_outputs],
                batch.starts_sentence[i],
                batch.starts_paragraph[i],
                batch.parent_label[i])
            pred_trees.append(tree.convert().linearize())

        return {'tree': pred_trees}

    def ensemble_greedy_tree(self, ensemble_rnn_output, starts_sentence, starts_paragraph, parent_label):
        sentence_length = len(ensemble_rnn_output[0])  # no padded
        for i, parser in enumerate(self.parsers):
            parser.starts_sentence = starts_sentence
            parser.starts_paragraph = starts_paragraph
            parser.rnn_output = parser.edge_pad(ensemble_rnn_output[i])

        @functools.lru_cache(maxsize=None)
        def helper(left, right, parent_label=None):
            assert 0 <= left < right <= sentence_length
            if right - left == 1:  # 終了条件
                tag, word = 'text', str(left)
                tree = rsttree.LeafParseNode(left, tag, word)
                return tree, torch.zeros(1, device=self.device).squeeze()

            ensemble_split_scores = []
            for parser in self.parsers:
                ensemble_split_scores.append(parser.get_split_scores(left, right))

            split_scores = self._ensemble(ensemble_split_scores)
            split, split_loss = parser.predict_split(split_scores, left, right)

            ensemble_ns_label_scores = []
            ensemble_rela_label_scores = []
            for parser in self.parsers:
                feature = parser.get_feature_embedding(left, right, split, parent_label)
                ensemble_ns_label_scores.append(parser.f_ns(feature))
                ensemble_rela_label_scores.append(parser.f_rela(feature))

            ns_label_scores = self._ensemble(ensemble_ns_label_scores)
            rela_label_scores = self._ensemble(ensemble_rela_label_scores)
            ns, ns_loss = parser.predict_label(ns_label_scores, left, right,
                                               sentence_length, parser.ns_vocab, 0)
            rela, rela_loss = parser.predict_label(rela_label_scores, left, right,
                                                   sentence_length, parser.rela_vocab, 1)

            left_trees, left_loss = helper(left, split, (ns, rela))
            right_trees, right_loss = helper(split, right, (ns, rela))
            children = rsttree.InternalParseNode((':'.join((ns, rela)),), [left_trees, right_trees])
            return children, ns_loss + rela_loss + split_loss + left_loss + right_loss

        pred_tree, loss = helper(0, sentence_length, parent_label)
        return pred_tree, loss

    @staticmethod
    def _ensemble(ensemble_scores):
        # ensemble_scores: list of scores tensor
        ensemble_scores = [softmax(scores, dim=0) for scores in ensemble_scores]
        scores = torch.sum(torch.stack(ensemble_scores), dim=0)
        # scores = scores.data.cpu().numpy()  # numpyとして返す
        return scores
