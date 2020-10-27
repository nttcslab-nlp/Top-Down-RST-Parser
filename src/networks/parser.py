import functools
import torch
import torch.nn as nn
from networks.layers import BiLSTM, FeedForward, DeepBiAffine
from networks.embedder import TextEmbedder, Embeddings
import dataset.trees as rsttree
from nltk import Tree
from trainer.checkpointer import Checkpointer
from dataset.data_loader import load_vocab


class SpanBasedParser(nn.Module):
    def __init__(self, text_embedder, hidden_size, margin, dropout,
                 ns_vocab, relation_vocab, device, hierarchical_type,
                 label_type, use_parent_label, use_hard_boundary):
        super(SpanBasedParser, self).__init__()
        self.hidden_size = hidden_size
        self.margin = margin
        self.dropout = dropout
        self.device = device
        self.hierarchical_type = hierarchical_type
        self.label_type = label_type
        self.use_parent_label = use_parent_label
        self.use_hard_boundary = use_hard_boundary

        self.ns_vocab = ns_vocab
        self.rela_vocab = relation_vocab

        # Embeddings
        """ Text embedder """
        self.text_embedder = text_embedder
        embed_size = self.text_embedder.get_embed_size()

        """ Parent label embedder """
        self.label_embed_size = 10
        self.ns_embedder = Embeddings(len(ns_vocab), self.label_embed_size,
                                      0.0, padding_idx=ns_vocab.stoi['<pad>'])
        self.rela_embedder = Embeddings(len(relation_vocab), self.label_embed_size,
                                        0.0, padding_idx=relation_vocab.stoi['<pad>'])
        """ Bound embedder """
        self.bound_embed_size = 10
        self.bound_embedder = Embeddings(16, self.bound_embed_size, 0.0)

        # BiLSTM
        self.bilstm = BiLSTM(embed_size, self.hidden_size, self.dropout)
        self.edge_pad = nn.ZeroPad2d((0, 0, 1, 1))
        # Scoring functions
        """ Split Scoring """
        span_embed_size = self.hidden_size*2 + self.bound_embed_size
        self.span_embed_size = span_embed_size
        self.f_split = DeepBiAffine(span_embed_size, self.dropout)
        """ Label Scoring """
        feature_embed_size = span_embed_size*4
        if use_parent_label:
            feature_embed_size += self.label_embed_size*2
        self.f_ns = FeedForward(feature_embed_size, [self.hidden_size], len(ns_vocab), self.dropout)
        self.f_rela = FeedForward(feature_embed_size, [self.hidden_size], len(relation_vocab), self.dropout)

    def freeze(self):
        self.text_embedder.freeze()
        self.bilstm.freeze()

    @classmethod
    def build_model(cls, config, fields):
        embedder = TextEmbedder.build_model(config, fields)
        hidden = config.hidden
        margin = config.margin
        dropout = config.dropout
        ns_vocab = load_vocab(config.data_root / config.ns_vocab, specials=['<pad>'], fmt='tsv')
        rela_vocab = load_vocab(config.data_root / config.relation_vocab, specials=['<pad>'], fmt='tsv')
        device = torch.device('cpu') if config.cpu else torch.device('cuda:0')
        hierarchical_type = config.hierarchical_type
        label_type = getattr(config, 'label_type', 'full')
        use_parent_label = getattr(config, 'parent_label_embed', True)
        use_hard_boundary = getattr(config, 'use_hard_boundary', False)
        model = cls(embedder, hidden, margin, dropout, ns_vocab, rela_vocab, device,
                    hierarchical_type, label_type, use_parent_label, use_hard_boundary)
        model.to(device)
        return model

    @classmethod
    def load_model(cls, model_path, config, fields):
        print('load model: {}'.format(model_path))
        device = torch.device('cpu') if config.cpu else torch.device('cuda:0')
        model_state = Checkpointer.restore(model_path, device=device)
        model_config = model_state['config']
        model_config.cpu = config.cpu
        if config.hdf_file is not None:
            model_config.hdf_file = config.hdf_file
        model_config.use_hard_boundary = config.use_hard_boundary
        model_param = model_state['model']
        if hasattr(fields['tokenized_strings'][1], 'vocab'):
            # use vocab for test
            embed_key = 'text_embedder.word_embedder.word_embed.embed.weight'
            model_param[embed_key] = fields['tokenized_strings'][1].vocab.vectors
        else:
            # if test fields don't have vocab, we use fields of train and valid
            fields = model_state['fields']
        model = cls.build_model(model_config, fields)
        model.load_state_dict(model_param)
        model.eval()
        return model

    @classmethod
    def load_pretrained_model(cls, model_path, config, fields):
        print('load model: {}'.format(model_path))
        device = torch.device('cpu') if config.cpu else torch.device('cuda:0')
        model_state = Checkpointer.restore(model_path, device=device)
        model_config = model_state['config']
        model_config.cpu = config.cpu
        if config.hdf_file is not None:
            model_config.hdf_file = config.hdf_file
        model_config.label_type = 'full'
        model_config.parent_label_embed = True
        model_param = model_state['model']
        # copy params
        for key, value in vars(model_config).items():
            if not hasattr(config, key):
                setattr(config, key, getattr(model_config, key))

        if hasattr(fields['tokenized_strings'][1], 'vocab'):
            # use vocab for test
            embed_key = 'text_embedder.word_embedder.word_embed.embed.weight'
            model_param[embed_key] = fields['tokenized_strings'][1].vocab.vectors
        else:
            # if test fields don't have vocab, we use fields of train and valid
            fields = model_state['fields']

        model = cls.build_model(model_config, fields)
        model.load_state_dict(model_param)

        if config.freeze:
            model.freeze()

        return model

    @classmethod
    def load_old_model(cls, old_model_path, config, fields):
        device = torch.device('cpu') if config.cpu else torch.device('cuda:0')
        ns_vocab_path = '/home/lr/kobayasi/Projects/forAAAI/SpanBasedRSTParser_Gate/data/Corpora/vocab.label'
        rela_vocab_path = '/home/lr/kobayasi/Projects/forAAAI/SpanBasedRSTParser_Gate/data/Corpora/vocab.relation'
        model_state = Checkpointer.restore(old_model_path, device=device)

        def replace(params):
            new_params = type(params)()
            for name, vec in params.items():
                if name.startswith('gate_lstm.'):
                    new_name = 'text_embedder.' + name
                    new_params[new_name] = vec
                elif name.startswith('lstm'):
                    new_name = 'bilstm.' + '.'.join(name.split('.')[1:])
                    new_params[new_name] = vec
                elif name.startswith('ns_embed'):
                    new_name = 'ns_embedder.embed.' + '.'.join(name.split('.')[1:])
                    new_params[new_name] = vec
                elif name.startswith('rela_embed'):
                    new_name = 'rela_embedder.embed.' + '.'.join(name.split('.')[1:])
                    new_params[new_name] = vec
                elif name.startswith('bound_embed'):
                    new_name = 'bound_embedder.embed.' + '.'.join(name.split('.')[1:])
                    new_params[new_name] = vec
                else:
                    new_params[name] = vec

            return new_params

        model_param = replace(model_state['model'])
        embed_key = 'text_embedder.word_embedder.word_embed.embed.weight'
        model_param[embed_key] = fields['tokenized_strings'][1].vocab.vectors

        embedder = TextEmbedder.build_model(config, fields)
        hidden = 250
        margin = 1.0
        dropout = 0.4
        ns_vocab = load_vocab(ns_vocab_path, specials=['<pad>'], fmt='pickle')
        rela_vocab = load_vocab(rela_vocab_path, specials=['<pad>'], fmt='pickle')
        model = cls(embedder, hidden, margin, dropout, ns_vocab, rela_vocab, device, 'd2e', False)
        model.to(device)
        model.load_state_dict(model_param)
        model.eval()
        return model

    def parse(self, doc):
        batch = doc.to_batch(self.device)
        output = self.forward(batch)
        tree = output['tree'][0]
        tree = Tree.fromstring(tree)
        return tree

    def forward(self, batch):
        rnn_outputs = self.embed(batch)

        losses = []
        pred_trees = []
        for i in range(len(batch)):
            tree, loss = self.greedy_tree(
                rnn_outputs[i],
                batch.starts_sentence[i],
                batch.starts_paragraph[i],
                batch.parent_label[i],
                batch.tree[i].convert() if batch.tree is not None else None)

            losses.append(loss)
            pred_trees.append(tree.convert().linearize())

        loss = torch.mean(torch.stack(losses))

        return {
            'loss': loss,
            'tree': pred_trees,
        }

    def embed(self, batch):
        # batch.tree: list of Tree (batch)
        # batch.word: tuple
        # batch.word[0]: tensor (batch, num_edus, num_words)
        # batch.word[1]: edu lengths (batch)
        # batch.word[2]: word lengths (batch, num_edus)
        # batch.elmo_word: raw input text for elmo embedding

        edu_embeddings = self.text_embedder(batch.word, batch.elmo_word, batch.starts_sentence, batch.spans)
        lstm_outputs = self.bilstm(edu_embeddings, batch.word[1])

        # lstm_outputs: (batch, num_edus, hidden)
        lstm_outputs = [output[:l] for output, l in zip(lstm_outputs, batch.word[1])]
        return lstm_outputs

    def greedy_tree(self, rnn_output, starts_sentence, starts_paragraph, parent_label, gold_tree=None):
        rnn_output = self.edge_pad(rnn_output)
        sentence_length = len(rnn_output) - 2
        self.rnn_output = rnn_output
        self.starts_sentence = starts_sentence
        self.starts_paragraph = starts_paragraph
        self.gold_tree = gold_tree
        is_train = self.training

        @functools.lru_cache(maxsize=None)
        def helper(left, right, parent_label=None):
            assert 0 <= left < right <= sentence_length
            if right - left == 1:  # 終了条件
                tag, word = 'text', str(left)
                tree = rsttree.LeafParseNode(left, tag, word)
                return tree, torch.zeros(1, device=self.device).squeeze()

            split_scores = self.get_split_scores(left, right)
            split, split_loss = self.predict_split(split_scores, left, right)

            feature = self.get_feature_embedding(left, right, split, parent_label)
            ns_label_scores = self.f_ns(feature)
            rela_label_scores = self.f_rela(feature)
            ns, ns_loss = self.predict_label(ns_label_scores, left, right,
                                             sentence_length, self.ns_vocab, 0)
            rela, rela_loss = self.predict_label(rela_label_scores, left, right,
                                                 sentence_length, self.rela_vocab, 1)

            left_trees, left_loss = helper(left, split, (ns, rela))
            right_trees, right_loss = helper(split, right, (ns, rela))
            children = rsttree.InternalParseNode((':'.join((ns, rela)),), [left_trees, right_trees])

            if self.label_type == 'skelton':
                loss = split_loss + left_loss + right_loss
            elif self.label_type == 'ns':
                loss = ns_loss + split_loss + left_loss + right_loss
            elif self.label_type == 'full':
                loss = ns_loss + rela_loss + split_loss + left_loss + right_loss

            return children, loss
        pred_tree, loss = helper(0, sentence_length, parent_label)
        if is_train:
            assert gold_tree.convert().linearize() == pred_tree.convert().linearize()
        return pred_tree, loss

    def get_split_scores(self, left, right):
        left_encodings = []
        right_encodings = []
        for k in range(left + 1, right):
            left_encodings.append(self.get_span_embedding(left, k))
            right_encodings.append(self.get_span_embedding(k, right))

        left_encodings = torch.stack(left_encodings)
        right_encodings = torch.stack(right_encodings)
        split_scores = self.f_split(left_encodings, right_encodings)
        split_scores = split_scores.view(len(left_encodings))

        if self.use_hard_boundary:
            paragraph_split = [not f for f in self.starts_paragraph[left+1:right]]
            sentence_split = [not f for f in self.starts_sentence[left+1:right]]
            min_value = min(split_scores) - 10.0
            if not all(paragraph_split):
                split_scores[paragraph_split] = min_value
            if not all(sentence_split):
                split_scores[sentence_split] = min_value

        return split_scores

    def get_span_embedding(self, left, right):
        if left == right:
            return torch.zeros([self.span_embed_size], device=self.device)

        forward = (
            self.rnn_output[right][:self.hidden_size] -
            self.rnn_output[left][:self.hidden_size])
        backward = (
            self.rnn_output[left + 1][self.hidden_size:] -
            self.rnn_output[right + 1][self.hidden_size:])

        bound_embedding = self.get_boundary_embedding(left, right)
        span_embedding = torch.cat([forward, backward, bound_embedding])
        return span_embedding

    def get_feature_embedding(self, left, right, split, parent_label):
        left_span = self.get_span_embedding(left, split)
        right_span = self.get_span_embedding(split, right)
        label_embedding = self.get_label_embedding(parent_label)

        N = len(self.rnn_output) - 2
        out_left_span = self.get_span_embedding(0, left)
        out_right_span = self.get_span_embedding(right, N)
        if self.use_parent_label:
            feature = torch.cat([left_span, right_span, label_embedding,
                                 out_left_span, out_right_span], dim=0)
        else:
            feature = torch.cat([left_span, right_span,
                                 out_left_span, out_right_span], dim=0)
        return feature

    def get_boundary_embedding(self, left, right):
        is_start_sentence = self.starts_sentence[left]
        is_start_paragraph = self.starts_paragraph[left]
        cross_sentence = any(self.starts_sentence[left+1: right])
        cross_paragraph = any(self.starts_paragraph[left+1: right])
        bound = int(is_start_sentence*1 + is_start_paragraph*2 + cross_sentence*4 + cross_paragraph*8)
        bound = torch.tensor(bound, dtype=torch.long, device=self.device)
        bound_embedding = self.bound_embedder(bound)
        return bound_embedding

    def get_label_embedding(self, label):
        if label is None:
            label = '<pad>:<pad>'
        if isinstance(label, str):
            label = label.split(':')  # split 'NS:Relation' into ('NS', 'Relation')

        if self.label_type == 'skelton':
            label = ['<pad>', '<pad>']
        elif self.label_type == 'ns':
            label = [label[0], '<pad>']
        elif self.label_type == 'full':
            pass

        ns, relation = label
        ns_idx = torch.tensor(self.ns_vocab.stoi[ns], dtype=torch.long, device=self.device)
        ns_embedding = self.ns_embedder(ns_idx)
        rela_idx = torch.tensor(self.rela_vocab.stoi[relation], dtype=torch.long, device=self.device)
        rela_embedding = self.rela_embedder(rela_idx)

        label_embedding = torch.cat([ns_embedding, rela_embedding], dim=-1)
        return label_embedding

    def augment(self, scores, oracle_index):
        assert len(scores.size()) == 1
        increment = torch.ones_like(scores) + self.margin
        increment[oracle_index] = 0
        return scores + increment

    def predict_split(self, split_scores, left, right):
        is_train = self.training
        if is_train:
            oracle_split = min(self.gold_tree.oracle_splits(left, right))
            oracle_split_index = oracle_split - (left + 1)
            split_scores = self.augment(split_scores, oracle_split_index)

        split_scores_np = split_scores.data.cpu().numpy()
        argmax_split_index = int(split_scores_np.argmax())
        argmax_split = argmax_split_index + (left + 1)

        if is_train:
            split = oracle_split
            split_loss = (
                split_scores[argmax_split_index] - split_scores[oracle_split_index]
                if argmax_split != oracle_split else torch.zeros(1, device=self.device).squeeze())
        else:
            split = argmax_split
            split_loss = split_scores[argmax_split_index]

        return split, split_loss

    def predict_label(self, label_scores, left, right, sentence_length, vocab, label_idx):
        is_train = self.training
        if is_train:
            oracle_label = self.gold_tree.oracle_label(left, right)[0].split(':')[label_idx]
            oracle_label_index = vocab.stoi[oracle_label]
            label_scores = self.augment(label_scores, oracle_label_index)

        label_scores_np = label_scores.data.cpu().numpy()
        argmax_label_index = int(
            label_scores_np.argmax() if right - left < sentence_length else
            label_scores_np[1:].argmax() + 1)
        argmax_label = vocab.itos[argmax_label_index]

        if is_train:
            label = oracle_label
            label_loss = (
                label_scores[argmax_label_index] - label_scores[oracle_label_index]
                if argmax_label != oracle_label else torch.zeros(1, device=self.device).squeeze())
        else:
            label = argmax_label
            label_loss = label_scores[argmax_label_index]

        return label, label_loss
