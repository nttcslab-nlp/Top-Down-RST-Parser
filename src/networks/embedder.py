import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from allennlp.commands.elmo import ElmoEmbedder
from networks.layers import BiLSTM, SelectiveGate
from dataset.hdf import HDF


class WordEmbedder(nn.Module):
    def __init__(self, vocab, embed_dim, dropout, use_elmo=False, elmo_batch_size=128, device=None, hdf_file=None):
        super(WordEmbedder, self).__init__()
        self.vocab = vocab
        self.device = device
        if hasattr(vocab, 'vectors'):
            self.word_embed = Embeddings.from_pretrained(vocab.vectors, dropout)
        else:
            pad_idx = vocab.stoi.get('<pad>', None)
            self.word_embed = Embeddings(len(vocab), embed_dim, dropout, padding_idx=pad_idx)

        self.use_elmo = use_elmo
        if use_elmo:
            if device is None or device.type == 'cpu':
                device_id = -1
            else:
                device_id = device.index if device.index is not None else 0

            self.elmo_embed = ElmoEmbeddings(device_id, dropout, elmo_batch_size)

        self.use_hdf = hdf_file is not None
        if self.use_hdf:
            assert use_elmo, 'hdf includes elmo vector, please set --elmo-embed'
            self.hdf = HDF(hdf_file)

    @classmethod
    def build_model(cls, config, fields):
        embed_dim = getattr(config, 'embed', 300)
        dropout = getattr(config, 'dropout', 0.4)
        use_elmo = getattr(config, 'elmo_embed', True)
        elmo_batch_size = getattr(config, 'elmo_batch_size', 128)
        hdf_file = getattr(config, 'hdf_file', None)
        vocab = fields['tokenized_strings'][1].vocab
        device = torch.device('cpu') if config.cpu else torch.device('cuda:0')
        embedder = cls(vocab, embed_dim, dropout, use_elmo, elmo_batch_size, device, hdf_file)
        embedder.to(device)
        return embedder

    def forward(self, inputs, raw_inputs, starts_sentence, spans):
        if self.use_hdf:
            batch_embeddings = []
            batch_spans = spans
            # num_batch = len(batch_spans)
            # num_edus = max([len(spans) for spans in batch_spans])
            # num_words = max([max([span[1] - span[0] for span in spans]) for spans in batch_spans])
            # assert inputs.size() == (num_batch, num_edus, num_words)
            num_words = inputs.size(-1)

            for spans in batch_spans:
                # spans list of span (start_offset, end_offset, doc_id)
                doc_id = spans[0][2]
                word_vectors = self.hdf.stov(doc_id, self.device)
                embeddings = []
                for span in spans:
                    start_offset, end_offset, _doc_id = span
                    assert doc_id == _doc_id
                    vector = word_vectors[start_offset: end_offset]
                    # vector = self.hdf.stov('{}_{}-{}'.format(doc_id, str(start_offset), str(end_offset)), self.device)[:, -(300+1024*3):]
                    embeddings.append(vector)
                embeddings = pad_sequence(embeddings, batch_first=True)
                diff = num_words - embeddings.size(1)
                embeddings = torch.nn.functional.pad(embeddings, (0, 0, 0, diff))
                # embeddings: num_edus, num_words, embed_dim)
                batch_embeddings.append(embeddings)

            batch_embeddings = pad_sequence(batch_embeddings, batch_first=True)
            # batch_embeddings: (num_batch, num_edus, num_words, embed_dim)
            # assert batch_embeddings.size() == (num_batch, num_edus, num_words, 300+1024*3)
            batch_embeddings = self.word_embed.dropout(batch_embeddings)
            return batch_embeddings

        word_embeddings = self.word_embed(inputs)
        if self.use_elmo:
            elmo_embeddings = self.elmo_embed(inputs, raw_inputs, starts_sentence)
            word_embeddings = torch.cat([elmo_embeddings, word_embeddings], dim=-1)

        return word_embeddings

    def get_embed_size(self):
        embed_size = self.word_embed.get_embed_size()
        if self.use_elmo:
            embed_size += self.elmo_embed.get_embed_size()

        return embed_size

    def embed_for_sentences(self, sentences):
        indices = []
        for sentence in sentences:
            indices.append(torch.tensor(
                [self.vocab.stoi.get(word, self.vocab.stoi['<unk>']) for word in sentence],
                dtype=torch.long, device=self.device))
        indices = pad_sequence(indices, batch_first=True)
        word_embeddings = self.word_embed(indices)
        if self.use_elmo:
            elmo_embeddings = self.elmo_embed.forward_for_sentences(sentences)
            B, L, E = word_embeddings.size()
            _B, _L, _E = elmo_embeddings.size()
            assert B == _B
            assert L == _L
            embeddings = torch.cat([elmo_embeddings, word_embeddings], axis=2)
        return embeddings


class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout, padding_idx=None, vectors=None):
        super(Embeddings, self).__init__()
        if vectors is not None:
            self.embed = nn.Embedding.from_pretrained(vectors)
        else:
            self.embed = nn.Embedding(input_dim, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)

    @classmethod
    def from_pretrained(cls, vectors, dropout):
        input_dim, embed_dim = vectors.size()
        return cls(input_dim, embed_dim, dropout, vectors=vectors)

    def forward(self, indices):
        if isinstance(indices, tuple):
            indices = indices[0]
            # indices[1]: lengths
        embeddings = self.embed(indices)
        return self.dropout(embeddings)

    def get_embed_size(self):
        return self.embed.weight.size(1)


class ElmoEmbeddings(nn.Module):
    def __init__(self, device_id, dropout, batch_size=128):
        super(ElmoEmbeddings, self).__init__()
        self.batch_size = batch_size
        self.elmo = ElmoEmbedder(cuda_device=device_id)
        self.dropout = nn.Dropout(dropout)

    def forward(self, indices, raw_text, starts_sentence):
        assert len(raw_text) == len(starts_sentence)
        all_sentences = []
        for edus, edu_starts_sentence in zip(raw_text, starts_sentence):
            ends_sentence = edu_starts_sentence[1:] + [True]
            sentences, sentence = [], []
            for edu_words, end_of_sentence in zip(edus, ends_sentence):
                sentence.extend(edu_words)
                if end_of_sentence:
                    sentences.append(sentence)
                    sentence = []

            all_sentences.extend(sentences)

        # Run ELMo Embedder
        sentence_embeddings = []
        for min_batch in self.batch_iter(all_sentences, self.batch_size):
            sentence_embeddings.extend(self._forward(min_batch))

        # Sentence embeddings -> EDU embeddings
        sentence_idx = 0
        batch_edu_embeddings = []
        for edus, edu_starts_sentence in zip(raw_text, starts_sentence):
            ends_sentence = edu_starts_sentence[1:] + [True]
            edu_offset = 0
            edu_embeddings = []
            for edu_words, end_of_sentence in zip(edus, ends_sentence):
                edu_length = len(edu_words)
                edu_embedding = sentence_embeddings[sentence_idx][edu_offset: edu_offset+edu_length]
                edu_embeddings.append(edu_embedding)

                edu_offset += edu_length
                if end_of_sentence:
                    sentence_idx += 1
                    edu_offset = 0

            # edu_embeddings: Num_edus, Num_words, embedding_size
            edu_embeddings = pad_sequence(edu_embeddings, batch_first=True, padding_value=0)
            max_num_words = indices.size(2)
            diff = max_num_words - edu_embeddings.size(1)
            edu_embeddings = torch.nn.functional.pad(edu_embeddings, (0, 0, 0, diff))
            batch_edu_embeddings.append(edu_embeddings)

        embeddings = pad_sequence(batch_edu_embeddings, batch_first=True, padding_value=0)

        B, E, W, _ = embeddings.size()
        _B, _E, _W = indices.size()
        assert B == _B
        assert E == _E
        assert W == _W
        return self.dropout(embeddings)

    def forward_for_sentences(self, sentences):
        vectors = []
        max_length = max([len(sentence) for sentence in sentences])
        for min_batch in self.batch_iter(sentences, self.batch_size):
            embeddings = self._forward(min_batch)
            diff = max_length - embeddings.size(1)
            embeddings = torch.nn.functional.pad(embeddings, (0, 0, 0, diff))
            vectors.append(embeddings)

        vectors = torch.cat(vectors, dim=0)
        return vectors

    def batch_iter(self, iterable, batch_size=1):
        l = len(iterable)
        for offset in range(0, l, batch_size):
            yield iterable[offset:min(offset + batch_size, l)]

    def _forward(self, raw_text):
        elmo_vectors, _ = self.elmo.batch_to_embeddings(raw_text)
        B, _, L, E = elmo_vectors.size()
        elmo_vectors = elmo_vectors.transpose(1, 2)  # Bx3xLxE -> BxLx3xE
        elmo_vectors = elmo_vectors.contiguous().view(B, L, -1)  # BxLx3xE -> BxLx3*E
        return elmo_vectors

    def get_embed_size(self):
        return 1024 * 3


class TextEmbedder(nn.Module):
    def __init__(self, word_embedder, hidden_size, dropout, use_gate=False):
        super(TextEmbedder, self).__init__()
        self.word_embedder = word_embedder
        self.use_gate = use_gate
        if use_gate:
            embed_size = word_embedder.get_embed_size()
            self.gate_lstm = SelectiveGate(BiLSTM(embed_size, hidden_size, dropout))

    @classmethod
    def build_model(cls, config, fields):
        word_embedder = WordEmbedder.build_model(config, fields)
        hidden_size = getattr(config, 'hidden', 250)
        dropout = getattr(config, 'dropout', 0.4)
        use_gate = getattr(config, 'gate_embed', True)
        text_embedder = cls(word_embedder, hidden_size, dropout, use_gate)
        device = torch.device('cpu') if config.cpu else torch.device('cuda:0')
        text_embedder.to(device)
        return text_embedder

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs, raw_inputs, starts_sentence, spans=None):
        word_embeddings = None
        indices = inputs[0]
        edu_lengths = inputs[1]
        word_lengths = inputs[2]
        # edu_lengts: (batch, ), 各バッチを構成するEDUの数
        # word_lengths: (batch, num_edus), 各EDUを構成する単語の数

        word_embeddings = self.word_embedder(indices, raw_inputs, starts_sentence, spans)
        # word_embeddings: (batch, num_edus, num_words, embed_dim)

        edu_embeddings = None
        if self.use_gate:
            # batchを展開
            edu_embeddings = []
            for _embeddings, num_edus, lengths in zip(word_embeddings, edu_lengths, word_lengths):
                gated_rnn_outputs, sGate = self.gate_lstm(_embeddings[:num_edus], lengths[:num_edus])
                gated_sum_embeddings = torch.sum(gated_rnn_outputs, dim=1)
                gated_mean_embeddings = gated_sum_embeddings / lengths[:num_edus].unsqueeze(-1).float()
                edu_embeddings.append(gated_mean_embeddings)

            # 再構築
            edu_embeddings = torch.nn.utils.rnn.pad_sequence(edu_embeddings, batch_first=True)
        else:
            edu_embeddings = torch.sum(word_embeddings, dim=2) / word_lengths.unsqueeze(-1).float()

        return edu_embeddings

    def get_embed_size(self):
        if self.use_gate:
            embed_size = self.gate_lstm.lstm.hidden_size
            if self.gate_lstm.lstm.bidirectional:
                embed_size *= 2
        else:
            embed_size = self.word_embedder.get_embed_size()

        return embed_size
