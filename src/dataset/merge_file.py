from torchtext.data import Example, Batch, Dataset


class Doc():
    def __init__(self, tokens, starts_sentence, starts_edu, starts_paragraph, parent_label, doc_id=None):
        self.tokens = tokens
        self.starts_edu = starts_edu
        self.starts_sentence = starts_sentence
        self.starts_paragraph = starts_paragraph
        self.parent_label = parent_label
        self.doc_id = doc_id

    def set_fields(self, fields):
        self.fields = fields

    def __repr__(self):
        N = 10
        tokens = 'tokens   :\t{} ... {}'.format(self.tokens[:N], self.tokens[-N:])
        edu_flags = 'edu flag :\t{} ... {}'.format(self.starts_edu[:N], self.starts_edu[-N:])
        sent_flags = 'sent flag:\t{} ... {}'.format(self.starts_sentence[:N], self.starts_sentence[-N:])
        para_flags = 'para flag:\t{} ... {}'.format(self.starts_paragraph[:N], self.starts_paragraph[-N:])
        label = 'parent_label:\t{}'.format(self.parent_label)
        txt = '\n'.join([tokens, edu_flags, sent_flags, para_flags, label])
        return txt

    def to_batch(self, device, fields=None, parent_label=None, x2y='d2e', index=0):
        if fields is None:
            fields = self.fields

        assert self.fields is not None, 'you need to call after setting fields with Doc.set_fields()'

        if x2y in ['d2e', 'd2p', 'd2s']:
            starts_sentence = self.starts_sentence
            starts_paragraph = self.starts_paragraph
            tokens = self.tokens
            if x2y == 'd2e':
                starts_xxx = self.starts_edu
            elif x2y == 'd2s':
                starts_xxx = self.starts_sentence
            elif x2y == 'd2p':
                starts_xxx = self.starts_paragraph

        start_offset = 0
        if x2y in ['p2s', 's2e']:
            def get_span(hierarchical_type, word_starts_sentence, word_starts_paragraph, index):
                if hierarchical_type.startswith('p'):
                    flags = word_starts_paragraph
                elif hierarchical_type.startswith('s'):
                    flags = word_starts_sentence

                # flags: [True, False, False, False, True, False, True, ...]
                # index: 0 -> start, end = 0, 4
                # index: 1 -> start, end = 4, 6
                count = 0
                start, end = 0, len(flags)
                for i, f in enumerate(flags):
                    if not f:
                        continue
                    if count == index:
                        start = i
                    if count == index + 1:
                        end = i
                        break
                    count += 1
                return start, end

            start, end = get_span(x2y, self.starts_sentence, self.starts_paragraph, index)
            start_offset = start  # to make text span
            starts_sentence = self.starts_sentence[start:end]
            starts_paragraph = self.starts_paragraph[start:end]
            tokens = self.tokens[start:end]

            if x2y == 'p2s':
                starts_xxx = self.starts_sentence[start:end]
            elif x2y == 's2e':
                starts_xxx = self.starts_edu[start:end]

        # tokenized_strings: List of edus
        # raw_tokenized_strings: List of edus splitted by white-space
        # starts_*: List of bool value representing start of *
        tokenized_strings = self.make_edu(tokens, starts_xxx)
        raw_tokenized_strings = [edu.split() for edu in tokenized_strings]
        starts_sentence = self.make_starts(starts_xxx, starts_sentence)
        starts_paragraph = self.make_starts(starts_xxx, starts_paragraph)
        parent_label = self.parent_label if parent_label is None else parent_label
        spans, _ = self.make_text_span(tokenized_strings, start_offset, self.doc_id)

        assert len(tokenized_strings) == len(starts_sentence) == len(starts_paragraph), \
            'num input seqs not same'

        example = Example.fromdict({
            'doc_id': self.doc_id,
            'labelled_attachment_tree': '(nucleus-nucleus:Elaboration (text 1) (text 2))',  # DummyTree
            'tokenized_strings': tokenized_strings,
            'raw_tokenized_strings': raw_tokenized_strings,
            'spans': spans,
            'starts_sentence': starts_sentence,
            'starts_paragraph': starts_paragraph,
            'parent_label': parent_label,
        }, fields)

        if isinstance(fields, dict):  # copy from torchtext.data.TabularDataset
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        dataset = Dataset([example], fields)
        batch = Batch([example], dataset, device=device)
        batch.tree = None
        return batch

    def make_edu(self, tokens, is_starts):
        assert len(tokens) == len(is_starts)
        edu_strings = []
        edu = []
        for token, is_start in zip(tokens + [''], is_starts + [True]):
            if is_start and edu:
                edu_strings.append(' '.join(edu))
                edu = []
            edu.append(token)

        return edu_strings

    def make_starts(self, base, target):
        starts = []
        for a, b in zip(base, target):
            if a:
                starts.append(b)

        return starts

    def make_text_span(self, edu_strings, starts_offset=0, doc_id=None):
        spans = []
        offset = starts_offset
        for edu_stirng in edu_strings:
            words = edu_stirng.split()
            n_words = len(words)
            if doc_id is not None:
                span = (offset, offset + n_words, doc_id)
            else:
                span = (offset, offset + n_words)
            spans.append(span)
            offset += n_words

        assert len(spans) == len(edu_strings)
        return spans, offset

    @classmethod
    def from_merge_file(cls, file_path):
        tokens = []
        starts_sentence = []
        starts_paragraph = []
        starts_edu = []
        _sentence_idx = -1
        _edu_idx = 0
        _paragraph_idx = 0
        with open(file_path) as f:
            for line in f:
                elms = line.strip().split('\t')
                assert len(elms) == 11
                sentence_idx = int(elms[0])  # start from 0
                token_idx = int(elms[1])  # start from 1
                token = elms[2]
                # pos = elms[4]
                # lemma = elms[3]
                edu_idx = int(elms[9])  # start from 1
                paragraph_idx = int(elms[10])  # start from 1

                tokens.append(token)
                starts_sentence.append(sentence_idx != _sentence_idx)
                starts_edu.append(edu_idx == _edu_idx + 1)
                starts_paragraph.append(paragraph_idx == _paragraph_idx + 1)

                _sentence_idx = sentence_idx
                _edu_idx = edu_idx
                _paragraph_idx = paragraph_idx

        return cls(tokens, starts_sentence, starts_edu, starts_paragraph, None)

    @classmethod
    def from_batch(cls, batch):
        assert len(batch) == 1
        tokenized_edu_strings = batch.elmo_word[0]
        edu_starts_sentence = batch.starts_sentence[0]  # edu_starts_sentence
        edu_starts_paragraph = batch.starts_paragraph[0]  # edu_starts_paragraph
        assert len(tokenized_edu_strings) == len(edu_starts_sentence) == len(edu_starts_paragraph)
        doc_id = batch.doc_id[0]
        parent_label = batch.parent_label[0]
        # edu -> word
        tokens = sum(tokenized_edu_strings, [])
        starts_edu = sum([[True] + [False] * (len(edu)-1) for edu in tokenized_edu_strings], [])
        starts_sentence = sum([[is_start] + [False] * (len(edu)-1) for edu, is_start in zip(tokenized_edu_strings, edu_starts_sentence)], [])
        starts_paragraph = sum([[is_start] + [False] * (len(edu)-1) for edu, is_start in zip(tokenized_edu_strings, edu_starts_paragraph)], [])
        assert len(tokens) == len(starts_edu) == len(starts_sentence) == len(starts_paragraph)
        return cls(tokens, starts_sentence, starts_edu, starts_paragraph, parent_label, doc_id)
