from torchtext import data
from dataset.rst_tree import load_tree_from_string


def rstdt_fields():
    # Define Fields
    DOC_ID = data.RawField(is_target=False)
    TREE = data.RawField(preprocessing=load_tree_from_string, is_target=True)
    WORD = data.NestedField(data.Field(batch_first=True), include_lengths=True)
    WORD.glove = True  # load the GloVe embeddings using the torchtext lib
    RAW_WORD = data.RawField(is_target=False)
    SPAN = data.RawField(is_target=False)
    BOUND_FLAG = data.RawField(is_target=False)
    PARENT_LABEL = data.RawField(is_target=False)

    fields = {
        'doc_id': ('doc_id', DOC_ID),
        'labelled_attachment_tree': ('tree', TREE),
        'tokenized_strings': ('word', WORD),
        'raw_tokenized_strings': ('elmo_word', RAW_WORD),
        'spans': ('spans', SPAN),
        'starts_sentence': ('starts_sentence', BOUND_FLAG),
        'starts_paragraph': ('starts_paragraph', BOUND_FLAG),
        'parent_label': ('parent_label', PARENT_LABEL)
    }

    return fields
