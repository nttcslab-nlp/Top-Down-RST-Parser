from nltk import Tree
from evaluate.tree_function import get_brackets
from evaluate.tree_function import convert2rst_tree


def rst_parseval(pred_trees, gold_trees, eval_type='full', gold_segmentation=True):
    # pred_trees: list of NLTK Tree
    # gold_trees: list of NLTK Tree

    sum_match = 0
    sum_pred = 0
    sum_gold = 0
    for p_tree, g_tree in zip(pred_trees, gold_trees):
        assert isinstance(p_tree, Tree)
        assert isinstance(g_tree, Tree)
        if p_tree.label() != 'ROOT':
            p_tree = convert2rst_tree(p_tree)
        if g_tree.label() != 'ROOT':
            g_tree = convert2rst_tree(g_tree)

        pred_spans = get_brackets(p_tree, eval_type)
        gold_spans = get_brackets(g_tree, eval_type)

        pred_cnt = len(pred_spans)
        gold_cnt = len(gold_spans)
        assert pred_cnt == gold_cnt
        match_cnt = len([span for span in pred_spans if span in gold_spans])

        sum_match += match_cnt
        sum_pred += pred_cnt
        sum_gold += gold_cnt

    # micro average f1
    micro_recall = sum_match / sum_pred
    micro_precision = sum_match / sum_gold
    micro_f1 = 2 * (micro_recall * micro_precision) / (micro_recall + micro_precision)
    if gold_segmentation:
        assert sum_pred == sum_gold
    return micro_f1


def original_parseval(pred_trees, gold_trees):
    pass
