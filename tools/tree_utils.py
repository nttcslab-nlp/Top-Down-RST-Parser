from nltk import Tree


def binarize(tree):
    if len(tree) == 1:
        # End of recursion
        return tree
    if len(tree) == 2:
        # Binary structure
        left_tree = binarize(tree[0])
        right_tree = binarize(tree[1])
    else:
        # Non-Binary structure
        labels = [tree[i].label() for i in range(len(tree))]
        is_polynuclear = all(map(lambda x: x == labels[0], labels))
        if is_polynuclear:
            # Polynuclear relation label such as:
            # same-unit, list, etc...
            # -> convert to right heavy structure
            left_tree = binarize(tree[0])
            right_tree = binarize(
                Tree(tree[0].label(), [tree[i] for i in range(1, len(tree))]))
        else:
            # Non Binary structure without Polynuclear label
            # S/N/S -> left heavy
            left_tree = binarize(Tree('nucleus:span', [tree[0], tree[1]]))
            right_tree = binarize(tree[2])

    return Tree(tree.label(), [left_tree, right_tree])


def re_categorize(rst_tree):
    # RST Treeであることを判定
    assert rst_tree.label() == 'ROOT'

    for position in rst_tree.treepositions():
        if type(rst_tree[position]) == str:
            continue
        if rst_tree[position].label() == 'text':
            continue
        if len(rst_tree[position]) < 2:
            continue

        sub_tree = rst_tree[position]

        # labelを抽出
        l_ns, l_relation = sub_tree[0].label().split(':')
        r_ns, r_relation = sub_tree[1].label().split(':')

        def _re_categorize(relation):
            # suffixを取り除く
            while relation[-2:] in ['-s', '-e', '-n']:
                relation = relation[:-2]
            return RELATION_TABLE[relation]

        # relation読み替える
        l_relation = _re_categorize(l_relation)
        r_relation = _re_categorize(r_relation)

        # アノテーションミスを修正
        if l_ns == r_ns == 'nucleus':
            # N-Nの時の場合, relationは左右等しいが一件のみ例外
            if l_relation != r_relation:
                # l_relation: Cause, r_relation: Span
                r_relation = 'Cause'
            assert l_relation == r_relation

        # 新しいrelationを付与
        rst_tree[position][0].set_label(':'.join([l_ns, l_relation]))
        rst_tree[position][1].set_label(':'.join([r_ns, r_relation]))

    return rst_tree


def convert2labelled_attachment_tree(rst_tree, top=True):
    if top:
        # RST Treeであることを判定
        assert rst_tree.label() == 'ROOT'

    if len(rst_tree) == 1:
        return rst_tree[0]

    left_rst_tree = rst_tree[0]
    right_rst_tree = rst_tree[1]
    l_ns, l_relation = left_rst_tree.label().split(':')
    r_ns, r_relation = right_rst_tree.label().split(':')
    ns = '-'.join([l_ns, r_ns])
    relation = l_relation if l_relation != 'Span' else r_relation
    label = ':'.join([ns, relation])

    return Tree(label, [convert2labelled_attachment_tree(rst_tree[0], top=False),
                        convert2labelled_attachment_tree(rst_tree[1], top=False)])


def convert2rst_tree(attach_tree):
    def helper(tree, parent_label, position):
        # labelの分解
        ns, relation = parent_label.split(':')
        left_ns, right_ns = ns.split('-')
        if ns == 'satellite-nucleus':
            left_relation = relation
            right_relation = 'Span'
        elif ns == 'nucleus-satellite':
            left_relation = 'Span'
            right_relation = relation
        elif ns == 'nucleus-nucleus':
            left_relation = right_relation = relation
        elif ns == 'dummy-dummy':
            left_relation = right_relation = relation
        else:
            print('unknown label')
            exit()

        # labelの復元
        if position == 'left':
            label = ':'.join([left_ns, left_relation])
        elif position == 'right':
            label = ':'.join([right_ns, right_relation])
        else:  # position == 'ROOT':
            label = 'ROOT'

        # 再帰的に木を構築
        if len(tree) < 2:
            children = [tree]
        else:  # len(tree) == 2:
            left_tree = helper(tree[0], tree.label(), 'left')
            right_tree = helper(tree[1], tree.label(), 'right')
            children = [left_tree, right_tree]
        return Tree(label, children)

    rst_tree = helper(attach_tree, 'dummy-dummy:dummy', 'ROOT')
    return rst_tree


def get_brackets(tree, eval_type):
    spans = []
    for position in tree.treepositions():
        subtree = tree[position]
        if isinstance(subtree, str) or isinstance(subtree, int):
            continue
        label = subtree.label()
        if label in ['ROOT', 'text']:
            continue

        edu_indices = [int(idx) for idx in subtree.leaves()]
        boundary = (edu_indices[0], edu_indices[-1])
        ns, relation = label.split(':')
        if eval_type == 'full':
            span = (boundary, ns, relation)
        elif eval_type == 'relation':
            span = (boundary, relation)
        elif eval_type == 'ns':
            span = (boundary, ns)
        elif eval_type == 'span':
            span = (boundary)
        else:
            print('unknown eval_type')
            exit()
        spans.append(span)
    return spans


RELATION_TABLE = {
    "ROOT": "ROOT",
    "span": "Span",
    "attribution": "Attribution",
    "attribution-negative": "Attribution",
    "background": "Background",
    "circumstance": "Background",
    "cause": "Cause",
    "result": "Cause",
    "cause-result": "Cause",
    "consequence": "Cause",
    "comparison": "Comparison",
    "preference": "Comparison",
    "analogy": "Comparison",
    "proportion": "Comparison",
    "condition": "Condition",
    "hypothetical": "Condition",
    "contingency": "Condition",
    "otherwise": "Condition",
    "contrast": "Contrast",
    "concession": "Contrast",
    "antithesis": "Contrast",
    "elaboration-additional": "Elaboration",
    "elaboration-general-specific": "Elaboration",
    "elaboration-part-whole": "Elaboration",
    "elaboration-process-step": "Elaboration",
    "elaboration-object-attribute": "Elaboration",
    "elaboration-set-member": "Elaboration",
    "example": "Elaboration",
    "definition": "Elaboration",
    "enablement": "Enablement",
    "purpose": "Enablement",
    "evaluation": "Evaluation",
    "interpretation": "Evaluation",
    "conclusion": "Evaluation",
    "comment": "Evaluation",
    "evidence": "Explanation",
    "explanation-argumentative": "Explanation",
    "reason": "Explanation",
    "list": "Joint",
    "disjunction": "Joint",
    "manner": "Manner-Means",
    "means": "Manner-Means",
    "problem-solution": "Topic-Comment",
    "question-answer": "Topic-Comment",
    "statement-response": "Topic-Comment",
    "topic-comment": "Topic-Comment",
    "comment-topic": "Topic-Comment",
    "rhetorical-question": "Topic-Comment",
    "summary": "Summary",
    "restatement": "Summary",
    "temporal-before": "Temporal",
    "temporal-after": "Temporal",
    "temporal-same-time": "Temporal",
    "sequence": "Temporal",
    "inverted-sequence": "Temporal",
    "topic-shift": "Topic-Change",
    "topic-drift": "Topic-Change",
    "textualorganization": "Textual-organization",
    "same-unit": "Same-unit"
}
