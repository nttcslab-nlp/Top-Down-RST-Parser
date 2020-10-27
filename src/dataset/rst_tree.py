try:
    if __name__ == '__main__':
        from trees import InternalTreebankNode, LeafTreebankNode, InternalParseNode, LeafParseNode
    else:
        from dataset.trees import InternalTreebankNode, LeafTreebankNode, InternalParseNode, LeafParseNode
except ImportError:
    print(
        """
We use [trees.py](https://github.com/mitchellstern/minimal-span-parser/blob/master/src/trees.py) in our code.
Please put it in `src/dataset/`
"""
    )
    exit(-1)


def load_tree_from_string(tree_string):
    tokens = tree_string.replace("(", " ( ").replace(")", " ) ").split()

    def helper(index):
        trees = []
        while index < len(tokens) and tokens[index] == "(":
            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                trees.append(InternalTreebankNode(label, children))
            else:
                word = tokens[index]
                index += 1
                trees.append(LeafTreebankNode(label, word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

        return trees, index

    trees, index = helper(0)
    assert index == len(tokens)
    assert len(trees) == 1
    return trees[0]


if __name__ == '__main__':
    tree = load_tree_from_string(
        """
(nucleus-satellite:Elaboration
    (nucleus-satellite:Elaboration
        (nucleus-satellite:Elaboration
            (nucleus-nucleus:Temporal (text 0) (text 1))
            (nucleus-satellite:Elaboration
                (nucleus-nucleus:Same-Unit
                    (nucleus-satellite:Summary (text 2) (text 3))
                    (text 4))
                (nucleus-nucleus:Joint (text 5)
                                       (nucleus-nucleus:Joint (text 6) (text 7)))))
        (nucleus-satellite:Elaboration
            (nucleus-satellite:Elaboration
               (nucleus-nucleus:Same-Unit
                   (nucleus-satellite:Summary (text 8) (text 9))
                   (nucleus-satellite:Summary (text 10) (text 11)))
               (text 12))
            (text 13)))
    (nucleus-satellite:Elaboration
        (nucleus-nucleus:Same-Unit
            (nucleus-satellite:Summary (text 14) (text 15))
            (text 16))
        (nucleus-satellite:Elaboration (text 17) (text 18))))
"""
    )
    print(tree.linearize())
    print('--')
    print(tree.convert().convert().linearize())
    print('--')
    t = tree.convert()
    assert t.oracle_label(0, 8)[0] == ':'.join(('nucleus-satellite', 'Elaboration'))
    assert t.oracle_splits(0, 8)[0] == 2
    assert t.oracle_label(10, 12)[0] == ':'.join(('nucleus-satellite', 'Summary'))
    assert t.oracle_splits(10, 12)[0] == 11
    print(t.oracle_label(0, 19)[0])
    print(t.oracle_splits(0, 19)[0])

    print(t.oracle_label(1, 18))
    print(t.oracle_splits(1, 18))
