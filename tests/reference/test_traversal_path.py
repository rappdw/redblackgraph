from redblackgraph.reference.generation import get_traversal_path

def test_traversal_path():
    path = range(16)
    expected = [
        [],  # 0
        [],  # 1
        ['r'],  # 2
        ['b'],  # 3
        ['r', 'r'],  # 4
        ['r', 'b'],  # 5
        ['b', 'r'],  # 6
        ['b', 'b'],  # 7
        ['r', 'r', 'r'],  # 8
        ['r', 'r', 'b'],  # 9
        ['r', 'b', 'r'],  # 10
        ['r', 'b', 'b'],  # 11
        ['b', 'r', 'r'],  # 12
        ['b', 'r', 'b'],  # 13
        ['b', 'b', 'r'],  # 14
        ['b', 'b', 'b'],  # 15
    ]
    for i in path:
        assert get_traversal_path(i) == expected[i]
