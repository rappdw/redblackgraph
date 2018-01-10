from redblackgraph.simple.util import nz_min


def test_nz_min():
    mini = nz_min(0, 4)
    assert mini == 4