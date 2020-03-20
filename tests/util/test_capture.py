from redblackgraph.util.capture import capture

def test_capture():
    A = [[-1, 0, 0, 2, 0,13, 0],
         [ 0,-1, 0, 0, 0, 0, 0],
         [ 2, 0, 1, 0, 0, 0, 0],
         [ 0, 0, 0,-1, 0, 0, 0],
         [ 0, 2, 0, 0,-1, 0, 3],
         [ 0, 0, 0, 0, 0, 1, 0],
         [ 0, 0, 0, 0, 0, 0, 1]]
    data = capture(A)
    assert data == '''[
  [-1, 0, 0, 2, 0,13, 0],
  [ 0,-1, 0, 0, 0, 0, 0],
  [ 2, 0, 1, 0, 0, 0, 0],
  [ 0, 0, 0,-1, 0, 0, 0],
  [ 0, 2, 0, 0,-1, 0, 3],
  [ 0, 0, 0, 0, 0, 1, 0],
  [ 0, 0, 0, 0, 0, 0, 1]
]'''