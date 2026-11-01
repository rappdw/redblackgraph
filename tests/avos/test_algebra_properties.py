"""
Property tests for the theorems in docs/MATHEMATICAL_FOUNDATIONS.md.

Each test names the result it checks. Graph-level tests run on seeded random
Red Black Graphs (deterministic, so failures reproduce), and compare every
backend against a brute-force enumeration of all paths.

Encoding reminder: a nonzero value x encodes the word given by the bits of x
after its leading 1 (0 = red, 1 = black). r(u, v) is the shortlex-minimal word
over all paths u -> v. The legacy diagonal stores -1 (red) / 1 (black).
"""
import itertools
import random

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import redblackgraph as rb
from redblackgraph import core, sparse
from redblackgraph import reference as ref
from redblackgraph.sparse import rb_matrix
from redblackgraph.sparse.csgraph import _rbg_math

RED, BLACK = 0, 1
SIGMA = {RED: -1, BLACK: 1}
SEEDS = range(40)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def msb(x):
    return x.bit_length() - 1


def word(x):
    """Word encoded by positive x: the bits after the leading 1."""
    return bin(x)[3:]


def encode(w):
    return int("1" + w, 2)


def shortlex_key(w):
    return (len(w), w)


def random_rbg(seed, n=8, p=0.75):
    """
    A random Red Black Graph in the legacy encoding.

    Returns (R, colors, parents) where R has -1/1 on the diagonal and 2/3 for
    edges, colors[v] is RED/BLACK, and parents[u][color] is u's parent of that
    color (or None). Vertices are shuffled so the matrix is not triangular.
    """
    rng = random.Random(seed)
    colors = [rng.choice((RED, BLACK)) for _ in range(n)]
    parents = [{RED: None, BLACK: None} for _ in range(n)]
    for u in range(n):
        for c in (RED, BLACK):
            candidates = [v for v in range(u + 1, n) if colors[v] == c]
            if candidates and rng.random() < p:
                parents[u][c] = rng.choice(candidates)
    perm = list(range(n))
    rng.shuffle(perm)
    inv = {old: new for new, old in enumerate(perm)}
    colors = [colors[perm[i]] for i in range(n)]
    parents = [{c: (None if parents[perm[i]][c] is None else inv[parents[perm[i]][c]])
                for c in (RED, BLACK)} for i in range(n)]
    return to_matrix(colors, parents), colors, parents


def to_matrix(colors, parents):
    n = len(colors)
    R = [[0] * n for _ in range(n)]
    for u in range(n):
        R[u][u] = SIGMA[colors[u]]
        for c, v in parents[u].items():
            if v is not None:
                R[u][v] = 2 + c
    return R


def brute_force_closure(colors, parents):
    """r(u, v) by enumerating every path (Definition 7), legacy diagonal."""
    n = len(colors)
    best = [[None] * n for _ in range(n)]

    def walk(u, x, w):
        if best[u][x] is None or shortlex_key(w) < shortlex_key(best[u][x]):
            best[u][x] = w
        for c, y in parents[x].items():
            if y is not None:
                walk(u, y, w + str(c))

    for u in range(n):
        walk(u, u, "")
    return [[SIGMA[colors[u]] if u == v else (0 if best[u][v] is None else encode(best[u][v]))
             for v in range(n)] for u in range(n)]


def bfs_distance(parents, u, v):
    frontier, seen, d = {u}, {u}, 0
    while frontier:
        if v in frontier:
            return d
        frontier = {y for x in frontier for y in parents[x].values()
                    if y is not None and y not in seen}
        seen |= frontier
        d += 1
    return None


def dense(W):
    return np.asarray(W.toarray() if hasattr(W, "toarray") else W, dtype=np.int64)


def as_rb_matrix(R):
    return rb_matrix(csr_matrix(np.array(R, dtype=np.int32)))


def sparse_squaring_closure(R):
    return sparse.transitive_closure_squaring(as_rb_matrix(R))


CLOSURES = {
    "reference": lambda R: ref.transitive_closure(R),
    "core": lambda R: core.transitive_closure(rb.array(R, dtype=np.int32)),
    "sparse_floyd_warshall": sparse.transitive_closure_floyd_warshall,
    "sparse_dijkstra": sparse.transitive_closure_dijkstra,
    "sparse_dag": lambda R: sparse.transitive_closure_dag_sparse(as_rb_matrix(R)),
    "sparse_squaring": sparse_squaring_closure,
    "sparse_adaptive": sparse.transitive_closure_adaptive,
    "sparse_component_wise": sparse.component_wise_closure,
}

PRODUCTS = {
    "reference": ref.avos_product,
    "core": core.avos_product,
    "sparse": sparse.avos_product,
    "cython": _rbg_math.py_avos_product,
}

PATH_VALUES = [0] + list(range(2, 40))  # 0 plus nonzero non-identity values


# --------------------------------------------------------------------------
# Section 2: encoding and order
# --------------------------------------------------------------------------

def test_lemma2_integer_order_is_shortlex():
    for a, b in itertools.product(range(1, 256), repeat=2):
        assert (a < b) == (shortlex_key(word(a)) < shortlex_key(word(b)))


def test_lemma3_shortlex_compatible_with_concatenation():
    words = [word(x) for x in range(1, 64)]
    for x, y, z in itertools.product(words[:16], words, words):
        if shortlex_key(y) <= shortlex_key(z):
            assert shortlex_key(x + y) <= shortlex_key(x + z)
            assert shortlex_key(y + x) <= shortlex_key(z + x)


# --------------------------------------------------------------------------
# Section 3: the avos semiring (Theorem 1)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", PRODUCTS)
def test_definition5_product_is_word_concatenation(name):
    product = PRODUCTS[name]
    for x, y in itertools.product(range(2, 300), range(2, 300)):
        assert product(x, y) == encode(word(x) + word(y)), (x, y)
    for x in range(0, 50):
        assert product(0, x) == 0 and product(x, 0) == 0


def test_theorem1_sum_is_selective_commutative_monoid():
    values = list(range(0, 40))
    for x, y, z in itertools.product(values, repeat=3):
        assert ref.avos_sum(x, y) == ref.avos_sum(y, x)
        assert ref.avos_sum(ref.avos_sum(x, y), z) == ref.avos_sum(x, ref.avos_sum(y, z))
    for x in values:
        assert ref.avos_sum(x, 0) == x
        assert ref.avos_sum(x, x) == x
    for x, y in itertools.product(values, repeat=2):
        assert ref.avos_sum(x, y) in (x, y)


def test_theorem1_product_is_associative_on_paths():
    p = ref.avos_product
    for x, y, z in itertools.product(PATH_VALUES, repeat=3):
        assert p(p(x, y), z) == p(x, p(y, z)), (x, y, z)


def test_theorem1_product_distributes_over_sum():
    p, s = ref.avos_product, ref.avos_sum
    for x, y, z in itertools.product(PATH_VALUES, repeat=3):
        assert p(x, s(y, z)) == s(p(x, y), p(x, z)), (x, y, z)
        assert p(s(y, z), x) == s(p(y, x), p(z, x)), (x, y, z)


def test_theorem1_one_is_bounded():
    for x in range(0, 200):
        assert ref.avos_sum(1, x) == 1


def test_proposition4_legacy_identities_are_parity_filters():
    """The +/-1 rules of the legacy encoding (table in section 6)."""
    p = ref.avos_product
    for x in range(2, 200):
        assert p(-1, x) == x and p(1, x) == x  # left identity
        assert p(x, -1) == (x if x % 2 == 0 else 0)  # right filter, red
        assert p(x, 1) == (x if x % 2 == 1 else 0)  # right filter, black
    assert p(-1, -1) == -1 and p(1, 1) == 1
    assert p(-1, 1) == 0 and p(1, -1) == 0


# --------------------------------------------------------------------------
# Section 4: closure (Theorem 2) and section 6 (Proposition 4)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("backend", CLOSURES)
def test_theorem2_closure_is_shortlex_minimal_path(backend, seed):
    R, colors, parents = random_rbg(seed)
    expected = np.array(brute_force_closure(colors, parents))
    result = CLOSURES[backend]([row[:] for row in R])
    np.testing.assert_array_equal(dense(result.W), expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_theorem2_repeated_squaring_core(seed):
    """Theorem 2(d): ceil(log2 n) squarings of R reach the closure."""
    R, colors, parents = random_rbg(seed)
    M = rb.array(R, dtype=np.int32)
    for _ in range(max(1, (len(R) - 1).bit_length())):
        M = M @ M
    np.testing.assert_array_equal(np.asarray(M), brute_force_closure(colors, parents))


@pytest.mark.parametrize("seed", SEEDS)
def test_theorem2_repeated_squaring_sparse(seed):
    R, colors, parents = random_rbg(seed)
    M = as_rb_matrix(R)
    for _ in range(max(1, (len(R) - 1).bit_length())):
        M = M @ M
    np.testing.assert_array_equal(dense(M), brute_force_closure(colors, parents))


@pytest.mark.parametrize("seed", SEEDS)
def test_theorem2_dag_recurrence(seed):
    """R*_u = e_u (+) sum over edges (u, v) of A_uv (x) R*_v (the DAG algorithm)."""
    R, colors, parents = random_rbg(seed)
    W = brute_force_closure(colors, parents)
    n = len(R)
    for u in range(n):
        row = [SIGMA[colors[u]] if j == u else 0 for j in range(n)]
        for v in range(n):
            if u != v and R[u][v]:
                row = [ref.avos_sum(a, ref.avos_product(R[u][v], b)) for a, b in zip(row, W[v])]
        assert row == W[u]


# --------------------------------------------------------------------------
# Section 5: structure of the closure (Theorem 3)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_theorem3_row_and_column_structure(seed):
    R, colors, parents = random_rbg(seed)
    W = dense(ref.transitive_closure([row[:] for row in R]).W)
    n = len(R)
    for u in range(n):
        assert W[u][u] == SIGMA[colors[u]]
        nonzero = [int(W[u][v]) for v in range(n) if v != u and W[u][v]]
        assert len(nonzero) == len(set(nonzero)), "values in a row must be distinct"
        for v in range(n):
            x = int(W[u][v])
            if u == v or x == 0:
                continue
            assert x % 2 == colors[v], "parity is the color of the target"
            assert msb(x) == bfs_distance(parents, u, v), "MSB is the generation count"
            y = u  # walking the encoded path reaches v
            for bit in word(x):
                y = parents[y][int(bit)]
            assert y == v


@pytest.mark.parametrize("seed", SEEDS)
def test_theorem3_triangle_inequality(seed):
    R, colors, parents = random_rbg(seed)
    W = brute_force_closure(colors, parents)
    n = len(R)
    for u, v, w in itertools.product(range(n), repeat=3):
        if len({u, v, w}) < 3:
            continue
        via = ref.avos_product(W[u][v], W[v][w])
        if via:
            assert W[u][w] != 0 and W[u][w] <= via


DIAMETER_BUG = pytest.mark.xfail(
    strict=True,
    reason="diameter is taken from the largest value seen during the run, including "
           "candidates later replaced by a shorter path, so it can overstate the diameter")


@pytest.mark.parametrize("backend", [
    pytest.param(b, marks=DIAMETER_BUG) if b not in ("sparse_dijkstra", "sparse_squaring") else b
    for b in CLOSURES
])
def test_theorem3_diameter(backend):
    """The diameter is MSB of the largest value in the final closure."""
    wrong = []
    for seed in range(200):
        R, colors, parents = random_rbg(seed)
        result = CLOSURES[backend]([row[:] for row in R])
        if result.diameter != msb(int(dense(result.W).max())):
            wrong.append(seed)
    assert not wrong, f"wrong diameter for seeds {wrong[:10]}"


# --------------------------------------------------------------------------
# Section 7: incremental updates (Theorems 5 and 6)
# --------------------------------------------------------------------------

def _ref_edge(W, a, b, rel):
    return ref.edge_relational_composition(np.asarray(W).tolist(), a, b, rel)


def _core_edge(W, a, b, rel):
    return core.edge_relational_composition(rb.array(W, dtype=np.int32), a, b, rel)


def _sparse_edge(W, a, b, rel):
    return sparse.edge_relational_composition(rb.array(W, dtype=np.int32), a, b, rel)


EDGE_COMPOSITIONS = {"reference": _ref_edge, "core": _core_edge, "sparse": _sparse_edge}


def _valid_new_edges(W, colors, parents):
    n = len(colors)
    for a, b in itertools.product(range(n), repeat=2):
        if a != b and parents[a][colors[b]] is None and W[b][a] == 0:
            yield a, b


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("impl", EDGE_COMPOSITIONS)
def test_theorem5_edge_composition(impl, seed):
    R, colors, parents = random_rbg(seed, p=0.5)
    W = brute_force_closure(colors, parents)
    edges = list(_valid_new_edges(W, colors, parents))
    for a, b in random.Random(seed).sample(edges, min(4, len(edges))):
        new_parents = [dict(p) for p in parents]
        new_parents[a][colors[b]] = b
        expected = brute_force_closure(colors, new_parents)
        result = EDGE_COMPOSITIONS[impl]([row[:] for row in W], a, b, 2 + colors[b])
        np.testing.assert_array_equal(dense(result), expected)


def _cycle_edge(seed):
    R, colors, parents = random_rbg(seed, p=0.9)
    W = brute_force_closure(colors, parents)
    for a, b in itertools.product(range(len(R)), repeat=2):
        if a != b and W[b][a] not in (0,):
            return W, a, b, 2 + colors[b]
    return None


@pytest.mark.parametrize("impl", [
    pytest.param("reference", marks=pytest.mark.xfail(
        strict=True, reason="reference edge composition does not check R*[beta][alpha]")),
    "core",
    pytest.param("sparse", marks=pytest.mark.xfail(
        strict=True, reason="sparse edge composition does not check R*[beta][alpha]")),
])
def test_theorem5_edge_composition_rejects_cycles(impl):
    case = _cycle_edge(0)
    assert case is not None
    W, a, b, rel = case
    with pytest.raises(ValueError):
        EDGE_COMPOSITIONS[impl]([row[:] for row in W], a, b, rel)


def _vertex_case(seed):
    """A random new vertex: (color, parent row u, child column v), valid by Theorem 6."""
    R, colors, parents = random_rbg(seed, p=0.5)
    W = brute_force_closure(colors, parents)
    n = len(R)
    rng = random.Random(seed + 1000)
    color = rng.choice((RED, BLACK))
    u = [0] * n
    for c in (RED, BLACK):
        candidates = [v for v in range(n) if colors[v] == c]
        if candidates and rng.random() < 0.7:
            u[rng.choice(candidates)] = 2 + c
    # ancestors of the new vertex: its parents and everything above them
    ancestors = {j for j in range(n) if any(u[k] and W[k][j] for k in range(n))}
    v = [0] * n
    for child in range(n):
        if parents[child][color] is None and rng.random() < 0.3:
            # the child and everything below it become descendants of the new vertex;
            # Theorem 6: acyclic iff no vertex is both an ancestor and a descendant
            descendants = {i for i in range(n) if W[i][child]}
            if not descendants & ancestors:
                v[child] = 2 + color
    new_colors = colors + [color]
    new_parents = [dict(p) for p in parents] + [{RED: None, BLACK: None}]
    for j in range(n):
        if u[j]:
            new_parents[n][colors[j]] = j
        if v[j]:
            new_parents[j][color] = n
    return W, u, v, SIGMA[color], brute_force_closure(new_colors, new_parents)


VERTEX_COMPOSITIONS = {
    "reference": lambda W, u, v, c: ref.vertex_relational_composition(
        [u], np.asarray(W).tolist(), [[x] for x in v], c),
    "core": lambda W, u, v, c: core.vertex_relational_composition(
        rb.array([u], dtype=np.int32), rb.array(W, dtype=np.int32),
        rb.array([[x] for x in v], dtype=np.int32), c),
    "sparse": lambda W, u, v, c: sparse.vertex_relational_composition(
        rb.array([u], dtype=np.int32), rb.array(W, dtype=np.int32),
        rb.array([[x] for x in v], dtype=np.int32), c),
}


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("impl", VERTEX_COMPOSITIONS)
def test_theorem6_vertex_composition(impl, seed):
    W, u, v, c, expected = _vertex_case(seed)
    result = VERTEX_COMPOSITIONS[impl]([row[:] for row in W], u, v, c)
    np.testing.assert_array_equal(dense(result), expected)


# --------------------------------------------------------------------------
# Cycle detection (note after Theorem 2)
# --------------------------------------------------------------------------

CYCLE = [[-1, 2, 0],
         [0, -1, 2],
         [2, 0, -1]]


@pytest.mark.parametrize("backend", [
    b if b != "sparse_squaring" else pytest.param(b, marks=pytest.mark.xfail(
        strict=True,
        reason="squaring never inspects the strict closure A(R*); the -1/1 diagonal hides the cycle"))
    for b in CLOSURES
])
def test_cycles_are_rejected(backend):
    with pytest.raises(Exception) as info:
        CLOSURES[backend]([row[:] for row in CYCLE])
    assert "cycle" in (type(info.value).__name__ + str(info.value)).lower()


# --------------------------------------------------------------------------
# Section 8: relationship calculation
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="calculate_relationship minimises the raw sum a + b, "
                                       "not the generation count MSB(a) + MSB(b)")
def test_section8_nearest_common_ancestor_minimises_generations():
    # ancestor 0 is at distances (1, 4); ancestor 1 is at (3, 3) but has the smaller raw sum
    a = [2, 8, 0]
    b = [16, 8, 0]
    assert ref.calculate_relationship(a, b).common_ancestor == 0


# --------------------------------------------------------------------------
# Section 10: standard ("canonical") ordering
# --------------------------------------------------------------------------

def _collapsed_pedigree():
    """
    Vertex 0's parents are 1 and 4, where 4 is also 1's great-grandparent
    (chain 1 -> 2 -> 3 -> 4 -> 5). Valid, but 0 has fewer total ancestor
    generations than its parent 1.
    """
    colors = [RED, RED, RED, RED, BLACK, RED]
    parents = [{RED: None, BLACK: None} for _ in colors]
    parents[0] = {RED: 1, BLACK: 4}
    for child, parent in ((1, 2), (2, 3), (3, 4), (4, 5)):
        parents[child][colors[parent]] = parent
    return brute_force_closure(colors, parents)


CANONICAL_ORDERINGS = {
    "reference": lambda W: ref.avos_canonical_ordering(W),
    "sparse": lambda W: sparse.avos_canonical_ordering(rb.array(W, dtype=np.int32)),
}


@pytest.mark.parametrize("impl", CANONICAL_ORDERINGS)
def test_section10_canonical_ordering_is_upper_triangular_random(impl):
    for seed in SEEDS:
        R, colors, parents = random_rbg(seed)
        A = dense(CANONICAL_ORDERINGS[impl](brute_force_closure(colors, parents)).A)
        assert np.array_equal(A, np.triu(A)), seed


@pytest.mark.parametrize("impl", CANONICAL_ORDERINGS)
@pytest.mark.xfail(strict=True, reason="ordering sorts by total ancestor generations, which can be "
                                       "smaller for a child than for its parent under pedigree collapse")
def test_section10_canonical_ordering_is_upper_triangular_collapse(impl):
    A = dense(CANONICAL_ORDERINGS[impl](_collapsed_pedigree()).A)
    assert np.array_equal(A, np.triu(A))
