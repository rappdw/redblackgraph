from ._shortest_path import shortest_path, floyd_warshall
from .transitive_closure import transitive_closure, transitive_closure_dijkstra, transitive_closure_floyd_warshall
from ._components import (
    find_components,
    find_components_sparse,
    extract_submatrix,
    merge_component_matrices,
    get_component_vertices,
)
from ._ordering import avos_canonical_ordering, _get_permutation
from ._permutation import permute, permute_sparse
from ._topological_sort import topological_sort, topological_ordering, is_upper_triangular
from ._relational_composition import vertex_relational_composition, edge_relational_composition
from .cycleerror import CycleError

# Sparse utilities
from ._sparse_format import (
    is_sparse,
    is_dense,
    get_density,
    get_nnz,
    ensure_csr,
    ensure_csc,
    ensure_coo,
    to_dense,
    to_dense_if_needed,
    get_format,
    csr_csc_pair,
    empty_csr,
    identity_csr,
)
from ._density import (
    DensityMonitor,
    DensificationError,
    DensificationWarning,
    check_density,
    assert_sparse,
)
