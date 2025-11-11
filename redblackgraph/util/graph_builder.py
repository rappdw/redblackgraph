from typing import Sequence
from pathlib import Path
import json

import numpy as np
import redblackgraph as rb
import time

from fscrawler import AbstractGraphBuilder, RelationshipDbReader
from scipy.sparse import coo_matrix
from redblackgraph.sparse.csgraph import avos_canonical_ordering


class RbgGraphBuilder(AbstractGraphBuilder):

    def __init__(self, sparse_threshold: int = 1000):
        super().__init__(sparse_threshold)
        self.idx = None
        self.val = None
        self.row = None
        self.col = None
        self.genders = None
        self.graph = None
        self.nv = None

    def init_builder(self, nv: int, ne: int):
        self._init_status(nv, ne)
        self.nv = nv
        print(f"DEBUG: init_builder called with {nv:,} vertices, {ne:,} edges")
        if nv > self.sparse_threshold:
            self.graph = None
            self.val = np.zeros(ne + nv, dtype=np.int32)
            self.row = np.zeros(ne + nv, dtype=np.int32)
            self.col = np.zeros(ne + nv, dtype=np.int32)
            self.genders = np.zeros(nv, dtype=np.int32)
            self.idx = ne + nv - 1
        else:
            self.idx = None
            self.val = None
            self.row = None
            self.col = None
            self.genders = None
            self.graph = rb.array(np.zeros((nv, nv), dtype=np.int32))

    def get_ordering(self) -> Sequence[int]:
        if self.graph is None:
            self.graph = rb.rb_matrix(coo_matrix((self.val, (self.row, self.col)), shape=(self.nv, self.nv)))
        
        # For very large graphs, skip canonical ordering to avoid memory issues
        # TODO: Implement sparse transitive closure or alternative ordering method
        if self.nv > self.sparse_threshold:
            import logging
            logging.getLogger(__name__).warning(
                f"Graph too large ({self.nv:,} vertices) for canonical ordering. "
                f"Using topological ordering instead."
            )
            # Return identity ordering (or could use topological sort if available)
            return list(range(self.nv))
        
        start_time = time.time()
        print(f"Computing transitive closure...")
        closure = self.graph.transitive_closure().W
        duration = time.time() - start_time
        print(f"Transitive closure computed, duration: {duration:.2f}s")

        start_time = time.time()
        print(f"Computing canonical ordering...")
        ordering = avos_canonical_ordering(closure)
        duration = time.time() - start_time
        print(f"Canonical ordering computed, duration: {duration:.2f}s")
        return ordering.label_permutation

    def add_vertex(self, vertex_id: int, color: int):
        self._track_vertex()
        if self.graph is not None:
            self.graph[vertex_id, vertex_id] = color
        else:
            self.val[self.idx] = color
            self.row[self.idx] = vertex_id
            self.col[self.idx] = vertex_id
            self.idx -= 1

    def add_edge(self, source_id: int, dest_id: int):
        self._track_edge()
        if self.graph is not None:
            self.graph[source_id, dest_id] = 3 if self.graph[dest_id, dest_id] == 1 else 2
        else:
            self.val[self.idx] = 3 if self.genders[dest_id] == 1 else 2
            self.row[self.idx] = source_id
            self.col[self.idx] = dest_id
            self.idx -= 1

    def add_gender(self, vertex_id: int, color: int):
        if self.graph is None:
            self.genders[vertex_id] = color

    def build(self):
        self._build_status()
        if self.graph is not None:
            rtn_graph = self.graph
        else:
            rtn_graph = rb.rb_matrix(coo_matrix((self.val, (self.row, self.col))))
            self.val = None
            self.row = None
            self.col = None
            self.genders = None
        return rtn_graph
    
    def save_cache(self, graph, cache_path: Path, metadata: dict):
        """Save the built graph to a cache file using NPZ format.
        
        Args:
            graph: The rb_matrix or rb.array graph object to cache
            cache_path: Path where the cache file should be saved (base name without extension)
            metadata: Dictionary containing metadata like {'hops': int or None}
        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure consistent naming: remove any extension and explicitly add .npz
        base_path = cache_path.with_suffix('')
        npz_path = base_path.with_suffix('.npz')
        json_path = base_path.with_suffix('.json')
        
        # Determine if sparse or dense and extract underlying data
        save_data = {}
        
        # Check if it's a sparse matrix (has .tocoo method)
        if hasattr(graph, 'tocoo'):
            # Convert to COO format for saving
            coo = graph.tocoo()
            save_data['format'] = np.array(['sparse'], dtype='U10')
            save_data['data'] = coo.data
            save_data['row'] = coo.row
            save_data['col'] = coo.col
            save_data['shape'] = np.array(coo.shape)
            vertices = int(coo.shape[0])
            edges = int(len(coo.data))
        else:
            # Dense array - extract underlying numpy array
            save_data['format'] = np.array(['dense'], dtype='U10')
            # Handle rb.array wrapper
            if hasattr(graph, 'W'):
                array_data = graph.W
            else:
                array_data = np.asarray(graph)
            save_data['array'] = array_data
            vertices = int(array_data.shape[0])
            edges = int(np.count_nonzero(array_data))
        
        # Save matrix data to NPZ (numpy will NOT add .npz if we include it in the path)
        np.savez_compressed(str(npz_path), **save_data)
        
        # Save metadata to companion JSON file with graph statistics
        metadata_with_version = {
            **metadata,
            'version': '1.0',
            'cache_format': 'npz',
            'vertices': vertices,
            'edges': edges
        }
        with open(json_path, 'w') as f:
            json.dump(metadata_with_version, f, indent=2)
        
        self.logger.info(f"Saved graph cache to {npz_path} + {json_path}")
    
    def load_cache(self, cache_path: Path, expected_metadata: dict):
        """Load a graph from a cache file in NPZ format.
        
        Args:
            cache_path: Path to the cache file (base name, will look for .npz and .json)
            expected_metadata: Dictionary with expected metadata like {'hops': int or None}
            
        Returns:
            The loaded rb_matrix or rb.array graph object
            
        Raises:
            ValueError: If metadata doesn't match expected values
            FileNotFoundError: If cache file doesn't exist
        """
        # Ensure consistent naming: remove any extension and explicitly add .npz/.json
        base_path = cache_path.with_suffix('')
        npz_path = base_path.with_suffix('.npz')
        json_path = base_path.with_suffix('.json')
        
        if not npz_path.exists():
            raise FileNotFoundError(f"Cache file not found: {npz_path}")
        
        # Load and verify metadata
        if not json_path.exists():
            raise FileNotFoundError(f"Cache metadata file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            cached_metadata = json.load(f)
        
        # Verify metadata matches
        for key, expected_value in expected_metadata.items():
            cached_value = cached_metadata.get(key)
            if cached_value != expected_value:
                raise ValueError(
                    f"Cache metadata mismatch for '{key}': "
                    f"expected {expected_value}, got {cached_value}"
                )
        
        # Load matrix data from NPZ
        npz_data = np.load(npz_path)
        matrix_format = str(npz_data['format'][0])
        
        if matrix_format == 'sparse':
            # Reconstruct sparse matrix
            data = npz_data['data']
            row = npz_data['row']
            col = npz_data['col']
            shape = tuple(npz_data['shape'])
            coo = coo_matrix((data, (row, col)), shape=shape)
            graph = rb.rb_matrix(coo)
            vertices = int(shape[0])
            edges = int(len(data))
        else:
            # Reconstruct dense array
            array = npz_data['array']
            graph = rb.array(array)
            vertices = int(array.shape[0])
            edges = int(np.count_nonzero(array))
        
        # Update JSON if it's missing vertex/edge counts (backward compatibility)
        if 'vertices' not in cached_metadata or 'edges' not in cached_metadata:
            self.logger.info(f"Updating cache metadata with vertices={vertices}, edges={edges}")
            cached_metadata['vertices'] = vertices
            cached_metadata['edges'] = edges
            with open(json_path, 'w') as f:
                json.dump(cached_metadata, f, indent=2)
        
        self.logger.info(f"Loaded graph cache from {npz_path} ({vertices} vertices, {edges} edges)")
        return graph
    
    def is_cache_valid(self, cache_path: Path, reference_path: Path) -> bool:
        """Check if cache files exist and are newer than the reference file.
        
        Args:
            cache_path: Base path for the cache file (will check .npz and .json)
            reference_path: Path to reference file (e.g., database) to compare modification times
            
        Returns:
            True if both cache files exist and are newer than reference_path, False otherwise
        """
        # Normalize to base path and add extensions
        base_path = cache_path.with_suffix('')
        npz_path = base_path.with_suffix('.npz')
        json_path = base_path.with_suffix('.json')
        
        # Both files must exist
        if not npz_path.exists() or not json_path.exists():
            return False
        
        if not reference_path.exists():
            return False
        
        # Cache is valid if it's newer than the reference file
        cache_mtime = npz_path.stat().st_mtime
        ref_mtime = reference_path.stat().st_mtime
        return cache_mtime > ref_mtime


def main():
    """Main entry point for graph builder script."""
    import argparse
    import time
    import logging
    
    # Configure logging to see fs-crawler messages
    logging.basicConfig(
        format='%(levelname)s [%(name)s]: %(message)s',
        level=logging.INFO
    )
    
    parser = argparse.ArgumentParser(
        description="Load a relationship database and compute its transitive closure"
    )
    parser.add_argument(
        "-d", "--database",
        type=str,
        default="/mnt/nas/data/rbg/rappdw-01-13-22/rappdw.db",
        help="Path to the relationship database file (default: %(default)s)"
    )
    parser.add_argument(
        "-c", "--hopcount",
        type=int,
        default=4,
        help="Number of hops to include in graph (default: %(default)s)"
    )
    
    args = parser.parse_args()
    
    start = time.time()
    print(f"Loading graph from DB: {args.database} (hopcount={args.hopcount})")
    builder = RbgGraphBuilder()
    print(f"DEBUG: Creating RelationshipDbReader with hops={args.hopcount}")
    rdr = RelationshipDbReader(args.database, args.hopcount, builder)
    print(f"DEBUG: RelationshipDbReader.hops = {rdr.hops}")
    g = rdr.read(use_cache=True)
    print(f"Graph loaded, duration: {time.time() - start:.2f}s")
    # Handle both dense and sparse matrices
    nnz = g.nnz if hasattr(g, 'nnz') else np.count_nonzero(g)
    print(f"Graph shape: {g.shape[0]:,} x {g.shape[1]:,}, {nnz:,} non-zero entries")

if __name__ == "__main__":
    main()
