#!/usr/bin/env python3
"""
Export canonical graphs to Excel/CSV with vertex names as row/column labels.

This script loads a canonical graph from cache and exports it with proper vertex
labels retrieved from the database.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import redblackgraph as rb
from scipy.sparse import coo_matrix
from fscrawler import RelationshipDbReader
from redblackgraph.util.graph_builder import RbgGraphBuilder
from redblackgraph.util.relationship_file_io import RedBlackGraphWriter


def load_canonical_graph(cache_path: Path):
    """Load a canonical graph from NPZ cache file.
    
    Args:
        cache_path: Path to the canonical cache file
        
    Returns:
        tuple: (graph object, metadata dict, label_permutation or None)
    """
    base_path = cache_path.with_suffix('')
    npz_path = base_path.with_suffix('.npz')
    json_path = base_path.with_suffix('.json')
    
    if not npz_path.exists():
        raise FileNotFoundError(f"Cache file not found: {npz_path}")
    
    # Load metadata
    if not json_path.exists():
        raise FileNotFoundError(f"Cache metadata file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    if not metadata.get('canonical'):
        logging.warning(f"Graph at {cache_path} may not be in canonical form")
    
    # Load matrix data
    npz_data = np.load(npz_path)
    matrix_format = str(npz_data['format'][0])
    
    if matrix_format == 'sparse':
        data = npz_data['data']
        row = npz_data['row']
        col = npz_data['col']
        shape = tuple(npz_data['shape'])
        coo = coo_matrix((data, (row, col)), shape=shape)
        graph = rb.rb_matrix(coo)
    else:
        array = npz_data['array']
        graph = rb.array(array)
    
    # Load label permutation if present
    label_permutation = None
    if 'label_permutation' in npz_data:
        label_permutation = npz_data['label_permutation']
    
    return graph, metadata, label_permutation


def get_vertex_names(db_path: Path, hops: int):
    """Get vertex names from the database.
    
    Args:
        db_path: Path to the SQLite database
        hops: Hop count for filtering
        
    Returns:
        dict: Mapping of vertex_id to (external_id, name) tuples
    """
    builder = RbgGraphBuilder()
    reader = RelationshipDbReader(str(db_path), hops, builder)
    return reader.get_vertex_key()


class CanonicalVertexInfo:
    """Adapter to provide vertex info for canonical graphs."""
    
    def __init__(self, vertex_key: dict):
        self.vertex_key = vertex_key
    
    def get_vertex_key(self):
        return self.vertex_key


def export_graph(graph, vertex_key: dict, output_path: Path, max_vertices: int = None):
    """Export graph to Excel with vertex labels.
    
    Args:
        graph: The graph to export
        vertex_key: Dictionary mapping vertex_id to (external_id, name)
        output_path: Output file path (.xlsx)
        max_vertices: Maximum number of vertices to export (for large graphs)
    """
    vertices = graph.shape[0]
    
    if max_vertices and vertices > max_vertices:
        logging.warning(f"Graph has {vertices:,} vertices, limiting export to {max_vertices:,}")
        # Create a subgraph with the first N vertices
        if hasattr(graph, 'tocsr'):
            # Sparse matrix
            subgraph = graph[:max_vertices, :max_vertices]
        else:
            # Dense array
            subgraph = rb.array(graph[:max_vertices, :max_vertices])
        
        # Filter vertex key to match
        filtered_key = {i: vertex_key[i] for i in range(max_vertices) if i in vertex_key}
        vertex_info = CanonicalVertexInfo(filtered_key)
    else:
        subgraph = graph
        vertex_info = CanonicalVertexInfo(vertex_key)
    
    # Export using RedBlackGraphWriter
    writer = RedBlackGraphWriter(vertex_info)
    logging.info(f"Exporting graph to {output_path}...")
    writer.write(subgraph, str(output_path))
    logging.info(f"Export complete: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export canonical graphs to Excel with vertex names",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export canonical graph for hop count 10
  %(prog)s --hop-count 10
  
  # Export with custom paths
  %(prog)s --hop-count 10 \\
      --cache-dir /mnt/nas/data/rbg/rappdw-01-13-22 \\
      --db-path /mnt/nas/data/rbg/rappdw-01-13-22/rappdw.db \\
      --output output.xlsx
  
  # Limit large graph export to first 1000 vertices
  %(prog)s --hop-count 12 --max-vertices 1000
        """
    )
    parser.add_argument(
        '--hop-count',
        type=int,
        required=True,
        help='Hop count of the canonical graph to export'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path('/mnt/nas/data/rbg/rappdw-01-13-22'),
        help='Directory containing cached canonical graphs (default: %(default)s)'
    )
    parser.add_argument(
        '--db-path',
        type=Path,
        default=Path('/mnt/nas/data/rbg/rappdw-01-13-22/rappdw.db'),
        help='Path to SQLite database with vertex names (default: %(default)s)'
    )
    parser.add_argument(
        '--base-name',
        type=str,
        default='rappdw',
        help='Base name for graph files (default: %(default)s)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output file path (.xlsx). Default: {base_name}_canonical_hops{N}.xlsx in cache dir'
    )
    parser.add_argument(
        '--max-vertices',
        type=int,
        default=None,
        help='Maximum number of vertices to export (useful for large graphs)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO
    )
    
    # Construct paths
    canonical_cache = args.cache_dir / f"{args.base_name}_canonical_hops{args.hop_count}.npz"
    
    if args.output:
        output_path = args.output
    else:
        # Place output in the same directory as the input cache file
        output_path = args.cache_dir / f"{args.base_name}_canonical_hops{args.hop_count}.xlsx"
    
    # Check Excel size limits
    MAX_EXCEL_COLS = 16384
    
    if not canonical_cache.exists():
        logging.error(f"Canonical graph cache not found: {canonical_cache}")
        logging.info(f"Run compute_canonical_forms.py first to generate canonical graphs")
        sys.exit(1)
    
    if not args.db_path.exists():
        logging.error(f"Database not found: {args.db_path}")
        sys.exit(1)
    
    try:
        # Load canonical graph
        logging.info(f"Loading canonical graph from {canonical_cache}")
        graph, metadata, label_permutation = load_canonical_graph(canonical_cache)
        vertices = graph.shape[0]
        nnz = graph.nnz if hasattr(graph, 'nnz') else np.count_nonzero(graph)
        logging.info(f"Loaded: {vertices:,} vertices, {nnz:,} edges")
        
        # Check size limits
        if vertices > MAX_EXCEL_COLS and not args.max_vertices:
            logging.error(f"Graph has {vertices:,} vertices, exceeds Excel column limit ({MAX_EXCEL_COLS:,})")
            logging.info(f"Use --max-vertices to export a subset")
            sys.exit(1)
        
        # Get vertex names from database
        logging.info(f"Loading vertex names from {args.db_path}")
        vertex_key = get_vertex_names(args.db_path, args.hop_count)
        logging.info(f"Loaded {len(vertex_key):,} vertex names")
        
        # Apply permutation to vertex labels if available
        if label_permutation is not None:
            logging.info("Applying canonical ordering permutation to vertex labels")
            # Create new vertex_key with reordered labels
            # label_permutation[i] = original_vertex_id for new position i
            reordered_vertex_key = {}
            for new_pos in range(len(label_permutation)):
                original_pos = label_permutation[new_pos]
                if original_pos in vertex_key:
                    reordered_vertex_key[new_pos] = vertex_key[original_pos]
            vertex_key = reordered_vertex_key
        
        # Export
        export_graph(graph, vertex_key, output_path, args.max_vertices)
        
        logging.info(f"✓ Export complete!")
        
    except Exception as e:
        logging.error(f"✗ Export failed: {e}", exc_info=args.verbose)
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
