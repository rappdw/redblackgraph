#!/usr/bin/env python3
"""
Script to compute canonical forms of cached graphs for performance benchmarking.

Reads cached base graphs, computes their canonical form (transitive closure + canonical ordering),
and saves the result with 'canonical' and hop count in the filename for hardware acceleration comparison.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from collections import deque

import numpy as np
import redblackgraph as rb
from scipy.sparse import coo_matrix
from redblackgraph.sparse.csgraph import avos_canonical_ordering
from fscrawler import RelationshipDbReader
from redblackgraph.util.graph_builder import RbgGraphBuilder


def decode_avos_value(edge_value):
    """Decode AVOS edge value to gender and generation information.
    
    AVOS values encode:
    - Gender via parity: even=male, odd=female
    - Generation via power of 2: 2=1st gen, 4=2nd gen, 8=3rd gen, etc.
    
    This is NOT the relationship type (BiologicalParent, etc.) - that's stored separately.
    
    Args:
        edge_value: Integer edge value from the graph
        
    Returns:
        str: Decoded gender and generation (e.g., "male parent", "female grandparent")
    """
    edge_val = int(edge_value)
    
    # Identity values
    if edge_val == -1:
        return "RED_ONE (male identity)"
    elif edge_val == 1:
        return "BLACK_ONE (female identity)"
    elif edge_val == 0:
        return "NULL"
    
    # Decode gender from parity
    is_female = (edge_val & 1) == 1
    gender = "female" if is_female else "male"
    
    # Calculate generation from value (2^n)
    import math
    abs_val = abs(edge_val)
    
    if abs_val <= 1:
        return f"{gender} self"
    
    generation = int(math.log2(abs_val))
    
    # Build generation description
    if generation == 1:
        return f"{gender} parent"
    elif generation == 2:
        return f"{gender} grandparent"
    elif generation == 3:
        return f"{gender} great-grandparent"
    else:
        greats = "great-" * (generation - 2)
        return f"{gender} {greats}grandparent"


def find_cycle_path(graph_matrix, start_vertex):
    """Find and return the cycle path starting from a given vertex using DFS.
    
    Ignores self-loops with identity values (-1, 0, 1) as these are valid in AVOS algebra.
    
    Args:
        graph_matrix: Sparse or dense matrix representing the graph
        start_vertex: Vertex ID where cycle was detected
        
    Returns:
        list: Path of vertices forming the cycle (length > 1), or None if no cycle found
    """
    # Convert to adjacency list for easier traversal
    if hasattr(graph_matrix, 'tocsr'):
        csr = graph_matrix.tocsr()
    else:
        csr = graph_matrix
    
    n = csr.shape[0]
    
    # Identity values that are allowed on diagonal
    IDENTITY_VALUES = {-1, 0, 1}
    
    def get_edge_value(i, j):
        """Get edge value from matrix."""
        if hasattr(csr, 'tocsr'):
            return csr[i, j]
        else:
            return csr[i, j] if len(csr.shape) > 1 else csr[i]
    
    def bfs_find_cycle(start):
        """BFS to find any cycle from start vertex, excluding identity self-loops."""
        from collections import deque
        
        queue = deque([(start, [start])])
        visited_paths = {start: [start]}
        
        while queue:
            current, path = queue.popleft()
            
            # Get neighbors
            if hasattr(csr, 'getrow'):
                row = csr.getrow(current)
                neighbors = [(idx, row.data[i]) for i, idx in enumerate(row.indices)]
            else:
                if len(csr.shape) > 1:
                    row_data = csr[current].toarray().flatten() if hasattr(csr[current], 'toarray') else csr[current]
                else:
                    row_data = csr[current]
                neighbors = [(idx, row_data[idx]) for idx in range(len(row_data)) if row_data[idx] != 0]
            
            for neighbor, edge_val in neighbors:
                # Skip identity self-loops (these are allowed)
                if neighbor == current and edge_val in IDENTITY_VALUES:
                    continue
                
                if neighbor == start and len(path) > 1:
                    # Found a cycle back to start
                    return path + [neighbor]
                
                if neighbor not in visited_paths or len(path) + 1 < len(visited_paths[neighbor]):
                    new_path = path + [neighbor]
                    visited_paths[neighbor] = new_path
                    queue.append((neighbor, new_path))
        
        return None
    
    # Try BFS from start vertex
    cycle = bfs_find_cycle(start_vertex)
    if cycle and len(cycle) > 2:  # Exclude trivial self-loops
        return cycle
    
    # If not found, try from all vertices
    for v in range(min(n, 1000)):  # Limit search to avoid taking too long
        cycle = bfs_find_cycle(v)
        if cycle and len(cycle) > 2:
            return cycle
    
    return None


def load_exclusions(config_path: Path):
    """Load exclusion configuration from JSON file.
    
    Args:
        config_path: Path to exclusions.json file
        
    Returns:
        tuple: (excluded_edges set, excluded_vertices set)
               where excluded_edges contains (from_id, to_id) tuples
               and excluded_vertices contains vertex IDs
    """
    if not config_path or not config_path.exists():
        return set(), set()
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        excluded_edges = set()
        for edge in config.get('excluded_edges', []):
            if 'from_id' in edge and 'to_id' in edge:
                excluded_edges.add((edge['from_id'], edge['to_id']))
        
        excluded_vertices = set()
        for vertex in config.get('excluded_vertices', []):
            if 'vertex_id' in vertex:
                excluded_vertices.add(vertex['vertex_id'])
        
        if excluded_edges or excluded_vertices:
            logging.info(f"Loaded exclusions: {len(excluded_edges)} edges, {len(excluded_vertices)} vertices")
        
        return excluded_edges, excluded_vertices
    except Exception as e:
        logging.warning(f"Could not load exclusions config: {e}")
        return set(), set()


def apply_exclusions(graph, vertex_key: dict, excluded_edges: set, excluded_vertices: set):
    """Apply exclusions to graph by removing specified edges and vertices.
    
    Args:
        graph: The graph object
        vertex_key: Mapping of vertex_id to (external_id, name) tuples
        excluded_edges: Set of (from_id, to_id) tuples to remove
        excluded_vertices: Set of vertex IDs to remove (removes all edges to/from)
        
    Returns:
        Filtered graph object
    """
    if not excluded_edges and not excluded_vertices:
        return graph
    
    logging.info("Applying exclusions...")
    
    # Get the underlying matrix
    graph_matrix = graph.W if hasattr(graph, 'W') else graph
    
    # Convert to COO for manipulation
    if hasattr(graph_matrix, 'tocoo'):
        coo = graph_matrix.tocoo()
    else:
        rows, cols = np.where(graph_matrix != 0)
        data = graph_matrix[rows, cols]
        coo = coo_matrix((data, (rows, cols)), shape=graph_matrix.shape)
    
    # Build reverse map: external_id -> internal vertex index
    ext_to_internal = {}
    for internal_idx, (ext_id, _) in vertex_key.items():
        ext_to_internal[ext_id] = internal_idx
    
    keep_mask = np.ones(len(coo.data), dtype=bool)
    removed_counts = {'edges': 0, 'vertices': 0}
    
    for idx, (i, j, val) in enumerate(zip(coo.row, coo.col, coo.data)):
        # Skip identity values on diagonal (always keep)
        if i == j and val in [-1, 0, 1]:
            continue
        
        # Get external IDs for this edge
        if i in vertex_key and j in vertex_key:
            ext_id_i, _ = vertex_key[i]
            ext_id_j, _ = vertex_key[j]
            
            # Check if this specific edge should be excluded
            if (ext_id_i, ext_id_j) in excluded_edges:
                keep_mask[idx] = False
                removed_counts['edges'] += 1
                continue
            
            # Check if either vertex should be excluded
            if ext_id_i in excluded_vertices or ext_id_j in excluded_vertices:
                keep_mask[idx] = False
                removed_counts['vertices'] += 1
                continue
    
    # Create filtered matrix
    filtered_data = coo.data[keep_mask]
    filtered_row = coo.row[keep_mask]
    filtered_col = coo.col[keep_mask]
    filtered_matrix = coo_matrix((filtered_data, (filtered_row, filtered_col)), shape=coo.shape)
    
    total_removed = len(coo.data) - len(filtered_data)
    if total_removed > 0:
        logging.info(f"  Removed {removed_counts['edges']} specific edges")
        logging.info(f"  Removed {removed_counts['vertices']} edges connected to excluded vertices")
        logging.info(f"  Total edges removed: {total_removed:,}")
    
    # Convert back to rb_matrix
    return rb.rb_matrix(filtered_matrix)


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


def get_relationship_types(db_path: Path, vertex_pairs: list):
    """Get relationship types from the database for specific vertex pairs.
    
    Args:
        db_path: Path to the SQLite database
        vertex_pairs: List of (source_id, dest_id) tuples (vertex IDs, not external IDs)
        
    Returns:
        dict: Mapping of (src_id, dst_id) -> relationship_type string
    """
    import sqlite3
    
    relationship_map = {}
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        for src_id, dst_id in vertex_pairs:
            # Query for relationship type between two vertices by their IDs
            query = """
                SELECT EDGE.type 
                FROM EDGE 
                WHERE EDGE.source = ? AND EDGE.destination = ?
            """
            cursor.execute(query, (src_id, dst_id))
            result = cursor.fetchone()
            
            if result:
                relationship_map[(src_id, dst_id)] = result[0]
        
        conn.close()
    except Exception as e:
        logging.warning(f"Could not load relationship types: {e}")
    
    return relationship_map


def load_cached_graph(cache_path: Path):
    """Load a graph from NPZ cache file.
    
    Args:
        cache_path: Path to the cache file (base name, will look for .npz and .json)
        
    Returns:
        tuple: (graph object, metadata dict)
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
    
    return graph, metadata


def save_canonical_graph(graph, cache_path: Path, original_metadata: dict, label_permutation=None):
    """Save canonical graph to NPZ cache file.
    
    Args:
        graph: The canonical rb_matrix or rb.array graph object
        cache_path: Path where the cache should be saved
        original_metadata: Metadata from the original graph
        label_permutation: Optional array mapping new positions to original vertex IDs
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    base_path = cache_path.with_suffix('')
    npz_path = base_path.with_suffix('.npz')
    json_path = base_path.with_suffix('.json')
    
    # Prepare data for saving
    save_data = {}
    
    if hasattr(graph, 'tocoo'):
        # Sparse matrix
        coo = graph.tocoo()
        save_data['format'] = np.array(['sparse'], dtype='U10')
        save_data['data'] = coo.data
        save_data['row'] = coo.row
        save_data['col'] = coo.col
        save_data['shape'] = np.array(coo.shape)
        vertices = int(coo.shape[0])
        edges = int(len(coo.data))
    else:
        # Dense array
        save_data['format'] = np.array(['dense'], dtype='U10')
        if hasattr(graph, 'W'):
            array_data = graph.W
        else:
            array_data = np.asarray(graph)
        save_data['array'] = array_data
        vertices = int(array_data.shape[0])
        edges = int(np.count_nonzero(array_data))
    
    # Add label permutation to save data if provided
    if label_permutation is not None:
        save_data['label_permutation'] = np.asarray(label_permutation, dtype=np.int32)
    
    # Save NPZ file
    np.savez_compressed(str(npz_path), **save_data)
    
    # Save metadata
    metadata = {
        **original_metadata,
        'canonical': True,
        'vertices': vertices,
        'edges': edges,
        'has_permutation': label_permutation is not None
    }
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Saved canonical graph to {npz_path} ({vertices:,} vertices, {edges:,} edges)")


def compute_canonical_form(input_path: Path, output_path: Path, db_path: Path = None, 
                          exclusions_path: Path = None):
    """Compute canonical form of a graph and save it.
    
    Args:
        input_path: Path to input cached graph
        output_path: Path where canonical graph should be saved
        db_path: Optional path to database for vertex name resolution
        exclusions_path: Optional path to exclusions.json file
    """
    logging.info(f"Loading base graph from {input_path}")
    start_time = time.time()
    graph, metadata = load_cached_graph(input_path)
    load_duration = time.time() - start_time
    
    vertices = graph.shape[0]
    nnz = graph.nnz if hasattr(graph, 'nnz') else np.count_nonzero(graph)
    logging.info(f"Loaded graph: {vertices:,} vertices, {nnz:,} edges (duration: {load_duration:.2f}s)")
    
    # Apply exclusions if configured
    if exclusions_path and db_path:
        excluded_edges, excluded_vertices = load_exclusions(exclusions_path)
        
        if excluded_edges or excluded_vertices:
            # Load vertex information
            hop_count = metadata.get('hop_count')
            if hop_count is None:
                import re
                match = re.search(r'hops(\d+)', str(input_path))
                if match:
                    hop_count = int(match.group(1))
            
            if hop_count:
                logging.info("Loading vertex information for exclusions...")
                vertex_key = get_vertex_names(db_path, hop_count)
                graph = apply_exclusions(graph, vertex_key, excluded_edges, excluded_vertices)
                
                # Update edge count after filtering
                nnz = graph.nnz if hasattr(graph, 'nnz') else np.count_nonzero(graph)
                logging.info(f"After exclusions: {nnz:,} edges remain")
    
    # Compute transitive closure
    logging.info("Computing transitive closure...")
    start_time = time.time()
    try:
        closure = graph.transitive_closure()
        closure_duration = time.time() - start_time
    except ValueError as e:
        closure_duration = time.time() - start_time
        # Extract vertex from error message
        error_msg = str(e)
        if 'cycle detected' in error_msg.lower() and 'Vertex' in error_msg:
            # Try to extract vertex number from error
            import re
            match = re.search(r'Vertex (\d+)', error_msg)
            if match:
                problem_vertex = int(match.group(1))
                logging.error(f"Cycle detected during transitive closure at vertex {problem_vertex}")
                
                # Load vertex names and prepare for relationship lookup
                vertex_names = {}
                vertex_key = {}
                if db_path and db_path.exists():
                    try:
                        logging.info("Loading vertex names from database...")
                        # Try to extract hop count from filename or metadata
                        hop_count = metadata.get('hop_count')
                        if hop_count is None:
                            # Try to extract from input_path filename (e.g., rappdw_hops10.npz)
                            import re
                            match = re.search(r'hops(\d+)', str(input_path))
                            if match:
                                hop_count = int(match.group(1))
                                logging.debug(f"Extracted hop count {hop_count} from filename")
                            else:
                                hop_count = 10  # fallback default
                        
                        vertex_key = get_vertex_names(db_path, hop_count)
                        # vertex_key maps vertex_id -> (external_id, name)
                        vertex_names = {vid: f"{name} ({ext_id})" for vid, (ext_id, name) in vertex_key.items()}
                        logging.info(f"Loaded {len(vertex_names):,} vertex names")
                    except Exception as db_err:
                        logging.warning(f"Could not load vertex names: {db_err}")
                
                logging.info("Tracing cycle path...")
                
                # Find the cycle path
                graph_matrix = graph.W if hasattr(graph, 'W') else graph
                cycle_path = find_cycle_path(graph_matrix, problem_vertex)
                
                if cycle_path and len(cycle_path) > 2:
                    logging.error(f"\nCycle path found ({len(cycle_path)} vertices in cycle):")
                    logging.error(f"NOTE: The cycle error occurs during transitive closure computation,")
                    logging.error(f"      meaning the AVOS product along this path creates a non-identity value.")
                    logging.error(f"      Self-loops with identity values (-1, 0, 1) are allowed.\n")
                    
                    # Format path with names
                    path_str_parts = []
                    for v in cycle_path:
                        if v in vertex_names:
                            path_str_parts.append(f"{v} [{vertex_names[v]}]")
                        else:
                            path_str_parts.append(str(v))
                    logging.error(f"  Cycle: {' -> '.join(path_str_parts)}")
                    
                    # Load relationship types from database
                    relationship_types = {}
                    if vertex_key:
                        # Build list of vertex pairs to query (using vertex IDs directly)
                        vertex_pairs = []
                        for i in range(len(cycle_path) - 1):
                            v1, v2 = cycle_path[i], cycle_path[i + 1]
                            if v1 in vertex_key and v2 in vertex_key:
                                # vertex_key maps internal index -> (external_id, name)
                                # But we need the database vertex ID (external_id in the graph, but "id" column in DB)
                                ext_id1, _ = vertex_key[v1]
                                ext_id2, _ = vertex_key[v2]
                                vertex_pairs.append((ext_id1, ext_id2))
                        
                        if vertex_pairs:
                            logging.info("Loading relationship types from database...")
                            relationship_types = get_relationship_types(db_path, vertex_pairs)
                    
                    # Show relationships
                    logging.error(f"\n  Direct edges in cycle:")
                    for i in range(len(cycle_path) - 1):
                        v1, v2 = cycle_path[i], cycle_path[i + 1]
                        if hasattr(graph_matrix, 'tocsr'):
                            edge_val = graph_matrix[v1, v2]
                        else:
                            edge_val = graph_matrix[v1, v2]
                        
                        v1_name = vertex_names.get(v1, f"Vertex {v1}")
                        v2_name = vertex_names.get(v2, f"Vertex {v2}")
                        
                        # Get relationship type from database and AVOS decoding
                        avos_decoded = decode_avos_value(edge_val)
                        rel_type = None
                        if v1 in vertex_key and v2 in vertex_key:
                            ext_id1, _ = vertex_key[v1]
                            ext_id2, _ = vertex_key[v2]
                            rel_type = relationship_types.get((ext_id1, ext_id2))
                        
                        logging.error(f"    {v1_name}")
                        if rel_type:
                            logging.error(f"      -> Relationship: {rel_type}")
                            logging.error(f"      -> AVOS: {avos_decoded} (value: {edge_val})")
                        else:
                            logging.error(f"      -> AVOS: {avos_decoded} (value: {edge_val})")
                            logging.error(f"      -> [Relationship type not found in database]")
                        logging.error(f"      -> {v2_name}")
                else:
                    logging.warning("Could not trace multi-step cycle path.")
                    logging.warning("This may indicate data quality issues in the source relationships.")
            raise
        else:
            raise
    
    # Get the underlying matrix (handle both W attribute and direct matrix)
    closure_matrix = closure.W if hasattr(closure, 'W') else closure
    closure_nnz = closure_matrix.nnz if hasattr(closure_matrix, 'nnz') else np.count_nonzero(closure_matrix)
    logging.info(f"Transitive closure computed: {closure_nnz:,} edges (duration: {closure_duration:.2f}s)")
    
    # Compute canonical ordering
    logging.info("Computing canonical ordering...")
    start_time = time.time()
    ordering = avos_canonical_ordering(closure_matrix)
    ordering_duration = time.time() - start_time
    logging.info(f"Canonical ordering computed (duration: {ordering_duration:.2f}s)")
    
    # Get canonical matrix and permutation
    canonical_graph = ordering.A
    label_permutation = ordering.label_permutation
    
    # Save canonical graph with permutation
    save_canonical_graph(canonical_graph, output_path, metadata, label_permutation)
    
    total_duration = load_duration + closure_duration + ordering_duration
    logging.info(f"Total processing time: {total_duration:.2f}s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute canonical forms of cached graphs for performance benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a specific hop count
  %(prog)s --hop-count 10
  
  # Process multiple hop counts
  %(prog)s --hop-count 10 11 12
  
  # Specify custom input/output directories
  %(prog)s --hop-count 10 --input-dir /custom/path --output-dir /output/path
  
  # Process with custom base name
  %(prog)s --hop-count 10 --base-name custom_graph
        """
    )
    parser.add_argument(
        '--hop-count',
        type=int,
        nargs='+',
        required=True,
        help='Hop count(s) to process (e.g., 10 or 10 11 12)'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('/mnt/nas/data/rbg/rappdw-01-13-22'),
        help='Directory containing cached base graphs (default: %(default)s)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory for canonical graph output (default: same as input-dir)'
    )
    parser.add_argument(
        '--base-name',
        type=str,
        default='rappdw',
        help='Base name for graph files (default: %(default)s)'
    )
    parser.add_argument(
        '--db-path',
        type=Path,
        default=Path('/mnt/nas/data/rbg/rappdw-01-13-22/rappdw.db'),
        help='Path to SQLite database for vertex name resolution (default: %(default)s)'
    )
    parser.add_argument(
        '--exclusions',
        type=Path,
        help='Path to exclusions.json file listing edges/vertices to exclude (default: config/exclusions.json if exists)'
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
    
    # Determine exclusions file
    exclusions_path = args.exclusions
    if not exclusions_path:
        # Try default location
        default_exclusions = Path('config/exclusions.json')
        if default_exclusions.exists():
            exclusions_path = default_exclusions
            logging.info(f"Using default exclusions file: {exclusions_path}")
    
    # Use input dir as output dir if not specified
    output_dir = args.output_dir if args.output_dir else args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each hop count
    for hop_count in args.hop_count:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing hop count: {hop_count}")
        logging.info(f"{'='*60}")
        
        # Construct input and output paths
        input_file = args.input_dir / f"{args.base_name}_hops{hop_count}.npz"
        output_file = output_dir / f"{args.base_name}_canonical_hops{hop_count}.npz"
        
        if not input_file.exists():
            logging.warning(f"Input file not found: {input_file}")
            continue
        
        if output_file.exists():
            logging.info(f"Output file already exists: {output_file}")
            response = input("Overwrite? (y/n): ").strip().lower()
            if response != 'y':
                logging.info("Skipping...")
                continue
        
        try:
            compute_canonical_form(input_file, output_file, args.db_path, exclusions_path)
            logging.info(f"✓ Successfully processed hop count {hop_count}")
        except Exception as e:
            logging.error(f"✗ Failed to process hop count {hop_count}: {e}", exc_info=args.verbose)
            if args.verbose:
                raise
    
    logging.info(f"\n{'='*60}")
    logging.info("Processing complete!")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
