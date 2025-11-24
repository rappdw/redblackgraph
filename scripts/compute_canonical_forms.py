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

import numpy as np
import redblackgraph as rb
from scipy.sparse import coo_matrix
from redblackgraph.sparse.csgraph import avos_canonical_ordering


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


def compute_canonical_form(input_path: Path, output_path: Path):
    """Compute canonical form of a graph and save it.
    
    Args:
        input_path: Path to input cached graph
        output_path: Path where canonical graph should be saved
    """
    logging.info(f"Loading base graph from {input_path}")
    start_time = time.time()
    graph, metadata = load_cached_graph(input_path)
    load_duration = time.time() - start_time
    
    vertices = graph.shape[0]
    nnz = graph.nnz if hasattr(graph, 'nnz') else np.count_nonzero(graph)
    logging.info(f"Loaded graph: {vertices:,} vertices, {nnz:,} edges (duration: {load_duration:.2f}s)")
    
    # Compute transitive closure
    logging.info("Computing transitive closure...")
    start_time = time.time()
    closure = graph.transitive_closure()
    closure_duration = time.time() - start_time
    
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
            compute_canonical_form(input_file, output_file)
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
