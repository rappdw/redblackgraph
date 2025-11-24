#!/usr/bin/env python3
"""
Export cached graphs (base or canonical) to Excel/CSV.

This script loads a graph from its NPZ cache file and exports it to Excel.
Optionally uses the database for vertex names if available.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import redblackgraph as rb
from scipy.sparse import coo_matrix
import xlsxwriter


MAX_COLUMNS_EXCEL = 16384
ROTATE_90 = 'rotate-90'


def load_cached_graph(cache_path: Path):
    """Load a graph from NPZ cache file.
    
    Args:
        cache_path: Path to the cache file
        
    Returns:
        tuple: (graph object, metadata dict)
    """
    base_path = cache_path.with_suffix('')
    npz_path = base_path.with_suffix('.npz')
    json_path = base_path.with_suffix('.json')
    
    if not npz_path.exists():
        raise FileNotFoundError(f"Cache file not found: {npz_path}")
    
    # Load metadata
    metadata = {}
    if json_path.exists():
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


def get_vertex_names_from_db(db_path: Path, hops: int):
    """Get vertex names from database if available.
    
    Args:
        db_path: Path to SQLite database
        hops: Hop count for filtering
        
    Returns:
        dict: Mapping of vertex_id to (external_id, name) tuples, or None if unavailable
    """
    try:
        from fscrawler import RelationshipDbReader
        from redblackgraph.util.graph_builder import RbgGraphBuilder
        
        if not db_path.exists():
            return None
        
        builder = RbgGraphBuilder()
        reader = RelationshipDbReader(str(db_path), hops, builder)
        return reader.get_vertex_key()
    except Exception as e:
        logging.warning(f"Could not load vertex names from database: {e}")
        return None


def calc_width(len_of_max_string):
    """Calculate column width for Excel."""
    return max(len_of_max_string + 0.83, 2.67)


def export_to_excel(graph, output_path: Path, vertex_key: dict = None, max_vertices: int = None):
    """Export graph to Excel.
    
    Args:
        graph: The graph to export
        output_path: Output file path (.xlsx)
        vertex_key: Optional dictionary mapping vertex_id to (external_id, name)
        max_vertices: Maximum number of vertices to export
    """
    vertices = graph.shape[0]
    
    # Check size limits
    if vertices > MAX_COLUMNS_EXCEL:
        if not max_vertices:
            raise ValueError(
                f"Graph has {vertices:,} vertices, exceeds Excel limit ({MAX_COLUMNS_EXCEL:,}). "
                f"Use --max-vertices to export a subset."
            )
        vertices = min(vertices, max_vertices)
        logging.warning(f"Limiting export to first {vertices:,} vertices")
    
    # Create subgraph if needed
    if max_vertices and graph.shape[0] > max_vertices:
        if hasattr(graph, 'tocsr'):
            graph = graph[:max_vertices, :max_vertices]
        else:
            if hasattr(graph, 'W'):
                graph = rb.array(graph.W[:max_vertices, :max_vertices])
            else:
                graph = rb.array(graph[:max_vertices, :max_vertices])
        vertices = max_vertices
    
    # Create workbook
    workbook = xlsxwriter.Workbook(str(output_path))
    worksheet = workbook.add_worksheet()
    
    # Formats
    rotate_90_format = workbook.add_format({'rotation': 90})
    font_red = workbook.add_format({'font_color': '#FF0000'})
    
    # Hide unused rows
    worksheet.set_default_row(hide_unused_rows=True)
    
    # Write headers
    max_key_len = 0
    max_value = 0
    
    worksheet.write(0, 0, ' ')
    for idx in range(vertices):
        if vertex_key and idx in vertex_key:
            ext_id, name = vertex_key[idx]
            cell_data = f"{ext_id} - {name}"
        else:
            cell_data = f"{idx}"
        
        max_key_len = max(max_key_len, len(cell_data))
        worksheet.write(0, idx + 1, cell_data, rotate_90_format)
        worksheet.write(idx + 1, 0, cell_data)
    
    # Write data
    is_sparse = isinstance(graph, rb.sparse.rb_matrix)
    
    if is_sparse:
        for i, j in zip(*graph.nonzero()):
            if i < vertices and j < vertices:
                value = graph[i, j]
                if value != 0:
                    max_value = max(max_value, abs(value))
                    if value == -1:
                        worksheet.write(i + 1, j + 1, 1, font_red)
                    else:
                        worksheet.write(i + 1, j + 1, value)
    else:
        # Dense array
        if hasattr(graph, 'W'):
            data = graph.W
        else:
            data = np.asarray(graph)
        
        for i in range(vertices):
            for j in range(vertices):
                value = int(data[i, j])
                if value != 0:
                    max_value = max(max_value, abs(value))
                    if value == -1:
                        worksheet.write(i + 1, j + 1, 1, font_red)
                    else:
                        worksheet.write(i + 1, j + 1, value)
    
    # Set column widths
    worksheet.freeze_panes(1, 1)
    worksheet.set_column(0, 0, calc_width(max_key_len))
    worksheet.set_column(1, vertices, calc_width(len(str(max_value))))
    worksheet.set_column(vertices + 1, MAX_COLUMNS_EXCEL - 1, None, None, {'hidden': True})
    
    workbook.close()
    logging.info(f"Exported {vertices} x {vertices} graph to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export cached graphs to Excel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export base graph
  %(prog)s --cache /mnt/nas/data/rbg/rappdw-01-13-22/rappdw_hops5.npz
  
  # Export canonical graph
  %(prog)s --cache /mnt/nas/data/rbg/rappdw-01-13-22/rappdw_canonical_hops5.npz
  
  # Export with vertex names from database
  %(prog)s --cache /mnt/nas/data/rbg/rappdw-01-13-22/rappdw_canonical_hops5.npz \\
      --db /mnt/nas/data/rbg/rappdw-01-13-22/rappdw.db \\
      --hop-count 5
  
  # Export to specific file
  %(prog)s --cache rappdw_canonical_hops5.npz --output my_graph.xlsx
  
  # Limit large graph
  %(prog)s --cache rappdw_hops10.npz --max-vertices 1000
        """
    )
    parser.add_argument(
        '--cache',
        type=Path,
        required=True,
        help='Path to cached graph file (.npz)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output Excel file (.xlsx). Default: based on cache filename'
    )
    parser.add_argument(
        '--db',
        type=Path,
        default=None,
        help='Optional: SQLite database path for vertex names'
    )
    parser.add_argument(
        '--hop-count',
        type=int,
        default=None,
        help='Hop count (required if using --db for vertex names)'
    )
    parser.add_argument(
        '--max-vertices',
        type=int,
        default=None,
        help='Maximum number of vertices to export'
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
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.cache.with_suffix('.xlsx')
    
    try:
        # Load graph
        logging.info(f"Loading graph from {args.cache}")
        graph, metadata = load_cached_graph(args.cache)
        vertices = graph.shape[0]
        nnz = graph.nnz if hasattr(graph, 'nnz') else np.count_nonzero(graph)
        
        is_canonical = metadata.get('canonical', False)
        hops = metadata.get('hops', 'unknown')
        
        logging.info(f"Loaded {'canonical' if is_canonical else 'base'} graph: "
                    f"{vertices:,} vertices, {nnz:,} edges (hops: {hops})")
        
        # Get vertex names if database provided
        vertex_key = None
        if args.db:
            if not args.hop_count:
                logging.warning("--db specified but --hop-count not provided, using metadata hop count")
                hop_count = metadata.get('hops')
                if not hop_count:
                    logging.error("No hop count in metadata and --hop-count not specified")
                    sys.exit(1)
            else:
                hop_count = args.hop_count
            
            logging.info(f"Loading vertex names from {args.db}")
            vertex_key = get_vertex_names_from_db(args.db, hop_count)
            if vertex_key:
                logging.info(f"Loaded {len(vertex_key):,} vertex names")
        
        # Export
        export_to_excel(graph, output_path, vertex_key, args.max_vertices)
        logging.info(f"✓ Export complete: {output_path}")
        
    except Exception as e:
        logging.error(f"✗ Export failed: {e}", exc_info=args.verbose)
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
