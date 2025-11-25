#!/usr/bin/env python3
"""
Detect and optionally break cycles in genealogy graphs.

Strategies for cycle detection and breaking:
1. Identify cycles before transitive closure
2. Rank edges by reliability/confidence
3. Remove weakest edges to break cycles
4. Export report for manual review
"""

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from collections import defaultdict

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix


def load_graph_from_npz(npz_path: Path):
    """Load graph from NPZ cache file."""
    npz_data = np.load(npz_path)
    matrix_format = str(npz_data['format'][0])
    
    if matrix_format == 'sparse':
        from scipy.sparse import coo_matrix
        data = npz_data['data']
        row = npz_data['row']
        col = npz_data['col']
        shape = tuple(npz_data['shape'])
        return coo_matrix((data, (row, col)), shape=shape)
    else:
        return npz_data['array']


def find_all_cycles(graph_matrix):
    """Find all simple cycles in the graph (before transitive closure).
    
    Returns:
        list: List of cycles, where each cycle is a list of vertex indices
    """
    # Convert to NetworkX directed graph (ignore identity self-loops)
    G = nx.DiGraph()
    
    if hasattr(graph_matrix, 'tocsr'):
        csr = graph_matrix.tocsr()
    else:
        csr = graph_matrix
    
    # Add edges (skip identity values -1, 0, 1 on diagonal)
    for i in range(csr.shape[0]):
        if hasattr(csr, 'getrow'):
            row = csr.getrow(i)
            for j, val in zip(row.indices, row.data):
                # Skip identity self-loops
                if i == j and val in [-1, 0, 1]:
                    continue
                if val != 0:
                    G.add_edge(i, j, weight=val)
        else:
            for j in range(csr.shape[1]):
                val = csr[i, j]
                if i == j and val in [-1, 0, 1]:
                    continue
                if val != 0:
                    G.add_edge(i, j, weight=val)
    
    # Find all simple cycles
    try:
        cycles = list(nx.simple_cycles(G))
        return cycles
    except:
        logging.warning("Could not enumerate all cycles (graph may be too large)")
        return []


def score_edge_reliability(src_id, dst_id, db_path: Path):
    """Score an edge's reliability based on relationship type.
    
    Reliability ranking (higher = more reliable):
    1. BiologicalParent (highest confidence)
    2. AssumedBiological
    3. UnspecifiedParentType
    4. UntypedParent
    5. AdoptiveParent (correct but not biological)
    6. StepParent (lowest confidence for biological cycle)
    7. FosterParent
    8. GuardianParent
    
    Returns:
        int: Reliability score (higher is better), or 0 if not found
    """
    reliability_scores = {
        'BiologicalParent': 100,
        'AssumedBiological': 80,
        'UnspecifiedParentType': 60,
        'UntypedParent': 50,
        'AdoptiveParent': 40,
        'StepParent': 20,
        'FosterParent': 10,
        'GuardianParent': 10,
    }
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        query = """
            SELECT EDGE.type 
            FROM EDGE 
            JOIN VERTEX AS v_src ON EDGE.source = v_src.id
            JOIN VERTEX AS v_dst ON EDGE.destination = v_dst.id
            WHERE v_src.external_id = ? AND v_dst.external_id = ?
        """
        cursor.execute(query, (src_id, dst_id))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            rel_type = result[0]
            return reliability_scores.get(rel_type, 30)  # Default medium score
        return 0
    except:
        return 0


def break_cycles_strategy(cycles, graph_matrix, vertex_key, db_path: Path, strategy='weakest_link'):
    """Determine which edges to remove to break cycles.
    
    Strategies:
    - 'weakest_link': Remove edge with lowest reliability score in each cycle
    - 'step_adoptive': Remove StepParent/AdoptiveParent/Foster/Guardian edges first
    - 'untyped': Remove UntypedParent edges first
    - 'manual': Just report, don't auto-remove
    
    Returns:
        dict: Report with cycle info and edges to remove
    """
    report = {
        'cycles_found': len(cycles),
        'cycles': [],
        'edges_to_remove': set()
    }
    
    for cycle_idx, cycle in enumerate(cycles):
        cycle_info = {
            'cycle_id': cycle_idx,
            'length': len(cycle),
            'vertices': cycle,
            'edges': [],
            'suggested_break': None
        }
        
        # Analyze each edge in the cycle
        edge_scores = []
        for i in range(len(cycle)):
            v1 = cycle[i]
            v2 = cycle[(i + 1) % len(cycle)]
            
            # Get edge info
            edge_val = graph_matrix[v1, v2] if hasattr(graph_matrix, '__getitem__') else 0
            
            # Get external IDs for database lookup
            ext_id1 = vertex_key[v1][0] if v1 in vertex_key else None
            ext_id2 = vertex_key[v2][0] if v2 in vertex_key else None
            
            score = 0
            if ext_id1 and ext_id2:
                score = score_edge_reliability(ext_id1, ext_id2, db_path)
            
            edge_info = {
                'from_vertex': v1,
                'to_vertex': v2,
                'avos_value': int(edge_val),
                'reliability_score': score,
                'external_ids': (ext_id1, ext_id2)
            }
            cycle_info['edges'].append(edge_info)
            edge_scores.append((score, v1, v2, ext_id1, ext_id2))
        
        # Apply strategy to choose which edge to break
        if strategy == 'weakest_link':
            # Remove edge with lowest score
            edge_scores.sort(key=lambda x: x[0])
            if edge_scores:
                score, v1, v2, ext_id1, ext_id2 = edge_scores[0]
                cycle_info['suggested_break'] = {
                    'from': v1,
                    'to': v2,
                    'reason': f'Lowest reliability score: {score}',
                    'external_ids': (ext_id1, ext_id2)
                }
                if ext_id1 and ext_id2:
                    report['edges_to_remove'].add((ext_id1, ext_id2))
        
        elif strategy in ['step_adoptive', 'untyped']:
            # Remove specific relationship types first
            # This would need more sophisticated filtering
            pass
        
        report['cycles'].append(cycle_info)
    
    report['edges_to_remove'] = list(report['edges_to_remove'])
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Detect and analyze cycles in genealogy graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Cycle Breaking Strategies:
  1. weakest_link - Remove edges with lowest reliability scores
  2. step_adoptive - Prioritize removing non-biological relationships
  3. untyped - Remove untyped/unverified relationships first
  4. manual - Report only, manual review required

Examples:
  # Detect cycles and report
  %(prog)s --hop-count 10 --strategy manual
  
  # Detect and suggest breaks using weakest link
  %(prog)s --hop-count 10 --strategy weakest_link --output cycles_report.json
        """
    )
    parser.add_argument(
        '--hop-count',
        type=int,
        required=True,
        help='Hop count of the graph to analyze'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('/mnt/nas/data/rbg/rappdw-01-13-22'),
        help='Directory containing cached graphs'
    )
    parser.add_argument(
        '--db-path',
        type=Path,
        default=Path('/mnt/nas/data/rbg/rappdw-01-13-22/rappdw.db'),
        help='Path to SQLite database'
    )
    parser.add_argument(
        '--base-name',
        type=str,
        default='rappdw',
        help='Base name for graph files'
    )
    parser.add_argument(
        '--strategy',
        choices=['manual', 'weakest_link', 'step_adoptive', 'untyped'],
        default='manual',
        help='Strategy for breaking cycles'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output JSON report file (default: print to console)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO
    )
    
    # Load graph
    input_file = args.input_dir / f"{args.base_name}_hops{args.hop_count}.npz"
    if not input_file.exists():
        logging.error(f"Graph file not found: {input_file}")
        return 1
    
    logging.info(f"Loading graph from {input_file}")
    graph = load_graph_from_npz(input_file)
    logging.info(f"Graph loaded: {graph.shape[0]:,} vertices")
    
    # Load vertex names
    from fscrawler import RelationshipDbReader
    from redblackgraph.util.graph_builder import RbgGraphBuilder
    
    logging.info("Loading vertex information from database...")
    builder = RbgGraphBuilder()
    reader = RelationshipDbReader(str(args.db_path), args.hop_count, builder)
    vertex_key = reader.get_vertex_key()
    logging.info(f"Loaded {len(vertex_key):,} vertices")
    
    # Find cycles
    logging.info("Searching for cycles...")
    cycles = find_all_cycles(graph)
    logging.info(f"Found {len(cycles)} cycles")
    
    if len(cycles) == 0:
        logging.info("✓ No cycles detected!")
        return 0
    
    # Analyze cycles
    logging.info(f"Analyzing cycles with strategy: {args.strategy}")
    report = break_cycles_strategy(cycles, graph, vertex_key, args.db_path, args.strategy)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        logging.info(f"Report written to {args.output}")
    else:
        print("\n" + "="*80)
        print(f"CYCLE DETECTION REPORT")
        print("="*80)
        print(f"Total cycles found: {report['cycles_found']}")
        print(f"Edges suggested for removal: {len(report['edges_to_remove'])}")
        
        for cycle in report['cycles'][:5]:  # Show first 5
            print(f"\nCycle {cycle['cycle_id']} (length {cycle['length']}):")
            print(f"  Vertices: {' -> '.join(map(str, cycle['vertices']))}")
            if cycle['suggested_break']:
                sb = cycle['suggested_break']
                print(f"  Suggested break: {sb['from']} -> {sb['to']}")
                print(f"    Reason: {sb['reason']}")
        
        if len(cycles) > 5:
            print(f"\n... and {len(cycles) - 5} more cycles")
    
    return 0


if __name__ == "__main__":
    exit(main())
