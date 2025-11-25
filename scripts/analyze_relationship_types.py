#!/usr/bin/env python3
"""
Analyze relationship types in the database to identify potential cycle sources.
"""

import argparse
import sqlite3
from pathlib import Path
from collections import Counter


def analyze_relationship_types(db_path: Path):
    """Analyze distribution of relationship types in the database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    print("="*80)
    print("RELATIONSHIP TYPE ANALYSIS")
    print("="*80)
    
    # Count by type
    cursor.execute("SELECT type, COUNT(*) FROM EDGE GROUP BY type ORDER BY COUNT(*) DESC")
    print("\nRelationship Type Distribution:")
    print("-" * 60)
    total = 0
    for rel_type, count in cursor.fetchall():
        print(f"  {rel_type:30s} {count:8,}")
        total += count
    print("-" * 60)
    print(f"  {'TOTAL':30s} {total:8,}")
    
    # Find potential self-loops
    cursor.execute("""
        SELECT v.id, v.given_name || ' ' || v.surname AS name, e.type
        FROM EDGE e
        JOIN VERTEX v ON e.source = v.id
        WHERE e.source = e.destination
        LIMIT 20
    """)
    self_loops = cursor.fetchall()
    if self_loops:
        print("\n⚠️  Self-Loops Found (person as their own parent):")
        print("-" * 60)
        for person_id, name, rel_type in self_loops[:10]:
            print(f"  {name} ({person_id}) - {rel_type}")
        if len(self_loops) > 10:
            print(f"  ... and {len(self_loops) - 10} more")
    
    # Find chains that might create cycles
    cursor.execute("""
        SELECT 
            v1.id AS p1_id,
            v1.given_name || ' ' || v1.surname AS p1_name,
            v2.id AS p2_id,
            v2.given_name || ' ' || v2.surname AS p2_name,
            e1.type AS rel_type
        FROM EDGE e1
        JOIN VERTEX v1 ON e1.source = v1.id
        JOIN VERTEX v2 ON e1.destination = v2.id
        JOIN EDGE e2 ON e2.source = v2.id
        WHERE e2.destination = v1.id
        LIMIT 20
    """)
    mutual = cursor.fetchall()
    if mutual:
        print("\n⚠️  Mutual Parent Relationships Found (A->B and B->A):")
        print("-" * 60)
        for p1_id, p1_name, p2_id, p2_name, rel_type in mutual[:10]:
            print(f"  {p1_name} ({p1_id}) <-> {p2_name} ({p2_id}) [{rel_type}]")
        if len(mutual) > 10:
            print(f"  ... and {len(mutual) - 10} more")
    
    # Find vertices with high out-degree (many children = potential hub in cycle)
    cursor.execute("""
        SELECT v.id, v.given_name || ' ' || v.surname AS name, COUNT(*) as child_count
        FROM VERTEX v
        JOIN EDGE e ON e.destination = v.id
        WHERE e.type IN ('BiologicalParent', 'AssumedBiological', 'UntypedParent')
        GROUP BY v.id
        HAVING child_count > 20
        ORDER BY child_count DESC
        LIMIT 10
    """)
    high_degree = cursor.fetchall()
    if high_degree:
        print("\n⚠️  Vertices with Unusually High Child Count (>20):")
        print("-" * 60)
        for person_id, name, count in high_degree:
            print(f"  {name} ({person_id}) - {count} children")
    
    # Find non-biological relationships that might be causing issues
    cursor.execute("""
        SELECT 
            v1.id,
            v1.given_name || ' ' || v1.surname AS name,
            e.type,
            COUNT(*) as count
        FROM EDGE e
        JOIN VERTEX v1 ON e.source = v1.id
        WHERE e.type IN ('StepParent', 'FosterParent', 'GuardianParent', 'AdoptiveParent')
        GROUP BY v1.id, e.type
        ORDER BY count DESC
        LIMIT 20
    """)
    non_bio = cursor.fetchall()
    if non_bio:
        print("\nNon-Biological Relationships (potential cycle sources):")
        print("-" * 60)
        for person_id, name, rel_type, count in non_bio[:10]:
            print(f"  {name} ({person_id}) - {count}x {rel_type}")
    
    conn.close()


def generate_mutual_exclusions(db_path: Path, output: Path):
    """Generate exclusions.json for mutual parent relationships.
    
    For each mutual relationship (A->B and B->A), exclude the one with
    the larger hop count.
    
    Args:
        db_path: Path to database
        output: Output path for exclusions.json
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Find all mutual relationships
    query = """
        SELECT 
            e1.source AS p1_id,
            v1.given_name || ' ' || v1.surname AS p1_name,
            e1.type AS p1_to_p2_type,
            e2.source AS p2_id,
            v2.given_name || ' ' || v2.surname AS p2_name,
            e2.type AS p2_to_p1_type
        FROM EDGE e1
        JOIN EDGE e2 ON e1.source = e2.destination AND e1.destination = e2.source
        JOIN VERTEX v1 ON e1.source = v1.id
        JOIN VERTEX v2 ON e1.destination = v2.id
        WHERE e1.source < e2.source  -- Avoid duplicates
        ORDER BY p1_id, p2_id
    """
    
    cursor.execute(query)
    mutual_relationships = cursor.fetchall()
    
    print(f"\nFound {len(mutual_relationships)} mutual parent relationships")
    
    if len(mutual_relationships) == 0:
        conn.close()
        return
    
    # For each mutual pair, determine which to exclude based on hop count
    exclusions = {
        "description": "Auto-generated exclusions for mutual parent relationships",
        "strategy": "Remove relationship with larger hop count from each mutual pair",
        "excluded_edges": []
    }
    
    for p1_id, p1_name, p1_to_p2_type, p2_id, p2_name, p2_to_p1_type in mutual_relationships:
        # Query iteration (hop count) for each edge
        # The iteration column indicates how far from the starting vertex
        q = """
            SELECT v.iteration 
            FROM VERTEX v 
            WHERE v.id = ?
        """
        
        cursor.execute(q, (p1_id,))
        p1_result = cursor.fetchone()
        p1_iteration = p1_result[0] if p1_result and p1_result[0] is not None else 0
        
        cursor.execute(q, (p2_id,))
        p2_result = cursor.fetchone()
        p2_iteration = p2_result[0] if p2_result and p2_result[0] is not None else 0
        
        # Determine which edge to exclude
        # Remove the edge pointing "inward" (from higher hop to lower hop)
        # Keep the edge pointing "outward" (from lower hop to higher hop)
        # If equal, exclude p1->p2 (arbitrary but consistent)
        
        if p2_iteration > p1_iteration:
            # p2 is further out, so exclude the edge p2->p1 (pointing inward)
            exclude_from = p2_id
            exclude_to = p1_id
            exclude_from_name = p2_name
            exclude_to_name = p1_name
            reason = f"Mutual relationship: {p1_name} (hop {p1_iteration}) <-> {p2_name} (hop {p2_iteration}). Removing {p2_name}->{p1_name}"
        elif p1_iteration > p2_iteration:
            # p1 is further out, so exclude the edge p1->p2 (pointing inward)
            exclude_from = p1_id
            exclude_to = p2_id
            exclude_from_name = p1_name
            exclude_to_name = p2_name
            reason = f"Mutual relationship: {p1_name} (hop {p1_iteration}) <-> {p2_name} (hop {p2_iteration}). Removing {p1_name}->{p2_name}"
        else:
            # Equal hop counts or both 0, exclude p1->p2 (arbitrary)
            exclude_from = p1_id
            exclude_to = p2_id
            exclude_from_name = p1_name
            exclude_to_name = p2_name
            reason = f"Mutual relationship: {p1_name} (hop {p1_iteration}) <-> {p2_name} (hop {p2_iteration}). Equal hops, removing {p1_name}->{p2_name}"
        
        exclusions["excluded_edges"].append({
            "from_id": exclude_from,
            "to_id": exclude_to,
            "reason": reason,
            "comment": f"{exclude_from_name} -> {exclude_to_name}"
        })
    
    conn.close()
    
    # Add empty excluded_vertices for completeness
    exclusions["excluded_vertices"] = []
    
    # Write to file
    import json
    with open(output, 'w') as f:
        json.dump(exclusions, f, indent=2)
    
    print(f"\n✓ Generated exclusions file: {output}")
    print(f"  {len(exclusions['excluded_edges'])} edges to exclude")
    print(f"\nTo apply these exclusions, run:")
    print(f"  python scripts/compute_canonical_forms.py --hop-count N --exclusions {output}")


def export_edges_for_removal(db_path: Path, relationship_types: list, output: Path):
    """Export edges of specific types for potential removal."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    placeholders = ','.join('?' * len(relationship_types))
    query = f"""
        SELECT 
            v1.id AS source_id,
            v1.given_name || ' ' || v1.surname AS source_name,
            v2.id AS dest_id,
            v2.given_name || ' ' || v2.surname AS dest_name,
            e.type
        FROM EDGE e
        JOIN VERTEX v1 ON e.source = v1.id
        JOIN VERTEX v2 ON e.destination = v2.id
        WHERE e.type IN ({placeholders})
        ORDER BY e.type, source_name
    """
    
    cursor.execute(query, relationship_types)
    
    with open(output, 'w') as f:
        f.write("# Edges for potential removal\n")
        f.write(f"# Relationship types: {', '.join(relationship_types)}\n\n")
        for src_id, src_name, dst_id, dst_name, rel_type in cursor.fetchall():
            f.write(f"{src_id}\t{src_name}\t{dst_id}\t{dst_name}\t{rel_type}\n")
    
    conn.close()
    print(f"\nExported edges to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze relationship types in genealogy database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  %(prog)s --db-path /path/to/rappdw.db
  
  # Export non-biological relationships for review
  %(prog)s --db-path /path/to/rappdw.db \
      --export-types StepParent FosterParent GuardianParent \
      --output non_bio_edges.tsv
  
  # Generate exclusions file for mutual parent relationships
  %(prog)s --db-path /path/to/rappdw.db \
      --generate-mutual-exclusions \
      --output config/exclusions.json
        """
    )
    parser.add_argument(
        '--db-path',
        type=Path,
        required=True,
        help='Path to SQLite database'
    )
    parser.add_argument(
        '--export-types',
        nargs='+',
        help='Export edges of specific relationship types'
    )
    parser.add_argument(
        '--generate-mutual-exclusions',
        action='store_true',
        help='Generate exclusions.json file for mutual parent relationships'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for exported edges (TSV) or exclusions (JSON)'
    )
    
    args = parser.parse_args()
    
    if not args.db_path.exists():
        print(f"Error: Database not found: {args.db_path}")
        return 1
    
    # Generate mutual exclusions if requested
    if args.generate_mutual_exclusions:
        if not args.output:
            print("Error: --output is required when using --generate-mutual-exclusions")
            return 1
        generate_mutual_exclusions(args.db_path, args.output)
        return 0
    
    # Run analysis
    analyze_relationship_types(args.db_path)
    
    # Export if requested
    if args.export_types and args.output:
        export_edges_for_removal(args.db_path, args.export_types, args.output)
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("1. Review any self-loops (person as own parent) - these are always errors")
    print("2. Check mutual parent relationships - likely data entry errors")
    print("3. Consider filtering non-biological relationships (Step/Foster/Guardian)")
    print("4. Investigate vertices with unusually high child counts")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
