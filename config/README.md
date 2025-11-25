# Exclusions Configuration

This directory contains configuration for excluding specific edges or vertices during canonical form computation to prevent cycles.

## Quick Start

1. **Identify problematic edges/vertices** using cycle detection:
   ```bash
   python scripts/compute_canonical_forms.py --hop-count 10
   # If cycles are detected, note the vertex IDs and relationships shown in error
   ```

2. **Add to exclusions.json**:
   ```json
   {
     "excluded_edges": [
       {
         "from_id": "I12345",
         "to_id": "I67890",
         "reason": "Creates cycle - John Smith -> Mary Jones"
       }
     ],
     "excluded_vertices": [
       {
         "vertex_id": "I99999",
         "reason": "Duplicate person with bad data"
       }
     ]
   }
   ```

3. **Rerun canonical form computation**:
   ```bash
   python scripts/compute_canonical_forms.py --hop-count 10
   # Automatically uses config/exclusions.json if it exists
   ```

## Exclusions File Format

The `exclusions.json` file has two main sections:

### Excluded Edges

Remove specific relationships between two individuals:

```json
{
  "excluded_edges": [
    {
      "from_id": "I12345",
      "to_id": "I67890",
      "reason": "Creates cycle in transitive closure",
      "comment": "Optional: any notes about why this edge is excluded"
    }
  ]
}
```

- **`from_id`**: Source vertex ID (from database VERTEX.id column)
- **`to_id`**: Destination vertex ID
- **`reason`**: Human-readable explanation (for documentation)
- **`comment`**: Optional additional notes

### Excluded Vertices

Remove all edges connected to specific individuals:

```json
{
  "excluded_vertices": [
    {
      "vertex_id": "I99999",
      "reason": "Duplicate person causing cycles"
    }
  ]
}
```

- **`vertex_id`**: The vertex ID to exclude (from database VERTEX.id column)
- **`reason`**: Human-readable explanation

When a vertex is excluded, **all edges to and from that vertex are removed**.

## Usage Examples

### Use Default Exclusions File

```bash
# Automatically uses config/exclusions.json if it exists
python scripts/compute_canonical_forms.py --hop-count 10
```

### Use Custom Exclusions File

```bash
python scripts/compute_canonical_forms.py \
    --hop-count 10 \
    --exclusions /path/to/my_exclusions.json
```

### No Exclusions

```bash
# Don't apply any exclusions (even if config/exclusions.json exists)
python scripts/compute_canonical_forms.py \
    --hop-count 10 \
    --exclusions /dev/null
```

## Finding IDs to Exclude

### From Cycle Detection Output

When `compute_canonical_forms.py` detects a cycle, it prints vertex IDs and names:

```
ERROR: Cycle path found (4 vertices in cycle):
ERROR:   Cycle: 28833 [John Smith (I12345)] -> 15042 [Mary Jones (I67890)] -> ...
ERROR:   Direct edges in cycle:
ERROR:     John Smith (I12345)
ERROR:       -> Relationship: BiologicalParent
ERROR:       -> AVOS: male parent (value: 2)
ERROR:       -> Mary Jones (I67890)
```

Use the IDs in parentheses (e.g., `I12345`, `I67890`) in your exclusions file.

### Query Database Directly

```bash
# Find a person by name
sqlite3 /path/to/rappdw.db \
  "SELECT id, given_name, surname FROM VERTEX WHERE surname LIKE '%Smith%' LIMIT 10"

# Find edges between two people
sqlite3 /path/to/rappdw.db \
  "SELECT source, destination, type FROM EDGE WHERE source='I12345' AND destination='I67890'"
```

### Use Analysis Scripts

```bash
# Analyze relationship types and find issues
python scripts/analyze_relationship_types.py --db-path /path/to/rappdw.db

# Detect all cycles and get suggestions
python scripts/detect_and_break_cycles.py --hop-count 10 --output cycles.json
```

## Workflow

### Iterative Cycle Resolution

1. **Run canonical form computation**
2. **If cycle detected**, note the vertex IDs from error output
3. **Add problematic edge to exclusions.json**
4. **Rerun** - repeat until no cycles

### Example Session

```bash
# First attempt
python scripts/compute_canonical_forms.py --hop-count 10
# ERROR: Cycle detected! John Smith (I12345) -> Mary Jones (I67890) -> ...

# Edit config/exclusions.json to add:
# {"from_id": "I12345", "to_id": "I67890", "reason": "Creates cycle"}

# Try again
python scripts/compute_canonical_forms.py --hop-count 10
# Success! (or another cycle to fix)
```

## Best Practices

1. **Document reasons** - Always include a meaningful `reason` field
2. **Prefer edges over vertices** - Exclude specific edges when possible; only exclude entire vertices for duplicates or fundamentally bad data
3. **Version control** - Commit `exclusions.json` to track what you've excluded
4. **Keep notes** - Use `comment` fields to document your investigation
5. **Minimize exclusions** - Only exclude what's necessary to break cycles

## Edge vs. Vertex Exclusion

### When to Exclude an Edge

- Specific incorrect relationship (wrong parent assignment)
- Step/adoptive relationship incorrectly marked as biological
- Data entry error for one relationship

**Effect**: Only that specific relationship is removed

### When to Exclude a Vertex

- Duplicate person in the database
- Person with fundamentally corrupted data
- Test/placeholder data

**Effect**: All relationships to/from that person are removed

## Complete Example

```json
{
  "description": "Exclusions for rappdw genealogy - hop count 10",
  "last_updated": "2025-11-24",
  
  "excluded_edges": [
    {
      "from_id": "I12345",
      "to_id": "I67890",
      "reason": "Creates 3-person cycle with I11111",
      "comment": "John Smith incorrectly listed as parent of Mary Jones"
    },
    {
      "from_id": "I22222",
      "to_id": "I33333",
      "reason": "Step-parent relationship, not biological"
    }
  ],
  
  "excluded_vertices": [
    {
      "vertex_id": "I99999",
      "reason": "Duplicate of I88888, merge artifact",
      "comment": "Same person entered twice, causes cycles with real I88888"
    }
  ]
}
```

## Validation

The script will log what it excludes:

```
INFO: Loaded exclusions: 2 edges, 1 vertices
INFO: Loading vertex information for exclusions...
INFO: Applying exclusions...
INFO:   Removed 2 specific edges
INFO:   Removed 15 edges connected to excluded vertices
INFO:   Total edges removed: 17
INFO: After exclusions: 18,072,196 edges remain
```

## Troubleshooting

### "Loaded exclusions: 0 edges, 0 vertices"

- Check JSON syntax is valid
- Ensure `from_id`, `to_id`, `vertex_id` fields are present
- Verify IDs match database (case-sensitive)

### Exclusions not applied

- Ensure you're using `--db-path` argument
- Check that exclusions file path is correct
- Enable verbose mode: `-v`

### Still getting cycles

- You may need to exclude more edges
- Use `detect_and_break_cycles.py` to find all cycles
- Consider excluding a vertex instead of individual edges

## See Also

- `../docs/cycle_breaking_strategies.md` - Overall strategy guide
- `../scripts/compute_canonical_forms.py` - Main processing script
- `../scripts/detect_and_break_cycles.py` - Automated cycle detection
- `../scripts/analyze_relationship_types.py` - Database analysis
