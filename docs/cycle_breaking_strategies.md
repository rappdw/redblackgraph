# Cycle Breaking Strategies for Genealogy Graphs

## Overview

Cycles in genealogy graphs are data quality issues that prevent computing transitive closure. This document outlines strategies for detecting and resolving cycles.

## Types of Cycles

### 1. **Data Entry Errors**
- Person listed as their own ancestor
- Incorrect parent-child assignments
- Swapped relationships

**Solution**: Manual correction in source data

### 2. **Relationship Type Confusion**
- Step-relationships treated as biological
- Adoptive relationships creating false biological cycles
- Foster/Guardian relationships

**Solution**: Filter non-biological relationships or lower their weight

### 3. **Data Merge Artifacts**
- Duplicate persons incorrectly merged
- Same person with multiple IDs creating false cycles

**Solution**: Identity resolution before graph construction

## Automated Detection Strategies

### Strategy 1: Pre-Transitive Closure Detection
**When**: Before computing canonical forms
**How**: Use `detect_and_break_cycles.py`

```bash
# Detect all cycles
python scripts/detect_and_break_cycles.py --hop-count 10 --strategy manual

# Generate removal suggestions
python scripts/detect_and_break_cycles.py --hop-count 10 \
    --strategy weakest_link --output cycles_report.json
```

### Strategy 2: Relationship Type Filtering
**When**: During graph construction
**How**: Modify `rbg-graph-builder` to exclude certain relationship types

```python
# In graph builder, filter relationships:
EXCLUDED_TYPES = ['StepParent', 'FosterParent', 'GuardianParent']

# Only process biological relationships
BIOLOGICAL_TYPES = [
    'BiologicalParent',
    'AssumedBiological',
    'UnspecifiedParentType',
    'UntypedParent'
]
```

### Strategy 3: Confidence-Based Edge Weighting
**When**: Before transitive closure
**How**: Assign reliability scores and remove low-confidence edges in cycles

**Reliability Ranking** (highest to lowest):
1. `BiologicalParent` - Explicitly verified biological relationship
2. `AssumedBiological` - Strong evidence of biological relationship
3. `UnspecifiedParentType` - Relationship exists, type unknown
4. `UntypedParent` - Assumed to be biological
5. `AdoptiveParent` - Correct relationship, not biological
6. `StepParent` - Often confused with biological
7. `FosterParent` - Temporary, non-biological
8. `GuardianParent` - Legal, non-biological

## Manual Review Workflow

### 1. Export Cycle Report
```bash
python scripts/detect_and_break_cycles.py \
    --hop-count 10 \
    --strategy weakest_link \
    --output /tmp/cycles.json
```

### 2. Review Each Cycle
For each cycle in the report:
- Check external IDs in FamilySearch/genealogy source
- Verify relationship types are correct
- Look for obvious errors (person as own ancestor, etc.)

### 3. Correct in Source Database
```sql
-- Example: Remove incorrect edge
DELETE FROM EDGE 
WHERE source = (SELECT id FROM VERTEX WHERE external_id = 'I12345')
  AND destination = (SELECT id FROM VERTEX WHERE external_id = 'I67890');

-- Or update relationship type
UPDATE EDGE 
SET type = 'StepParent'
WHERE source = (SELECT id FROM VERTEX WHERE external_id = 'I12345')
  AND destination = (SELECT id FROM VERTEX WHERE external_id = 'I67890');
```

### 4. Rebuild Graph
```bash
# After database corrections
rbg-graph-builder -c 10
python scripts/compute_canonical_forms.py --hop-count 10
```

## Automated Breaking Strategies

### Option A: Remove Weakest Links
Break cycles by removing edges with lowest confidence scores.

**Pros**:
- Systematic and reproducible
- Preserves most reliable data
- Good for large graphs

**Cons**:
- May remove correct relationships
- Doesn't fix root cause

**Implementation**: Use `--strategy weakest_link`

### Option B: Filter Relationship Types
Exclude non-biological relationships from graph construction.

**Pros**:
- Prevents most cycles
- Clear semantic reasoning
- Easy to explain

**Cons**:
- Loses step/adoptive relationship data
- May still have biological data errors

**Implementation**: Modify graph builder configuration

### Option C: Connected Component Analysis
Break at articulation points to minimize graph damage.

**Pros**:
- Minimal connectivity loss
- Graph theory optimal

**Cons**:
- Complex to implement
- May not align with semantic meaning

### Option D: Temporal Ordering
Use birth/death dates to determine impossible relationships.

**Pros**:
- High confidence when dates available
- Catches obvious errors

**Cons**:
- Requires date data
- Date data often incomplete/incorrect

## Recommended Workflow

### For Initial Data Cleanup (hop count ≤ 10)
1. Run cycle detection with `--strategy manual`
2. Review top 10-20 cycles manually
3. Correct obvious errors in source database
4. Rebuild graph

### For Large Graphs (hop count > 10)
1. Run cycle detection with `--strategy weakest_link`
2. Export removal suggestions
3. Apply automated filtering for non-biological types
4. Manual review of remaining cycles
5. Rebuild graph incrementally

### For Production Pipelines
1. Pre-filter at graph construction (exclude StepParent, etc.)
2. Automated cycle detection as validation step
3. Alert on new cycles
4. Queue for manual review

## Database Queries for Analysis

### Find Most Common Cycle Participants
```sql
-- Vertices appearing in multiple cycles
SELECT v.external_id, v.name, COUNT(*) as cycle_count
FROM VERTEX v
WHERE v.id IN (
    -- Replace with cycle vertex IDs from report
    SELECT source FROM EDGE WHERE <cycle condition>
)
GROUP BY v.id
ORDER BY cycle_count DESC;
```

### Find Edges by Relationship Type
```sql
SELECT 
    v1.external_id AS source,
    v1.name AS source_name,
    v2.external_id AS dest,
    v2.name AS dest_name,
    e.type
FROM EDGE e
JOIN VERTEX v1 ON e.source = v1.id
JOIN VERTEX v2 ON e.destination = v2.id
WHERE e.type IN ('StepParent', 'FosterParent', 'GuardianParent')
ORDER BY e.type;
```

### Identify Potential Self-Loops
```sql
SELECT v.external_id, v.name, COUNT(*) as loop_count
FROM VERTEX v
JOIN EDGE e ON e.source = e.destination AND e.source = v.id
WHERE e.type NOT IN ('BiologicalParent', 'AssumedBiological')
GROUP BY v.id;
```

## Next Steps

1. **Try the detection script** on your current graph
2. **Review the cycle report** to understand patterns
3. **Choose a strategy** based on your data quality priorities
4. **Implement corrections** in source database or graph builder
5. **Validate** by rerunning canonical form computation

## See Also

- `scripts/detect_and_break_cycles.py` - Automated cycle detection
- `scripts/compute_canonical_forms.py` - Enhanced with cycle path tracing
- FamilySearch API documentation for relationship types
