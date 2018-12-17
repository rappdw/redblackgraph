# Keep Track of What I'm Working On

## Re-work main notebook
Remove the concept of pedigree number replace it with relationship function.

Editorial changes that result from the simplification

## Sparse implementation
Work on sparse implementations

## Optimization
Consider optimizations rather than Floyd-Warshall, e.g. topological sort and APSP for DAG. Should be O(VE) rather than O(V^3)

## Need to correctly handle overflow exceptions
Currently we are identifying the exception in the avos_product, but it isn't 
being handled properly in the calling code.