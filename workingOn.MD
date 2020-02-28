# Keep Track of What I'm Working On

## Re-work main notebook
Remove the concept of pedigree number replace it with relationship function.

Editorial changes that result from the simplification

## numpy-like implementation
Work on core implementations for:
 
 * transitive closure
 * canonicalization
 
## Sparse implementation
Work on sparse implementations

* transitive_closure
* relational_composition
* canonicalization

## Optimization
Consider optimizations rather than Floyd-Warshall, e.g. topological sort and APSP for DAG. Should be O(VE) rather than O(V^3)

## Need to correctly handle overflow exceptions
Currently we are identifying the exception in the avos_product, but it isn't 
being handled properly in the calling code.

## Reimplement canonical_sort
Do this by combining topological_sort with csgraph.find_components

## Work in Ahnentafel
The basic numbering scheme is the same as the Ahnen number (see: [Wikipedia Reference](https://en.wikipedia.org/wiki/Ahnentafel))

## Look at GPU implementation for Floyd-Warshall
See [Discussion thread on GPU impl of Floyd-Warshall](https://devtalk.nvidia.com/default/topic/534223/simple-cuda-implementation-of-the-floyd-warshall-graph-algorithm/)
it points out several implementations on github