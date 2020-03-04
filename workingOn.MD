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

* [APSP for Large Graphs on the GPU](https://www.cis.upenn.edu/~cis565/LECTURES/APSP.pdf) (Slide Deck)
* [APSP for Large Graphs on the GPU](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1213&context=hms) (Paper)
* [Efficient Multi-GPU Computation of APSP](https://public.lanl.gov/sunil/pubs/ipdps14.pdf) (paper)
* [APSP in quadratic time with high probability](https://www.youtube.com/watch?v=cn9W2zvD25U&list=ULXZyNMWCZht4&index=3867) (video)

## Distributed Analysis of Large Graphs

* [Experimental Analysis of Distributed Graph Systems](https://arxiv.org/pdf/1806.08082.pdf)
* [Pregel Slides](http://people.apache.org/~edwardyoon/documents/pregel.pdf)
* [Pregel: A System for Large-Scale Graph Processing](https://kowshik.github.io/JPregel/pregel_paper.pdf)
* [Solving APSP Problem in Large Graphs Using Apache Spark](https://arxiv.org/pdf/1902.04446.pdf) (paper)
