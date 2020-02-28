# Dynamic Algorithms
[Dynamic Graph Alogrithms](http://cs.ioc.ee/ewscs/2012/italiano/dynamic1.pdf)

# Formalization of Math 
Introduced to the notion of matrix distance products in [All Pairs Shortest Paths using Bridging Sets
and Rectangular Matrix Multiplication](https://arxiv.org/pdf/cs/0008011.pdf). Distance product for a rectangular 
matrix ==> $c_ij = min{a_ik + b_kj} for 1 <= i,j <= n$

Introduced to Strassen's method of matrix multiplication in [Introduction to Algorithms](http://mitpress.mit.edu/algorithms)

Looking into Strassen's method, it seems that avos multiplicaiton is likely a semiring [Strassen algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm)

# Formalization of Set for Red Black Graphs

A = -1, 0, 1, 2, 3, ...
Removed reference to unsigned types in sparsetools... Rather than do this, 
red 1 should be defined as binary 11111...1 for word width w and then only operate 
on unsigned types. This makes the min function a bit odd, but works well.

# Idea: Compare hop cardinality from various sampled roots
Take a sample of individuals and explore hop cardinality, components identified by hop, etc.

# Observation: Sampled crawls exhibit similar characteristics as the entire tree
We see a large component that contains that vast majority of individuals, there are a handful of smaller compoennts
and a lot of islands. The next largest component is much smaller in size than the largest. (Todo: compare diameters)