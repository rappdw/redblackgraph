RedBlackGraph: A binary tree graph
----------------------------------

# Introduction

## Constraints
A RedBlack Graph is a DAG with the following constraints. 
1) Any given node is assigned a color, either red or black. 
2) Any given node can have at most one outbound arc to a node of a given color.

## Concepts
Additionally, we introduce two concepts, for any given node in such a graph, there are two
"views" for a given node *N*.
1) *Downstream* - The subgraph, constructed by taking *N* and selecting
those nodes that can be linked by following outbound arcs.
2) *Upstream* - The subgraph, constructed by taking the union of all nodes that have *N* in their downstream view. 
Once, *N* is encountered no further outbound arcs are followed.

These constraints ensure that while the resulting graph is a DAG, all
downstreem views are binary trees, and any node, *N* can appear in
an arbitrary number of downstream views of other nodes.

## Motivation
As motivation for considering these constraints, realize that
human familial relationships can be modeled as a RedBlack graph.
If we assign gender of female to "red" and male to "black", and 
represent an output arc from one node to another as parentage, e.g.
outbound arc from node *s* to node *m*, where *m* is "red" indicates
that *m* is the mother of *s* while an outbound arc from node *s*
to node *f* where *f* is "black" indicates that *f* is the father of 
*s*. In this case a *downstream* view of node *s* represents the 
ancestry of node *s*, while the *upstream* view of node *s* represents
all individuals represented in the graph that have *s* as an ancester
(or a descendency view).

# Linear Algebra
Adjacency matrices are sometimes used to represent graphs. **Add brief overview here**.

## Representation
For a RedBlack graph, we shall deviate from the standard representation of an 
adjacency matrix in two distinct ways:
1) The diagonal will hold a *0* if the node is black and a *1* if the node is red
2) Rather than using a *1* to indicate a direct relationship, an arc to a black node
is represented by a *2* if it connects to a black node and a *3* if it connects to a 
red node.

So, to capture the following relationships, myself (*s*, node 0), my father (*f*, node 1), my mother (*m*, node 2) and my paternal 
grandfather (*pgf*, node 3), and my daughter (*d*, node 4)
in a RedBlack Graph, the following adjacency matrix would do so. (The first row and first column in the following
table are not part of the matrix. They label a given row or column)

|       | *s*   | *f*   | *m*   | *pgf* | *d* |
| ----- | ----- | ----- | ----- | ----- | --- |
| *s*   | 0     | 2     | 3     |       |     |
| *f*   |       | 0     |       | 2     |     |
| *m*   |       |       | 1     |       |     |
| *pgf* |       |       |       | 0     |     |
| *d*   | 2     |       |       |       | 1   |

(Notice: the row vectors in this matrix represent the *downstream* view of a 
given node, while the column vectors represent the *upstream* view of a given node.)

## Binary Tree Functions, Products & Algebra
You are probably familar with a [pedigree chart](http://pedigreechart.com). 
It is a form of binary tree used for family history research. Each slot in the
pedigree has a numeric label, 2 for father, 3 for mother, 4 for paternal grandfather, 5 for
paternal grandmother, 6 for maternal grandfather, 7 for maternal grandmother, etc.

### *Pedigree_numbers* and *generation* function
Let's call this labeling of nodes in a binary tree the *pedigree_number*. The
and intorduce the *generation* function. This function takes a *pedigree_number* and returns
the number of generations removed from the root, e.g. 0 for the root (0, 1), 1 for a father or mother 
(2, 3), 2 for grandparents (4, 5, 6, 7), etc. It is obvious that the *generation* function
is simply the integral portion of log2(*pedigree_number*) implemented as:
```
template <class T>
const T generation(T x)
{
    int gen = 0;
    while (x >>= 1) ++gen;
    return gen;
}
```

### *avos* Product for Scalers
Refer back to the Adjacency Matrix above. I see that I am 
related to my father with a *pedigree_number* of 2. I also see
that he is related to his father with a *pedigree_number* of 2. In order
to complete the definition of RedBlack Graph Adjacency Matrix multiplication, 
we need to define a new product that captures is the "transitive" relationship. 
In other words, if I am related to *x* by *pedigree_number*, **A**, and
*x* is related to *y* by *pedigree_number**, **B**, then I am related to *y* by
*pedigree_number*, **C**. The *avos* (latin for ancestor) product defines:

**A** *avos* **B** = **C**. 

In order to uncover the algebra behind this product, let's look at the
*pedigree_numbers* for a few examples, first in base 10, then in binary:

| Use Case           | **A**<sub>base10</sub> | **B**<sub>base10</sub> | **C**<sub>base10</sub> | **A**<sub>binary</sub> | **B**<sub>binary</sub> | **C**<sub>binary</sub> |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ----- |
| My Father's Father | 2     | 2     | 4     | 10    | 10    | 100   |
| My Father's Mother | 2     | 3     | 5     | 10    | 11    | 101   |
| My Mother's Father | 3     | 2     | 6     | 11    | 10    | 110   |
| My Father's PGF    | 2     | 4     | 8     | 10    | 100   | 1000  |

While not obvious, by examining the binary representation we can see that the
*avos* product simply replaces the left most significant digit of **B**
with the value of **A**. Implented as:
```
template <class T>
const T avos(const T& lhs, const T& rhs)
{
    T generationNumber = generation(rhs);
    if (lhs == 0 || lhs == 1) {
        if (generationNumber == 0 && lhs != rhs) {
            throw std::domain_error("Undefined avos." );
        }
        return rhs;
    }
    return (rhs & (T)(pow(2, generationNumber) - 1)) | (lhs << generationNumber);
}
```
## *avos* Product for Vectors
With *pedigree_numbers*, the *generation* function and the *avos* product, we can explore linear algebra.

Examining the example adjacency matrix above, we see that 5th row vector represents the
*downstream* for for my daughter while the third column vector represents the upstream view
for my mother. The *avos* product for these vectors, ideally, would provide
a scaler that is the *pedigree_number* that represents how my daughter is related to my mother,
e.g. 5. The vector dot product, summing the element-wise products of both vectors, results in a scaler 
value of 6. suming the element-wise *avos* product, in this case, results in the *pedigree_number*. It is possible,
both theoretically and practically, for there to be multiple paths to the same ancestor. Arbitrarily, we choose the
most direct relationship as the desired representation. In doing so, we replace summing the
element-wise *avos* product with the non-zero minimum of the element-wise *avos* product.

## Squaring the Adjaceny Matrix and *Completeness*

Repeatedly squaring an adjancency matrix results in a representtation that repeately
expands both the downstream and upstream views by one generation. When *A*<sup>2</sup> = *A*, the
adjacency matrix if fully expanded, and each row represents the complete downstream view and column
represents the complete upstream view. We designate such a matrix as *complete*.

An interesting property of the *complete* matrix is that row vectors representing siblings will be identical, 
whereas column vectors of siblings will be independant.

## *Simple* Vectors
Within the context of *A<sub>complete</sub>*, define *simple*, e.g. an indepdendant 
row or column vector that has non-zero elements representing one-hop relationships to nodes in 
*A<sub>complete</sub>*. Designated as *v<sub>simple</sub>*. *Simple* vectors have the
following properties

* *v<sub>row</sub>* - At most one non-zero element of *2*. At most one non-zero element of *3*.
* *v<sub>column</sub>* - Any number of identical non-zero elements that are *2* if the node represented is *black* and *3* of the node is *red*.
* The *avos* product of a *simple* row vector, *v* and a *complete* adjacency matrix is a *complete* row vector.

    *v<sub>simple</sub>* *avos* *A<sub>complete</sub>* = *v<sub>complete</sub>*
* The *avos* product of a *complete* adjacency marix and a *simple* column vector is a *complete* column vector.

    *A<sub>complete</sub>* *avos* *v<sub>simple</sub>* = *v<sub>complete</sub>*

## Relational Composition
Given *A<sub>complete</sub>* of size *N*, a relational composition is generation of *A<sup>'</sup><sub>complete</sub>* of size *N+1* from
the independant *simple* row/column vectors representing a new node in the graph. Conceptually:

*A<sup>'* = *u* *A* *v*

This composition is accomplished as follows:

*u<sup>'</sup>* = *u* *avos* *A*
 
*v<sup>'</sup>* = *A* *avos* *v*

*A<sup>'</sup>* is composed by "appending" *u<sup>'</sup>* as a new row, *v<sup>'</sup>* as a new column, and then
adding a diagonal element A<sub>N+1,N+1</sub> of either 0 or 1 (0 if the node is *black*, 1 if the node is *red*).

At this point, *A<sup>'</sup>* is composed, but not *complete*. To complete *A<sup>'</sup>*, to the following for each row where


# Old Stuff
A RedBlack Graph is a DAG such that any node is colored either red or black with the following constraint: any node may
have at most 1 outbound arc to a given colored node.

Properties of the graph:
 - The "view" of the graph outbound from any given node is a binary tree
 - Use the following numbering scheme on this binary tree (base 2):
   - root node is numbered 0 if red, 1 if black, root of a parent node is 10 if red, 11 if black, etc.
   - any node in the tree can be uniquely identified such that:
    < fill in here >>
 - Identification of direct ancestry is O(1) operation
 - Identification of relationship is O(Log(n)) operation


It's commercial open-source software, released under the MIT license.
