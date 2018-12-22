
# coding: utf-8

# # Introduction
# 
# When representing relationships such as familial relationships a number of approaches may be utilized including: ad-hoc, general graphs, specialized tables or charts. As I've explored different representations, a specialized graph representation has emerged which both has interesting properties and is able to efficiently provide solutions to problems that arise including relationship calculation, iterative modification of the relationship set, loop detection/prevention, etc.
# 
# The underlying model is a directed, acyclic graph of multiple, interleaved binary trees, I designate as "Red Black Graphs". The name, Red Black Graph, derives from superficial similarity to Red Black Trees. Red Black Trees are binary trees such that each node has an extra, color bit (red or black). This color bit us used to balance the tree as modifications are made. In a Red Black Graph each vertex also has a an extra, color bit, rather than utilizing the color bit for balancing, the color bit is used to constrain edges between vertices.
# 
# A formal definition of Red Black Graphs is provided. Interesting emergent properties are explored. Several applications of the underlying math are provided to illustrate the utility and benefites of using Red Black Graphs to model familial relationships.
# 
# ## Formal Definition
# 
# A Red Black Graph is a directed acyclic graph with the following constraints:
# 
# 1. Any given vertex must have a color, either red or black
# 2. Any given vertex can have at most one outbound edge to a vertex of a given color
# 
# ## Motivation
# 
# The relationships resulting from sexual reproduction can be modeled by a Red Black Graph, arbitrarily assigning vertices that are male as Red and vertices that are female as Black with direction of edges being from the offspring to the parent.  
# 
# <!-- to generate latex using nbconvert use the image markdown, for display use the inline html --> 
# <img src="img/simple-graph.png" alt="Sample Graph" style="width: 300px;"/>
# <!-- ![Simple Red Black Graph](img/simple-graph.png){ width=50% } -->
# 
# ## Observation
# For a given vertex in a Red Black graph there are two distinct sub-graphs or "views" or perspectives, Ancestry and Descendency. 
# 
# **Ancestry** is the sub-graph for a given vertex that consists of all the vertices and edges reachable by following out-bound edges. This sub-graph is a well-defined binary tree. 
# 
# <img src="img/simple-graph-ancestry.png" alt="Ancestry View" style="width: 300px;"/>
# <!-- ![Ancestry View for Female Offspring](img/simple-graph-ancestry.png){ width=50% } -->
# 
# **Descendency** is the sub-graph fro a given vertex that consists of all vertices and edges that can follow a graph traversal and arrive at the given node.
# 
# <img src="img/simple-graph-descendency.png" alt="Descendency View" style="width: 300px;"/>
# <!-- ![Descendency View for Female Progenitor](img/simple-graph-descendency.png){ width=50% } -->
# 

# # Underlying Mathematics
# 
# ## Adjacency Matrix
# 
# An adjacency matrix is a square matrix used to represent a graph. The elements of the matrix are 1 if there is an edge between the vertices represented by the column index and the row index. Slightly more formally, with a graph of vertex set *V*, the adjacency matrix is a square |*V*| x |*V*| matrix, $A$, such that $A_{ij}$ is one if there exists an edge from $vertex_{i}$ to $vertex_{j}$ and zero otherwise.
# 
# Given the above sample graph and chosing indices for the vertices as follows: 0 - Female Offspring, 1 - Male Offsprint, 2 - Male Progenitor, 3 - Female Progenitor, the graph would be represented by the following adjacency matrix.
# 
# $$A = \begin{bmatrix}
# 0 & 0 & 1 & 1 \\
# 0 & 0 & 1 & 1 \\
# 0 & 0 & 0 & 0 \\
# 0 & 0 & 0 & 0 \\
# \end{bmatrix}$$
# 
# ## Red Black Graph Adjacency Matrix
# 
# The Red Black Graph adjacency matrix, $R$ is defined as follows: 
# $$R[i,j] = \begin{cases}
#         -1, & \text{if }i=j\text{, and }i\text{ is a red vertex},\\
#         1, & \text{if }i=j\text{, and }i\text{ is a black vertex},\\
#         2, & \text{if there exists an edge from } i \text{ to } j \text{, and } R[j, j] == -1,\\
#         3, & \text{if there exists an edge from } i \text{ to } j \text{, and } R[j, j] == 1,\\
#         0, & \text{otherwise}
# \end{cases}$$
# 
# With this definition, the simple Red Black Graph above is represented as: 
# 
# $$R = \begin{bmatrix}
# 1 & 0 & 2 & 3 \\
# 0 & -1 & 2 & 3 \\
# 0 & 0 & -1 & 0 \\
# 0 & 0 & 0 & 1 \\
# \end{bmatrix}$$
# 
# Observe the following properties:
# 
# $$trace(R) = |V_{black}| - |V_{red}|$$
# 
# $$|V| = |V_{black}| + |V_{red}|$$
# 
# $$|V_{black}| = \frac{|V| + trace(R)}{2}$$
# 
# $$|V_{red}| = \frac{|V| - trace(R)}{2}$$

# ## Transitive Closure
# 
# Computing the transitive closure of an adjacency matrix, $A$, results in the reachability a matrix, $A^+$, that shows all vertices that are reachable from any given vertex. If $A_{ij} == 1$ there is a path from $vertex_{i}$ to $vertex_{j}$.
# 
# The transitive closure of a Red Black adjacency matrix, $R$, is defined to be the complete ancestry/decendency matrix, $R^+$. It indicates reacability, as well as further information, including distance of relationship and explicit traversal path. If $R^+_{ij} == x$ and $x$ is non-zero, then from $x$ we can derive the number of generations of seperation, i.e. 1 for parental relationship, 2 for grand-parental relationship, etc., as well as the relationship "path", e.g. my father's mother's father.
# 
# In recording ancestral relationships it is common to use a pedigree chart. See fig. "Pedigree Chart". 
# 
# <img src="img/pedigree-1.png" alt="Pedigree Chart" style="width: 800px;"/>
# <!-- ![Pedigree Chart](img/pedigree-1.png){ width=80% } -->
# 
# Note that in the pedigree chart, the father's position is labeled "2", the mothers's "3", the paternal grandfather's "4", etc. We will refer to this labeling of the binary tree as the pedigree number, $n_{p}$. The transitive closure of a Red Black adjacency matrix, $R^+$ is defined as:
# $$R^+[i,j] = \begin{cases}
#         -1, & \text{if }i=j\text{, and }i\text{ is a red vertex},\\
#         1, & \text{if }i=j\text{, and }i\text{ is a black vertex},\\
#         n_{p}^{i \rightarrow j}, & \text{if there exists aa relationship from } vertex_i \text{ to } vetex_j\\
#         0, & \text{otherwise}
# \end{cases}$$
# 
# **Note on notation**: $n_{p}^{i \rightarrow j}$ indicates the pedigree number, $n_{p}$, that represents $vertex_{j}$'s position in $vertex_{i}$'s pedigree.
# 
# As an example consider the following relationship graph, where each node has been labeled with a vertex index:
# 
# <img src="img/simple-graph-transitive-closure.png" alt="Graph for Transitive Closure" style="width: 100px;"/>
# <!-- ![Red Black Graph Example for Transitive Closure](img/simple-graph-transitive-closure.png){ width=20% } -->
# 
# By inspection:
# 
# $$R = \begin{bmatrix}
# -1 & 2 & 3 & 0 & 0 \\
# 0 & -1 & 0 & 2 & 0 \\
# 0 & 0 & 1 & 0 & 0 \\
# 0 & 0 & 0 & -1 & 0 \\
# 2 & 0 & 0 & 0 & 1 \\
# \end{bmatrix}\text{, }R^+ = \begin{bmatrix}
# -1 & 2 & 3 & 4 & 0 \\
# 0 & -1 & 0 & 2 & 0 \\
# 0 & 0 & 1 & 0 & 0 \\
# 0 & 0 & 0 & -1 & 0 \\
# 2 & 4 & 5 & 8 & 1 \\
# \end{bmatrix}$$
# 
# Before proceeding to the math operations necessary to transform $R$ to $R^+$, consider the following observations:
# 
# * $n_{p}$ for males (red) are even; $n_{p}$ for females (black) are odd.
# * The generation function for a given pedigree number, $g(n_{p})$, is defined as the number of edges that must be followed to connect the root vertex (position 1 in the pedigree chart) with a vertex designated by $n_{p}$. The generation function is trivially derived by taking the integral portion of $log_{2}$ of $n_{p}$.
# * The traversal path from the root vertex to a given vertex can be derived from the pedigree number by successively right shifting out bits (of a $base_2$ integer representation) and using that bit to "walk" the traversal edge to a red vertex or black vertex, i.e. a shifted 1 bit indicates that the edge to the black vertex should be followed, a 0 shifted bit indicates that the edge to the red vertex should be followed.
# * The diameter or a Red Black Graph is given by the generation function of the maximum $n_{p}$ in $R^+$.
# 
# A simple python implementations of the *generation* and *traversal* functions follow:

# In[1]:


# %load ../redblackgraph/reference/generation.py
def get_traversal_path(pedigree_number):
    '''Given a pedigree_number, representing a relationship from a "root" vertex to an
    "ancester" vertex, return the traversal path of edges to red or black vertices
    to "walk" from the "root" to the "ancesster".

    For example, input of 14 results in ['b', 'b', 'r'] which indicates that starting at
    the "root" vertex, follow the edge to the black vertex, then the edge to the black
    vertex then the edge to the red vertex.'''
    x = pedigree_number
    path = []
    mask = 1
    while (x > 1):
        path.insert(0, 'b' if x & mask else 'r')
        x >>= 1
    return path


# In[2]:


get_traversal_path(12)


# This example of the traversal path for $n_p = 12$ results in the graph walk from a given starting vertex of first following the edge to the black vertex, from there the edge to the red vertex, and, finally, from there the edge to the red vertex; arriving at the maternal grandfather's father's vertex.

# ## Transitive Relationship Function
# 
# As constructing $R^+$ by instpection is cumbersome for non-trivial cases, I've determined an approach to mathematically derive $R^+$ from $R$. The first step requires defining a transitive relationship function. To illustrate, consider the following case of 3 vertices: $vertex_{a}$, $vertex_{b}$ and $vertex_{c}$. Further, assume that $vertex_{b}$ is related to $vertex_{a}$ as defined by $n_{p}^{a \rightarrow b}$, $\pmb{x}$, and that $vertex_{c}$ is related to $vertex_{b}$ by $n_{p}^{b \rightarrow c}$, $\pmb{y}$. Therefore $vertex_{c}$ is related to $vertex_{a}$ by some $n_{p}^{a \rightarrow c}$, $\pmb{z}$. The transitive realtionship function (named the avos product and designated by $\lor$) is defined as: $\pmb{z}=\pmb{x}\lor\pmb{y}$. 
# 
# To explore the avos product, consider this concrete example: $vertex_{b}$ is $vertex_a$'s paternal grandfather ($n_{p}=4$); $vertex_{c}$ is $vertex_{b}$'s maternal grandmother ($n_{p}=7$). If we were to transcribe $vertex_{b}$'s pedigree into the proper place in $vertex_{a}$'s pedigree chart we'd see that $vertex_{c}$ has $n_{p}=19$ in $vertex_{a}$'s pedigree. In other words, $19=4\lor7$.
# 
# To uncover the arithmetic for the avos product, we will examine the some examples. The first column describes $vertex_{a}$'s relationship to $vertex_{b}$, the second column describes $vertex_{b}$'s relationship to $vertex_{c}$, the third column is the $n_{p}^{a \rightarrow b}$, the forth column is the $n_{p}^{b \rightarrow c}$, the final column is the $n_{p}^{a \rightarrow c}$.
# 
# | b's relationship to a  | c's relationship to b | $n_p^{a \rightarrow b}$ | $n_p^{b \rightarrow c}$ | $n_p^{a \rightarrow c}$ |
# | ---------------------- | --------------------- | --------- | -------- | --------- |
# | father                 | father                | 2         | 2        | 4         |
# | father                 | mother                | 2         | 3        | 5         |
# | mother                 | father                | 3         | 2        | 6         |
# | mother                 | mother                | 3         | 3        | 7         |
# | father                 | paternal grandfather  | 2         | 4        | 8         |
# | maternal grandmother   | paternal grandfather  | 7         | 4        | 28        |
# 
# While there appears to be some sort of "counting" going on, it is not obvious what aritmetic should be used to define the avos product. Let's examine the same information, but recast $n_p$ into $base_2$.
# 
# | b's relationship to a  | c's relationship to b | $n_{p, base2}^{a \rightarrow b}$ | $n_{p, base2}^{b \rightarrow c}$ | $n_{p, base2}^{a \rightarrow c}$ |
# | ---------------------- | --------------------- | --------- | -------- | --------- |
# | father                 | father                | 10        | 10       | 100       |
# | father                 | mother                | 10        | 11       | 101       |
# | mother                 | father                | 11        | 10       | 110       |
# | mother                 | mother                | 11        | 11       | 111       |
# | father                 | paternal grandfather  | 10        | 100      | 1000      |
# | maternal grandmother   | paternal grandfather  | 111       | 100      | 11100     |
# 
# While perhaps not obvious, upon examination of the binary representation, the avos product simply replaces the left most significant digit of $n_{p, base2}^{b \rightarrow c}$ with the value of $n_{p, base2}^{a \rightarrow b}$.
# 
# With this observation, a simple implementation of the avos product follows:

# In[3]:


# %load ../redblackgraph/reference/avos.py
from redblackgraph.reference.util import compute_sign, MSB

def avos_sum(x: int, y: int) -> int:
    '''
    The avos sum is the non-zero minumum of x and y unless x == -y in which case the result is 0
    :param x: operand 1
    :param y: operand 2
    :return: avos sum
    '''
    if x == -y:
        return 0
    if x == 0:
        return y
    if y == 0:
        return x
    if x < y:
        return x
    return y

def avos_product(x: int, y: int) -> int:
    '''
    The avos product replaces the left most significant bit of operand 2 with operand 1
    :param x: operand 1
    :param y: operand 2
    :return: avos product
    '''

    sign = compute_sign(x, y)
    x, y = abs(x), abs(y)

    # The zero property of the avos product
    if x == 0 or y == 0:
        return 0

    bit_position = MSB(y)
    return sign * ((y & (2 ** bit_position - 1)) | (x << bit_position))


# ## Transitive Closure for Red-Black Adjacency Matrix
# 
# Transitive closure of an adjacency matrix can be computed a number of ways, a simple approach is the [Floyd-Warshall Algorithm](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm).
# 
# Summarized, this algorithm is a tripple loop across the matrix indices continously updating the current transitive relationship, $A^+_{i,j}$, if there is a relationship from $A_{i,k}$ and a relationship from $A_{k,j}$. For $R^+$, the Floyd-Warshall algorithm can be modified so that the transitive relationship for $R^+_{i,j}$ is defined as $R^+_{i,j} = R_{i,k} \lor R_{k,j}$. An element in $R^+$ is only updated if the transitive relationship is less than the current transitive relationship. To facilitate this, a **nz_min** function is defined that takes multilple arguments as input and returns the non-zero minimum of those arguments. 
# 
# A number of matrix operations utilize a "sum of products" pattern (matrix mutliplication, some versions of Floyd-Warshall, et.c). A similar pattern of "non-zero minimum of avos products" is present in operations on Red Black matrices. An example is present in the following implementation.

# In[4]:


# %load ../redblackgraph/reference/warshall.py
import numpy as np
from dataclasses import dataclass
from typing import Sequence
from redblackgraph.reference import avos_sum, avos_product, MSB


@dataclass
class WarshallResult:
    W: np.array
    diameter: int

def warshall(M: Sequence[Sequence[int]]) :
    '''Computes the transitive closure of a Red Black adjacency matrix and as a side-effect,
    the diameter.'''

    # Modification of stardard warshall algorithm:
    # * Replaces innermost loop's: `W[i][j] = W[i][j] or (W[i][k] and W[k][j])`
    # * Adds diameter calculation
    n = len(M)
    W = np.array(M, copy=True)
    diameter = 0
    for k in range(n):
        for i in range(n):
            for j in range(n):
                W[i][j] = avos_sum(W[i][j], avos_product(W[i][k], W[k][j]))
                diameter = max(diameter, W[i][j])
    return WarshallResult(W, MSB(diameter))


# The example above where $R^+$ was derived by inspection, can now be computed:

# In[5]:


import redblackgraph as rb
R = [[-1,  2,  3,  0,  0],
     [ 0, -1,  0,  2,  0],
     [ 0,  0,  1,  0,  0],
     [ 0,  0,  0, -1,  0],
     [ 2,  0,  0,  0,  1]]
warshall(R).W


# ## Observations
# 
# Given $R^+$, observe that:
# 
# * row vectors represent the complete ancestry view for a given vertex
# * column vectors represent the complete descendency view for a given vertex
# * row vectors representing siblings will be identical
# * column vectors representing siblings will be independant if either of the siblings have offspring represented in the graph
# * determining whether *y* is an ancestor of *x* is **O**(1) and provided by $R^+[x,y]$

# # Applications of $R^+$
# 
# ## Calculating Relationship Between Two Rows in $R^+$
# 
# With $R^+$ there exists an efficient way to determining full kinship (see: [consanguinity](https://en.wikipedia.org/wiki/Consanguinity)) between any two vertices. 
# 
# 1. Given two row vectors from $R^+$, $\vec a$ and $\vec b$, find the minimum of $\vec a_{i} + \vec b_{i}$ where both $\vec a_{i}$ and $\vec b_{i}$ are non-zero. This yields two pedigree numbers, $n_{p}^{a \rightarrow i}$ and $n_{p}^{b \rightarrow i}$ expressing the relationship of $vertex_{a}$ and $vertex_{b}$ to the nearest common ancestor, $vertex_{i-min}$
# 2. Determine the generational distance to the common ancestor, $d_{a}=g(n_{p}^{a \rightarrow i})$ and $d_{b}=g(n_{p}^{b \rightarrow i})$.
# 3. Using a Table of Consanguinity, calculate the relationship
# 
# ### Observation
# 
# Determining whether $x$ is related to $y$ is **O**($m$) where $m$ is the expected number of ancestors and $m << |V|$ (assuming an efficient sparse matrix representation, otherwise it is **O**($|V|$)). Empirically, $m$ is on the order of $ln(|V|)$.
# 
# A simple implementation follows:

# In[6]:


# %load ../redblackgraph/reference/calc_relationship.py
from dataclasses import dataclass
from typing import Sequence
from redblackgraph.reference import MSB


@dataclass
class Relationship:
    common_ancestor: int
    relationship: str

def lookup_relationship(da: int, db: int) -> str:
    '''
    This is a very rudimentary implementation of a Consanguinity lookup and doesn't handle many
    cases correctly.
    :param da: generational distance from u to common ancestor
    :param db: generational distance from v to common ancester
    :return: a string designating relationship
    '''
    da, db = abs(da), abs(db)
    removal = abs(da - db)
    if da == 0 or db == 0:
        # direct ancestor
        if removal == 1:
            return "parent"
        if removal == 2:
            return "grandparent"
        if removal == 3:
            return "great grandparent"
        return f"{removal - 2} great grandparent"
    else:
        # cousin
        generational = min(da, db)
        return f"{generational - 1} cousin {removal} removed"


def calculate_relationship(a: Sequence[int], b: Sequence[int]) -> Relationship:
    '''
    Determine if a relationship exists between u, v where u, v are row vectors of the transitive
    closure of a Red Black adjacency matrix
    :param a: row vector for vertex u
    :param b: row vector for vertex v
    :return: (Relationship designation, common ancestor vertex)
    '''

    common_ancestor, (x, y) = min([e for e in enumerate(zip(a, b))
                                   if not e[1][0] == 0 and not e[1][1] == 0],
                                  key=lambda x: x[1][0] + x[1][1],
                                  default=(-1, (0, 0)))

    if common_ancestor == -1:
        return Relationship(-1, "No Relationship")
    return Relationship(common_ancestor,
                        lookup_relationship(MSB(x), MSB(y)))


# # Linear Algebra
# 
# ## Introduction
# Having provided a formal definition for a Red Black Graph, looked at its adjacency matrix, $R$, the transitive closure of its adjacency matrix, $R^+$, and the avos product, lets extend these observations into a more general discussion of how principles of linear algebra can be applied to Red Black Graphs.
# 
# ## Vector Classes
# Within the context of a Red Black Graph and its matrix representations, $R$ and $R^+$, the following vector classes are defined:
# 
# * *row* vector - represented as $\vec u$. These vectors represent ancestry for a given vertex. Values for elements in these vectors are constrained to whole numbers and -1 where any number, aside from 0, may appear in an alement at most once and where either -1 or 1 must appear as an element but not both. 
# * *column* vector - represented by $\vec v$. These vectors represent descendency for a given vertex. Values for elements in these vectors are constrained to whole numbers and -1 where either -1 or 1 must appear as an element but not both. Futhermore if -1 appears as an element any further non-zero integer elements must be even and if 1 appears as an element any further non-zero integer elements must be odd.
# * *simple row vector* - represented by $\vec u_{s}$. Row vectors for which elements are constrained to {-1, 0, 1, 2, 3}. These represent a given vertex and it's immediate ancestry only.
# * *simple column vector* - represented by $\vec v_{s}$. Column vectors for which elements are constrained to {-1, 0, 1, 2, 3}. These represent a given vertex and it's immediate descendency only.
# * *closed row vector* - represented by $\vec u_{c}$. Row vectors from $R^+$. These represent the complete ancestry for a given vertex.
# * *closed column vector* - represented by $\vec v_{c}$. Column vectors from $R^+$. These represent the complete descendency for a given vertex.
# * *compositional vectors* - represented by $\vec u^{c}_{s}$ or $\vec v^{c}_{s}$. Compositional vectors conform to the constrainst of simple row or column vectors with the following additional constraint: neiter 1 nor -1 appear as an element. The color of the vector is inherent to the vector but not carried as an element. Any consraints due the color are present as if the color were present as an element. Color, if significant, is represented notationally by replacing the supersscript $c$ with the color designation, either $r$ or $b$.
# 
# 
# ## Avos Product for Vectors
# 
# Consider what an avos vector product might represent. Given a row vector and a column vector, the avos product is defined to yield $n_{p}$ representing the relationship between the vertices representing the row and column vectors respectively. 
# 
# Consider the $R$ from the transitive closure example:
# 
# $$\begin{bmatrix}
# -1 & 2 & 3 & 0 & 0 \\
# 0 & -1 & 0 & 2 & 0 \\
# 0 & 0 & 1 & 0 & 0 \\
# 0 & 0 & 0 & -1 & 0 \\
# 2 & 0 & 0 & 0 & 1 \\
# \end{bmatrix}$$
# 
# The 4th row vector of $R$ is $u_{s}$ for $vertex_{4}$ while the 2nd column vector is $v_{s}$ for $vertex_{2}$. It is observable by inspection that $vertex_{4}$ is related to $vertex_{2}$ by $n_{p} == 5$. We desire to define $\vec{u_{s}}\lor\vec{v_{s}}=n^{4 \rightarrow 2}_{p}$ or
# 
# $$\begin{bmatrix}
# 2 & 0 & 0 & 0 & 1 \\
# \end{bmatrix}
# \lor
# \begin{bmatrix}
# 3 \\
# 0 \\
# 1 \\
# 0 \\
# 0 \\
# \end{bmatrix}
# = 5$$
# 
# The vector dot product, summing element-wise products, results in a scaler value of 6. Summing element-wise avos products does yield 5, which represents a relationship. However, it is possible for there to be multiple paths through the graph bewteen two nodes. Should that be the case, summing the element-wise avos products would not result in the $n_{p}$. For the avos vector product, rather than summing the element-wise avos products the non-zero minimum of element-wise product is chosen, thus representing the "closest" relationship between $vertex_{a}$ and $vertex_{c}$.
# 
# A simple implementation of the avos vector product follows:

# In[7]:


# %load ../redblackgraph/reference/vec_avos.py
from functools import reduce
from redblackgraph.reference import avos_product, avos_sum

def vec_avos(u, v):
    '''Given two vectors, compute the avos product.'''
    return reduce( avos_sum, [avos_product(a, b) for a, b in zip(u, v)])


# ## Observation - Simple/Complete Relationship
# 
# The product of a simple row vector and the transitive closure of a Red Black adjacency matrix is a closed row vector
# $$\vec{u_{s}} \lor R^+ = \vec{u_{c}}$$
# 
# The product of the transitive closure of a Red Black adjacency matrix and a simple column vector is a closed column vector
# $$R^+ \lor \vec{v_{s}} = \vec{v_{c}}$$
# 
# **TODO**: Need to walk through an explanation of why this is so.
# 
# ## Avos product for Matrices
# 
# With scaler and vector avos products defined, extension to matrices is elementary. Given $A$ and $B$, both matrices following the constraints defined for $R$, and $C = A \lor B$, the elements of $C_{ij}$ are given by the vector avos product of $u_{i}$ from A and $v_{j}$ from B
# 
# Avos matrix multiplication of general matrices seems a little abstract so consider the following practical example. $R \lor R$ shows all vertices directly related by following up to 2 relationship edges, $R \lor R \lor R$ shows all vertices related by following up to 3 relationship edges, etc. For some $m <= |V|$ there will be a $\prod_{n=1}^{m} R == R^+$.
# 
# A simple implementation of the avos matrix product follows:

# In[8]:


# %load ../redblackgraph/reference/mat_avos.py
from functools import reduce
from redblackgraph.reference import avos_product, avos_sum


def mat_avos(A, B):
    '''Given two matrices, compute the "avos" product.'''
    return [[reduce( avos_sum, [avos_product(a, b) for a, b in zip(A_row, B_col)]) for B_col in zip(*B)] for A_row in A]


# ## Relational Composition
# ### Adding a Vertex to $R^+$
# 
# Consider the case of adding a new vertex to a red black graph. The new vertex, $\lambda$, may introduce edges to/from vertices in the graph and the corresponding row/column vectors conform to the compositional vector classes defined above. Specifically if adding a red vertex to the graph, the vectors $u^{r}_{\lambda,s}$ and $v^{r}_{\lambda,s}$ define the composition, or if adding a black vertex to the graph, the vectors $u^{b}_{\lambda,s}$ and $v^{b}_{\lambda,s}$ define the composition. These compositional vectors have non-zero elements only for immediate ancestry/descendency. The operation of adding a new vertex to a graph is designated the "vertex relational composition" and is defined where $R^+$ is a square matrix of dimension $N$ and $R_{\lambda}^+$ is a square matrix of dimension $N + 1$ and the colors of $\vec{u^{c}_{\lambda,s}}$ and $\vec{v^{c}_{\lambda,s}}$ must be the same. The notation of the vertex relational composition is:
# 
# $$R_{\lambda}^+ = {\vec{u^{c}_{\lambda,s}} R^+ \vec{v^{c}_{\lambda,s}}}_{color}$$
# 
# The simple/complete relationship observation above can be applied in this instance. $\vec{u^{c}_{\lambda,s}} = \vec{u^{c}_{\lambda,c}}$ and $R^+ \lor \vec{v^{c}_{\lambda,s}} = \vec{v^{c}_{\lambda,c}}$. $\vec{u^{c}_{\lambda,c}}$ and $\vec{v^{c}_{\lambda,c}}$ are the row and column, respectively, that need to be appended to $R^+$ (along with the final diagonal element corresponding to $\lambda$'s color) to compose $R_{\lambda}^+$. Appending the complete compositional vectors to $R^+$ isn't sufficient to compose $R_{\lambda}^+$. The "body" of $R^+$ needs to be "updated" to ensure that $R_{\lambda}^+$ is also transitively closed. For each row in $R^+$, every element in that row is set to the avos product of the corresponding column element in $\vec{v^{c}_{\lambda,c}}$ and the corresponding row element in $\vec{u^{c}_{\lambda,c}}$.
# 
# Expressing this algorithmically:
# 
# 1. generate $\vec{u^{c}_{\lambda,c}} = \vec{u^{c}_{\lambda,s}} \lor R^+$
# 2. generate $\vec{v^{c}_{\lambda,c}} = R^+ \lor \vec{v^{c}_{\lambda,s}}$
# 2. Compose $R_{\lambda}^+$ by:
#     1. appending $u^{c}_{\lambda,c}$ to $R^+$ as a new row 
#     2. appending $v^{c}_{\lambda,c}$ to $R^+$ as a new column
#     3. setting the diagnoal element ${R_{\lambda}^+}_{N+1, N+1}$ to either 1 or -1 depending on the color of the composition.
#     4. For each row, $i$, and each column, $j$, where $\vec{u^{c}_{\lambda,c}}_{j} \neq 0$, set ${R_{\lambda}^{+}}_{i,j} = \vec{u^{c}_{\lambda,s}}_{j} \lor \vec{v^{c}_{\lambda,c}}_{i}$
#     
# ### Adding an Edge to $R^+$
# Consider the case of adding a new edge to a red black graph. The operation of adding a new edge to a graph is designated the "edge relational composition". The new edge is added between two existing vertices, $vertex_\alpha$ and $vertex_\beta$. The notation of the edge relational composition is:
# 
# $$R^+_\lambda = R^+ \lor_{\alpha, \beta} n_p^{\alpha \rightarrow \beta}$$
# 
# As in the vertex relational composition, we'll make use of the simple/complete relational observation. In this case, the row representing $vertex_\alpha$ is replaced with the avos product of itself (with $element_\beta$ replaced with $n_p^{\alpha \rightarrow \beta}$) and $R^+$. Notationally: $R^{+'} = R^+ +_\alpha ((vertex_\alpha +_\beta n_p^{\alpha \rightarrow \beta}) \lor R^+)$ where $+_i$ designates replacement of element $i$ in the LHS with the value of the RHS. As in the vertex relational composition, replacing row vector $\alpha$ with it's complete form isn't sufficient to compose $R_{\lambda}^+$. The remainder of the row vectors need to be closed with the new relationship. For each row, $i$, in $R^{+'}$ excluding $\alpha$, every element, $j$ in that row is set to $R^{+'}_{i,\alpha} \lor R^{+'}_{\alpha,j}$.
# 
# Expressing this algorithmically:
# 
# 1. generate $\vec{u^{'}_\alpha} = \vec{u_\alpha} +_{\beta} n_p^{\alpha \rightarrow \beta}$, where $\vec{u_\alpha}$ is row $\alpha$ in $R^+$
# 2. generate $\vec{u^{c'}_\alpha} = \vec{u^{'}_\alpha} \lor R^+$
# 2. Compose $R_{\lambda}^+$ by:
#     1. replacing row $\alpha$ in $R^+$: $R^{+'} = R^+ +_{\beta} \vec{u^{c'}_a\alpha}$ 
#     2. For each row, $i$, and each column, $j$, where $i \neq \alpha$, set ${R_{\lambda}^{+}}_{i,j} = R^{+'}_{i,\alpha} \lor R^{+'}_{\alpha,j}$
# 
# ### Simple Implementations

# In[9]:


# %load ../redblackgraph/reference/rel_composition.py
from redblackgraph.reference import avos_sum, avos_product, mat_avos
import copy


def vertex_relational_composition(u, R, v, color):
    '''
    Given simple row vector u, transitively closed matrix R, and simple column vector v where
    u and v represent a vertex, lambda, not currently represented in R, compose R_{\lambda}
    which is the transitive closure for the graph with lambda included
    :param u: simple row vector for new vertex, lambda
    :param R: transitive closure for Red Black graph
    :param v: simple column vector for new vertex, lambda
    :param color: color of the node either -1 or 1
    :return: transitive closure of the graph, R, with new node, lambda
    '''
    N = len(R)
    uc_lambda = mat_avos(u, R)
    vc_lambda = mat_avos(R, v)
    R_lambda = copy.deepcopy(R)
    R_lambda.append(uc_lambda[0])
    for i in range(N):
        R_lambda[i].append(vc_lambda[i][0])
        for j in range(N):
            if uc_lambda[0][j] != 0:
                R_lambda[i][j] = avos_sum(avos_product(vc_lambda[i][0], uc_lambda[0][j]), R_lambda[i][j])
    R_lambda[N].append(color)
    return R_lambda

def edge_relational_composition(R, alpha, beta, relationship):
    '''
    Given a transitively closed graph, two vertices in that graph, alpha and beta, and the
    relationship from alpha to beta, compose R'', which is the transitive closure with the
    new edge included
    :param R:
    :param alpha: a vertex in the graph (row index)
    :param beta: a vertex in the grpah (column index)
    :param relationship: the relationship (beta's pedigree number in alpha's pedigree)
    :return: transitive closure of the grpah, R, with new edge
    '''
    N = len(R)
    u_lambda = [R[alpha]]
    u_lambda[0][beta] = relationship
    u_lambda = mat_avos(u_lambda, R)
    R_lambda = copy.deepcopy(R)
    R_lambda[alpha] = u_lambda[0]
    for i in range(N):
        for j in range(N):
            if R_lambda[alpha][j] != 0:
                R_lambda[i][j] = avos_sum(avos_product(R_lambda[i][alpha], R_lambda[alpha][j]), R_lambda[i][j])
    return R_lambda


# # Applications of avos Linear Algebra
# ## Loop Prevention
# 
# An issue that can be encountered in systems that represent familial relationships is the inadvertent injection of graph cycles, resulting in the ["I am my own Grandpa"](https://en.wikipedia.org/wiki/I%27m_My_Own_Grandpa) case. While this is impossible when relationships model sexual reproduction, the introduction of step-relationships, etc. would make this a possibility. Often times there is ambiguity in the available historical records. If a researcher isn't careful, cylces may result as a genealogical model is created. Modifications to both forms of the relational composition algorithms can prevent the introduction of cycles into the graph. 
# 
# ### Vertex Relational Composition Loop Prevention
# 
# As vertices are added to an existing graph via relational composition, the intermedite, complete compositional vectors, $\vec{u^{c}_{\lambda, c}}$ and $\vec{v^{c}_{\lambda, s}}$ represent the complete ancestry and complete descedency for the new vertex $\lambda$ respectively. The cycle constraint would be invalidated should there be any vertex that simultaneously appears in the ancestry and descendency for a given vertex.
# 
# Given $\vec{u^{c}_{\lambda, c}}$ and $\vec{v^{c}_{\lambda, s}}$ of dimension $n$, the **vertex relational composition** is undefined if there exists a dimension $i$ where $i \neq n \land \vec{u^{c}_{\lambda, c}}_{i} \neq 0 \land \vec{v^{c}_{\lambda, s}}_{i} \neq 0$ and is well-formed otherwise.
# 
# ### Edge Relational Composition Loop Prevention
# 
# This case is trivial with a transitively closed matrix. Given $R^+$ and $n_p^{\alpha \rightarrow \beta}$, the **edge relational composition** is undefined if $n_p^{\beta \rightarrow \alpha} \neq 0$ and well-formed otherwise.
# 
# ## Connected Component Identification
# 
# As Red Black Graphs are used to represent family relationships, an interesting case is determining how many disjoint trees are represetned within a graph. Tarjan's algorithm is typically used to compute the connected components of a graph. In the case of a transitively closed adjacency matrix, the depth first search used in Tarjan's algorithm is inherently "pre-computed". Because of this property, Tarjan's algorithm can be simplified.
# 

# In[10]:


# %load ../redblackgraph/reference/components.py
def find_components(A):
    """
    Given an input adjacency matrix compute the connected components
    :param A: input adjacency matrix (transitively closed)
    :return: a vector with matching length of A with the elements holding the connected component id of
    the identified connected components
    """
    n = len(A)
    u = [0] * n
    component_number = 1
    u[0] = component_number
    for i in range(n):
        if u[i] == 0:
            component_number += 1
            u[i] = component_number
        row_component_number = u[i]
        for j in range(n):
            if A[i][j] != 0:
                if u[j] == 0:
                    u[j] = row_component_number
                elif u[j] != row_component_number:
                    # There are a couple cases here. We implicitely assume a new row
                    # is a new component, so we need to back that out (iterate from 0
                    # to j), but we could also encounter a row that "merges" two
                    # components (need to sweep the entire u vector)
                    for k in range(n):
                        if u[k] == row_component_number:
                            u[k] = u[j]
                    component_number -= 1
                    row_component_number = u[j]
                    u[i] = row_component_number
    return u


# Consider the following graph
# 
# <img src="img/find-components.png" alt="Graph with Components" style="width: 200px;"/>
# <!-- ![Graph with Components](img/find-components.png){ width=50% } -->
# 
# By inspection, there are two components and the application of the simplified Tarjan's algorithm identifies which vertices belong to which components.

# In[11]:


R = [[-1, 0, 0, 2, 0, 3, 0],
     [ 0,-1, 0, 0, 0, 0, 0],
     [ 2, 0, 1, 0, 0, 0, 0],
     [ 0, 0, 0,-1, 0, 0, 0],
     [ 0, 2, 0, 0,-1, 0, 3],
     [ 0, 0, 0, 0, 0, 1, 0],
     [ 0, 0, 0, 0, 0, 0, 1]]
find_components(R)


# With an efficient sparse representation this algorithm is also $\mathbf{O}(|V| + |E|)$. 
# 
# ## Canonical Form
# 
# Returning to the example, it is obvious from inspection that one component consists of 4 nodes, the other of 3 and that the diamter of the larger component is 2, while the diamter of the smaller is 1. As this information is readily available in the Red Black Graph, it is easily added to the $find\_components$ algorithm (see the following $find\_components\_extended$ algorithm). With the observation that symetrically permuting a matrix corresponds to relabeling the vertices of the associated graph, I will show that with an appropriate relabeling of the graph vertices the Red Black graph adjacency matrix is upper triangular, $R^{+_c}$ or canonical form, and that $R^{+_c} = \mathbf{P} R^+ \mathbf{P}^\top$ where $\mathbf{P}$ is a permutation matrix derived from the count of the vertices in a component, the identity of the encompasing component for a vertex, and the maximum $n_p$ for each vertex. 
# 
# To arrive at $\mathbf{P}$ the list of nodes is sorted (in reverse order) first on the size of the encompassing connected component, secondly on the identifier of the connected component and finally on the maximum $n_p$ for the vertex. The vertices are then labeled based on this sorting, e.g. the $zero^{th}$ vertex is the vetex from the largest connected component that has the greatest $n_p$ (or most distant ancestor) on down to the $n^{th}$ vertex which is the vertex from the smallest connected component with no (or nearest) ancestor. (Ordering is arbitrary for vertices with identical sort keys.)
# 
# A simple implementation of triangularizing $R$ based on the properties inherent in the adjacency matrix and the extended $find\_components$ algorithm follows.

# In[12]:


# %load ../redblackgraph/reference/triangularization.py
import numpy as np

from dataclasses import dataclass
from typing import Dict, Iterator, Sequence, Tuple
from collections import defaultdict
from typing import Tuple

@dataclass
class Components:
    ids: Sequence[int]
    max_pedigree_number: Sequence[int]
    size_map: Dict[int, int] # keyed by component id, valued by size of component

    def get_permutation_basis(self):
        # this yeilds a list of tuples where each tuple is the size of the component, the component id of the vertex,
        # the max np for the vertex and the id of the vertex. We want the nodes
        # ordered by components size, componente id, max np, finally by vertex id
        return sorted(
            [(self.size_map[element[1][0]],) + element[1] + (element[0],)
             for element in enumerate(zip(self.ids, self.max_pedigree_number))],
            reverse=True
        )

@dataclass
class Triangularization:
    A: Sequence[Sequence[int]]
    label_permutation: Sequence[int]

def find_components_extended(A: Sequence[Sequence[int]]) -> Components:
    """
    Given an input adjacency matrix (assumed to be transitively closed), canonical_sort the
    matrix (simply a relabeling of the graph)
    :param A: input adjacency matrix
    :return: a tuple of:
      [0] - a vector matching length of A with the elements holding the connected component id of
      the identified connected components - labeled u
      [1] - a vector matching length of A with the elements holding the max n_p for the corresponding
      row - labeled v
      [2] - a dictionary keyed by component id and valued by size of component
    """
    n = len(A)
    u = [0] * n
    v = [0] * n
    q = defaultdict(lambda: 0)
    component_number = 1
    u[0] = component_number
    q[component_number] += 1
    for i in range(n):
        row_max = -2
        if u[i] == 0:
            component_number += 1
            u[i] = component_number
            q[component_number] += 1
        row_component_number = u[i]
        for j in range(n):
            if A[i][j] != 0:
                row_max = max(A[i][j], row_max)
                if u[j] == 0:
                    u[j] = row_component_number
                    q[row_component_number] += 1
                elif u[j] != row_component_number:
                    # There are a couple cases here. We implicitely assume a new row
                    # is a new component, so we need to back that out (iterate from 0
                    # to j), but we could also encounter a row that "merges" two
                    # components (need to sweep the entire u vector)
                    for k in range(n):
                        if u[k] == row_component_number:
                            u[k] = u[j]
                            q[row_component_number] -= 1
                            q[u[j]] += 1
                    component_number -= 1
                    row_component_number = u[j]
        v[i] = row_max
    return Components(u, v, {k:v for k,v in q.items() if v != 0})

def _get_triangularization_permutation_matrices(A: Sequence[Sequence[int]]):
    """
    u, v, and q are computed via find_components_extended, and then used to compute a
    permutation matrix, P, and P_transpose
    :param A:
    :return: the permutation matrices that will canonical_sort A
    """
    permutation_basis = find_components_extended(A).get_permutation_basis()

    # from the permutation basis, create the permutation matrix
    n = len(permutation_basis)
    P = np.zeros(shape=(n, n), dtype=np.int32)
    P_transpose = np.zeros(shape=(n, n), dtype=np.int32)
    # label_permutation can be calculated as P @ np.arrange(n), but since we are running the index do it here
    label_permutation = np.arange(n)
    for idx, element in enumerate(permutation_basis):
        label_permutation[idx] = element[3]
        P[idx][element[3]] = 1
        P_transpose[element[3]][idx] = 1
    return P, P_transpose, label_permutation


def triangularize(A: Sequence[Sequence[int]]) -> Triangularization:
    """
    canonical_sort the matrix. Uses P and P_transpose if provided, otherwise computes
    the permutation matrices
    :param A:
    :param P: the transposition matrices (P and P_transpose)
    :return: a triangular matrix that is symmetrical to A (a relabeling of the graph vertices)
    """
    P, P_t, label_permutation = _get_triangularization_permutation_matrices(A)

    # triagularize A
    return Triangularization((P @ A @ P_t), label_permutation)


# In[13]:


triangularize(warshall(R).W).A


# <img src="img/canonical.png" alt="Graph with Components (Canonical Form)" style="width: 200px;"/>
# <!-- ![Graph with Components (Canonical Form)](img/canonical.png){ width=50% } -->
# 

# # Appendix A
# ## Determinants
# 
# Let's explore the determinants of the class of matrices that represent Red Black Graphs. Staring with the simple case of a $2 x 2$ matrix.
# 
# $$\begin{vmatrix}A\end{vmatrix} = \begin{vmatrix}a & b \\ c & d \\ \end{vmatrix} = ad - cb$$
# 
# As per formal definition, $a$ and $d$ $\in \left\{ {-1, 1}\right\}$; $b$ defines the relationship from the vertex represented by the first row to the vertex represented by the second row; $c$ defines the relationship from the vertex represented by the second row to the vertex represented by the first row. 
# 
# As per constraints (no cycles) if $b$ is non-zero then $c$ must be zero and conversely if $c$ is non-zero, $b$ must be zero. Therefore for a $2 x 2$ matrix, $A$, $det(A) \in \left\{ {-1, 1}\right\}$.
# 
# Consider the case of a $3 x 3$ matrix.
# 
# $$\begin{vmatrix}A\end{vmatrix} = \begin{vmatrix}a & b & c \\ d & e & f \\ g & h & i \end{vmatrix} = aei + bfg + cdh - ceg - bdi - afh$$
# 
# As in the $2 x 2$ case, the product of the diagonals is constrained to $\left\{ {-1, 1}\right\}$ and all other terms will be zero as they either represent the cycle of path length 2 or path length 1. Let's label the vertex representing by the first row as $\alpha$, the second row as $\beta$ and the third row as $\gamma$. Let's look at the $bfg$ term. $b$ represents the relationship from $\alpha$ to $\beta$, $f$ represents the relationship from $\beta$ to $\gamma$ and $g$ represents the relationship from $\gamma$ to $\alpha$. This term defines a cycle of path length 2 and at least one of the terms must be zero by constraint. 
# 
# Let's look at the $ceg$ term. $c$ represents the relationship from $\alpha$ to $\gamma$, $e$ represents the relationship from $\beta$ to itself and $g$ represents the relationship from $\gamma$ to $\alpha$. Again, by constraint, either $c$ or $g$ must be zero. Therfore $ceg$ will be zero. Likewise, bdi and afh terms will be zero.
# 
# While the $2 x 2$ and $3 x 3$ cases are interesting, this line of reasoning doesn't extend to finding the determinant of higher dimensional matrices. As any Red Black graph can be represented in it's canonical form, an upper triangular matrix, we observe that:
# $\det R =
#     \begin{cases}
#             1, &         \text{if } |V_{red}| \text{ is even},\\
#             -1, &         \text{if } |V_{red}| \text{ is odd}.
#     \end{cases}$

# # Appendix B
# ## Implementation Notes on Numpy Extension
# 
# Following are some python examples. In addition to the simple implementation presented above in pure python, for performance optimized linear algebra operations, extension modules are provided by Numpy, SciPy, etc. The redblackgraph module also provides extension model implementations that are described below
# 
# ## rb.array / rb.matrix
# The redblackgraph module provides two Numpy extensions, one for array and one for matrix.
# 
# The distinctive characteristics of these classes are matrix multiplication has been overridden to support the avos product, as well as methods defined for transitive_closure and relational_composition
# 
# To motivate the examples, let's model my familial relationshps. I'm (D) the child of Ewald (E) and Regina (R). Ewald and Marta (M) also have a child, my half-brother, Harald (H). Ewald's parents were Michael (Mi) and Amalie (A). Regenia's parents were John (J) and Inez (I). John also had a son Donald (Do) with Evelyn (Ev). Michael's parents were George (G) and Mariea (Ma). Finally, John's parents were Samuel (S) and Emeline (Em).
# 
# This set of relationships is represented by the graph below
# 
# <img src="img/small-graph.png" alt="Graph for Exploring Python Implementation" style="width: 300px;"/>
# <!-- ![Graph for Exploring Python Implementation](img/small-graph.png){ width=75% } -->
# 
# We'll model this as a RedBlackGraph denoting each vertex numerically in the order introduced in the above narrative, e.g. D:0, E:1, R:2, M:3, H:4, Mi:5, A:6, J:7, I:8, Do:9, Ev:10, G:11, Ma:12, S:13, Em14
# 
# In these examples, we'll first calculate transitive closure then we'll remove the node (row/column) for John, create the simple row and column vectors for John and use a relational composition to recostruct a transitive closure equivalent. Finally we'll get some timings to compare implementations.

# ## Simple Implementation
# ### Transitive Closure

# In[14]:


import numpy as np
import redblackgraph.reference as smp
import copy
#      D   E   R   M   H  Mi   A   J   I  Do  Ev   G  Ma   S  Em
A = [[-1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # D
     [ 0, -1,  0,  0,  0,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0], # E
     [ 0,  0,  1,  0,  0,  0,  0,  2,  3,  0,  0,  0,  0,  0,  0], # R
     [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # M
     [ 0,  2,  0,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # H
     [ 0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  2,  3,  0,  0], # Mi
     [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0], # A
     [ 0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  2,  3], # J
     [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0], # I
     [ 0,  0,  0,  0,  0,  0,  0,  2,  0, -1,  3,  0,  0,  0,  0], # Do
     [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0], # Ev
     [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0], # G
     [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0], # Ma
     [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0], # S
     [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]  # Em
    ]
B = copy.deepcopy(A)
res = smp.warshall(B)
print(f"A_star:\n{res.W} \ndiameter: {res.diameter}")


# ### Vertex Relational Composition
# For illustrative purposes, let's remove John from the rb.array representation of the graph

# In[15]:


#       D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em
A1 = [[-1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # D
      [ 0, -1,  0,  0,  0,  2,  3,  0,  0,  0,  0,  0,  0,  0], # E
      [ 0,  0,  1,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0], # R
      [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # M
      [ 0,  2,  0,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0], # H
      [ 0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  2,  3,  0,  0], # Mi
      [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0], # A
      [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0], # I
      [ 0,  0,  0,  0,  0,  0,  0,  0, -1,  3,  0,  0,  0,  0], # Do
      [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0], # Ev
      [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0], # G
      [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0], # Ma
      [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0], # S
      [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]  # Em
     ]
B1 = copy.deepcopy(A1)
res = smp.warshall(B1)
print(f"A1_star:\n{res.W} \ndiameter: {res.diameter}")


# **Observation**: I am no longer related to Samuel nor Emeline, but that the diameter is still 3 (my relationship to George and Mariea).
# 
# Let's look at the row (u) and column (v) vectors that would define John in relationship to A1 as well as the relational_composition of A1 with u and v.

# In[16]:


#      D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em
u = [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  3]] 
v = [[ 0], # D  
     [ 0], # E
     [ 2], # R
     [ 0], # M
     [ 0], # H
     [ 0], # Mi
     [ 0], # A
     [ 0], # I
     [ 2], # Do
     [ 0], # Ev
     [ 0], # G
     [ 0], # Ma
     [ 0], # S
     [ 0], # Em
    ]
A_lambda = smp.vertex_relational_composition(u, A1, v, -1)
A_lambda


# ### Edge Transitive Closure
# Using the above example, remove the relationship from Regina to John

# In[17]:


#        D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em   J
R1 = [[ -1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # D
      [  0, -1,  0,  0,  0,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0],  # E
      [  0,  0,  1,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0],  # R
      [  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # M
      [  0,  2,  0,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # H
      [  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  2,  3,  0,  0,  0],  # Mi
      [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],  # A
      [  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],  # I
      [  0,  0,  0,  0,  0,  0,  0,  0, -1,  3,  0,  0,  0,  0,  2],  # Do
      [  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],  # Ev
      [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0],  # G
      [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],  # Ma
      [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],  # S
      [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0], # Em
      [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  3, -1]  # J
      ]
R = smp.warshall(R1).W
# Missing edge is R -> J, 2
A_lambda = smp.edge_relational_composition(R, 2, 14, 2)
A_lambda


# ### Timings

# In[18]:


get_ipython().run_cell_magic('timeit', '', 'B1 = copy.deepcopy(A1)\nres = smp.warshall(B1)')


# In[19]:


get_ipython().run_cell_magic('timeit', '', 'A_lambda = smp.vertex_relational_composition(u, A1, v, -1)')


# In[20]:


get_ipython().run_cell_magic('timeit', '', 'A_lambda = smp.edge_relational_composition(R, 2, 14, 2)')


# ## Optimized Implementation
# ### Transitive Closure

# In[21]:


#               D   E   R   M   H  Mi   A   J   I  Do  Ev   G  Ma   S  Em
A = rb.array([[-1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # D
              [ 0, -1,  0,  0,  0,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0], # E
              [ 0,  0,  1,  0,  0,  0,  0,  2,  3,  0,  0,  0,  0,  0,  0], # R
              [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # M
              [ 0,  2,  0,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # H
              [ 0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  2,  3,  0,  0], # Mi
              [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0], # A
              [ 0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  2,  3], # J
              [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0], # I
              [ 0,  0,  0,  0,  0,  0,  0,  2,  0, -1,  3,  0,  0,  0,  0], # Do
              [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0], # Ev
              [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0], # G
              [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0], # Ma
              [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0], # S
              [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]  # Em
             ], dtype=np.int32)


# In[22]:


result = A.transitive_closure()
print(f"A_star:\n{result.W} \ndiameter: {result.diameter}")


# ### Vertex Relational Composition
# For illustrative purposes, let's remove John from the rb.array representation of the graph

# In[23]:


#                D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em
A1 = rb.array([[-1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # D
               [ 0, -1,  0,  0,  0,  2,  3,  0,  0,  0,  0,  0,  0,  0], # E
               [ 0,  0,  1,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0], # R
               [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # M
               [ 0,  2,  0,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0], # H
               [ 0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  2,  3,  0,  0], # Mi
               [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0], # A
               [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0], # I
               [ 0,  0,  0,  0,  0,  0,  0,  0, -1,  3,  0,  0,  0,  0], # Do
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0], # Ev
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0], # G
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0], # Ma
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0], # S
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]  # Em
              ], dtype=np.int32)
result = A1.transitive_closure()
print(f"A1_star:\n{result.W} \ndiameter: {result.diameter}")
A1_star = result.W


# **Observation**: I am no longer related to Samuel nor Emeline, but that the diameter is still 3 (my relationship to George and Mariea).
# 
# Let's look at the row (u) and column (v) vectors t

# In[24]:


#               D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em
u = rb.array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  3]], dtype=np.int32) 
v = rb.array([[ 0],  
              [ 0],
              [ 2],
              [ 0],
              [ 0],
              [ 0],
              [ 0],
              [ 0],
              [ 2],
              [ 0],
              [ 0],
              [ 0],
              [ 0],
              [ 0],
             ], dtype=np.int32) 

u_lambda = u @ A1_star
v_lambda = A1_star @ v
print(f"u_lambda:\n{u_lambda}")
print(f"v_lambda:\n{v_lambda}")

A_lambda = A1_star.vertex_relational_composition(u, v, -1)
print(f"A_lambda:\n{A_lambda}")


# ### Edge Transitive Closure
# Using the above example, remove the relationship from Regina to John

# In[25]:


#                 D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em   J
R1 = rb.array([[ -1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # D
               [  0, -1,  0,  0,  0,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0],  # E
               [  0,  0,  1,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0],  # R
               [  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # M
               [  0,  2,  0,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # H
               [  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  2,  3,  0,  0,  0],  # Mi
               [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],  # A
               [  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],  # I
               [  0,  0,  0,  0,  0,  0,  0,  0, -1,  3,  0,  0,  0,  0,  2],  # Do
               [  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],  # Ev
               [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0],  # G
               [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],  # Ma
               [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],  # S
               [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0],  # Em
               [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  3, -1]   # J
      ])
R = R1.transitive_closure().W
# Missing edge is R -> J, 2
A_lambda = R.edge_relational_composition(2, 14, 2)
A_lambda


# ### Timings

# In[26]:


get_ipython().run_cell_magic('timeit', '', 'result = A.transitive_closure()\nA1_star = result.W')


# In[27]:


get_ipython().run_cell_magic('timeit', '', 'A_lambda = A1_star.vertex_relational_composition(u, v, -1)')


# In[28]:


get_ipython().run_cell_magic('timeit', '', 'A_lambda = R.edge_relational_composition(2, 14, 2)')


# ## Miscellaneous Linear Algebra

# In[29]:


from numpy.linalg import det
det(A_lambda)


# In[30]:


A_lambda.cardinality()

