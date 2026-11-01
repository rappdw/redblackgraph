# Mathematical Foundations (draft)

> **Status:** Draft rewrite of the "Formal Definition", "Red Black Arithmetic", "Parity Identity
> Constraints" and "Mathematical Foundations: Algebraic Structure" sections of
> *Red Black Graph – A DAG of Multiple, Interleaved Binary Trees*. It also supplies the proofs
> left as TODOs in "Observation – Simple/Complete Relationship" and "Adding an Edge to $R^+$".
>
> **Main change:** The previous text concluded that the avos algebra is *not* a semiring. That
> conclusion came from storing a vertex's color in the diagonal of the matrix (the values
> $-1$/$1$), not from the algebra itself. Once color is kept separately, the avos sum and product
> form a semiring: the **shortlex path semiring** over the free monoid on two letters. Standard
> semiring results then apply. The legacy $\pm 1$ diagonal is kept as a display convention and
> is shown to give identical results (§6).
>
> Every theorem below was also checked numerically against the reference implementation
> on random graphs.

---

## 1. Red Black Graphs

Let $C = \{\mathsf r, \mathsf b\}$ be the two colors (red, black). We identify them with the bits
$\mathsf r = 0$, $\mathsf b = 1$.

**Definition 1 (Red Black Graph).** A *Red Black Graph* is a finite directed graph $G = (V, E)$
together with a coloring $c : V \to C$ such that

1. **(Acyclic)** $G$ has no directed cycles. In particular $E$ has no self-loops.
2. **(One edge per color)** For every $u \in V$ and every color $k \in C$, there is at most one
   edge $(u, v) \in E$ with $c(v) = k$.

In the genealogical model, vertices are individuals, $c(v) = \mathsf r$ for males and
$\mathsf b$ for females, and an edge $u \to v$ means "$v$ is a parent of $u$". Constraint 2 says
each person has at most one father and at most one mother. Constraint 1 says nobody is their own
ancestor.

**Definition 2 (Path word).** For a directed path $p = (v_0 \to v_1 \to \dots \to v_n)$, its
*word* is the sequence of colors of the vertices it enters:
$$w(p) = c(v_1)\,c(v_2)\cdots c(v_n) \in C^*.$$
The empty path at $v_0$ has the empty word $\varepsilon$, and $|w(p)| = n$ is the path length.

**Lemma 1 (Determinism).** For every $u \in V$ and every word $w \in C^*$, there is at most one
path that starts at $u$ and has word $w$.

*Proof.* Induction on $|w|$. The empty word gives only the empty path. If $w = w'k$, the path
for $w'$ (if any) ends at a unique vertex $x$, and by constraint 2 $x$ has at most one edge to a
vertex of color $k$. $\square$

So a Red Black Graph is a partial deterministic automaton over the alphabet $C$. Every state
can be a start state, and reading the letter $k$ always leads to a state of color $k$. We
write $\delta(u, w)$ for the endpoint of the path from $u$ with word $w$, when it exists.

> **Note on "the ancestry is a binary tree".** Because of Lemma 1, the *unfolding* of the
> ancestry of $u$ (one node per word $w$ with $\delta(u,w)$ defined) is a binary tree. The
> ancestry *subgraph* itself is a tree only when no ancestor is reachable by two different
> words. Pedigree collapse (implex), e.g. descent from a common ancestor through both parents,
> breaks this, and the subgraph is then a DAG. The definitions below handle both cases.

---

## 2. Encoding paths as integers (Ahnentafel numbering)

**Definition 3 (Encoding).** Define $\iota : C^* \to \mathbb Z_{\ge 1}$ by
$$\iota(\varepsilon) = 1, \qquad \iota(wk) = 2\,\iota(w) + k.$$
Equivalently, $\iota(w)$ is the binary numeral "$1$" followed by the bits of $w$.

$\iota$ is a bijection onto the positive integers. Its inverse writes $x$ in binary and drops
the leading $1$. This is the classical **Ahnentafel** (Eytzinger / Sosa–Stradonitz) numbering
used in genealogical pedigree charts: self $= 1$, father $= 2$, mother $= 3$, paternal
grandfather $= 4$, …, and in general the father of $n$ is $2n$ and the mother is $2n+1$.

Write $\operatorname{MSB}(x) = \lfloor \log_2 x \rfloor$ for $x \ge 1$. Then
$$|\iota^{-1}(x)| = \operatorname{MSB}(x), \qquad \text{last letter of } \iota^{-1}(x) = x \bmod 2 \quad (x \ge 2).$$

**Definition 4 (Shortlex order).** For words $w, w'$, write $w <_{\mathrm{sl}} w'$ if $|w| < |w'|$,
or $|w| = |w'|$ and $w$ comes before $w'$ in lexicographic order with $\mathsf r < \mathsf b$.

**Lemma 2 (The integer order is shortlex).** $\iota(w) < \iota(w') \iff w <_{\mathrm{sl}} w'$.

*Proof.* Words of length $n$ map to the interval $[2^n, 2^{n+1})$, so shorter words map to
smaller integers. Within one interval, the leading $1$ is shared and the integers compare by their
remaining $n$ bits, which is lexicographic order on $w$. $\square$

**Lemma 3 (Shortlex is compatible with concatenation).** If $y \le_{\mathrm{sl}} z$ then
$xy \le_{\mathrm{sl}} xz$ and $yx \le_{\mathrm{sl}} zx$ for every word $x$.

*Proof.* If $|y| < |z|$, both conclusions hold by length. If $|y| = |z|$:
$xy$ and $xz$ share the prefix $x$, so they compare as $y$ and $z$ do. $yx$ and $zx$ have equal
length, and their first difference falls inside the first $|y|$ letters, so they also compare as
$y$ and $z$ do. $\square$

---

## 3. The avos semiring

Let $S = \mathbb Z_{\ge 0}$. The value $0$ means "no path", and every positive integer encodes a
word through $\iota$.

**Definition 5 (Avos sum and product).**
$$
x \oplus y = \begin{cases} y & x = 0 \\ x & y = 0 \\ \min(x, y) & \text{otherwise} \end{cases}
\qquad
x \otimes y = \begin{cases} 0 & x = 0 \text{ or } y = 0 \\
\iota\big(\iota^{-1}(x)\,\iota^{-1}(y)\big) & \text{otherwise.} \end{cases}
$$
Concretely, with $m = \operatorname{MSB}(y)$:
$$x \otimes y = (x \ll m) \;\big|\; (y - 2^m).$$
That is, the leading $1$ of $y$ is replaced by all the bits of $x$. This is the bit operation
already used in the implementation.

**Theorem 1 (Avos semiring).** $(S, \oplus, \otimes, 0, 1)$ is a semiring. Moreover:

- $\oplus$ is commutative, idempotent and *selective* ($x \oplus y \in \{x, y\}$);
- $1 \oplus x = 1$ for all $x$. In the terminology of Gondran & Minoux and Mohri the semiring is
  *bounded*, or *0-closed*.

*Proof.*

- **$(S, \oplus, 0)$ is a commutative monoid.** $\oplus$ is the minimum for the total order on $S$
  in which $0$ is placed above every positive integer. So $\oplus$ is associative,
  commutative, idempotent and selective, and $0$ is its identity.
- **$(S, \otimes, 1)$ is a monoid with absorbing $0$.** On $S \setminus \{0\}$, $\otimes$ is
  concatenation of words transported through the bijection $\iota$. So it is associative, and
  $1 = \iota(\varepsilon)$ is a two-sided identity. Adjoining $0$ as an absorbing element
  keeps both properties.
- **Distributivity.** For nonzero $x, y, z$, Lemmas 2 and 3 give
  $x \otimes \min(y,z) = \min(x\otimes y, x \otimes z)$ and
  $\min(y,z) \otimes x = \min(y \otimes x, z \otimes x)$. If any of the operands is $0$, the
  identity follows directly from $0$ being the $\oplus$-identity and the $\otimes$-absorber.
- **Boundedness.** $1$ is the smallest positive integer. $\square$

This is the semiring of shortlex-minimal words over $\{\mathsf r, \mathsf b\}$, a special case of
the lexicographic semirings used for weighted automata (Roark, Sproat & Shafran, 2011). It is not
commutative under $\otimes$ ($2 \otimes 3 = 5$, $3 \otimes 2 = 6$), which is expected since path
composition depends on order.

> **Where the old non-associativity came from.** The counterexample
> $(2 \otimes \mathrm{BLACK\_ONE}) \otimes 3 = 0 \ne 5 = 2 \otimes (\mathrm{BLACK\_ONE} \otimes 3)$
> uses $1$ as a *color filter* ("father's female self"). In Theorem 1, $1$ is the empty path,
> and $2 \otimes 1 = 2$. Color is not stored in the value. It is a property of the vertex,
> and Theorem 3 shows the value already records it.

---

## 4. Matrices, closure and the relationship function

Number the vertices $1, \dots, n$. Matrices over $S$ use the induced operations
$(XY)_{uv} = \bigoplus_w X_{uw} \otimes Y_{wv}$ and $(X \oplus Y)_{uv} = X_{uv} \oplus Y_{uv}$.
The identity matrix $I$ has $1$ on the diagonal and $0$ elsewhere.

**Definition 6 (Adjacency matrix).** $A_{uv} = \iota(c(v)) = 2 + c(v)$ if $(u, v) \in E$, and $0$
otherwise. In words, $2$ marks an edge to a red parent and $3$ an edge to a black parent.

**Definition 7 (Relationship).**
$$r(u, v) = \begin{cases} \iota\big(\min_{\mathrm{sl}} \{\, w(p) : p \text{ a path } u \to v \,\}\big) & \text{if a path exists} \\ 0 & \text{otherwise.} \end{cases}$$
So $r(u,u) = 1$. For $u \ne v$, $r(u, v)$ encodes the *shortest* path from $u$ to $v$, with ties
broken lexicographically ($\mathsf r$ before $\mathsf b$, i.e. paternal line first). This replaces
the old rule 5, which did not specify tie-breaking and whose step $x \leftarrow x \ll 0$ for red
vertices left $x$ unchanged.

**Theorem 2 (Powers and closure).**

(a) $(A^k)_{uv}$ is $\iota$ of the shortlex-minimal word over paths $u \to v$ of length exactly
$k$, or $0$ if there are none.

(b) Let $R^* = I \oplus A \oplus A^2 \oplus \cdots \oplus A^{n-1}$. Then $R^*_{uv} = r(u, v)$ for
all $u, v$.

(c) $R^*$ satisfies $R^* = I \oplus A R^* = I \oplus R^* A$, and it is the least solution of
$X = I \oplus AX$.

(d) $R^* = (I \oplus A)^{m}$ for every $m \ge n - 1$.

*Proof.*

- **(a)** Induction on $k$. We have
  $(A^k)_{uv} = \bigoplus_x (A^{k-1})_{ux} \otimes A_{xv}$. Every length-$k$ path factors uniquely
  as a length-$(k-1)$ path to some $x$ followed by the edge $x \to v$. By Lemma 3, the
  shortlex-minimum over paths through a fixed $x$ is the minimal prefix word followed by
  $c(v)$. Taking $\oplus$ over $x$ gives the overall minimum.
- **(b)** A DAG on $n$ vertices has no path longer than $n-1$, and $\oplus$ takes the minimum over
  all lengths.
- **(c)** Expand the right-hand side using (b). Leastness is the standard argument for the
  algebraic path problem (Lehmann 1977; Gondran & Minoux 2008, ch. 4).
- **(d)** In any semiring with idempotent $\oplus$,
  $(I \oplus A)^m = \bigoplus_{j=0}^{m} A^j$. Every binomial coefficient collapses to one
  term. $A^j = 0$ for $j \ge n$ in a DAG. $\square$

> **Graphs with cycles.** Boundedness ($1 \oplus x = 1$) makes the closure exist for *any*
> finite graph. Removing a cycle from a path gives a strictly shorter, hence shortlex-smaller,
> word, so the minimum is always reached by a simple path, and (b) holds unchanged. Cycles are
> detected by the strict closure $A^+ = A R^*$. The graph is acyclic iff
> $\operatorname{diag}(A^+) = 0$, and otherwise $A^+_{uu}$ encodes the shortest cycle through $u$.

**Consequences for the algorithms in this repository.**

| Algorithm | Justification |
|---|---|
| Floyd–Warshall with $W_{ij} \leftarrow W_{ij} \oplus W_{ik} \otimes W_{kj}$ | Correct in any semiring in which the closure exists (Lehmann 1977). Theorem 1 provides one. |
| Repeated squaring, $\lceil \log_2 n \rceil$ squarings (CPU and GPU) | Theorem 2(d). |
| DAG closure in reverse topological order | Theorem 2(c) row by row: $R^*_{u,:} = e_u \oplus \bigoplus_{(u,v)\in E} A_{uv} \otimes R^*_{v,:}$. Rows of all parents are final before row $u$ is computed. |
| "Simple/complete relationship" (old TODO) | The row form of 2(c): $\vec u_s \otimes R^* = \vec u_c$ with $\vec u_s = e_u \oplus A_{u,:}$. The column version is $R^* = I \oplus R^* A$. |

---

## 5. Structure of the closure

**Theorem 3 (Row and column structure).** Let $u \ne v$ with $r(u, v) \ne 0$.

1. **(Color)** $r(u, v) \bmod 2 = c(v)$. The parity of a relationship is the color of the target.
2. **(Generation)** $\operatorname{MSB}(r(u, v))$ is the length of the shortest path from $u$ to $v$.
3. **(Path)** Reading the bits of $r(u, v)$ after the leading $1$ gives the colors along the
   chosen path. By Lemma 1 this determines the path's vertices.
4. **(Injective rows)** If $r(u, v) = r(u, v') \ne 0$ then $v = v'$. Each nonzero value appears
   at most once in a row.
5. **(Triangle inequality)** $r(u, w) \le r(u, v) \otimes r(v, w)$ whenever the right side is
   nonzero, with equality iff the chosen path from $u$ to $w$ passes through $v$.

*Proof.*

1. The last letter of the word is $c(v)$, and the last bit of $\iota(w)$ is the last letter of $w$.
2. Follows from Lemma 2: shortlex compares by length first.
3. Follows from Definition 3.
4. Both values encode the same word $w$, and $v = \delta(u, w) = v'$ by Lemma 1.
5. The concatenation of the two chosen paths is a path from $u$ to $w$, and $r(u,w)$ is the
   minimum over all such paths. $\square$

Consequences:

- The **diameter** (longest shortest path) is $\operatorname{MSB}\big(\max_{u,v} R^*_{uv}\big)$.
- Column $v$ holds every descendant's relationship to $v$, so by part 1 all of its nonzero
  off-diagonal entries have parity $c(v)$. Rows, by contrast, mix both parities.

---

## 6. Relation to the legacy $\pm1$ encoding

The original presentation and the current code store a matrix
$$R = A + \operatorname{diag}(\sigma), \qquad \sigma_v = \begin{cases} -1 & c(v) = \mathsf r \\ \phantom{-}1 & c(v) = \mathsf b \end{cases}$$
and use a product with extra rules for $\pm 1$:

| Legacy name | Legacy rule | Role in this text |
|---|---|---|
| RED\_ONE $=-1$, BLACK\_ONE $=1$ | diagonal entries | the identity $1$ on the diagonal, plus the color vector $\sigma$ |
| $\sigma_u \otimes x = x$ | left identity | $1 \otimes x = x$ |
| $x \otimes \sigma_v = x$ if $x \bmod 2 = c(v)$, else $0$ | right "parity filter" | $x \otimes 1 = x$ |
| $\sigma \otimes \sigma' = 0$ for different colors | cross-color rule | never reached on valid input |

**Proposition 4 (Equivalence).** For every Red Black Graph, the legacy closure $R^+$ and $R^*$
agree off the diagonal, and $\operatorname{diag}(R^+) = \sigma$.

*Proof.* In the legacy products, the special values appear only as diagonal entries. The left
rule is the identity. The right rule is applied only to entries $R_{uv}$ with $u\ne v$ and to
$\sigma_v$, and by Theorem 3.1 every nonzero entry ending at $v$ already has parity $c(v)$. So
the filter never changes a value, and the cross-color rule needs two different colors on the same
diagonal entry, which cannot happen. $\square$

In tests on random valid graphs, the parity filter and the cross-color rule changed no result in
over 345,000 products. The $\pm 1$ diagonal is therefore a display convention: it lets one matrix
also show each vertex's color. The color identities

$$\operatorname{tr}(R) = |V_{\mathsf b}| - |V_{\mathsf r}|, \qquad |V_{\mathsf b}| = \tfrac{1}{2}\big(|V| + \operatorname{tr} R\big), \qquad |V_{\mathsf r}| = \tfrac{1}{2}\big(|V| - \operatorname{tr} R\big)$$

are statements about $\sigma$ in ordinary integer arithmetic. They are not avos operations.

> **Signed versus unsigned storage.** With signed integers, $\min(-1, x) = -1$. With unsigned
> storage (RED\_ONE $= 2^w - 1$), $\min(-1, x) = x$. The two disagree only when an off-diagonal
> value is combined into a diagonal entry, which happens only on a cycle. Cycles are rejected
> explicitly before that point, so both backends agree on valid input. With the encoding in this
> text, the question does not arise.

---

## 7. Incremental updates and cycle prevention

Let $R^*$ be the closure of a Red Black Graph.

**Theorem 5 (Adding an edge).** Let $\alpha, \beta \in V$ with $\alpha \ne \beta$, where $\alpha$
has no edge to a vertex of color $c(\beta)$. Let $a = 2 + c(\beta)$. Adding the edge
$\alpha \to \beta$ keeps the graph acyclic iff $R^*_{\beta\alpha} = 0$. In that case the new
closure is
$$R'^{*}_{uv} = R^*_{uv} \;\oplus\; R^*_{u\alpha} \otimes a \otimes R^*_{\beta v}.$$
In matrix form, $R'^* = R^* \oplus (R^* e_\alpha) \otimes a \otimes (e_\beta^{\top} R^*)$, a
rank-one update.

*Proof.* A cycle through the new edge needs a path $\beta \to \alpha$. In the new acyclic graph
every path uses the new edge at most once, so a new path factors as
$(u \to \alpha)(\alpha \to \beta)(\beta \to v)$. Its minimum over these factorizations is the
second term, by distributivity and Lemma 3. Old paths contribute $R^*_{uv}$. $\square$

**Theorem 6 (Adding a vertex).** Add a vertex $\lambda$ of color $k$, with

- a row vector $a$ for its parents: $a_v = 2 + c(v)$ for each parent $v$, at most one per color;
- a column vector $b$ for its children: $b_u = 2 + k$ for each child $u$, where $u$ has no
  existing parent of color $k$.

Let $\hat u = a R^*$ (the ancestors of $\lambda$) and $\hat v = R^* b$ (the descendants of
$\lambda$). The result is acyclic iff there is no $x$ with $\hat u_x \ne 0$ and $\hat v_x \ne 0$.
In that case
$$R'^{*} = \begin{pmatrix} R^* \oplus \hat v \otimes \hat u & \hat v \\ \hat u & 1 \end{pmatrix},
\qquad (\hat v \otimes \hat u)_{xy} = \hat v_x \otimes \hat u_y .$$

*Proof.* Same argument as Theorem 5. Every new path passes through $\lambda$ exactly once, and
a cycle exists iff some vertex is both an ancestor and a descendant of $\lambda$. $\square$

These are exactly the "vertex relational composition" and "edge relational composition" of the
original text, now with a precise statement and proof. Both run in $O(n^2)$ on a dense closure.
On a sparse closure they run in time proportional to the number of affected entries.

---

## 8. Relationship calculation

For vertices $u, v$, a **common ancestor** is any $x$ with $R^*_{ux} \ne 0$ and $R^*_{vx} \ne 0$
(including $x = u$ or $x = v$). Its generational distances are
$d_u = \operatorname{MSB}(R^*_{ux})$ and $d_v = \operatorname{MSB}(R^*_{vx})$.

A **nearest common ancestor** minimizes $d_u + d_v$. Ties can be broken by
$R^*_{ux} \oplus R^*_{vx}$ or by vertex id. The pair $(d_u, d_v)$ then gives the degree of
kinship (parent, sibling, $n$-th cousin $m$ times removed) from the usual table of consanguinity.

> The original text minimized the raw sum $R^*_{ux} + R^*_{vx}$. Since $R^*_{ux} \in [2^{d_u}, 2^{d_u+1})$
> this is not monotone in $d_u + d_v$. For example, an ancestor at distances $(3,3)$ with values
> $(8, 8)$ beats one at $(1,4)$ with values $(2,16)$. This can happen with pedigree collapse.

**Limits of the closure.** $R^*$ keeps one path per pair, the shortlex-minimal one. It cannot:

- tell full from half relationships, since full siblings have two nearest common ancestors and
  half siblings have one. Counting all nearest common ancestors at the minimal distance fixes
  this.
- compute Wright's coefficient of relationship, which sums over *all* pairs of paths. That needs
  a different semiring, e.g. $(\mathbb R_{\ge 0}, +, \times)$ with edge weight $\tfrac12$, or the
  product of the avos semiring with a counting semiring.

---

## 9. Finite word size

A value of path length $d$ needs $d + 1$ bits. A $w$-bit unsigned integer therefore holds
ancestry up to $w - 1$ generations: 63 for `uint64`, and 62 for `int64` once the sign bit and
the reserved $-1$ are taken into account. Products that exceed this must raise an overflow error,
as the current implementation does. They must not wrap.

---

## 10. Other parts of the original text

- **"Canonical form".** The ordering is not *canonical*: ties are broken by component id and
  vertex id, which depend on the original labeling, so isomorphic graphs can produce different
  matrices. True canonical labeling of DAGs is as hard as graph isomorphism. Suggested name:
  *standard form* or *component-sorted topological form*.

  The ordering is also not guaranteed to be upper triangular. Neither the original key (largest
  relationship value in the row) nor the current one (total ancestor generations,
  $\sum_x \operatorname{MSB}(R^*_{ux})$) always increases from parent to child. Both measure
  *shortest* paths, and under pedigree collapse a child can reach its parent's ancestors by
  shorter routes than the parent does. For example, if $u$'s parents are $v$ and $v$'s
  great-grandparent, $u$ can have both a smaller largest value and a smaller total than $v$. A
  key that always respects the edges is the *height* $h(u)$, the length of the longest path from
  $u$ to an ancestor with no parents: $h(u) = 1 + \max_{\text{parents } p} h(p)$, which is
  strictly larger than the height of each parent. The closure keeps only shortest paths, so $h$
  has to be computed from $A$ (one pass in topological order), not read from $R^*$.
- **Connected components.** These are weakly connected components, found by union-find or BFS.
  Tarjan's algorithm is for strongly connected components, and a DAG's are all singletons. The
  current implementation is $O(\operatorname{nnz}(R^*) + n \cdot \#\text{merges})$, not
  $O(|V| + |E|)$.
- **Determinants (Appendix A).** In ordinary integer arithmetic, $R$ is similar to a triangular
  matrix with diagonal $\sigma$, so $\det R = (-1)^{|V_{\mathsf r}|}$ for every $n$. The
  $2\times2$ and $3\times3$ case analysis is unnecessary. The avos semiring has no subtraction,
  so the determinant is not an avos notion. Suggest removing the appendix or reducing it to
  one sentence.
- **Fast matrix multiplication.** Strassen-style algorithms need additive inverses, so they apply
  to rings but not to the avos semiring, whose sum is a minimum and has no subtraction. The same is
  true of tropical (min, +) matrix products. Fast algorithms for distance products, such as
  Zwick's bridging-set method for all-pairs shortest paths, rely on small integer edge weights
  and min-plus specifics, and do not carry over directly. In practice the sparse and DAG
  algorithms are the effective route, since genealogical graphs are very sparse.
- **Graded and category-theoretic readings.** These can be dropped. Parity is not a
  $\mathbb Z/2$ grading, because $\operatorname{parity}(x \otimes y) = \operatorname{parity}(y)$
  rather than the sum of the parities. The natural categorical statement is simpler: $r$ is the
  shortlex-minimal image of the free category on $G$ under the functor that labels each edge with
  its target's color.

---

## 11. Implementation status

`tests/avos/test_algebra_properties.py` checks each result above against every backend, using
brute-force enumeration of all paths as ground truth. All closure backends, the dense and sparse
matrix products, and the reference, core and sparse vertex and edge compositions agree with
Theorems 2, 3, 5 and 6. The tests also record the following gaps as strict expected failures:

| Gap | Affected | Result it violates |
|---|---|---|
| Reported diameter can be too large: it is the largest value seen while the algorithm runs, including candidates later replaced by a shorter path | reference, core, sparse Floyd–Warshall, sparse DAG, adaptive, component-wise (Dijkstra and squaring are correct) | Theorem 3.2 |
| Edge composition accepts an edge that creates a cycle | reference, sparse (core checks) | Theorem 5 |
| Closure by repeated squaring does not detect cycles; the $\pm1$ diagonal hides them | sparse squaring | note after Theorem 2 |
| Relationship calculation minimizes the raw sum of values, not generations | `calculate_relationship` | §8 |
| "Canonical" ordering is not always upper triangular under pedigree collapse | reference, sparse | §10 |

---

## References

- Carré, B. A. (1971). An algebra for network routing problems. *IMA J. Appl. Math.* 7(3).
- Lehmann, D. J. (1977). Algebraic structures for transitive closure. *Theoretical Computer Science* 4(1).
- Gondran, M., & Minoux, M. (2008). *Graphs, Dioids and Semirings*. Springer.
- Mohri, M. (2002). Semiring frameworks and algorithms for shortest-distance problems. *J. Automata, Languages and Combinatorics* 7(3).
- Roark, B., Sproat, R., & Shafran, I. (2011). Lexicographic semirings for exact automata encoding of sequence models. *ACL*.
- Ahnentafel numbering (Eytzinger 1590; Sosa 1676; Kekulé von Stradonitz 1898).
- Wright, S. (1922). Coefficients of inbreeding and relationship. *American Naturalist* 56.
