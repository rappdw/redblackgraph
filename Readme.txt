A RedBlack Graph is a DAG such that any node is colored either red or black with the following constraint: any node may
have at most 1 outbound arc to a given colored node.

Properties of the graph:
 - The "view" of the graph outbound from any given node is a binary tree
 - Use the following numbering scheme on this binary tree (base 2):
   - root node is numbered 0 if red, 1 if black, root of a parent node is 10 if red, 11 if black, etc.
   - any node in the tree can be uniquely identified such that:
    < fill in here >>
 - Identification of direct ancestry is O(1) operation
 - Identification of relationship is O(Log2(n)) operation

