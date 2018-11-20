[Dynamic Graph Alogrithms](http://cs.ioc.ee/ewscs/2012/italiano/dynamic1.pdf)

Introduced to the notion of matrix distance products in [All Pairs Shortest Paths using Bridging Sets
and Rectangular Matrix Multiplication](https://arxiv.org/pdf/cs/0008011.pdf). Distance product for a rectangular 
matrix ==> $c_ij = min{a_ik + b_kj} for 1 <= i,j <= n$

Introduced to Strassen's method of matrix multiplication in [Introduction to Algorithms](http://mitpress.mit.edu/algorithms)

Looking into Strassen's method, it seems that avos multiplicaiton is likely a semi-ring [Strassen algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm)

Removed reference to unsigned types in sparsetools... Rather than do this, 
red 1 should be defined as binary 11111...1 for word width w and then only operate 
on unsigned types. This makes the min function a bit odd, but works well.