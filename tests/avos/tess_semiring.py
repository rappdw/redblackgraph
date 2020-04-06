from redblackgraph.reference import avos_sum, avos_product

# from "Fun with Semirings"... Formal definition
# Set R with two distinguishing elements, 0, and 1 and two binary operations +, *, satisfying
# the following relations for any a, b, c elements of R
#
#          a + b = b + a         (I)
#    a + (b + c) = (a + b) + c   (II)
#          a + 0 = a             (III)
#    a * (b * c) = (a * b) * c   (IV)
#          a * 0 = 0             (V)
#          a * 1 = a             (VI)
#    a * (b + c) = a * b + a * c (VII)
#    (a + b) * c = a * c + b * c (VIII)
#
# In the case of avos, there are three distingushing elements, 0, 1, -1 (named "red 1"), + is avos_sum
# * is avos_product, 1 and -1 can be used interchangeably, R is the set of -1, 0 and all positive integers
# where 1 * -1 = -1 and -1 * 1 = -1 ("red 1" is more of an identity that "black 1")
#
# following are the test cases that verify this for the number -1, 0, 1, ... 15

r = range(-1, 16)

def test_I():
    #          a + b = b + a         (I)
    for a in r:
        for b in r:
            assert avos_sum(a, b) == avos_sum(b, a)

def test_II():
    #    a + (b + c) = (a + b) + c   (II)
    for a in r:
        for b in r:
            for c in r:
                assert avos_sum(a, avos_sum(b, c) == avos_sum(avos_sum(a, b), c))

def test_III():
    #          a + 0 = a             (III)
    for a in r:
        assert avos_sum(a, 0) == a

def test_IV():
    #    a * (b * c) = (a * b) * c   (IV)
    for a in r:
        for b in r:
            for c in r:
                assert avos_product(a, avos_product(b, c)) == avos_product(avos_product(a, b), c)

def test_V():
    #          a * 0 = 0             (V)
    for a in r:
        assert avos_product(a, 0) == 0

def test_VI():
    #          a * 1 = a             (VI)
    for a in r:
        assert avos_product(a, 1) == a
        if a == 1:
            assert avos_product(a, -1) == -1
        else:
            assert avos_product(a, -1) == a

def test_VII():
    #    a * (b + c) = a * b + a * c (VII)
    for a in r:
        for b in r:
            for c in r:
                assert avos_product(a, avos_sum(b, c)) == avos_sum(avos_product(a, b), avos_product(a, c))

def test_VIII():
    #    (a + b) * c = a * c + b * c (VIII)
    for a in r:
        for b in r:
            for c in r:
                assert avos_product(avos_sum(a, b), c) == avos_sum(avos_product(a, c), avos_product(b, c))

# Closure:
#    a_star = 1 + a * a_star = 1 + a_star * a
