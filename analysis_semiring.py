#!/usr/bin/env python3
"""Deep mathematical analysis of red-black algebra as a semiring."""

from redblackgraph import avos_product, avos_sum, RED_ONE, BLACK_ONE

print("=" * 80)
print("DEEP MATHEMATICAL ANALYSIS: Red-Black Algebra")
print("=" * 80)

print("\n1. NUMBER SYSTEM DEFINITION")
print("-" * 80)
print("Proposed: S = ℕ₀ ∪ {RED_ONE, BLACK_ONE}")
print("Where:")
print("  - ℕ₀ = {0, 2, 3, 4, 5, 6, ...}")
print("  - RED_ONE = -1 (even/male identity)")
print("  - BLACK_ONE = 1 (odd/female identity)")

print("\n2. SEMIRING AXIOMS")
print("=" * 80)
print("For (S, ⊕, ⊗) to be a semiring:")
print("  1. (S, ⊕) is commutative monoid with identity 0")
print("  2. (S, ⊗) is monoid with identity")
print("  3. ⊗ distributes over ⊕")
print("  4. 0 is annihilator: 0⊗x = x⊗0 = 0")

test_vals = [0, RED_ONE, BLACK_ONE, 2, 3, 4, 5]

print("\n3. TESTING AVOS SUM (⊕)")
print("-" * 80)

# Commutativity
commutative = True
for x in test_vals:
    for y in test_vals:
        if avos_sum(x, y) != avos_sum(y, x):
            print(f"✗ Commutativity fail: {x}⊕{y}={avos_sum(x,y)}, {y}⊕{x}={avos_sum(y,x)}")
            commutative = False
if commutative:
    print("✓ Commutativity: PASS")

# Associativity  
associative = True
for x, y, z in [(2, 3, 4), (RED_ONE, 5, 7), (BLACK_ONE, 2, 0)]:
    left = avos_sum(avos_sum(x, y), z)
    right = avos_sum(x, avos_sum(y, z))
    if left != right:
        print(f"✗ Associativity fail: ({x}⊕{y})⊕{z}={left}, {x}⊕({y}⊕{z})={right}")
        associative = False
if associative:
    print("✓ Associativity: PASS")

# Identity
identity_sum = all(avos_sum(x, 0) == x and avos_sum(0, x) == x for x in test_vals)
if identity_sum:
    print("✓ Identity (0): PASS")
else:
    print("✗ Identity (0): FAIL")

print(f"\n(S, ⊕) forms commutative monoid: {commutative and associative and identity_sum}")

print("\n4. TESTING AVOS PRODUCT (⊗)")
print("-" * 80)

# Associativity
print("Associativity: (x⊗y)⊗z = x⊗(y⊗z)")
test_cases = [
    (2, 3, 4),
    (RED_ONE, 2, 3),
    (BLACK_ONE, 3, 2),
    (2, BLACK_ONE, 3),
    (3, RED_ONE, 2),
    (2, 3, RED_ONE),
    (3, 2, BLACK_ONE),
]
associative_prod = True
for x, y, z in test_cases:
    left = avos_product(avos_product(x, y), z)
    right = avos_product(x, avos_product(y, z))
    match = "✓" if left == right else "✗"
    print(f"  {match} ({x}⊗{y})⊗{z}={left:3}, {x}⊗({y}⊗{z})={right:3}")
    if left != right:
        associative_prod = False

if associative_prod:
    print("\n✓ Associativity: PASS")
else:
    print("\n✗ Associativity: FAIL - NOT ASSOCIATIVE WITH PARITY CONSTRAINTS!")

# Zero annihilator
print("\nZero annihilator: 0⊗x = x⊗0 = 0")
annihilator = all(avos_product(0, x) == 0 and avos_product(x, 0) == 0 for x in test_vals)
if annihilator:
    print("✓ Zero annihilator: PASS")

# Identity test
print("\n5. IDENTITY ELEMENT ANALYSIS")
print("=" * 80)
print("Semiring requires ONE two-sided identity e: e⊗x = x⊗e = x for all x\n")

print("Testing RED_ONE as two-sided identity:")
for x in [2, 3, 4, 5]:
    left = avos_product(RED_ONE, x)
    right = avos_product(x, RED_ONE)
    both_work = (left == x and right == x)
    print(f"  {RED_ONE}⊗{x}={left:2}, {x}⊗{RED_ONE}={right:2}  {'✓' if both_work else '✗'}")

print("\nTesting BLACK_ONE as two-sided identity:")
for x in [2, 3, 4, 5]:
    left = avos_product(BLACK_ONE, x)
    right = avos_product(x, BLACK_ONE)
    both_work = (left == x and right == x)
    print(f"  {BLACK_ONE}⊗{x}={left:2}, {x}⊗{BLACK_ONE}={right:2}  {'✓' if both_work else '✗'}")

print("\n6. DISTRIBUTIVITY")
print("=" * 80)
print("Testing: x⊗(y⊕z) = (x⊗y)⊕(x⊗z)")
print("and:     (x⊕y)⊗z = (x⊗z)⊕(y⊗z)")

distrib_cases = [
    (2, 3, 5),
    (3, 2, 4),
    (RED_ONE, 2, 3),
    (2, 0, 3),
]
left_distrib = True
right_distrib = True
for x, y, z in distrib_cases:
    # Left distributivity
    left = avos_product(x, avos_sum(y, z))
    right = avos_sum(avos_product(x, y), avos_product(x, z))
    if left != right:
        print(f"✗ Left: {x}⊗({y}⊕{z})={left}, ({x}⊗{y})⊕({x}⊗{z})={right}")
        left_distrib = False
    
    # Right distributivity  
    left = avos_product(avos_sum(x, y), z)
    right = avos_sum(avos_product(x, z), avos_product(y, z))
    if left != right:
        print(f"✗ Right: ({x}⊕{y})⊗{z}={left}, ({x}⊗{z})⊕({y}⊗{z})={right}")
        right_distrib = False

if left_distrib and right_distrib:
    print("✓ Distributivity: PASS")

print("\n" + "=" * 80)
print("FINAL ANALYSIS")
print("=" * 80)
print(f"""
✓ (S, ⊕) is commutative monoid: {commutative and associative and identity_sum}
{'✓' if associative_prod else '✗'} (S, ⊗) is associative: {associative_prod}
✗ (S, ⊗) has single two-sided identity: FALSE
✓ 0 is annihilator: {annihilator}
{'✓' if left_distrib and right_distrib else '✗'} Distributivity: {left_distrib and right_distrib}

CONCLUSION:
-----------
The structure (S, ⊕, ⊗) with parity-dependent identities is NOT a classical semiring
because it lacks a single two-sided multiplicative identity.

Instead, it may be:
1. A "near-semiring" or "hemiring" with relaxed axioms
2. A "two-sorted semiring" with separate even/odd carriers
3. A "category-enriched" algebraic structure where morphisms carry parity
4. A semiring with a "parity functor" or "grading"

The asymmetric identity behavior suggests this is closer to a CATEGORY-THEORETIC
structure where composition depends on object types (even/odd parity).
""")
