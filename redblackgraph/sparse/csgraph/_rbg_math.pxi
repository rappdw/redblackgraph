cdef inline unsigned short MSB(DTYPE_t x):
    '''
    Given an integer, return the bit position of it's most significant bit
    :param x: operand
    :return: bit position of leftmost significant bit
    '''
    cdef unsigned short bit_position = 0
    while (x > 1):
        x >>= 1
        bit_position += 1
    return bit_position

cdef inline bint avos_lt(DTYPE_t x, DTYPE_t y):
    if y == 0:
        return x != 0
    return x < y

cdef inline DTYPE_t avos_sum(DTYPE_t x, DTYPE_t y):
    '''
    The avos sum is the non-zero minimum of x and y
    :param x: operand 1
    :param y: operand 2
    :return: avos sum
    '''
    if x == 0:
        return y
    if y == 0:
        return x
    if x < y:
        return x
    return y

cdef inline DTYPE_t avos_product(DTYPE_t lhs, DTYPE_t rhs):
    '''
    The avos product with parity identity constraints.
    
    Identity semantics (asymmetric):
    - LEFT identity: starting point marker, no filtering
    - RIGHT identity: gender/parity filter
      * RED_ONE (-1) filters for even (odd → 0)
      * BLACK_ONE (1) filters for odd (even → 0)
    
    :param lhs: left operand
    :param rhs: right operand
    :return: avos product
    '''
    cdef UDTYPE_t x = <UDTYPE_t>lhs
    cdef UDTYPE_t y = <UDTYPE_t>rhs

    # The zero property of the avos product
    if x == 0 or y == 0:
        return 0
    
    # Identity ⊗ Identity special cases (must come before other checks)
    # Same-gender self-loops
    if lhs == -1 and rhs == -1:
        return -1  # RED_ONE ⊗ RED_ONE = RED_ONE (male self-loop)
    if lhs == 1 and rhs == 1:
        return 1   # BLACK_ONE ⊗ BLACK_ONE = BLACK_ONE (female self-loop)
    
    # Cross-gender identity cases (RED_ONE is even/male, BLACK_ONE is odd/female)
    if lhs == -1 and rhs == 1:
        return 0  # RED_ONE ⊗ BLACK_ONE: male's female self is undefined
    if lhs == 1 and rhs == -1:
        return 0  # BLACK_ONE ⊗ RED_ONE: female's male self is undefined
    
    # Identity on LEFT: just a starting point marker, treat as 1 for bit-shift
    if lhs == -1:
        x = 1  # Treat RED_ONE as 1 for composition
    
    # Identity on RIGHT: acts as gender/parity filter
    # When rhs is RED_ONE (-1): filters for even values only
    if rhs == -1:
        if lhs & 1:  # lhs is odd (use original lhs, not converted x)
            return 0  # Odd values have no male self
        else:
            return x  # Even values' male self is themselves
    
    # When rhs is BLACK_ONE (1): filters for odd values only
    if rhs == 1:
        if lhs & 1:  # lhs is odd
            return x  # Odd values' female self is themselves
        else:
            return 0  # Even values have no female self

    cdef unsigned short bit_position = MSB(y)
    return ((y & (2 ** bit_position - 1)) | (x << bit_position))
