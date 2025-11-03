def MSB(x: int) -> int:
    '''
    Given an integer, return the bit position of it's most significant bit
    :param x: operand
    :return: bit position of leftmost significant bit
    '''
    bit_position = 0
    while (x > 1):
        x >>= 1
        bit_position += 1
    return bit_position

def avos_sum(x: int, y: int) -> int:
    '''
    The avos sum is the non-zero minumum of x and y
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

def avos_product(x: int, y: int) -> int:
    '''
    The avos product operation in the red-black semiring.
    
    Parity identity constraints:
    - RED_ONE (-1) is identity for even values, annihilator for odd values
    - BLACK_ONE (1) is identity for odd values, annihilator for even values
    
    Rules:
    - even ⊗ RED_ONE = even (identity)
    - odd ⊗ RED_ONE = 0 (annihilator)
    - odd ⊗ BLACK_ONE = odd (identity)
    - even ⊗ BLACK_ONE = 0 (annihilator)
    
    :param x: operand 1
    :param y: operand 2
    :return: avos product
    '''

    # negative values are invalid (aside from -1)
    if x < -1 or y < -1:
        raise ValueError(f"Invalid input. Negative values (aside from -1) are not allowed. x: {x}, y:{y}")
    
    # The zero property of the avos product
    if x == 0 or y == 0:
        return 0
    
    # Identity ⊗ Identity special cases (must come before other checks)
    # Same-gender self-loops
    if x == -1 and y == -1:
        return -1  # RED_ONE ⊗ RED_ONE = RED_ONE (male self-loop)
    if x == 1 and y == 1:
        return 1   # BLACK_ONE ⊗ BLACK_ONE = BLACK_ONE (female self-loop)
    
    # Cross-gender identity cases (RED_ONE is even/male, BLACK_ONE is odd/female)
    if x == -1 and y == 1:
        return 0  # RED_ONE ⊗ BLACK_ONE: male's female self is undefined
    if x == 1 and y == -1:
        return 0  # BLACK_ONE ⊗ RED_ONE: female's male self is undefined
    
    # Identity on LEFT (lhs): Just a starting point marker, treat as 1 for bit-shift
    # Example: RED_ONE ⊗ 3 = "male-me's mother" = 3
    if x == -1:
        x = 1  # Treat RED_ONE as 1 for composition
    
    # Identity on RIGHT (rhs): Acts as gender/parity filter
    # Example: 3 ⊗ RED_ONE = "mother's male self" = 0 (undefined, mother is female/odd)
    #          3 ⊗ BLACK_ONE = "mother's female self" = 3 (mother is female/odd)
    
    # When y is RED_ONE (-1): Filters for even values only
    if y == -1:
        if x & 1:  # x is odd
            return 0  # Odd values have no male self
        else:
            return x  # Even values' male self is themselves
    
    # When y is BLACK_ONE (1): Filters for odd values only
    if y == 1:
        if x & 1:  # x is odd
            return x  # Odd values' female self is themselves
        else:
            return 0  # Even values have no female self

    # Normal case: bit shifting operation
    bit_position = MSB(y)
    return ((y & (2 ** bit_position - 1)) | (x << bit_position))
