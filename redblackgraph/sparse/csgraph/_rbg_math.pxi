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

cdef inline DTYPE_t avos_product(DTYPE_t lhs, DTYPE_t rhs):
    '''
    The avos product replaces the left most significant bit of operand 2 with operand 1
    :param x: operand 1
    :param y: operand 2
    :return: avos product
    '''
    cdef UDTYPE_t red_one = -1
    cdef UDTYPE_t x = <UDTYPE_t>lhs
    cdef UDTYPE_t y = <UDTYPE_t>rhs

    # The zero property of the avos product
    if x == 0 or y == 0:
        return 0
    # Special case -1 * 1 or -1 * -1
    if x == red_one:
        if y == 1:
            return -1
        x = 1
    if y == red_one:
        if x == 1:
            return -1
        y = 1

    cdef unsigned short bit_position = MSB(y)
    return ((y & (2 ** bit_position - 1)) | (x << bit_position))
