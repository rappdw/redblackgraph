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

def leftmost_significant_bit_position(x: int) -> int:
    '''
    Given an integer, return the bit position of it's leftmost significant bit
    :param x: operand
    :return: bit position of leftmost significant bit
    '''
    bit_position = 0
    while (x > 1):
        x >>= 1
        bit_position += 1
    return bit_position

def avos_product(x: int, y: int) -> int:
    '''
    The avos product replaces the left most significant bit of operand 2 with operand 1
    :param x: operand 1
    :param y: operand 2
    :return: avos product
    '''

    sign = 1 if (x >= 0) == (y >= 0) else -1
    x, y = abs(x), abs(y)

    # The zero property of the avos product
    if x == 0 or y == 0:
        return 0

    bit_position = leftmost_significant_bit_position(y)
    return sign * ((y & (2 ** bit_position - 1)) | (x << bit_position))
