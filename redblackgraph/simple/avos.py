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

def compute_sign(x: int, y:int) -> int:
    '''
    This is a little complicated to explain at this point, so for now, just trust me
    :param x:
    :param y:
    :return:
    '''

    a = x >= 0
    b = y >=0
    c = x == -1
    d = y == -1

    # with the truth states of a, b, c, d, sign will be:
    #   as of yet undetermined (u) - (!a or !b) and (!c and !d)
    #   negative (n) - (!a and !b) and !u
    #   positive (p) = !(n or u)
    if (not a or not b) and (not c and not d):
        sign = None
    elif not a and not b:
        sign = -1
    else:
        sign = 1

    return sign

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

    bit_position = leftmost_significant_bit_position(y)
    return sign * ((y & (2 ** bit_position - 1)) | (x << bit_position))
