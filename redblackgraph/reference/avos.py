from redblackgraph.reference.util import MSB

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
    The avos product replaces the left most significant bit of operand 2 with operand 1
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
    # Special case -1 * 1 or -1 * -1
    if x == -1:
        if y == 1:
            return -1
        x = 1
    if y == -1:
        if x == 1:
            return -1
        y = 1

    bit_position = MSB(y)
    return ((y & (2 ** bit_position - 1)) | (x << bit_position))
