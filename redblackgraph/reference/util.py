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

