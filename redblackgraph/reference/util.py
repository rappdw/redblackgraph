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
    elif (not a and not b) or (
            (not a or not b) and (x == 1 or y == 1)):
        # identity property first. if one of the operatns are 1 or -1, then the sign is the sign
        # of the other operand. If the operands are 1 and -1 then the sign is -
        sign = -1
    else:
        sign = 1

    return sign

