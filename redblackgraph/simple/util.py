def nz_min(*args, **kwargs):
    '''
    nz_min(iterable, *[, default=0, key=func]) -> value
    nz_min(arg1, arg2, *args, *[, key=func]) -> value

    With a single iterable argument, return its smallest non-zero item or 0 if iterable is empty. The
    default keyword-only argument specifies an object to return if
    the provided iterable is empty.
    With two or more arguments, return the smallest non-zero argument or 0 of no non-zero values in args.
    '''
    key = kwargs.get("key", lambda x: x)
    default = kwargs.get("default", 0)
    if len(args) == 1:
        args = args[0]
    mini = None
    for i in args:
        k_i = key(i)
        k_mini = key(mini)
        if mini == None or k_mini == 0 or (k_i < k_mini and not k_i == 0):
            mini = i
    return default if mini is None else mini