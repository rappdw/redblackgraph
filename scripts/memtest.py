import numpy as np
import os
import psutil
import sys
import traceback

PROCESS = psutil.Process(os.getpid())
MEGA = 1024 * 1024


def main():
    try:
        print_memory_usage()
        # alloc_max_str()
        alloc_max_array()
    except MemoryError as error:
        # Output expected MemoryErrors.
        log_exception(error)
    except Exception as exception:
        # Output unexpected Exceptions.
        log_exception(exception, False)


def alloc_max_array():
    """Allocates memory for maximum array.
    See: https://stackoverflow.com/a/15495136

    :return: None
    """
    base = 13 * 10000
    i = 0
    nbytes = 0
    while True:
        try:
            size = base + i * 1000
            collection = np.ones((size,size), dtype=np.int32)
            nbytes = collection.nbytes
            i += 1
            if i % 1 == 0:
                print(f"loop: {i}; size: {size:,}; allocated: {nbytes/(1024*1024*1024):,.2f} GB")
        except MemoryError as error:
            # Output expected MemoryErrors.
            log_exception(error)
            break
        except Exception as exception:
            # Output unexpected Exceptions.
            log_exception(exception, False)
            break
    print(f'Maximum array size: {nbytes:,}')
    print_memory_usage()


def log_exception(exception: BaseException, expected: bool = True):
    """Prints the passed BaseException to the console, including traceback.

    :param exception: The BaseException to output.
    :param expected: Determines if BaseException was expected.
    """
    output = "[{}] {}: {}".format('EXPECTED' if expected else 'UNEXPECTED', type(exception).__name__, exception)
    print(output)
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_tb(exc_traceback)


def print_memory_usage():
    """Prints current memory usage stats.
    See: https://stackoverflow.com/a/15495136

    :return: None
    """
    total, available, percent, used, free, active, inactive, wired = psutil.virtual_memory()
    total, available, used, free = total / MEGA, available / MEGA, used / MEGA, free / MEGA
    proc = PROCESS.memory_info()[1] / MEGA
    print(f'process = {proc:,.2f} total = {total:,.2f} available = {available:,.2f} used = {used:,.2f} free = {free:,.2f} percent = {percent}')


if __name__ == "__main__":
    main()