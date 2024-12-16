import numbers
import os
import uuid
import time


def expand_time_tokens(filename, unixtime):
    """Expand time tokens in a filename"""
    if not isinstance(unixtime, numbers.Number):
        raise ValueError(f"Unixtime but be numeric not {unixtime}")

    return os.path.abspath(time.strftime(filename, time.gmtime(unixtime)))

def create_directory(filename):
    """Creates all sub directories necessary to be able to write filename"""
    dir = os.path.dirname(filename)
    if dir != "":
        os.makedirs(dir, exist_ok=True)

def is_number(value):
    return isinstance(value, numbers.Number)

def get_workdir(path):
    v = uuid.uuid4()
    return path + "/" + str(v)
