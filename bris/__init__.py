__version__ = "0.1.0"


def main():
    import bris.__main__

    bris.__main__.main()


from . import callbacks
from . import output
from . import outputs
from . import conventions
from . import utils
