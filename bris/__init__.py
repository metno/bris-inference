__version__ = "0.1.1"


def main():
    import bris.__main__

    bris.__main__.main()


from . import callbacks, conventions, outputs, sources, utils
