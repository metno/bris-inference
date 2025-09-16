__version__ = "0.2.0"
import asyncio


def main():
    import bris.__main__

    asyncio.run(bris.__main__.main())


def inspect():
    import bris.inspect

    bris.inspect.inspect()
