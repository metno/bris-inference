import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    print("Hello world")


if __name__ == "__main__":
    main()
