from pathlib import Path

from tidysci.application import run


def main():
    here = Path(__file__).parent
    run([(here, here / "recipes")])


if __name__ == '__main__':
    main()
