from pathlib import Path

from tidysci.application import Application


def main():
    here = Path(__file__).parent
    app = Application([(here, here / "recipes")])
    app.run()


if __name__ == '__main__':
    main()
