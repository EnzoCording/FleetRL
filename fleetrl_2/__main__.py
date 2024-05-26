from pathlib import Path

from pyjob_todo.application import Application


def main():
    here = Path(__file__).parent
    app = Application(here, here / "templates",
                      full_package_name="FleetRL")
    app.run()


if __name__ == '__main__':
    main()
