import argparse

from .ui import App


def main():
    parser = argparse.ArgumentParser(description="Serve the StreamJoy UI")
    parser.add_argument("command", help="The command to run", choices=["ui"])
    args = parser.parse_args()

    if args.command == "ui":
        App().serve(port=8888, show=True)


if __name__ == "__main__":
    main()
