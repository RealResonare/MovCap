import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch MovCap GUI")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Configuration file"
    )
    args = parser.parse_args()

    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.gui.app import MovCapApp
    app = MovCapApp(config_path=args.config)
    app.run()


if __name__ == "__main__":
    main()
