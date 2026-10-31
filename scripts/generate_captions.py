"""Generate one caption per recording with a selected LALM.

Planned in E04-E10.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.cli import base_parser, init_run, not_implemented  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--model", help="model key from configs/models.yaml")
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)
    config = load_config("generation", args.overrides)
    logger.debug("Config: %s", config)
    return not_implemented(logger, "E04-E10")


if __name__ == "__main__":
    sys.exit(main())
