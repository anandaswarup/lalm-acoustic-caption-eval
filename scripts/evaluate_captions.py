"""Score model captions against human references (five- and four-reference conditions).

Planned in E11-E12, E14.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.cli import base_parser, init_run, not_implemented  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)
    config = load_config("metrics", args.overrides)
    logger.debug("Config: %s", config)
    return not_implemented(logger, "E11-E12, E14")


if __name__ == "__main__":
    sys.exit(main())
