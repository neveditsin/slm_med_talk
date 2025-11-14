"""Allow ``python -m oie_toolkit`` to behave like the CLI module."""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
