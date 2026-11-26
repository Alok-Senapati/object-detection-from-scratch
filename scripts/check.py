"""Run all setup checks from the project root on Windows or Linux."""

import subprocess
import sys


def main() -> None:
    """Run lint, formatting, typing and tests, stopping on the first failing command.

    Returns:
        None.

    Raises:
        subprocess.CalledProcessError: A required validation command fails.
    """
    for command in (
        ["ruff", "check", "."],
        ["ruff", "format", "--check", "."],
        ["mypy"],
        ["pytest"],
    ):
        subprocess.run([sys.executable, "-m", *command], check=True)


if __name__ == "__main__":
    main()
