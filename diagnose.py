"""Run read-only checks without importing the GUI in this process."""

import argparse
from pathlib import Path

from src.diagnostics import collect_report, format_report
from src.training.config import load_config, parse_config
from src.training.records import default_session_path, load_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, help="Optional saved settings.json to check")
    parser.add_argument("--report", type=Path, help="Write a local report (contains paths); existing files are not overwritten")
    args = parser.parse_args()
    config, extra = None, []
    try:
        if args.settings:
            config = load_config(args.settings)
        else:
            training = load_session(default_session_path()).get("training")
            config = parse_config(training) if training else None
    except (OSError, ValueError, TypeError) as error:
        extra.append({"status": "FAIL", "title": "Saved settings", "detail": f"{error}\nOriginal file left unchanged."})
    try:
        checks = extra + collect_report(config)
    except Exception as error:
        checks = extra + [{"status": "FAIL", "title": "Environment check interrupted", "detail": str(error)}]
    report = format_report(checks)
    print(report)
    if args.report:
        try:
            with args.report.open("x", encoding="utf-8") as file:
                file.write(report)
        except OSError as error:
            print(f"Could not save report: {error}")
            return 1
    return int(any(item["status"] == "FAIL" for item in checks))


if __name__ == "__main__":
    raise SystemExit(main())
