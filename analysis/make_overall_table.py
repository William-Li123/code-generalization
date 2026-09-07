#!/usr/bin/env python3
"""Rebuild the manuscript's overall main-experiment table body."""

from paper_artifacts import ensure_output, make_overall_table, parse_common


def main() -> None:
    args = parse_common(__doc__).parse_args()
    make_overall_table(args.package_root, ensure_output(args.output))


if __name__ == "__main__":
    main()
