#!/usr/bin/env python3
"""Rebuild the complete support-cell and largest-residual table bodies."""

from paper_artifacts import build_bundle, make_support_table, parse_common


def main() -> None:
    args = parse_common(__doc__).parse_args()
    output, bundle = build_bundle(args)
    make_support_table(bundle, output)


if __name__ == "__main__":
    main()
