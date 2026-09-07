#!/usr/bin/env python3
"""Rebuild the three-panel headline figure from formal seed CSVs."""

from paper_artifacts import build_bundle, make_headline_figure, parse_common


def main() -> None:
    args = parse_common(__doc__).parse_args()
    output, bundle = build_bundle(args)
    make_headline_figure(args.package_root, bundle, output)


if __name__ == "__main__":
    main()
