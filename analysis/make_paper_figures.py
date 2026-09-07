#!/usr/bin/env python3
"""Rebuild the main support-map and net-effect paper figures."""

from paper_artifacts import build_bundle, make_paper_figures, parse_common


def main() -> None:
    args = parse_common(__doc__).parse_args()
    output, bundle = build_bundle(args)
    make_paper_figures(args.package_root, bundle, output)


if __name__ == "__main__":
    main()
