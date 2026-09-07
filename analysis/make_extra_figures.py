#!/usr/bin/env python3
"""Rebuild the raw permutation-null and practitioner-map figures."""

from paper_artifacts import build_bundle, make_extra_figures, parse_common


def main() -> None:
    args = parse_common(__doc__).parse_args()
    output, bundle = build_bundle(args)
    make_extra_figures(bundle, output)


if __name__ == "__main__":
    main()
