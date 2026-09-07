#!/usr/bin/env python3
"""Rebuild residual and sensitivity-profile null analyses and figures."""

from paper_artifacts import build_bundle, make_residual_null, parse_common


def main() -> None:
    args = parse_common(__doc__).parse_args()
    output, bundle = build_bundle(args)
    make_residual_null(bundle, output)


if __name__ == "__main__":
    main()
