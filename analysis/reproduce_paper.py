#!/usr/bin/env python3
"""Build all main-paper tables, figures, and null statistics in one run."""

from paper_artifacts import (
    build_bundle,
    make_extra_figures,
    make_headline_figure,
    make_overall_table,
    make_paper_figures,
    make_residual_null,
    make_support_table,
    parse_common,
)


def main() -> None:
    args = parse_common(__doc__).parse_args()
    output, bundle = build_bundle(args)
    make_overall_table(args.package_root, output)
    make_support_table(bundle, output)
    make_paper_figures(args.package_root, bundle, output)
    make_headline_figure(args.package_root, bundle, output)
    make_extra_figures(bundle, output)
    make_residual_null(bundle, output)


if __name__ == "__main__":
    main()
