from __future__ import annotations

import argparse
import json
from pathlib import Path

from EDA.local_gate import run_domain


def _load_domain_paths(root: Path, selected: str | None, run_all: bool) -> list[Path]:
    config_dir = root / 'EDA' / 'configs'
    if run_all:
        return [config_dir / 'books.yaml', config_dir / 'movielens.yaml', config_dir / 'movies.yaml']
    if selected is None:
        raise ValueError('Provide --domain or --all.')
    return [config_dir / f'{selected}.yaml']


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the local EDA gate pipeline.')
    parser.add_argument('--domain', choices=['books', 'movielens', 'movies'])
    parser.add_argument('--all', action='store_true', dest='run_all')
    parser.add_argument('--skip-prepare', action='store_true', help='Skip prepare_data dependency runs if outputs already exist.')
    parser.add_argument('--max-target-rows', type=int, default=None, help='Optional smoke-test limit for target rows processed per domain.')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    summaries = []
    for domain_config in _load_domain_paths(project_root, args.domain, args.run_all):
        summaries.append(
            run_domain(
                project_root,
                domain_config,
                prepare=not args.skip_prepare,
                max_target_rows=args.max_target_rows,
            )
        )

    combined_dir = project_root / 'outputs' / 'local_gate' / 'combined'
    combined_dir.mkdir(parents=True, exist_ok=True)
    with (combined_dir / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summaries, handle, indent=2)

    lines = [
        '# Local gate pass/fail summary',
        '',
        '| domain | leakage | cell validation | ring | auc | logistic | negative control | quadrant | decision |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for summary in summaries:
        passes = summary['passes']
        lines.append(
            f"| {summary['domain']} | {'pass' if passes['leakage_audit'] else 'fail'} | {'pass' if passes['cell_validation'] else 'fail'} | {'pass' if passes['ring_coverage'] else 'fail'} | {'pass' if passes['auc_gate'] else 'fail'} | {'pass' if passes['logistic_gate'] else 'fail'} | {'pass' if passes['negative_control'] else 'fail'} | {'pass' if passes['quadrant'] else 'fail'} | {summary['decision']} |"
        )
    (combined_dir / 'pass_fail_summary.md').write_text(
        "\n".join(lines) + "\n",
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
