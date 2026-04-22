from __future__ import annotations

import ast
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score

from EDA.plotting import save_heatmap, save_line_chart
from generative_recommenders_pl.data.preprocessor import (
    LeakageAuditSummary,
    apply_source_target_leakage_filter,
    load_reference_interactions,
)

RING_CONFIGS = {
    'tight': (0.30, 0.50),
    'default': (0.25, 0.75),
    'loose': (0.10, 0.90),
}


def summarize_alignment_scores(
    scores: pd.DataFrame,
    *,
    domain_label: str,
    sample_size: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    positive_ser = (
        scores.loc[scores['y_ser'] == 1, ['F_alignment_rank_pct']]
        .dropna(subset=['F_alignment_rank_pct'])
        .copy()
    )
    if positive_ser.empty:
        return pd.DataFrame(
            [{'domain': domain_label, 'sample_size': 0, 'median_rank_pct': math.nan, 'mean_rank_pct': math.nan}]
        )

    effective_sample_size = min(sample_size, positive_ser.shape[0])
    sample = (
        positive_ser.sample(n=effective_sample_size, random_state=random_state)
        if positive_ser.shape[0] > effective_sample_size
        else positive_ser
    )
    return pd.DataFrame(
        [
            {
                'domain': domain_label,
                'sample_size': int(sample.shape[0]),
                'median_rank_pct': float(sample['F_alignment_rank_pct'].median()),
                'mean_rank_pct': float(sample['F_alignment_rank_pct'].mean()),
            }
        ]
    )


@dataclass
class DomainSpec:
    name: str
    label: str
    source_data_name: str
    target_data_name: str
    source_prepared_dir: Path
    target_prepared_dir: Path
    source_raw_path: Path
    target_raw_path: Path
    source_raw_kind: str
    target_raw_kind: str
    metadata_path: Path
    metadata_kind: str
    semantic_id_candidates: list[Path]
    positive_rating_threshold: float = 4.0
    primary_domain: bool = True


@dataclass
class CellBuildResult:
    item_cells: pd.DataFrame
    item_metadata: pd.DataFrame
    level_scores: dict[str, dict[str, float]]
    selected_level: str
    sid_coverage: float
    cell_source: str
    cell_profiles: dict[str, dict[str, dict[str, float]]]


class LocalGateRunner:
    def __init__(
        self,
        project_root: Path,
        domain_cfg_path: Path,
        *,
        prepare: bool = True,
        output_root: Path | None = None,
        max_target_rows: int | None = None,
    ) -> None:
        cfg = OmegaConf.to_container(OmegaConf.load(domain_cfg_path), resolve=True)
        self.root = project_root
        self.domain = DomainSpec(
            name=cfg['name'],
            label=cfg['label'],
            source_data_name=cfg['source_data_name'],
            target_data_name=cfg['target_data_name'],
            source_prepared_dir=project_root / cfg['source_prepared_dir'],
            target_prepared_dir=project_root / cfg['target_prepared_dir'],
            source_raw_path=project_root / cfg['source_raw_path'],
            target_raw_path=project_root / cfg['target_raw_path'],
            source_raw_kind=cfg['source_raw_kind'],
            target_raw_kind=cfg['target_raw_kind'],
            metadata_path=project_root / cfg['metadata_path'],
            metadata_kind=cfg['metadata_kind'],
            semantic_id_candidates=[project_root / candidate for candidate in cfg.get('semantic_id_candidates', [])],
            positive_rating_threshold=float(cfg.get('positive_rating_threshold', 4.0)),
            primary_domain=bool(cfg.get('primary_domain', True)),
        )
        self.prepare = prepare
        self.max_target_rows = max_target_rows
        self.output_root = output_root or (project_root / 'outputs' / 'local_gate' / self.domain.name)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.python = project_root / '.venv' / 'bin' / 'python'
        if not self.python.exists():
            self.python = Path(sys.executable)

    def ensure_prepared(self) -> None:
        if not self.prepare and (self.domain.source_prepared_dir / 'ratings.csv').exists() and (self.domain.target_prepared_dir / 'ratings.csv').exists():
            return
        for data_name in [self.domain.source_data_name, self.domain.target_data_name]:
            cmd = [
                str(self.python),
                'src/generative_recommenders_pl/scripts/prepare_data.py',
                f'data={data_name}',
            ]
            subprocess.run(cmd, cwd=self.root, check=True)

    def run(self) -> dict[str, Any]:
        self.ensure_prepared()
        source_ratings = pd.read_csv(self.domain.source_prepared_dir / 'ratings.csv')
        target_ratings = pd.read_csv(self.domain.target_prepared_dir / 'ratings.csv')
        if self.max_target_rows is not None and target_ratings.shape[0] > self.max_target_rows:
            target_ratings = target_ratings.sort_values('timestamp').head(self.max_target_rows).reset_index(drop=True)
        source_sequences = pd.read_csv(self.domain.source_prepared_dir / 'sasrec_format.csv')
        source_lookup = pd.read_csv(self.domain.source_prepared_dir / 'item_lookup.csv')
        target_lookup = self._read_lookup_file(self.domain.target_prepared_dir, 'item_lookup.csv')
        source_users = pd.read_csv(self.domain.source_prepared_dir / 'user_lookup.csv')
        target_users = self._read_lookup_file(self.domain.target_prepared_dir, 'user_lookup.csv')

        input_audit = self._input_audit(source_ratings, target_ratings, source_lookup, target_lookup, source_users, target_users)
        leakage_bundle = self._leakage_audit(source_ratings)
        cell_result = self._build_cells(source_ratings, target_ratings, source_lookup)
        ring_bundle = self._ring_bundle(source_ratings, target_ratings, cell_result)
        prior_bundle = self._prior_scores(source_ratings, target_ratings, cell_result, ring_bundle)
        evaluation = self._evaluate(prior_bundle, ring_bundle)
        summary = self._write_summary(input_audit, leakage_bundle, cell_result, ring_bundle, prior_bundle, evaluation)
        return summary

    def _read_lookup_file(self, preferred_dir: Path, filename: str) -> pd.DataFrame:
        preferred_path = preferred_dir / filename
        if preferred_path.exists():
            return pd.read_csv(preferred_path)
        fallback_path = self.domain.source_prepared_dir / filename
        if fallback_path.exists():
            return pd.read_csv(fallback_path)
        raise FileNotFoundError(
            f"Expected lookup file {filename} under {preferred_dir} or {self.domain.source_prepared_dir}"
        )

    def _load_raw_source(self) -> pd.DataFrame:
        if self.domain.source_raw_kind == 'amazon':
            frame = pd.read_csv(
                self.domain.source_raw_path,
                sep=',',
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                header=None,
                engine='python',
            )
            return frame
        frame = pd.read_csv(self.domain.source_raw_path)
        frame.columns = [str(column).lstrip('﻿') for column in frame.columns]
        frame.rename(columns={'userId': 'user_id', 'movieId': 'item_id'}, inplace=True)
        return frame[['user_id', 'item_id', 'rating', 'timestamp']]

    def _load_raw_target(self) -> pd.DataFrame:
        frame = pd.read_csv(self.domain.target_raw_path)
        frame.columns = [str(column).lstrip('﻿') for column in frame.columns]
        rename_map = {'userId': 'user_id', 'movieId': 'item_id'}
        frame.rename(columns=rename_map, inplace=True)
        if 'label' not in frame.columns and 's_ser_find' in frame.columns:
            ser_cols = [
                's_ser_find', 's_ser_imp', 's_ser_rec', 'm_ser_find', 'm_ser_imp', 'm_ser_rec'
            ]
            flags = frame[ser_cols].applymap(lambda value: str(value).lower() == 'true')
            frame['label'] = flags.any(axis=1).astype(int)
        return frame

    def _input_audit(self, source_ratings: pd.DataFrame, target_ratings: pd.DataFrame, source_lookup: pd.DataFrame, target_lookup: pd.DataFrame, source_users: pd.DataFrame, target_users: pd.DataFrame) -> pd.DataFrame:
        sid_path = next((path for path in self.domain.semantic_id_candidates if path.exists()), None)
        sid_coverage = 1.0 if sid_path is not None else 0.0
        audit = pd.DataFrame([
            {
                'domain': self.domain.label,
                'source_users': int(source_users.shape[0]),
                'source_items': int(source_lookup.shape[0]),
                'source_interactions': int(source_ratings.shape[0]),
                'target_users': int(target_users.shape[0]),
                'target_items': int(target_lookup.shape[0]),
                'target_pairs': int(target_ratings[['user_id', 'item_id']].drop_duplicates().shape[0]),
                'sid_coverage': sid_coverage,
                'timestamp_coverage': float(
                    (pd.to_numeric(source_ratings['timestamp'], errors='coerce').notna().mean() + pd.to_numeric(target_ratings['timestamp'], errors='coerce').notna().mean()) / 2.0
                ),
                'shared_user_space': bool(set(target_ratings['user_id']).issubset(set(source_users['normalized_user_id']))),
                'shared_item_space': bool(set(target_ratings['item_id']).issubset(set(source_lookup['normalized_item_id']))),
            }
        ])
        out_dir = self.output_root / 'input_audit'
        out_dir.mkdir(parents=True, exist_ok=True)
        audit.to_csv(out_dir / f'{self.domain.name}_input_audit.csv', index=False)
        return audit

    def _leakage_audit(self, processed_source_ratings: pd.DataFrame) -> dict[str, Any]:
        raw_source = self._load_raw_source()
        reference = load_reference_interactions(self.domain.target_raw_path)
        cleaned_source, summary, details = apply_source_target_leakage_filter(raw_source, reference, return_details=True)
        lookup_path = self.domain.source_prepared_dir / 'item_lookup.csv'
        user_lookup_path = self.domain.source_prepared_dir / 'user_lookup.csv'
        lookup = pd.read_csv(lookup_path)
        user_lookup = pd.read_csv(user_lookup_path)
        normalized_clean = self._normalize_with_lookups(cleaned_source, lookup, user_lookup)
        matches_processed = int(normalized_clean.shape[0] == processed_source_ratings.shape[0])

        summary_df = pd.DataFrame([
            {
                'domain': self.domain.label,
                'source_before': summary.source_before,
                'target_pairs': summary.target_pairs,
                'removed_target_pairs': summary.removed_target_pairs,
                'removed_post_target_interactions': summary.removed_post_target_interactions,
                'ambiguous': summary.ambiguous,
                'source_after': summary.source_after,
                'matches_processed_source_rows': matches_processed,
            }
        ])
        out_dir = self.output_root / 'leakage'
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_dir / 'leakage_summary.csv', index=False)
        for name, frame in details.items():
            frame.to_csv(out_dir / f'{name}.csv', index=False)
        return {'summary': summary_df, 'details': details}

    def _normalize_with_lookups(self, ratings: pd.DataFrame, item_lookup: pd.DataFrame, user_lookup: pd.DataFrame) -> pd.DataFrame:
        ratings = ratings.copy()
        ratings['user_id'] = ratings['user_id'].astype(str)
        ratings['item_id'] = ratings['item_id'].astype(str)
        user_map = {str(row.original_user_id).casefold(): int(row.normalized_user_id) for row in user_lookup.itertuples(index=False)}
        item_map = {str(row.original_item_id).casefold(): int(row.normalized_item_id) for row in item_lookup.itertuples(index=False)}
        ratings['user_id'] = ratings['user_id'].str.casefold().map(user_map)
        ratings['item_id'] = ratings['item_id'].str.casefold().map(item_map)
        ratings['timestamp'] = pd.to_numeric(ratings['timestamp'], errors='coerce')
        ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')
        ratings = ratings.dropna(subset=['user_id', 'item_id', 'timestamp']).copy()
        ratings['user_id'] = ratings['user_id'].astype(int)
        ratings['item_id'] = ratings['item_id'].astype(int)
        ratings['timestamp'] = ratings['timestamp'].astype(np.int64)
        if 'rating' in ratings.columns:
            ratings['rating'] = ratings['rating'].astype(float)
        return ratings.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    def _build_cells(self, source_ratings: pd.DataFrame, target_ratings: pd.DataFrame, item_lookup: pd.DataFrame) -> CellBuildResult:
        item_metadata = self._load_item_metadata(item_lookup)
        sid_frame = self._load_sid_frame(item_lookup)
        sid_coverage = 0.0
        cell_source = 'metadata_fallback'
        if sid_frame is not None:
            sid_coverage = float(sid_frame['level1'].notna().mean())
            item_cells = sid_frame[['normalized_item_id', 'level1', 'level2']].copy()
            cell_source = 'semantic_id'
        else:
            item_cells = item_metadata[['normalized_item_id', 'level1', 'level2']].copy()

        level_scores: dict[str, dict[str, float]] = {}
        cell_profiles: dict[str, dict[str, dict[str, float]]] = {}
        out_dir = self.output_root / 'cell_validation'
        out_dir.mkdir(parents=True, exist_ok=True)
        for level in ['level1', 'level2']:
            stats_df, curve_points, scores, profiles = self._cell_level_stats(
                source_ratings,
                item_metadata,
                item_cells,
                level=level,
            )
            stats_df.to_csv(out_dir / f'{self.domain.name}_cell_stats_{level}.csv', index=False)
            save_line_chart(
                curve_points,
                out_dir / f'{self.domain.name}_distance_transition_{level}.png',
                title=f'{self.domain.label} {level} distance vs transition probability',
                x_label='cell distance bin midpoint',
                y_label='mean transition probability',
            )
            level_scores[level] = scores
            cell_profiles[level] = profiles

        selected_level = 'level2'
        if level_scores['level2']['median_cell_size'] < 3 or level_scores['level2']['dead_cell_rate'] > 0.6:
            selected_level = 'level1'

        return CellBuildResult(
            item_cells=item_cells.rename(columns={'level1': 'cell_level1', 'level2': 'cell_level2'}),
            item_metadata=item_metadata,
            level_scores=level_scores,
            selected_level=selected_level,
            sid_coverage=sid_coverage,
            cell_source=cell_source,
            cell_profiles=cell_profiles,
        )

    def _load_sid_frame(self, item_lookup: pd.DataFrame) -> pd.DataFrame | None:
        sid_path = next((path for path in self.domain.semantic_id_candidates if path.exists()), None)
        if sid_path is None:
            return None
        raw = torch.load(sid_path, map_location='cpu')
        semantic_ids = raw['semantic_ids']
        if isinstance(semantic_ids, torch.Tensor):
            semantic_ids = semantic_ids.cpu().numpy()
        sid_rows = []
        lookup_map = {str(row.original_item_id).casefold(): int(row.normalized_item_id) for row in item_lookup.itertuples(index=False)}
        for item_id, code in zip(raw['item_ids'], semantic_ids):
            normalized_id = lookup_map.get(str(item_id).casefold())
            if normalized_id is None:
                continue
            code = np.asarray(code).tolist()
            level1 = f'sid:{code[0]}' if code else 'sid:unknown'
            if len(code) > 1:
                level2 = f'sid:{code[0]}/{code[1]}'
            else:
                level2 = level1
            sid_rows.append({'normalized_item_id': normalized_id, 'level1': level1, 'level2': level2})
        if not sid_rows:
            return None
        return pd.DataFrame(sid_rows)

    def _load_item_metadata(self, item_lookup: pd.DataFrame) -> pd.DataFrame:
        target_ids = {str(row.original_item_id).casefold() for row in item_lookup.itertuples(index=False)}
        records: dict[str, dict[str, Any]] = {}
        if self.domain.metadata_kind == 'amazon':
            with self.domain.metadata_path.open('r', encoding='utf-8', errors='ignore') as handle:
                for raw_line in handle:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        obj = ast.literal_eval(raw_line)
                    except Exception:
                        continue
                    asin = str(obj.get('asin', '')).casefold()
                    if asin not in target_ids:
                        continue
                    categories = obj.get('categories') or []
                    flat_categories = []
                    for path in categories:
                        flat_categories.extend([str(part) for part in path if str(part).strip()])
                    description = obj.get('description') or ''
                    if isinstance(description, list):
                        description = ' '.join(map(str, description))
                    records[asin] = {
                        'item_key': asin,
                        'title': str(obj.get('title') or ''),
                        'description': str(description),
                        'tags': flat_categories,
                    }
        else:
            with self.domain.metadata_path.open(newline='', encoding='utf-8', errors='ignore') as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    movie_id = str(row.get('movieId') or row.get('movie_id') or '').casefold()
                    if movie_id not in target_ids:
                        continue
                    genres = [genre.strip() for genre in str(row.get('genres', '')).split('|') if genre and genre != '(no genres listed)']
                    records[movie_id] = {
                        'item_key': movie_id,
                        'title': str(row.get('title') or ''),
                        'description': '',
                        'tags': genres,
                    }
        rows = []
        for row in item_lookup.itertuples(index=False):
            item_key = str(row.original_item_id).casefold()
            meta = records.get(item_key, {'item_key': item_key, 'title': '', 'description': '', 'tags': []})
            tags = [tag for tag in meta['tags'] if tag]
            meaningful = [tag for tag in tags if tag.lower() not in {'books', 'movies & tv', 'movies'}]
            if not meaningful:
                meaningful = tags or ['unknown']
            level1 = meaningful[0]
            level2 = ' / '.join(meaningful[:2]) if len(meaningful) > 1 else level1
            text = ' '.join(part for part in [meta['title'], meta['description'], ' '.join(tags)] if part).strip() or f'item {row.normalized_item_id}'
            rows.append(
                {
                    'normalized_item_id': int(row.normalized_item_id),
                    'original_item_id': str(row.original_item_id),
                    'title': meta['title'],
                    'description': meta['description'],
                    'tags': tags,
                    'text': text,
                    'level1': level1,
                    'level2': level2,
                }
            )
        return pd.DataFrame(rows)

    def _cell_level_stats(self, source_ratings: pd.DataFrame, item_metadata: pd.DataFrame, item_cells: pd.DataFrame, *, level: str):
        working = source_ratings.merge(item_cells[['normalized_item_id', level]], left_on='item_id', right_on='normalized_item_id', how='left')
        working[level] = working[level].fillna('unknown')
        cell_counts = working[level].value_counts().sort_index()
        cell_probs = cell_counts / cell_counts.sum()
        entropy = float(-(cell_probs * np.log(cell_probs + 1e-12)).sum() / math.log(max(len(cell_probs), 2)))
        meta = item_metadata.merge(
            item_cells[['normalized_item_id', level]].rename(columns={level: 'cell_level'}),
            on='normalized_item_id',
            how='left',
        )
        tag_purities = []
        intra_similarities = []
        centroids, tag_baseline_parts = self._build_cell_profiles(meta, 'cell_level')
        for cell, group in meta.groupby('cell_level', observed=True):
            counts = Counter()
            for tags in group['tags']:
                counts.update(tags or ['unknown'])
            total = sum(counts.values())
            purity = float(max(counts.values()) / total) if total else 0.0
            tag_purities.append(purity)
            centroid = centroids.get(str(cell), {})
            similarities = [self._item_to_centroid_similarity(tags or ['unknown'], centroid) for tags in group['tags']]
            intra_similarities.append(float(np.mean(similarities)) if similarities else 0.0)

        transitions = self._transition_probabilities(working[['user_id', 'timestamp', level]].rename(columns={level: 'cell'}))
        distances = []
        probs = []
        cells = sorted(centroids)
        for src_cell, dests in transitions.items():
            if src_cell not in centroids:
                continue
            for dest_cell, prob in dests.items():
                if dest_cell not in centroids:
                    continue
                distance = self._centroid_distance(
                    centroids[src_cell],
                    centroids[dest_cell],
                )
                distances.append(distance)
                probs.append(prob)
        corr = float(spearmanr(distances, probs).correlation) if len(distances) > 2 else 0.0
        curve = self._distance_probability_curve(distances, probs)

        stats_df = pd.DataFrame([
            {
                'domain': self.domain.label,
                'level': level,
                'number_of_cells': int(cell_counts.shape[0]),
                'mean_cell_size': float(cell_counts.mean()),
                'median_cell_size': float(cell_counts.median()),
                'dead_cell_rate': 0.0,
                'code_entropy': entropy,
                'category_purity': float(np.mean(tag_purities)) if tag_purities else 0.0,
                'category_purity_random_baseline': float(np.mean(tag_baseline_parts)) if tag_baseline_parts else 0.0,
                'intra_cell_embedding_similarity': float(np.mean(intra_similarities)) if intra_similarities else 0.0,
                'distance_transition_correlation': corr,
            }
        ])
        scores = stats_df.iloc[0].to_dict()
        return stats_df, curve, scores, centroids

    def _build_cell_profiles(
        self,
        meta: pd.DataFrame,
        cell_column: str,
    ) -> tuple[dict[str, dict[str, float]], list[float]]:
        random = np.random.default_rng(42)
        global_tag_counts = Counter()
        rows = []
        for row in meta.itertuples(index=False):
            tags = [str(tag).strip().lower() for tag in (row.tags or []) if str(tag).strip()]
            if not tags:
                tags = ['unknown']
            rows.append((str(getattr(row, cell_column)), tags))
            global_tag_counts.update(tags)

        cell_tag_counts: dict[str, Counter] = {}
        for cell, tags in rows:
            counter = cell_tag_counts.setdefault(cell, Counter())
            counter.update(tags)

        centroids: dict[str, dict[str, float]] = {}
        for cell, counter in cell_tag_counts.items():
            norm = math.sqrt(sum(float(value) ** 2 for value in counter.values()))
            if norm == 0:
                centroids[cell] = {}
            else:
                centroids[cell] = {
                    tag: float(value) / norm for tag, value in counter.items()
                }

        baseline_parts: list[float] = []
        all_tags = list(global_tag_counts)
        if all_tags:
            for cell, counter in list(cell_tag_counts.items())[: min(len(cell_tag_counts), 128)]:
                sample_size = min(max(len(counter), 1), len(all_tags))
                sample = random.choice(all_tags, size=sample_size, replace=False).tolist()
                baseline_parts.append(
                    self._item_to_centroid_similarity(sample, centroids.get(cell, {}))
                )
        return centroids, baseline_parts

    def _item_to_centroid_similarity(
        self,
        tags: list[str],
        centroid: dict[str, float],
    ) -> float:
        clean_tags = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
        if not clean_tags or not centroid:
            return 0.0
        unique_tags = sorted(set(clean_tags))
        numerator = sum(centroid.get(tag, 0.0) for tag in unique_tags)
        denominator = math.sqrt(len(unique_tags))
        return float(numerator / denominator) if denominator else 0.0

    def _centroid_distance(
        self,
        left: dict[str, float],
        right: dict[str, float],
    ) -> float:
        if not left or not right:
            return 1.0
        shared = set(left).intersection(right)
        cosine = sum(left[tag] * right[tag] for tag in shared)
        cosine = max(min(cosine, 1.0), -1.0)
        return 1.0 - float(cosine)

    def _distance_probability_curve(self, distances: list[float], probs: list[float]) -> list[tuple[float, float]]:
        if not distances:
            return []
        frame = pd.DataFrame({'distance': distances, 'prob': probs})
        bins = np.linspace(frame['distance'].min(), frame['distance'].max(), 6)
        if len(np.unique(bins)) < 2:
            return [(float(frame['distance'].mean()), float(frame['prob'].mean()))]
        frame['bin'] = pd.cut(frame['distance'], bins=bins, include_lowest=True, duplicates='drop')
        grouped = frame.groupby('bin', observed=True).agg(distance=('distance', 'mean'), probability=('prob', 'mean')).reset_index(drop=True)
        return list(zip(grouped['distance'].tolist(), grouped['probability'].tolist()))

    def _transition_probabilities(self, frame: pd.DataFrame) -> dict[str, dict[str, float]]:
        frame = frame.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        counts: dict[str, dict[str, int]] = {}
        for _, user_frame in frame.groupby('user_id'):
            cells = user_frame['cell'].astype(str).tolist()
            for left, right in zip(cells[:-1], cells[1:]):
                counts.setdefault(left, {})[right] = counts.setdefault(left, {}).get(right, 0) + 1
        probs: dict[str, dict[str, float]] = {}
        for left, destinations in counts.items():
            total = float(sum(destinations.values()))
            if total <= 0:
                continue
            probs[left] = {right: value / total for right, value in destinations.items()}
        return probs

    def _ring_bundle(self, source_ratings: pd.DataFrame, target_ratings: pd.DataFrame, cell_result: CellBuildResult) -> dict[str, Any]:
        level_col = 'cell_' + cell_result.selected_level
        item_cells = cell_result.item_cells.rename(columns={f'cell_{cell_result.selected_level}': 'cell'})[['normalized_item_id', 'cell']]
        source = source_ratings.merge(item_cells, left_on='item_id', right_on='normalized_item_id', how='left')
        target = target_ratings.merge(item_cells, left_on='item_id', right_on='normalized_item_id', how='left')
        source['cell'] = source['cell'].fillna('unknown')
        target['cell'] = target['cell'].fillna('unknown')

        centroids = cell_result.cell_profiles.get(cell_result.selected_level)
        if centroids is None:
            cell_metadata = cell_result.item_metadata.merge(
                item_cells,
                on='normalized_item_id',
                how='left',
            )
            centroids, _ = self._build_cell_profiles(cell_metadata, 'cell')

        transition_distances = []
        for _, user_frame in source.sort_values(['user_id', 'timestamp']).groupby('user_id'):
            cells = user_frame['cell'].astype(str).tolist()
            for left, right in zip(cells[:-1], cells[1:]):
                if left not in centroids or right not in centroids:
                    continue
                transition_distances.append(
                    self._centroid_distance(centroids[left], centroids[right])
                )
        if not transition_distances:
            transition_distances = [0.0, 1.0]

        results = {}
        out_dir = self.output_root / 'ring'
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, (low_q, high_q) in RING_CONFIGS.items():
            r_min, r_max = np.quantile(transition_distances, [low_q, high_q])
            scored = []
            source_histories = source.sort_values(['user_id', 'timestamp']).groupby('user_id')
            source_groups = {user_id: group for user_id, group in source_histories}
            for row in target.itertuples(index=False):
                history = source_groups.get(row.user_id)
                seen_cells: list[str] = []
                if history is not None:
                    seen_cells = history.loc[history['timestamp'] <= row.timestamp, 'cell'].astype(str).tolist()
                seen_set = set(seen_cells)
                ring_label = 'unknown'
                ring_flag = 0
                if pd.isna(row.cell) or row.cell == 'unknown':
                    ring_label = 'unknown'
                elif row.cell in seen_set:
                    ring_label = 'seen cell'
                elif not seen_set:
                    ring_label = 'far-unseen'
                else:
                    current_centroid = centroids.get(str(row.cell), {})
                    distances = [
                        self._centroid_distance(current_centroid, centroids.get(seen_cell, {}))
                        for seen_cell in seen_set
                        if seen_cell in centroids
                    ]
                    distance = min(distances) if distances else 1.0
                    if r_min <= distance <= r_max:
                        ring_label = 'near-unseen ring'
                        ring_flag = 1
                    else:
                        ring_label = 'far-unseen'
                scored.append({'user_id': row.user_id, 'item_id': row.item_id, 'ring_bucket': ring_label, 'ring_flag': ring_flag, 'y_ser': int(getattr(row, 'label', getattr(row, 'ser_label', 0))), 'rating': float(row.rating)})
            scored_df = pd.DataFrame(scored)
            coverage = scored_df.groupby('ring_bucket', observed=True).size().reset_index(name='count')
            coverage['share'] = coverage['count'] / coverage['count'].sum()
            coverage.to_csv(out_dir / f'{self.domain.name}_ring_coverage_{name}.csv', index=False)
            comparison = scored_df.groupby('y_ser')['ring_flag'].mean().reset_index(name='p_in_ring')
            comparison.to_csv(out_dir / f'{self.domain.name}_ring_ser_comparison_{name}.csv', index=False)
            results[name] = {'bounds': (float(r_min), float(r_max)), 'scored': scored_df, 'comparison': comparison}
        return results

    def _prior_scores(self, source_ratings: pd.DataFrame, target_ratings: pd.DataFrame, cell_result: CellBuildResult, ring_bundle: dict[str, Any]) -> dict[str, Any]:
        active_ring = ring_bundle['default']['scored'][['user_id', 'item_id', 'ring_bucket', 'ring_flag']]
        level_col = 'cell_' + cell_result.selected_level
        item_cells = cell_result.item_cells[['normalized_item_id', level_col]].rename(columns={level_col: 'cell'})
        source = source_ratings.merge(item_cells, left_on='item_id', right_on='normalized_item_id', how='left')
        target = target_ratings.merge(item_cells, left_on='item_id', right_on='normalized_item_id', how='left')
        source['cell'] = source['cell'].fillna('unknown')
        target['cell'] = target['cell'].fillna('unknown')
        source = source.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        transition_probs = self._transition_probabilities(source[['user_id', 'timestamp', 'cell']])
        cells = sorted(set(source['cell'].astype(str)).union(target['cell'].astype(str)))
        pop_counts = source['cell'].astype(str).value_counts(normalize=True).to_dict()
        cell_sizes = item_cells['cell'].astype(str).value_counts().to_dict()
        rng = np.random.default_rng(42)
        shuffled_destinations = cells.copy()
        rng.shuffle(shuffled_destinations)
        shuffle_map = {cell: shuffled for cell, shuffled in zip(cells, shuffled_destinations)}
        shuffled_transition: dict[str, dict[str, float]] = {}
        transition_rows = []
        for src_cell, destinations in transition_probs.items():
            for dest_cell, prob in destinations.items():
                transition_rows.append(
                    {
                        'source_cell': str(src_cell),
                        'target_cell': str(dest_cell),
                        'probability': float(prob),
                    }
                )
                shuffled_transition.setdefault(str(src_cell), {})[
                    shuffle_map[str(dest_cell)]
                ] = float(prob)

        scores = []
        source_groups = {
            user_id: (
                group['timestamp'].to_numpy(dtype=np.int64),
                group['cell'].astype(str).to_numpy(),
            )
            for user_id, group in source.groupby('user_id')
        }
        lambda_smoothing = 0.1
        gamma = 0.9
        for row in target.itertuples(index=False):
            history = source_groups.get(row.user_id)
            if history is None:
                recent_weights = {}
            else:
                history_timestamps, history_cells = history
                hist = history_cells[history_timestamps <= row.timestamp]
                recent_weights = {}
                reversed_cells = hist[::-1].tolist()
                for idx, cell in enumerate(reversed_cells):
                    recent_weights[cell] = recent_weights.get(cell, 0.0) + (gamma ** idx)
            total_recent = sum(recent_weights.values())
            current_cell = str(row.cell)
            recent_prob = recent_weights.get(current_cell, 0.0) / total_recent if total_recent else 0.0
            pop_prob = float(pop_counts.get(current_cell, 0.0))
            if total_recent:
                normalized_weights = {
                    cell: weight / total_recent for cell, weight in recent_weights.items()
                }
                score_map: dict[str, float] = {}
                shuffled_score_map: dict[str, float] = {}
                for seen_cell, weight in normalized_weights.items():
                    for dest_cell, prob in transition_probs.get(seen_cell, {}).items():
                        score_map[dest_cell] = score_map.get(dest_cell, 0.0) + weight * float(prob)
                    for dest_cell, prob in shuffled_transition.get(seen_cell, {}).items():
                        shuffled_score_map[dest_cell] = shuffled_score_map.get(dest_cell, 0.0) + weight * float(prob)
            else:
                score_map = {}
                shuffled_score_map = {}
            f_trans = float(score_map.get(current_cell, 0.0))
            e0 = 0.5 * recent_prob + 0.5 * pop_prob
            prior = pop_prob
            l0 = math.log((f_trans + lambda_smoothing * prior + 1e-12) / (e0 + lambda_smoothing * prior + 1e-12))
            shuffled_f = float(shuffled_score_map.get(current_cell, 0.0))
            shuffled_l0 = math.log((shuffled_f + lambda_smoothing * prior + 1e-12) / (e0 + lambda_smoothing * prior + 1e-12))
            alignment_pct = math.nan
            if int(getattr(row, 'label', getattr(row, 'ser_label', 0))) == 1:
                target_score = score_map.get(current_cell, 0.0)
                all_scores = np.array(list(score_map.values()), dtype=float)
                better = int((all_scores > target_score).sum())
                equal = int((all_scores == target_score).sum())
                zero_tail = max(len(cells) - len(score_map), 0)
                if target_score == 0.0:
                    equal += zero_tail
                alignment_pct = float((better + 0.5 * max(equal, 1)) / max(len(cells), 1))
            scores.append(
                {
                    'user_id': int(row.user_id),
                    'item_id': int(row.item_id),
                    'cell_id': current_cell,
                    'y_ser': int(getattr(row, 'label', getattr(row, 'ser_label', 0))),
                    'rating': float(row.rating),
                    'F_trans': f_trans,
                    'E0': e0,
                    'L0': l0,
                    'F_trans_shuffled': shuffled_f,
                    'L0_shuffled': shuffled_l0,
                    'F_alignment_rank_pct': alignment_pct,
                    'pop_cell': pop_prob,
                    'cell_size': int(cell_sizes.get(current_cell, 0)),
                }
            )
        score_df = pd.DataFrame(scores).merge(active_ring, on=['user_id', 'item_id'], how='left')
        score_df['ring_bucket'] = score_df['ring_bucket'].fillna('unknown')
        score_df['ring_flag'] = score_df['ring_flag'].fillna(0).astype(int)
        out_dir = self.output_root / 'priors'
        out_dir.mkdir(parents=True, exist_ok=True)
        score_df.to_csv(out_dir / f'{self.domain.name}_target_prior_scores.csv', index=False)
        pd.DataFrame(transition_rows).to_csv(
            out_dir / f'{self.domain.name}_cell_transition_matrix.csv',
            index=False,
        )
        return {
            'scores': score_df,
            'transition_matrix': transition_probs,
            'shuffled_transition': shuffled_transition,
        }

    def _evaluate(self, prior_bundle: dict[str, Any], ring_bundle: dict[str, Any]) -> dict[str, Any]:
        scores = prior_bundle['scores'].copy()
        scores = scores[scores['rating'] >= self.domain.positive_rating_threshold].copy()
        eval_dir = self.output_root / 'auc'
        eval_dir.mkdir(parents=True, exist_ok=True)
        if scores['y_ser'].nunique() < 2:
            auc_table = pd.DataFrame([{'domain': self.domain.label, 'auc_F_trans': np.nan, 'auc_neg_E0': np.nan, 'auc_L0': np.nan, 'auc_L0_shuffled': np.nan, 'top_bottom_lift': np.nan}])
            auc_table.to_csv(eval_dir / 'prior_auc.csv', index=False)
            return {'auc': auc_table}

        auc_table = pd.DataFrame([
            {
                'domain': self.domain.label,
                'auc_F_trans': float(roc_auc_score(scores['y_ser'], scores['F_trans'])),
                'auc_neg_E0': float(roc_auc_score(scores['y_ser'], -scores['E0'])),
                'auc_L0': float(roc_auc_score(scores['y_ser'], scores['L0'])),
                'auc_L0_shuffled': float(roc_auc_score(scores['y_ser'], scores['L0_shuffled'])),
            }
        ])
        decile = self._decile_lift(scores)
        auc_table['top_bottom_lift'] = decile['lift']
        auc_table.to_csv(eval_dir / 'prior_auc.csv', index=False)

        quadrant = self._quadrant(scores)
        logistic = self._logistic_gate(scores)
        controls = self._controls(scores)
        alignment = self._alignment(scores)
        ring_robustness = self._ring_robustness(ring_bundle)

        decile['table'].to_csv(self.output_root / 'decile' / 'decile_lift.csv', index=False)
        quadrant['table'].to_csv(self.output_root / 'quadrant' / 'quadrant_summary.csv', index=False)
        logistic['table'].to_csv(self.output_root / 'controls' / 'logistic_gate.csv', index=False)
        controls['cell_thresholds'].to_csv(self.output_root / 'controls' / 'cell_threshold_auc.csv', index=False)
        controls['popularity_buckets'].to_csv(self.output_root / 'controls' / 'popularity_bucket_auc.csv', index=False)
        alignment['table'].to_csv(self.output_root / 'auc' / 'alignment_spot_check.csv', index=False)
        ring_robustness.to_csv(self.output_root / 'ring' / 'ring_robustness.csv', index=False)

        return {
            'auc': auc_table,
            'decile': decile,
            'quadrant': quadrant,
            'logistic': logistic,
            'controls': controls,
            'alignment': alignment,
            'ring_robustness': ring_robustness,
            'scores': scores,
        }

    def _decile_lift(self, scores: pd.DataFrame) -> dict[str, Any]:
        decile_dir = self.output_root / 'decile'
        decile_dir.mkdir(parents=True, exist_ok=True)
        work = scores.copy()
        work['decile'] = pd.qcut(work['L0'].rank(method='first'), 10, labels=False, duplicates='drop') + 1
        table = work.groupby('decile', observed=True).agg(
            count=('y_ser', 'size'),
            ser_rate=('y_ser', 'mean'),
            avg_F=('F_trans', 'mean'),
            avg_E0=('E0', 'mean'),
            avg_L0=('L0', 'mean'),
            avg_pop=('pop_cell', 'mean'),
            avg_cell_size=('cell_size', 'mean'),
        ).reset_index()
        if table.empty:
            lift = math.nan
        else:
            top = float(table.loc[table['decile'] == table['decile'].max(), 'ser_rate'].iloc[0])
            bottom = float(table.loc[table['decile'] == table['decile'].min(), 'ser_rate'].iloc[0])
            lift = top / bottom if bottom > 0 else math.inf
        return {'table': table, 'lift': lift}

    def _quadrant(self, scores: pd.DataFrame) -> dict[str, Any]:
        out_dir = self.output_root / 'quadrant'
        out_dir.mkdir(parents=True, exist_ok=True)
        f_cut = scores['F_trans'].median()
        e_cut = scores['E0'].median()
        conditions = []
        for row in scores.itertuples(index=False):
            high_f = row.F_trans >= f_cut
            low_e = row.E0 < e_cut
            if high_f and low_e:
                quadrant = 'HighF/LowE'
            elif high_f and not low_e:
                quadrant = 'HighF/HighE'
            elif not high_f and low_e:
                quadrant = 'LowF/LowE'
            else:
                quadrant = 'LowF/HighE'
            conditions.append(quadrant)
        work = scores.copy()
        work['quadrant'] = conditions
        table = work.groupby('quadrant', observed=True).agg(
            count=('y_ser', 'size'),
            ser_rate=('y_ser', 'mean'),
            relevance_rate=('rating', lambda values: float((pd.Series(values) >= self.domain.positive_rating_threshold).mean())),
            avg_pop=('pop_cell', 'mean'),
            avg_cell_size=('cell_size', 'mean'),
            avg_ring=('ring_flag', 'mean'),
        ).reset_index()
        values = [
            [float(table.loc[table['quadrant'] == 'HighF/LowE', 'ser_rate'].iloc[0]) if 'HighF/LowE' in set(table['quadrant']) else 0.0,
             float(table.loc[table['quadrant'] == 'HighF/HighE', 'ser_rate'].iloc[0]) if 'HighF/HighE' in set(table['quadrant']) else 0.0],
            [float(table.loc[table['quadrant'] == 'LowF/LowE', 'ser_rate'].iloc[0]) if 'LowF/LowE' in set(table['quadrant']) else 0.0,
             float(table.loc[table['quadrant'] == 'LowF/HighE', 'ser_rate'].iloc[0]) if 'LowF/HighE' in set(table['quadrant']) else 0.0],
        ]
        annotations = [
            [f"HighF/LowE\\n{values[0][0]:.3f}", f"HighF/HighE\\n{values[0][1]:.3f}"],
            [f"LowF/LowE\\n{values[1][0]:.3f}", f"LowF/HighE\\n{values[1][1]:.3f}"],
        ]
        save_heatmap(values, ['Low E0', 'High E0'], ['High F', 'Low F'], out_dir / f'{self.domain.name}_quadrant_heatmap.png', title=f'{self.domain.label} quadrant serendipity rate', annotations=annotations)
        return {'table': table, 'f_cut': float(f_cut), 'e_cut': float(e_cut)}

    def _logistic_gate(self, scores: pd.DataFrame) -> dict[str, Any]:
        out_dir = self.output_root / 'controls'
        out_dir.mkdir(parents=True, exist_ok=True)
        design = pd.DataFrame(
            {
                'L0': scores['L0'].astype(float),
                'log_pop': np.log1p(scores['pop_cell'].astype(float)),
                'log_cell_size': np.log1p(scores['cell_size'].astype(float)),
                'ring_flag': scores['ring_flag'].astype(float),
            }
        )
        y = scores['y_ser'].astype(float).to_numpy()
        beta, se, p_values = self._fit_logistic(design.to_numpy(), y)
        corrected_p = min(float(p_values[0]) * 3.0, 1.0)
        table = pd.DataFrame(
            {
                'feature': ['L0', 'log_pop', 'log_cell_size', 'ring_flag'],
                'beta': beta[1:],
                'std_err': se[1:],
                'p_value': p_values[1:],
            }
        )
        table['bonferroni_p_value'] = table['p_value'] * 3.0
        l0_row = table.loc[table['feature'] == 'L0'].iloc[0]
        summary = pd.DataFrame([
            {
                'domain': self.domain.label,
                'beta_L0': float(l0_row['beta']),
                'p_value_L0': float(l0_row['p_value']),
                'bonferroni_p_value_L0': float(l0_row['bonferroni_p_value']),
                'passes_primary_logistic_gate': bool(l0_row['beta'] > 0 and l0_row['bonferroni_p_value'] < 0.05),
            }
        ])
        summary.to_csv(out_dir / 'logistic_gate_summary.csv', index=False)
        return {'table': table, 'summary': summary}

    def _fit_logistic(self, X: np.ndarray, y: np.ndarray):
        X = np.column_stack([np.ones(X.shape[0]), X])
        beta = np.zeros(X.shape[1], dtype=float)
        ridge = 1e-6
        for _ in range(100):
            logits = X @ beta
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            W = np.clip(probs * (1.0 - probs), 1e-6, None)
            z = logits + (y - probs) / W
            xtwx = X.T @ (W[:, None] * X) + ridge * np.eye(X.shape[1])
            xtwz = X.T @ (W * z)
            beta_new = np.linalg.solve(xtwx, xtwz)
            if np.max(np.abs(beta_new - beta)) < 1e-8:
                beta = beta_new
                break
            beta = beta_new
        logits = X @ beta
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        W = np.clip(probs * (1.0 - probs), 1e-6, None)
        fisher = X.T @ (W[:, None] * X) + ridge * np.eye(X.shape[1])
        cov = np.linalg.inv(fisher)
        se = np.sqrt(np.diag(cov))
        z_scores = beta / np.clip(se, 1e-9, None)
        p_values = 2.0 * norm.sf(np.abs(z_scores))
        return beta, se, p_values

    def _controls(self, scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
        out_dir = self.output_root / 'controls'
        out_dir.mkdir(parents=True, exist_ok=True)
        threshold_rows = []
        for threshold in [5, 10, 20]:
            subset = scores[scores['cell_size'] >= threshold]
            auc_value = float(roc_auc_score(subset['y_ser'], subset['L0'])) if subset['y_ser'].nunique() > 1 and not subset.empty else math.nan
            threshold_rows.append({'cell_size_threshold': threshold, 'count': int(subset.shape[0]), 'auc_L0': auc_value})
        cell_thresholds = pd.DataFrame(threshold_rows)

        work = scores.copy()
        work['popularity_bucket'] = pd.qcut(work['pop_cell'].rank(method='first'), 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        bucket_rows = []
        for bucket, subset in work.groupby('popularity_bucket', observed=True):
            auc_value = float(roc_auc_score(subset['y_ser'], subset['L0'])) if subset['y_ser'].nunique() > 1 else math.nan
            bucket_rows.append({'popularity_bucket': bucket, 'count': int(subset.shape[0]), 'auc_L0': auc_value})
        popularity_buckets = pd.DataFrame(bucket_rows)
        return {'cell_thresholds': cell_thresholds, 'popularity_buckets': popularity_buckets}

    def _alignment(self, scores: pd.DataFrame) -> dict[str, Any]:
        table = summarize_alignment_scores(scores, domain_label=self.domain.label)
        sample = (
            scores.loc[(scores['y_ser'] == 1) & scores['F_alignment_rank_pct'].notna()].copy()
        )
        return {'table': table, 'sample': sample}

    def _ring_robustness(self, ring_bundle: dict[str, Any]) -> pd.DataFrame:
        rows = []
        for name, bundle in ring_bundle.items():
            comparison = bundle['comparison']
            positive = comparison.loc[comparison['y_ser'] == 1, 'p_in_ring']
            negative = comparison.loc[comparison['y_ser'] == 0, 'p_in_ring']
            rows.append(
                {
                    'ring_setting': name,
                    'r_min': bundle['bounds'][0],
                    'r_max': bundle['bounds'][1],
                    'p_in_ring_y1': float(positive.iloc[0]) if not positive.empty else math.nan,
                    'p_in_ring_y0': float(negative.iloc[0]) if not negative.empty else math.nan,
                    'directional_pass': bool((float(positive.iloc[0]) if not positive.empty else -math.inf) > (float(negative.iloc[0]) if not negative.empty else math.inf)),
                }
            )
        return pd.DataFrame(rows)

    def _write_summary(self, input_audit: pd.DataFrame, leakage_bundle: dict[str, Any], cell_result: CellBuildResult, ring_bundle: dict[str, Any], prior_bundle: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
        quadrant_table = evaluation['quadrant']['table']
        high_low = quadrant_table.loc[quadrant_table['quadrant'] == 'HighF/LowE', 'ser_rate']
        low_low = quadrant_table.loc[quadrant_table['quadrant'] == 'LowF/LowE', 'ser_rate']
        auc_table = evaluation['auc'].iloc[0]
        logistic_summary = evaluation['logistic']['summary'].iloc[0]
        ring_default = evaluation['ring_robustness'].loc[evaluation['ring_robustness']['ring_setting'] == 'default'].iloc[0]

        passes = {
            'leakage_audit': bool(leakage_bundle['summary']['removed_target_pairs'].iloc[0] >= 0 and leakage_bundle['summary']['removed_post_target_interactions'].iloc[0] >= 0),
            'cell_validation': bool(cell_result.level_scores[cell_result.selected_level]['distance_transition_correlation'] < 0),
            'ring_coverage': bool(ring_default['directional_pass']),
            'auc_gate': bool(auc_table['auc_L0'] > 0.55),
            'component_gain': bool(auc_table['auc_L0'] > auc_table['auc_F_trans'] and auc_table['auc_L0'] > auc_table['auc_neg_E0']),
            'logistic_gate': bool(logistic_summary['passes_primary_logistic_gate']),
            'negative_control': bool(abs(auc_table['auc_L0_shuffled'] - 0.5) <= 0.1),
            'quadrant': bool((float(high_low.iloc[0]) if not high_low.empty else -math.inf) > (float(low_low.iloc[0]) if not low_low.empty else math.inf)),
        }
        if passes['auc_gate'] and passes['quadrant'] and passes['logistic_gate']:
            decision = 'PROCEED_TO_NEURAL_F'
        elif passes['cell_validation'] and passes['ring_coverage']:
            decision = 'REVISE_CELL_OR_RING'
        else:
            decision = 'STOP_OR_PIVOT'

        summary = {
            'domain': self.domain.label,
            'name': self.domain.name,
            'primary_domain': self.domain.primary_domain,
            'cell_source': cell_result.cell_source,
            'selected_cell_level': cell_result.selected_level,
            'sid_coverage': cell_result.sid_coverage,
            'passes': passes,
            'decision': decision,
            'auc_L0': float(auc_table['auc_L0']),
            'auc_F_trans': float(auc_table['auc_F_trans']),
            'auc_neg_E0': float(auc_table['auc_neg_E0']),
            'auc_L0_shuffled': float(auc_table['auc_L0_shuffled']),
            'top_bottom_lift': float(auc_table['top_bottom_lift']),
            'beta_L0': float(logistic_summary['beta_L0']),
            'bonferroni_p_value_L0': float(logistic_summary['bonferroni_p_value_L0']),
            'alignment_median_rank_pct': float(evaluation['alignment']['table']['median_rank_pct'].iloc[0]),
        }
        with (self.output_root / 'summary.json').open('w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)
        return summary


def run_domain(
    project_root: Path,
    domain_config: Path,
    *,
    prepare: bool = True,
    output_root: Path | None = None,
    max_target_rows: int | None = None,
) -> dict[str, Any]:
    runner = LocalGateRunner(
        project_root,
        domain_config,
        prepare=prepare,
        output_root=output_root,
        max_target_rows=max_target_rows,
    )
    return runner.run()
