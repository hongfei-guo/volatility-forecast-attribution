"""Select the paper's evaluation forecasts and combine probability and point panels."""
from pathlib import Path
import argparse

import pandas as pd
import yaml


def prepare(components, point_panels, start, end):
    probability = components.copy()
    probability['origin_date'] = pd.to_datetime(probability.origin_date)
    probability['mature_date'] = pd.to_datetime(probability.mature_date)
    probability = probability[probability.scope.eq('evaluation') & probability.origin_date.between(start, end)]
    frames = [probability]
    for panel in point_panels:
        point = panel[panel.forecast_kind.eq('variance_point_only')].copy()
        point['origin_date'] = pd.to_datetime(point.origin_date)
        point['mature_date'] = pd.to_datetime(point.mature_date)
        point = point[point.origin_date.between(start, end)]
        point['predictive_file'] = None
        point['scope'] = 'evaluation'
        frames.append(point)
    result = pd.concat(frames, ignore_index=True)
    keys = ['market', 'model_id', 'origin_date', 'horizon']
    if result.empty or result.duplicated(keys).any():
        raise ValueError('evaluation forecasts must be nonempty and unique')
    return result.sort_values(keys).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component-index', type=Path, required=True)
    parser.add_argument('--point-panels', type=Path, nargs='+', required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2]/'design/analysis.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())['sample']
    result = prepare(pd.read_parquet(args.component_index),
                     [pd.read_parquet(p) for p in args.point_panels],
                     pd.Timestamp(config['oos_start']), pd.Timestamp(config['oos_end']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(f'Prepared {len(result)} evaluation forecasts; warm-up records excluded.')


if __name__ == '__main__':
    main()
