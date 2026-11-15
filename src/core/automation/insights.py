""" Model scores across all leagues: training hold-out metrics, live track record (prediction ledger) and the preferred
    model of every league & market. Shared by the automation pipeline and the dashboard.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.automation.ledger import PredictionLedger, preferred_models
from src.core.database.model import ModelDatabase


def _eval_metric(report: Optional[pd.DataFrame], metric: str) -> float:
    if report is None or metric not in report.columns:
        return np.nan
    eval_report = report[report['Data'] == 'Eval'] if 'Data' in report.columns else report
    return float(eval_report[metric].iloc[0]) if not eval_report.empty else np.nan


def model_scores(ledger: Optional[PredictionLedger] = None) -> pd.DataFrame:
    """ One row per saved model (all leagues) with columns: league_id, market, model_id, classifier,
        holdout_log_loss, holdout_accuracy, skill_vs_bookmaker, train_samples, eval_samples,
        settled, accuracy, log_loss, bets, profit, roi (live record), preferred, preferred_basis.
    """

    # The model index holds every league's models; it is loaded once.
    index = ModelDatabase(league_id='').index

    rows = []
    for league_id, models in index.items():
        for model_id, model_config in models.items():
            classifier_config = model_config.classifier_config
            model_cls = classifier_config.get('model_cls')
            classifier = model_cls.__name__ if hasattr(model_cls, '__name__') else str(model_cls).rsplit(':', 1)[-1]
            report = model_config.train_report
            rows.append({
                'league_id': league_id,
                'market': classifier_config['target_type'].value,
                'model_id': model_id,
                'classifier': classifier,
                'holdout_log_loss': _eval_metric(report, 'Log Loss'),
                'holdout_accuracy': _eval_metric(report, 'Accuracy'),
                'skill_vs_bookmaker': _eval_metric(report, 'Log Loss Skill'),
                'train_samples': model_config.num_train_samples,
                'eval_samples': model_config.num_eval_samples,
            })

    scores = pd.DataFrame(rows, columns=[
        'league_id', 'market', 'model_id', 'classifier', 'holdout_log_loss', 'holdout_accuracy',
        'skill_vs_bookmaker', 'train_samples', 'eval_samples'
    ])
    records = (ledger or PredictionLedger()).model_records()
    scores = scores.merge(records, on=['league_id', 'market', 'model_id'], how='left')
    scores['settled'] = scores['settled'].fillna(0).astype(int)

    preferred = preferred_models(scores)
    keys = list(zip(scores['league_id'], scores['market']))
    scores['preferred'] = [preferred.get(key, (None, None))[0] == model_id for key, model_id in zip(keys, scores['model_id'])]
    scores['preferred_basis'] = [preferred.get(key, (None, ''))[1] for key in keys]
    return scores.sort_values(by=['league_id', 'market', 'holdout_log_loss']).reset_index(drop=True)


def preferred_lookup(scores: pd.DataFrame) -> Dict[Tuple[str, str], str]:
    """ {(league_id, market): preferred model id} """

    preferred = scores[scores['preferred']]
    return {(row.league_id, row.market): row.model_id for row in preferred.itertuples()}
