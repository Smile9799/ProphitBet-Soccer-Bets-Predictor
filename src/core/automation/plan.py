""" Automation plan: a JSON file describing the leagues and models to maintain. Running a plan creates any missing
    league or model (optionally with hyperparameter tuning), so a scheduled run can build everything unattended.

    See automation_plan.example.json for the format.
"""

import inspect
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from src.core.database.league import LeagueDatabase
from src.core.database.model import ModelDatabase
from src.core.leagues.league import League, LeagueConfig
from src.core.preprocessing.utils.targets import TargetType

logger = logging.getLogger('prophitbet.automation')

SUPPORTED_CLASSIFIERS = [
    'NaiveBayes', 'RandomForest', 'KNearestNeighbors', 'DiscriminantAnalysis', 'LogisticRegressor',
    'SupportVectorMachine', 'DecisionTree', 'XGBoost', 'LightGBM'
]

_LEAGUE_DEFAULTS = {
    'start_year': 2015,
    'match_history_window': 3,
    'goal_diff_margin': 2,
    'aggregation': 'sum',
    'stats_columns': 'all',
    'odd_1_mask_range': None,
    'odd_x_mask_range': None,
    'odd_2_mask_range': None
}
_MODEL_DEFAULTS = {
    'test_size': 400,
    'normalizer': 'Standard',
    'sampler': None,
    'calibrate_probabilities': True,
    'params': {},
    'tune_trials': 0,
    'tune_metric': 'Log Loss',
    'tune_algorithm': 'tpe',
    'seed': 0
}


@dataclass
class ModelSpec:
    model_id: str
    classifier: str
    target: TargetType
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LeagueSpec:
    league: League
    league_id: str
    options: Dict[str, Any]
    model_specs: List[Dict[str, Any]]


@dataclass
class Plan:
    leagues: List[LeagueSpec]
    predict_days: int = 7
    retrain_tolerance: float = 0.0

    @property
    def league_ids(self) -> List[str]:
        return [league_spec.league_id for league_spec in self.leagues]


def _sanitize(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9.]+', '-', text).strip('-')


def load_plan(filepath: str, league_database: LeagueDatabase) -> Plan:
    """ Loads and validates an automation plan file. """

    with open(filepath, mode='r', encoding='utf-8') as jsonfile:
        config = json.load(jsonfile)

    league_defaults = {**_LEAGUE_DEFAULTS, **config.get('league_defaults', {})}
    model_defaults = {**_MODEL_DEFAULTS, **config.get('model_defaults', {})}
    leagues_by_name = {(league.country.casefold(), league.name.casefold()): league for league in league_database.leagues}

    league_specs = []
    for entry in config.get('leagues', []):
        key = (str(entry.get('country', '')).casefold(), str(entry.get('name', '')).casefold())
        league = leagues_by_name.get(key)

        if league is None:
            available = sorted(f'{league.country}/{league.name}' for league in league_database.leagues)
            raise ValueError(f'Unknown league "{entry.get("country")}/{entry.get("name")}". Available: {available}')

        options = {**league_defaults, **{k: v for k, v in entry.items() if k in _LEAGUE_DEFAULTS}}
        league_id = entry.get('league_id') or f'{league.country}-{league.name}-01'
        model_specs = [{**model_defaults, **model_entry} for model_entry in entry.get('models', [])]

        for model_spec in model_specs:
            if model_spec.get('classifier') not in SUPPORTED_CLASSIFIERS:
                raise ValueError(f'Unknown classifier "{model_spec.get("classifier")}". Supported: {SUPPORTED_CLASSIFIERS}')

        league_specs.append(LeagueSpec(league=league, league_id=league_id, options=options, model_specs=model_specs))

    return Plan(
        leagues=league_specs,
        predict_days=int(config.get('predict_days', 7)),
        retrain_tolerance=float(config.get('retrain_tolerance', 0.0))
    )


class PlanExecutor:
    """ Creates the leagues and models of a plan that do not exist yet. """

    def __init__(self, league_database: LeagueDatabase):
        self._league_database = league_database

    def ensure_league(self, league_spec: LeagueSpec) -> bool:
        """ Creates the league if it does not exist. Returns whether it was created. """

        if self._league_database.league_exists(league_id=league_spec.league_id):
            return False

        from src.core.preprocessing.filters import filter_league_odds
        from src.core.preprocessing.statistics import StatisticsEngine

        options = league_spec.options
        stats_columns = options['stats_columns']
        if stats_columns == 'all':
            stats_columns = StatisticsEngine().stats_columns

        def as_range(value) -> Optional[tuple]:
            return None if value is None else (float(value[0]), float(value[1]))

        league_config = LeagueConfig(
            league=league_spec.league,
            league_id=league_spec.league_id,
            start_year=max(int(options['start_year']), league_spec.league.min_year),
            stats_columns=list(stats_columns),
            match_history_window=int(options['match_history_window']),
            goal_diff_margin=int(options['goal_diff_margin']),
            aggregation=options['aggregation'],
            odd_1_mask_range=as_range(options['odd_1_mask_range']),
            odd_x_mask_range=as_range(options['odd_x_mask_range']),
            odd_2_mask_range=as_range(options['odd_2_mask_range'])
        )

        logger.info('[%s] Creating league (%s from %d)...', league_spec.league_id, league_spec.league, league_config.start_year)
        df = self._league_database.download_league(league_config=league_config)
        filtered_df = filter_league_odds(
            df=df,
            odd_1_mask_range=league_config.odd_1_mask_range,
            odd_x_mask_range=league_config.odd_x_mask_range,
            odd_2_mask_range=league_config.odd_2_mask_range,
            validate_dataframe=True
        )
        if filtered_df is None:
            raise ValueError(f'League "{league_spec.league_id}" has no valid matches after applying the odd filters.')

        self._league_database.save_league(df=filtered_df, league_config=league_config)
        logger.info('[%s] Created with %d matches.', league_spec.league_id, filtered_df.shape[0])
        return True

    @staticmethod
    def expand_model_specs(league_spec: LeagueSpec, available_targets: List[TargetType]) -> List[ModelSpec]:
        """ Expands each plan model entry into one model per target (market). """

        model_specs = []
        for entry in league_spec.model_specs:
            targets = entry.get('targets', entry.get('target', TargetType.RESULT.value))
            if targets == 'all':
                targets = [target.value for target in available_targets]
            elif isinstance(targets, str):
                targets = [targets]

            for target_value in targets:
                target = TargetType(target_value)
                model_id = entry.get('model_id') if len(targets) == 1 and entry.get('model_id') else \
                    _sanitize(f'{target.value}-{entry["classifier"]}')
                model_specs.append(ModelSpec(model_id=model_id, classifier=entry['classifier'], target=target, options=entry))
        return model_specs

    def ensure_models(self, league_spec: LeagueSpec) -> List[str]:
        """ Trains the league's plan models that do not exist yet. Returns the created model ids. """

        import src.core.models.classifiers as classifiers
        from src.core.models.tuner import OptimizationMetric, TuningAlgorithm
        from src.core.models.utils import create_model
        from src.core.preprocessing.utils.normalization import NormalizerType
        from src.core.preprocessing.utils.sampling import SamplerType

        league_id = league_spec.league_id
        df = self._league_database.load_league(league_id=league_id)
        valid_df = df.dropna().sort_values(by='Date', ascending=False).reset_index(drop=True)
        available_targets = TargetType.available(valid_df)
        model_database = ModelDatabase(league_id=league_id)

        created = []
        for spec in self.expand_model_specs(league_spec=league_spec, available_targets=available_targets):
            if model_database.model_exists(model_id=spec.model_id):
                continue
            if spec.target not in available_targets:
                logger.warning('[%s] Skipping "%s": market "%s" is not available.', league_id, spec.model_id, spec.target.value)
                continue

            options = spec.options
            model_cls = getattr(classifiers, spec.classifier)
            accepted = inspect.signature(model_cls.__init__).parameters
            model_params: Dict[str, Any] = {
                'target_type': spec.target,
                'calibrate_probabilities': bool(options['calibrate_probabilities']),
                **options.get('params', {})
            }
            if 'normalizer' in accepted and options.get('normalizer'):
                model_params['normalizer'] = NormalizerType(options['normalizer'])
            if options.get('sampler'):
                model_params['sampler'] = SamplerType(options['sampler'])

            trials = int(options.get('tune_trials') or 0)
            tunable_params = {}
            if trials > 0:
                fixed = set(options.get('params', {})) | {'target_type'}
                tunable_params = {
                    name: values for name, values in model_cls.get_suggested_hyperparameters().items() if name not in fixed
                }
                for name in tunable_params:
                    model_params.pop(name, None)

            test_size: Union[int, float] = options['test_size']
            logger.info(
                '[%s] Training "%s" (%s, %s)%s...', league_id, spec.model_id, spec.classifier, spec.target.value,
                f' with {trials} tuning trials' if trials else ''
            )
            try:
                model_config, classifier = create_model(
                    league_id=league_id,
                    model_id=spec.model_id,
                    df=valid_df,
                    test_size=test_size,
                    model_cls=model_cls,
                    model_params=model_params,
                    tunable_params=tunable_params,
                    trials=max(trials, 1),
                    tune_algorithm=TuningAlgorithm(options['tune_algorithm']),
                    tune_metric=next(m for m in OptimizationMetric if m.value[0] == options['tune_metric']),
                    seed=int(options['seed']),
                    open_optuna_dashboard=False
                )
                model_database.save_model(model_config=model_config, classifier=classifier)
                report = model_config.train_report
                eval_row = report[report['Data'] == 'Eval'].iloc[0]
                logger.info(
                    '[%s] Saved "%s": hold-out accuracy %.3f, log loss %.4f.',
                    league_id, spec.model_id, eval_row['Accuracy'], eval_row['Log Loss']
                )
                created.append(spec.model_id)
            except Exception:
                logger.exception('[%s] Training "%s" failed.', league_id, spec.model_id)
        return created
