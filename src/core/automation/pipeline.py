""" Headless automation pipeline: refresh league data, retrain saved models and predict upcoming fixtures.

    It reuses the same core components as the GUI, so automated predictions match the GUI's predictions.
"""

import logging
import math
import os
import pickle
import shutil
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.database.league import LeagueDatabase
from src.core.database.model import ModelDatabase
from src.core.preprocessing.filters import filter_matches
from src.core.preprocessing.utils.inputs import (
    construct_inputs_by_match,
    latest_season_and_week,
    prepare_prediction_row,
)
from src.core.preprocessing.utils.normalization import NormalizerType
from src.core.preprocessing.utils.selection import train_test_split
from src.core.preprocessing.utils.targets import TargetType

logger = logging.getLogger('prophitbet.automation')


@dataclass(frozen=True)
class RetrainResult:
    """ Outcome of retraining a single saved model. Status is one of: replaced, kept, skipped, failed. """

    league_id: str
    model_id: str
    target: str
    status: str
    old_log_loss: Optional[float] = None
    new_log_loss: Optional[float] = None
    message: str = ''


@dataclass
class PredictionRun:
    """ Predictions of all models for the upcoming fixtures of a league, and the matches that were skipped. """

    league_id: str
    predictions: pd.DataFrame
    skipped: pd.DataFrame


def _normalizer_type(normalizer: Any) -> Any:
    """ Converts a fitted normalizer (stored in the model config after training) back to its unfitted type. """

    if normalizer is None or isinstance(normalizer, (str, NormalizerType)):
        return normalizer

    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

    mapping = {MinMaxScaler: NormalizerType.MIN_MAX, MaxAbsScaler: NormalizerType.MAX_ABS, StandardScaler: NormalizerType.STANDARD}
    for normalizer_cls, normalizer_type in mapping.items():
        if type(normalizer) is normalizer_cls:
            return normalizer_type
    raise ValueError(f'Unsupported fitted normalizer: "{type(normalizer).__name__}".')


def _eval_log_loss(report: pd.DataFrame) -> float:
    """ Extracts the evaluation (hold-out) log loss from a fit/evaluation report. """

    if 'Data' in report.columns:
        report = report[report['Data'] == 'Eval']
    return float(report['Log Loss'].iloc[0])


class AutomationPipeline:
    """ Runs the refresh -> retrain -> predict steps of one or more leagues without the GUI.

        Attributes
        ----------
        :param league_database: (LeagueDatabase | None) league database. A new one is created if None.
    """

    def __init__(self, league_database: Optional[LeagueDatabase] = None):
        self._league_database = league_database or LeagueDatabase()

    @property
    def league_database(self) -> LeagueDatabase:
        return self._league_database

    def get_league_ids(self) -> List[str]:
        return self._league_database.get_league_ids()

    # ------------------------------------------------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------------------------------------------------

    def refresh(self, league_id: str, rebuild: bool = False) -> pd.DataFrame:
        """ Downloads the latest matches of a league. Leagues stored before new markets were introduced are
            rebuilt automatically.

            Args
            ----
            :param league_id: (str) league identifier.
            :param rebuild: (bool) whether to re-download all seasons instead of only the latest one.
        """

        previous_rows = self._league_database.load_league(league_id=league_id).shape[0]

        if not rebuild and self._league_database._requires_rebuild(
                league_config=self._league_database.index[league_id],
                history_df=self._league_database.load_league(league_id=league_id)
        ):
            logger.info('[%s] League was created before the new markets were added: rebuilding it.', league_id)
            rebuild = True

        if rebuild:
            logger.info('[%s] Rebuilding league (re-downloading all seasons)...', league_id)
            df = self._league_database.rebuild_league(league_id=league_id)
        else:
            logger.info('[%s] Updating league data...', league_id)
            df = self._league_database.update_league(league_id=league_id)

        markets = ', '.join(target.value for target in TargetType.available(df))
        logger.info(
            '[%s] Matches: %d -> %d. Last match: %s. Markets: %s',
            league_id, previous_rows, df.shape[0], df['Date'].max(), markets
        )
        return df

    # ------------------------------------------------------------------------------------------------------------
    # Retrain
    # ------------------------------------------------------------------------------------------------------------

    def _backup_model(self, model_database: ModelDatabase, model_id: str, model_config: Any):
        """ Stores a copy of the current model (files & config) before it is replaced. Only the latest backup is kept. """

        model_directory = model_database.build_model_directory(model_id=model_id)
        backup_directory = f'{self._league_database._build_league_directory(model_database.league_id)}/model_backups/{model_id}'
        shutil.rmtree(backup_directory, ignore_errors=True)
        shutil.copytree(model_directory, backup_directory)

        with open(f'{backup_directory}/model_config.pkl', mode='wb') as pklfile:
            pickle.dump(model_config, pklfile)

    def retrain(
            self,
            league_id: str,
            model_ids: Optional[List[str]] = None,
            tolerance: float = 0.0,
            force: bool = False,
            seed: int = 0
    ) -> List[RetrainResult]:
        """ Retrains saved models on the latest league data using their stored hyperparameters and hold-out size.
            A retrained model replaces the saved one only if its hold-out log loss is not worse than the saved model's
            log loss on the same (most recent) hold-out matches, plus the specified tolerance.

            Args
            ----
            :param league_id: (str) league identifier.
            :param model_ids: (List[str] | None) models to retrain. Defaults to all models of the league.
            :param tolerance: (float) allowed log loss increase for the retrained model to still be accepted.
            :param force: (bool) whether to always replace the saved model.
            :param seed: (int) random seed.
        """

        df = self._league_database.load_league(league_id=league_id)
        valid_df = df.dropna().sort_values(by='Date', ascending=False).reset_index(drop=True)
        model_database = ModelDatabase(league_id=league_id)
        selected_ids = model_ids or model_database.get_model_ids()

        results = []
        for model_id in selected_ids:
            if not model_database.model_exists(model_id=model_id):
                results.append(RetrainResult(league_id, model_id, '', 'failed', message='Model does not exist.'))
                continue

            target_value = ''
            try:
                old_model, old_config = model_database.load_model(model_id=model_id)
                target_type = old_config.classifier_config['target_type']
                target_value = target_type.value

                if not target_type.is_available(valid_df):
                    results.append(RetrainResult(
                        league_id, model_id, target_value, 'skipped',
                        message=f'Market "{target_value}" is not available in the league data.'
                    ))
                    continue

                eval_size = int(old_config.num_eval_samples)
                if eval_size <= 0:
                    results.append(RetrainResult(
                        league_id, model_id, target_value, 'skipped', message='Model has no hold-out set to compare.'
                    ))
                    continue

                if not force and valid_df.shape[0] <= old_config.num_train_samples + eval_size:
                    results.append(RetrainResult(
                        league_id, model_id, target_value, 'skipped', message='No new matches since the model was trained.'
                    ))
                    continue

                train_df, eval_df = train_test_split(df=valid_df, test_size=eval_size)

                # Reconstructing the model with its stored (possibly tuned) hyperparameters and a fresh normalizer.
                params = dict(old_config.classifier_config)
                params.pop('model_cls')
                params['normalizer'] = _normalizer_type(params['normalizer'])
                new_model = type(old_model)(**params)

                logger.info(
                    '[%s] Retraining "%s" (%s) on %d matches, holding out the latest %d...',
                    league_id, model_id, target_value, train_df.shape[0], eval_df.shape[0]
                )
                train_report = new_model.fit(train_df=train_df, eval_df=eval_df, seed=seed)
                new_log_loss = _eval_log_loss(train_report)

                old_report, _ = old_model.evaluate(df=eval_df)
                old_log_loss = _eval_log_loss(old_report)

                accepted = force or new_log_loss <= old_log_loss + tolerance
                if not accepted:
                    results.append(RetrainResult(
                        league_id, model_id, target_value, 'kept', old_log_loss, new_log_loss,
                        'Retrained model was worse on the latest matches; saved model kept.'
                    ))
                    logger.info(
                        '[%s] "%s": kept saved model (log loss saved %.4f vs retrained %.4f).',
                        league_id, model_id, old_log_loss, new_log_loss
                    )
                    continue

                self._backup_model(model_database=model_database, model_id=model_id, model_config=old_config)
                new_config = replace(
                    old_config,
                    classifier_config=new_model.model_config,
                    train_report=train_report,
                    num_train_samples=train_df.shape[0],
                    num_eval_samples=eval_df.shape[0]
                )
                model_database.save_model(model_config=new_config, classifier=new_model)

                # Cached SHAP explanations belong to the replaced model.
                shap_filepath = f'{model_database.build_model_directory(model_id=model_id)}/shap.pkl'
                if os.path.exists(shap_filepath):
                    os.remove(shap_filepath)

                results.append(RetrainResult(
                    league_id, model_id, target_value, 'replaced', old_log_loss, new_log_loss,
                    'Forced replacement.' if force and new_log_loss > old_log_loss + tolerance else ''
                ))
                logger.info(
                    '[%s] "%s": replaced (log loss saved %.4f -> retrained %.4f).',
                    league_id, model_id, old_log_loss, new_log_loss
                )
            except Exception as error:
                logger.exception('[%s] Retraining "%s" failed.', league_id, model_id)
                results.append(RetrainResult(league_id, model_id, target_value, 'failed', message=str(error)))
        return results

    # ------------------------------------------------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------------------------------------------------

    def _upcoming_fixtures(
            self,
            fixture_url: str,
            today: date,
            days: int
    ) -> List[Tuple[str, pd.DataFrame]]:
        """ Scrapes the fixtures of the next N days. Returns a list of (match date, fixture dataframe). """

        from src.core.network.fixtures.dates import add_years_to_dates
        from src.core.network.fixtures.footystats import FootyStatsScraper

        scraper = FootyStatsScraper()
        tree = scraper.scrape(fixture_url=fixture_url)
        date_labels = scraper.get_available_table_dates(tree=tree)

        if not date_labels:
            return []

        last_day = today + timedelta(days=days)
        fixtures = []
        for table_index, label in enumerate(add_years_to_dates(date_items=date_labels, current_date=today)):
            match_date = pd.to_datetime(label, errors='coerce')

            if pd.isna(match_date) or not (today <= match_date.date() <= last_day):
                continue

            fixture_df = scraper.parse_league_table(tree=tree, table_index=table_index)
            if fixture_df is not None and not fixture_df.empty:
                fixtures.append((match_date.strftime('%Y-%m-%d'), fixture_df))
        return fixtures

    @staticmethod
    def _valid_odd(value: Any) -> Optional[float]:
        try:
            odd = float(value)
        except (TypeError, ValueError):
            return None
        return odd if math.isfinite(odd) and odd > 1.0 else None

    def _build_prediction_inputs(
            self,
            source_df: pd.DataFrame,
            fixtures: List[Tuple[str, pd.DataFrame]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ Builds model inputs for the scraped fixtures. Returns (inputs, match info, skipped matches). """

        from src.core.preprocessing.utils.fixtures import match_fixture_teams

        league_df = source_df.dropna(ignore_index=True)
        season, week = latest_season_and_week(league_df=source_df)

        input_rows, match_rows, skipped_rows = [], [], []
        for match_date, fixture_df in fixtures:
            fixture_df = fixture_df.dropna(subset=['Home', 'Away']).reset_index(drop=True)
            fixture_df = fixture_df[(fixture_df['Home'].astype(str).str.strip() != '')]
            original_names = fixture_df[['Home', 'Away']].copy()
            fixture_df = match_fixture_teams(league_df=source_df, fixture_df=fixture_df.copy())

            for index, fixture in fixture_df.iterrows():
                home, away = fixture['Home'], fixture['Away']
                fixture_home, fixture_away = original_names.loc[index, 'Home'], original_names.loc[index, 'Away']
                info = {
                    'Date': match_date, 'Home': home, 'Away': away,
                    'Fixture Home': fixture_home, 'Fixture Away': fixture_away
                }
                odds = [self._valid_odd(fixture[odd]) for odd in ('1', 'X', '2')]

                if home == away:
                    skipped_rows.append({**info, 'Reason': 'Team names matched to the same team.'})
                    continue
                if any(odd is None for odd in odds):
                    skipped_rows.append({**info, 'Reason': 'Odds are not available yet.'})
                    continue

                match = pd.Series([home, away, *odds], index=['Home', 'Away', '1', 'X', '2'])
                row = construct_inputs_by_match(league_df=league_df, match_df=match)
                if row is None:
                    skipped_rows.append({**info, 'Reason': 'No previous home match between these teams.'})
                    continue

                input_rows.append(prepare_prediction_row(row=row, match_date=match_date, season=season, week=week))
                match_rows.append({**info, '1': odds[0], 'X': odds[1], '2': odds[2]})

        inputs_df = pd.DataFrame(input_rows).reset_index(drop=True)
        matches_df = pd.DataFrame(match_rows)
        skipped_df = pd.DataFrame(skipped_rows, columns=['Date', 'Home', 'Away', 'Fixture Home', 'Fixture Away', 'Reason'])
        return inputs_df, matches_df, skipped_df

    def predict(
            self,
            league_id: str,
            days: int = 7,
            model_ids: Optional[List[str]] = None,
            today: Optional[date] = None
    ) -> PredictionRun:
        """ Predicts the upcoming fixtures (next N days) of a league with every saved model.

            Args
            ----
            :param league_id: (str) league identifier.
            :param days: (int) number of days ahead to predict.
            :param model_ids: (List[str] | None) models to use. Defaults to all models of the league.
            :param today: (date | None) reference date. Defaults to today.
        """

        today = today or date.today()
        league_config = self._league_database.index[league_id]
        source_df = self._league_database.load_league(league_id=league_id)

        logger.info('[%s] Fetching fixtures for %s .. %s', league_id, today, today + timedelta(days=days))
        fixtures = self._upcoming_fixtures(fixture_url=league_config.league.fixture_url, today=today, days=days)
        inputs_df, matches_df, skipped_df = self._build_prediction_inputs(source_df=source_df, fixtures=fixtures)
        logger.info(
            '[%s] Fixtures: %d matches to predict, %d skipped.',
            league_id, inputs_df.shape[0], skipped_df.shape[0]
        )

        model_database = ModelDatabase(league_id=league_id)
        selected_ids = model_ids or model_database.get_model_ids()
        prediction_frames = []

        if inputs_df.shape[0] > 0:
            for model_id in selected_ids:
                try:
                    model, model_config = model_database.load_model(model_id=model_id)
                    prediction_frames.append(self._predict_with_model(
                        model=model,
                        model_id=model_id,
                        model_config=model_config,
                        inputs_df=inputs_df,
                        matches_df=matches_df
                    ))
                except Exception:
                    logger.exception('[%s] Prediction with "%s" failed.', league_id, model_id)

        predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
        if not predictions_df.empty:
            predictions_df.insert(0, 'League', league_id)
            probability_columns = list(dict.fromkeys(
                f'Prob-{label}' for target in TargetType for label in target.get_mapper()
                if f'Prob-{label}' in predictions_df.columns
            ))
            leading_columns = ['League', 'Date', 'Home', 'Away', '1', 'X', '2', 'Market', 'Model', 'Prediction', 'Confidence']
            trailing_columns = ['Passes Filters', 'Fixture Home', 'Fixture Away']
            predictions_df = predictions_df[[*leading_columns, *probability_columns, *trailing_columns]]
        if not skipped_df.empty:
            skipped_df.insert(0, 'League', league_id)
        return PredictionRun(league_id=league_id, predictions=predictions_df, skipped=skipped_df)

    @staticmethod
    def _predict_with_model(
            model: Any,
            model_id: str,
            model_config: Any,
            inputs_df: pd.DataFrame,
            matches_df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Generates the predictions of one model, and flags matches passing the model's saved evaluation filters. """

        target_type: TargetType = model_config.classifier_config['target_type']
        labels = np.asarray(list(target_type.get_mapper().keys()))
        probabilities = np.asarray(model.predict_proba(df=inputs_df), dtype=float)
        prediction_ids = np.argmax(probabilities, axis=1)

        result_df = matches_df[['Date', 'Home', 'Away', '1', 'X', '2']].copy()
        result_df['Market'] = target_type.value
        result_df['Model'] = model_id
        result_df['Prediction'] = labels[prediction_ids]
        result_df['Confidence'] = probabilities.max(axis=1).round(3)
        for label, values in zip(labels, probabilities.T):
            result_df[f'Prob-{label}'] = values.round(3)

        # Saved evaluation filters (created in the GUI evaluation dialog). Double chance filters are not applied.
        eval_filters = [
            eval_filter for eval_filter in (getattr(model_config, 'eval_filters', None) or [])
            if not eval_filter.double_chance
        ]
        if eval_filters:
            filter_df = inputs_df.copy()
            for label, values in zip(labels, probabilities.T):
                filter_df[f'Prob-{label}'] = values
            filter_df['Prediction'] = prediction_ids
            passing_ids = filter_matches(matches_df=filter_df, eval_filters=eval_filters).index
            result_df['Passes Filters'] = np.where(result_df.index.isin(passing_ids), 'Y', 'N')
        else:
            result_df['Passes Filters'] = ''

        # Original FootyStats names, to verify the fuzzy team-name matching.
        result_df['Fixture Home'] = matches_df['Fixture Home']
        result_df['Fixture Away'] = matches_df['Fixture Away']
        return result_df


def export_predictions(runs: List[PredictionRun], output_directory: str, run_date: Optional[datetime] = None) -> List[str]:
    """ Exports predictions to CSV files (one per league) and a single Excel workbook. Returns the written paths. """

    run_date = run_date or datetime.now()
    directory = os.path.join(output_directory, run_date.strftime('%Y-%m-%d'))
    os.makedirs(directory, exist_ok=True)

    written = []
    for run in runs:
        if run.predictions.empty:
            continue
        filepath = os.path.join(directory, f'{run.league_id}.csv')
        run.predictions.to_csv(filepath, index=False)
        written.append(filepath)

    all_predictions = [run.predictions for run in runs if not run.predictions.empty]
    all_skipped = [run.skipped for run in runs if not run.skipped.empty]
    if all_predictions or all_skipped:
        workbook_filepath = os.path.join(directory, 'predictions.xlsx')
        with pd.ExcelWriter(workbook_filepath, engine='openpyxl') as writer:
            if all_predictions:
                pd.concat(all_predictions, ignore_index=True).to_excel(writer, sheet_name='Predictions', index=False)
            if all_skipped:
                pd.concat(all_skipped, ignore_index=True).to_excel(writer, sheet_name='Skipped', index=False)
        written.append(workbook_filepath)
    return written
