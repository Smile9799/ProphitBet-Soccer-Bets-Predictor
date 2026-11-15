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
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.database.league import LeagueDatabase
from src.core.database.model import ModelDatabase
from src.core.preprocessing.filters import filter_matches
from src.core.preprocessing.utils.inputs import (
    construct_inputs_by_form,
    construct_inputs_by_match,
    latest_season_and_week,
    prepare_prediction_row,
)
from src.core.preprocessing.utils.normalization import NormalizerType
from src.core.preprocessing.utils.selection import train_test_split
from src.core.preprocessing.utils.targets import TargetType

logger = logging.getLogger('prophitbet.automation')

# Odds columns of the predicted label, per market (used to compute the expected value of a bet).
MARKET_ODDS = {
    TargetType.RESULT: {'H': '1', 'D': 'X', 'A': '2'},
    TargetType.RESULT_UNDER_OVER: {'O': 'Over 2.5', 'U': 'Under 2.5'},
}
FIXTURE_COLUMNS = ['Date', 'Home', 'Away', '1', 'X', '2', 'Over 2.5', 'Under 2.5', 'Fixture Home', 'Fixture Away', 'Source']


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


def _valid_odd(value: Any) -> Optional[float]:
    try:
        odd = float(value)
    except (TypeError, ValueError):
        return None
    return odd if math.isfinite(odd) and odd > 1.0 else None


class AutomationPipeline:
    """ Runs the refresh -> retrain -> predict steps of one or more leagues without the GUI.

        Attributes
        ----------
        :param league_database: (LeagueDatabase | None) league database. A new one is created if None.
    """

    def __init__(self, league_database: Optional[LeagueDatabase] = None):
        self._league_database = league_database or LeagueDatabase()
        self._football_data_fixtures = None

        # After repeated FootyStats failures (it rate-limits with 403), it is skipped for the rest of the run.
        self._footystats_failures = 0
        self._max_footystats_failures = 2

    @property
    def league_database(self) -> LeagueDatabase:
        return self._league_database

    def get_league_ids(self) -> List[str]:
        return self._league_database.get_league_ids()

    # ------------------------------------------------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------------------------------------------------

    def refresh(self, league_id: str, rebuild: bool = False) -> Tuple[pd.DataFrame, bool]:
        """ Downloads the latest matches of a league. Returns the league data and whether it was rebuilt.
            Leagues stored before new markets / statistics fixes were introduced are rebuilt automatically.

            Args
            ----
            :param league_id: (str) league identifier.
            :param rebuild: (bool) whether to re-download all seasons instead of only the latest one.
        """

        history_df = self._league_database.load_league(league_id=league_id)
        previous_rows = history_df.shape[0]

        if not rebuild and self._league_database._requires_rebuild(
                league_config=self._league_database.index[league_id],
                history_df=history_df
        ):
            logger.info('[%s] League data predates new markets/statistics fixes: rebuilding it.', league_id)
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
        return df, rebuild

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
            :param force: (bool) whether to always retrain and replace the saved models (e.g. after the league's
                          statistics were recomputed, which makes the saved models' inputs outdated).
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
    # Fixtures
    # ------------------------------------------------------------------------------------------------------------

    def _footystats_fixtures(self, league, today: date, last_day: date) -> pd.DataFrame:
        """ Scrapes the league's fixtures from FootyStats (team names are not yet matched to the league's names). """

        from src.core.network.fixtures.dates import add_years_to_dates
        from src.core.network.fixtures.footystats import FootyStatsScraper

        scraper = FootyStatsScraper()
        tree = scraper.scrape(fixture_url=league.fixture_url)
        date_labels = scraper.get_available_table_dates(tree=tree) or []

        frames = []
        for table_index, label in enumerate(add_years_to_dates(date_items=date_labels, current_date=today)):
            match_date = pd.to_datetime(label, errors='coerce')
            if pd.isna(match_date) or not (today <= match_date.date() <= last_day):
                continue

            fixture_df = scraper.parse_league_table(tree=tree, table_index=table_index)
            if fixture_df is not None and not fixture_df.empty:
                frames.append(fixture_df[['Home', 'Away', '1', 'X', '2']].assign(Date=match_date.strftime('%Y-%m-%d')))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['Date', 'Home', 'Away', '1', 'X', '2'])

    def _football_data_fixtures_df(self, league, today: date, last_day: date) -> pd.DataFrame:
        from src.core.network.fixtures.footballdata import FootballDataFixtures

        if self._football_data_fixtures is None:
            self._football_data_fixtures = FootballDataFixtures()

        df = self._football_data_fixtures.get_fixtures(league=league)
        dates = pd.to_datetime(df['Date'], errors='coerce').dt.date
        return df[(dates >= today) & (dates <= last_day)].reset_index(drop=True)

    def get_fixtures(self, league_id: str, days: int = 7, today: Optional[date] = None) -> pd.DataFrame:
        """ Returns the league's upcoming fixtures of the next N days, with team names matched to the league's names.
            FootyStats is used first; football-data.co.uk is used as fallback when FootyStats fails or has no fixtures,
            and to fill missing odds (and add Over/Under 2.5 odds) of FootyStats matches.
        """

        from src.core.preprocessing.utils.fixtures import match_fixture_teams

        today = today or date.today()
        last_day = today + timedelta(days=days)
        league = self._league_database.index[league_id].league
        league_df = self._league_database.load_league(league_id=league_id)

        footystats_df = pd.DataFrame(columns=['Date', 'Home', 'Away', '1', 'X', '2'])
        if self._footystats_failures < self._max_footystats_failures:
            try:
                footystats_df = self._footystats_fixtures(league=league, today=today, last_day=last_day)
                self._footystats_failures = 0
            except Exception as error:
                self._footystats_failures += 1
                logger.warning('[%s] FootyStats fixtures failed (%s).', league_id, error)
                if self._footystats_failures == self._max_footystats_failures:
                    logger.warning('FootyStats failed repeatedly: using football-data.co.uk only for the rest of the run.')

        try:
            football_data_df = self._football_data_fixtures_df(league=league, today=today, last_day=last_day)
        except Exception as error:
            logger.warning('[%s] football-data.co.uk fixtures failed (%s).', league_id, error)
            football_data_df = pd.DataFrame(columns=['Date', 'Home', 'Away', '1', 'X', '2', 'Over 2.5', 'Under 2.5'])

        # FootyStats team names are fuzzy-matched to the league's names (football-data.co.uk already uses them).
        footystats_df = footystats_df.dropna(subset=['Home', 'Away'])
        footystats_df = footystats_df[footystats_df['Home'].astype(str).str.strip() != ''].reset_index(drop=True)
        if not footystats_df.empty:
            footystats_df['Fixture Home'] = footystats_df['Home']
            footystats_df['Fixture Away'] = footystats_df['Away']
            footystats_df = match_fixture_teams(league_df=league_df, fixture_df=footystats_df.copy())
            footystats_df['Source'] = 'FootyStats'
            for odd in ['1', 'X', '2']:
                footystats_df[odd] = footystats_df[odd].map(_valid_odd)

        if not football_data_df.empty:
            football_data_df = football_data_df.assign(
                **{'Fixture Home': football_data_df['Home'], 'Fixture Away': football_data_df['Away'], 'Source': 'football-data'}
            )

        if footystats_df.empty:
            fixtures_df = football_data_df
        elif football_data_df.empty:
            fixtures_df = footystats_df.assign(**{'Over 2.5': np.nan, 'Under 2.5': np.nan})
        else:
            # Filling missing FootyStats odds and adding U/O odds from football-data.co.uk (matched by teams).
            odds_df = football_data_df.drop_duplicates(subset=['Home', 'Away']).set_index(['Home', 'Away'])
            keys = pd.MultiIndex.from_frame(footystats_df[['Home', 'Away']])
            for odd in ['1', 'X', '2', 'Over 2.5', 'Under 2.5']:
                fallback = odds_df[odd].reindex(keys).to_numpy()
                current = footystats_df[odd].to_numpy(dtype=float) if odd in footystats_df else np.full(len(keys), np.nan)
                footystats_df[odd] = np.where(np.isnan(current), fallback, current)
            has_fallback = keys.isin(odds_df.index)
            footystats_df.loc[has_fallback, 'Source'] = 'FootyStats + football-data odds'

            # Matches published by football-data.co.uk only.
            known = set(zip(footystats_df['Home'], footystats_df['Away']))
            extra_df = football_data_df[[pair not in known for pair in zip(football_data_df['Home'], football_data_df['Away'])]]
            fixtures_df = pd.concat([footystats_df, extra_df], ignore_index=True)

        if fixtures_df.empty:
            return pd.DataFrame(columns=FIXTURE_COLUMNS)
        return fixtures_df[FIXTURE_COLUMNS].sort_values(by='Date', kind='stable').reset_index(drop=True)

    # ------------------------------------------------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------------------------------------------------

    def _build_prediction_inputs(
            self,
            league_id: str,
            fixtures_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ Builds model inputs for the fixtures. Returns (inputs, match info, skipped matches).
            Inputs are computed from the teams' current form; matches whose teams have not played enough matches in
            the current season fall back to the original method (the teams' last meeting at the same venue).
        """

        league_config = self._league_database.index[league_id]
        source_df = self._league_database.load_league(league_id=league_id)
        league_df = source_df.dropna(ignore_index=True)
        season, week = latest_season_and_week(league_df=source_df)

        skipped_rows = []
        valid_ids = []
        for index, fixture in fixtures_df.iterrows():
            info = fixture[['Date', 'Home', 'Away', 'Fixture Home', 'Fixture Away', 'Source']].to_dict()
            if fixture['Home'] == fixture['Away']:
                skipped_rows.append({**info, 'Reason': 'Team names matched to the same team.'})
            elif any(_valid_odd(fixture[odd]) is None for odd in ('1', 'X', '2')):
                skipped_rows.append({**info, 'Reason': 'Odds are not available yet.'})
            else:
                valid_ids.append(index)

        matches_df = fixtures_df.loc[valid_ids].reset_index(drop=True)
        form_df = construct_inputs_by_form(league_df=source_df, matches_df=matches_df, league_config=league_config) \
            if not matches_df.empty else pd.DataFrame(columns=source_df.columns)
        feature_columns = [column for column in league_config.stats_columns if column in form_df.columns]

        input_rows, match_rows = [], []
        for index, match in matches_df.iterrows():
            form_row = form_df.loc[index]
            if not form_row[feature_columns].isna().any():
                input_rows.append(form_row)
                match_rows.append({**match.to_dict(), 'Inputs': 'Current form'})
                continue

            last_meeting = construct_inputs_by_match(
                league_df=league_df,
                match_df=pd.Series([match['Home'], match['Away'], match['1'], match['X'], match['2']], index=['Home', 'Away', '1', 'X', '2'])
            )
            if last_meeting is None:
                skipped_rows.append({
                    **match[['Date', 'Home', 'Away', 'Fixture Home', 'Fixture Away', 'Source']].to_dict(),
                    'Reason': 'Not enough matches this season, and no previous home meeting between these teams.'
                })
                continue

            input_rows.append(prepare_prediction_row(row=last_meeting, match_date=match['Date'], season=season, week=week))
            match_rows.append({**match.to_dict(), 'Inputs': 'Last meeting (early season)'})

        inputs_df = pd.DataFrame(input_rows).reset_index(drop=True)
        if not inputs_df.empty:
            inputs_df = inputs_df.astype(league_df.dtypes.to_dict())
        info_df = pd.DataFrame(match_rows)
        skipped_df = pd.DataFrame(
            skipped_rows, columns=['Date', 'Home', 'Away', 'Fixture Home', 'Fixture Away', 'Source', 'Reason']
        )
        return inputs_df, info_df, skipped_df

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
        logger.info('[%s] Fetching fixtures for %s .. %s', league_id, today, today + timedelta(days=days))
        fixtures_df = self.get_fixtures(league_id=league_id, days=days, today=today)
        inputs_df, matches_df, skipped_df = self._build_prediction_inputs(league_id=league_id, fixtures_df=fixtures_df)
        sources = ', '.join(f'{source}: {count}' for source, count in fixtures_df['Source'].value_counts().items()) or 'none'
        logger.info(
            '[%s] Fixtures: %d matches to predict, %d skipped (sources: %s).',
            league_id, inputs_df.shape[0], skipped_df.shape[0], sources
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
            leading_columns = [
                'League', 'Date', 'Home', 'Away', '1', 'X', '2', 'Over 2.5', 'Under 2.5',
                'Market', 'Model', 'Prediction', 'Confidence', 'Odd', 'EV', 'Holdout Log Loss', 'Skill vs Bookmaker'
            ]
            trailing_columns = ['Passes Filters', 'Inputs', 'Source', 'Fixture Home', 'Fixture Away']
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
        """ Generates the predictions of one model, the expected value of betting on the prediction (when the market's
            odds are known), and flags matches passing the model's saved evaluation filters.
        """

        target_type: TargetType = model_config.classifier_config['target_type']
        labels = np.asarray(list(target_type.get_mapper().keys()))
        probabilities = np.asarray(model.predict_proba(df=inputs_df), dtype=float)
        prediction_ids = np.argmax(probabilities, axis=1)
        predictions = labels[prediction_ids]
        confidence = probabilities.max(axis=1)

        result_df = matches_df[['Date', 'Home', 'Away', '1', 'X', '2', 'Over 2.5', 'Under 2.5']].copy()
        result_df['Market'] = target_type.value
        result_df['Model'] = model_id
        result_df['Prediction'] = predictions
        result_df['Confidence'] = confidence.round(3)

        # Expected value (profit per unit stake) = probability * odd - 1, for markets with known odds.
        odds_columns = MARKET_ODDS.get(target_type)
        if odds_columns:
            odds = np.array([
                pd.to_numeric(matches_df.iloc[i][odds_columns[label]], errors='coerce')
                for i, label in enumerate(predictions)
            ], dtype=float)
        else:
            odds = np.full(predictions.shape[0], np.nan)
        result_df['Odd'] = odds
        result_df['EV'] = (confidence*odds - 1.0).round(3)

        # Hold-out quality of the model. "Skill vs Bookmaker" > 0 means its probabilities beat the bookmaker's on the
        # latest matches (available for the Result market). EV is only meaningful for such models.
        report = model_config.train_report
        eval_report = report[report['Data'] == 'Eval'] if 'Data' in report.columns else report
        result_df['Holdout Log Loss'] = float(eval_report['Log Loss'].iloc[0]) if not eval_report.empty else np.nan
        result_df['Skill vs Bookmaker'] = (
            float(eval_report['Log Loss Skill'].iloc[0])
            if 'Log Loss Skill' in eval_report.columns and not eval_report.empty else np.nan
        )

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

        for column in ['Inputs', 'Source', 'Fixture Home', 'Fixture Away']:
            result_df[column] = matches_df[column]
        return result_df


def export_predictions(runs: List[PredictionRun], output_directory: str, run_date: Optional[datetime] = None) -> List[str]:
    """ Exports predictions to CSV files (one per league) and a single Excel workbook with the sheets: Predictions,
        Value Bets (positive expected value from models that beat the bookmaker on their hold-out matches), Skipped.
        Returns the written paths.
    """

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
                predictions_df = pd.concat(all_predictions, ignore_index=True)
                predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
                value_mask = (predictions_df['EV'] > 0) & (predictions_df['Skill vs Bookmaker'] > 0)
                value_df = predictions_df[value_mask].sort_values(by='EV', ascending=False)
                value_df.to_excel(writer, sheet_name='Value Bets', index=False)
            if all_skipped:
                pd.concat(all_skipped, ignore_index=True).to_excel(writer, sheet_name='Skipped', index=False)
        written.append(workbook_filepath)
    return written
