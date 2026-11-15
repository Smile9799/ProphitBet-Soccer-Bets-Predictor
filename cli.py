""" ProphitBet command line interface: create leagues, train models, refresh data, retrain models and predict fixtures
    without the GUI.

    Examples:
        python cli.py list
        python cli.py plan                       # create the leagues/models of the plan that do not exist yet
        python cli.py run --plan                 # plan -> refresh -> retrain -> predict (use this for scheduling)
        python cli.py run --all                  # refresh -> retrain -> predict of every existing league
        python cli.py refresh --league England-Premier-League-01 --rebuild
        python cli.py retrain --all --tolerance 0.005
        python cli.py predict --all --days 7

    The plan defaults to storage/configs/automation_plan.json (see automation_plan.example.json).
    Do not run it while the GUI is open: both keep their own copy of the model index and would overwrite each other.
"""

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIRECTORY = PROJECT_ROOT / 'storage' / 'logs'
PREDICTIONS_DIRECTORY = PROJECT_ROOT / 'storage' / 'predictions'
DEFAULT_PLAN_PATH = PROJECT_ROOT / 'storage' / 'configs' / 'automation_plan.json'
FIXTURE_REQUEST_DELAY_SECONDS = 5.0


def _configure_logging(verbose: bool) -> logging.Logger:
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s %(levelname)-7s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(LOG_DIRECTORY / 'automation.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger('prophitbet')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers = [file_handler, console_handler]
    logger.propagate = False
    return logger


@contextmanager
def _single_instance_lock():
    """ Prevents overlapping scheduled runs (no-op on platforms without fcntl). """

    try:
        import fcntl
    except ImportError:
        yield
        return

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIRECTORY / 'automation.lock', 'w') as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another ProphitBet automation run is in progress.')
        yield


def _load_plan(pipeline, args):
    from src.core.automation.plan import load_plan

    plan_path = Path(args.plan) if args.plan else DEFAULT_PLAN_PATH
    if not plan_path.is_file():
        raise SystemExit(
            f'Plan file not found: {plan_path}. Copy automation_plan.example.json to '
            f'storage/configs/automation_plan.json and edit it.'
        )
    return load_plan(filepath=str(plan_path), league_database=pipeline.league_database)


def _selected_leagues(pipeline, args, plan=None) -> list:
    league_ids = pipeline.get_league_ids()

    if getattr(args, 'all', False):
        return league_ids
    if plan is not None:
        return [league_id for league_id in plan.league_ids if league_id in league_ids]
    if not args.league:
        raise SystemExit('Specify --league <id> (repeatable) or --all. Use "python cli.py list" to see leagues.')

    unknown = [league_id for league_id in args.league if league_id not in league_ids]
    if unknown:
        raise SystemExit(f'Unknown league(s): {unknown}. Available: {league_ids}')
    return args.league


def _command_list(pipeline, _args, _logger) -> int:
    from src.core.database.model import ModelDatabase
    from src.core.preprocessing.utils.targets import TargetType

    for league_id in pipeline.get_league_ids():
        df = pipeline.league_database.load_league(league_id=league_id)
        markets = [target.value for target in TargetType.available(df)]
        print(f'{league_id}: {df.shape[0]} matches, last match {df["Date"].max()}')
        print(f'    markets: {", ".join(markets)}')

        model_database = ModelDatabase(league_id=league_id)
        for model_id in model_database.get_model_ids():
            model_config = model_database.index[league_id][model_id]
            target = model_config.classifier_config['target_type'].value
            print(f'    model: {model_id} [{target}] trained on {model_config.num_train_samples} matches')
    return 0


def _run_plan(pipeline, plan, logger) -> tuple:
    """ Creates the plan's missing leagues and models. Returns (ok, {league id: created model ids}). """

    from src.core.automation.plan import PlanExecutor

    executor = PlanExecutor(league_database=pipeline.league_database)
    ok = True
    created_models = {}
    for league_spec in plan.leagues:
        try:
            executor.ensure_league(league_spec=league_spec)
            created = executor.ensure_models(league_spec=league_spec)
            created_models[league_spec.league_id] = set(created)
            if created:
                logger.info('[%s] Created models: %s', league_spec.league_id, ', '.join(created))
        except Exception:
            logger.exception('[%s] Plan step failed.', league_spec.league_id)
            ok = False
    return ok, created_models


def _run_refresh(pipeline, league_ids, args, logger) -> tuple:
    ok = True
    rebuilt = set()
    for league_id in league_ids:
        try:
            _df, was_rebuilt = pipeline.refresh(league_id=league_id, rebuild=args.rebuild)
            if was_rebuilt:
                rebuilt.add(league_id)
        except Exception:
            logger.exception('[%s] Refresh failed.', league_id)
            ok = False
    return ok, rebuilt


def _run_retrain(pipeline, league_ids, args, logger, forced_leagues=frozenset(), new_models=None) -> bool:
    from src.core.database.model import ModelDatabase

    ok = True
    results = []
    for league_id in league_ids:
        force = args.force or league_id in forced_leagues
        if league_id in forced_leagues and not args.force:
            logger.info('[%s] League statistics were recomputed: retraining all of its models.', league_id)

        # Models created in this run are already trained on the latest data.
        model_ids = args.model or ModelDatabase(league_id=league_id).get_model_ids()
        model_ids = [model_id for model_id in model_ids if model_id not in (new_models or {}).get(league_id, set())]
        if not model_ids:
            continue

        results.extend(pipeline.retrain(
            league_id=league_id,
            model_ids=model_ids,
            tolerance=args.tolerance,
            force=force
        ))

    for result in results:
        if result.status == 'failed':
            ok = False
        losses = ''
        if result.old_log_loss is not None and result.new_log_loss is not None:
            losses = f' (log loss {result.old_log_loss:.4f} -> {result.new_log_loss:.4f})'
        logger.info(
            'Retrain summary: %s / %s [%s]: %s%s %s',
            result.league_id, result.model_id, result.target, result.status.upper(), losses, result.message
        )
    return ok


def _run_predict(pipeline, league_ids, args, logger) -> bool:
    import pandas as pd
    from src.core.automation.pipeline import export_predictions

    ok = True
    runs = []
    for position, league_id in enumerate(league_ids):
        if position > 0:
            time.sleep(FIXTURE_REQUEST_DELAY_SECONDS)  # be polite to FootyStats (it rate-limits with 403)
        try:
            runs.append(pipeline.predict(league_id=league_id, days=args.days, model_ids=args.model))
        except Exception:
            logger.exception('[%s] Prediction failed.', league_id)
            ok = False

    written = export_predictions(runs=runs, output_directory=args.output)
    predictions = [run.predictions for run in runs if not run.predictions.empty]

    if predictions:
        summary_columns = ['League', 'Date', 'Home', 'Away', 'Market', 'Model', 'Prediction', 'Confidence', 'EV', 'Passes Filters']
        summary_df = pd.concat(predictions, ignore_index=True)[summary_columns]
        with pd.option_context('display.max_rows', 500, 'display.width', 200, 'display.max_columns', 20):
            print(summary_df.to_string(index=False))
    else:
        logger.info('No predictions were generated (no upcoming fixtures with odds, or no saved models).')

    for filepath in written:
        logger.info('Saved: %s', filepath)
    return ok


def _command_plan(pipeline, args, logger) -> int:
    ok, _created = _run_plan(pipeline, _load_plan(pipeline, args), logger)
    return 0 if ok else 1


def _command_refresh(pipeline, args, logger) -> int:
    ok, _rebuilt = _run_refresh(pipeline, _selected_leagues(pipeline, args), args, logger)
    return 0 if ok else 1


def _command_retrain(pipeline, args, logger) -> int:
    return 0 if _run_retrain(pipeline, _selected_leagues(pipeline, args), args, logger) else 1


def _command_predict(pipeline, args, logger) -> int:
    return 0 if _run_predict(pipeline, _selected_leagues(pipeline, args), args, logger) else 1


def _command_run(pipeline, args, logger) -> int:
    ok = True
    plan = None

    if args.plan is not None:
        plan = _load_plan(pipeline, args)
        if args.days is None:
            args.days = plan.predict_days
        if args.tolerance is None:
            args.tolerance = plan.retrain_tolerance

    args.days = 7 if args.days is None else args.days
    args.tolerance = 0.0 if args.tolerance is None else args.tolerance

    # Existing leagues are refreshed first, so that models created by the plan train on up-to-date data.
    refresh_ok, rebuilt = _run_refresh(pipeline, _selected_leagues(pipeline, args, plan=plan), args, logger)
    ok = refresh_ok and ok

    new_models = {}
    if plan is not None:
        plan_ok, new_models = _run_plan(pipeline, plan, logger)
        ok = plan_ok and ok
    league_ids = _selected_leagues(pipeline, args, plan=plan)

    if not args.skip_retrain:
        ok = _run_retrain(pipeline, league_ids, args, logger, forced_leagues=rebuilt, new_models=new_models) and ok

    ok = _run_predict(pipeline, league_ids, args, logger) and ok
    logger.info('Run finished %s.', 'successfully' if ok else 'with errors (see log)')
    return 0 if ok else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='ProphitBet automation (create leagues/models, refresh, retrain, predict).')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    def add_league_args(subparser):
        subparser.add_argument('--league', action='append', help='League id (repeatable).')
        subparser.add_argument('--all', action='store_true', help='Use all created leagues.')

    def add_plan_arg(subparser, help_text):
        subparser.add_argument('--plan', nargs='?', const='', default=None, help=help_text)

    def add_refresh_args(subparser):
        subparser.add_argument('--rebuild', action='store_true', help='Re-download all seasons (not only the latest).')

    def add_retrain_args(subparser, default_tolerance):
        subparser.add_argument('--model', action='append', help='Model id to use (repeatable). Defaults to all.')
        subparser.add_argument(
            '--tolerance', type=float, default=default_tolerance,
            help='Accept a retrained model whose hold-out log loss is at most this much worse. Default: 0.'
        )
        subparser.add_argument('--force', action='store_true', help='Always replace saved models with retrained ones.')

    def add_predict_args(subparser, include_model: bool, default_days):
        subparser.add_argument('--days', type=int, default=default_days, help='Predict fixtures of the next N days. Default: 7.')
        subparser.add_argument('--output', default=str(PREDICTIONS_DIRECTORY), help='Output directory.')
        if include_model:
            subparser.add_argument('--model', action='append', help='Model id to use (repeatable). Defaults to all.')

    list_parser = subparsers.add_parser('list', help='List leagues, available markets and saved models.')
    list_parser.set_defaults(handler=_command_list)

    plan_parser = subparsers.add_parser('plan', help='Create the leagues and models of the plan that do not exist yet.')
    plan_parser.add_argument('--plan', default='', help=f'Plan file. Default: {DEFAULT_PLAN_PATH.relative_to(PROJECT_ROOT)}')
    plan_parser.set_defaults(handler=_command_plan)

    refresh_parser = subparsers.add_parser('refresh', help='Download the latest matches.')
    add_league_args(refresh_parser)
    add_refresh_args(refresh_parser)
    refresh_parser.set_defaults(handler=_command_refresh)

    retrain_parser = subparsers.add_parser('retrain', help='Retrain saved models on the latest data.')
    add_league_args(retrain_parser)
    add_retrain_args(retrain_parser, default_tolerance=0.0)
    retrain_parser.set_defaults(handler=_command_retrain)

    predict_parser = subparsers.add_parser('predict', help='Predict upcoming fixtures with saved models.')
    add_league_args(predict_parser)
    add_predict_args(predict_parser, include_model=True, default_days=7)
    predict_parser.set_defaults(handler=_command_predict)

    run_parser = subparsers.add_parser('run', help='(Plan ->) refresh -> retrain -> predict (for scheduled runs).')
    add_league_args(run_parser)
    add_plan_arg(run_parser, 'Run the plan first and process its leagues (optionally a plan file path).')
    add_refresh_args(run_parser)
    add_retrain_args(run_parser, default_tolerance=None)
    add_predict_args(run_parser, include_model=False, default_days=None)
    run_parser.add_argument('--skip-retrain', action='store_true', help='Only refresh data and predict.')
    run_parser.set_defaults(handler=_command_run)
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    # Storage paths in the configs are relative to the project root.
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    logger = _configure_logging(verbose=args.verbose)

    from src.core.automation.pipeline import AutomationPipeline

    with _single_instance_lock():
        return args.handler(AutomationPipeline(), args, logger)


if __name__ == '__main__':
    mp.freeze_support()
    sys.exit(main())
