"""
Master runner: full wind power forecasting pipeline for all four seasons.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    ALL_MODEL_LABELS,
    FORECAST_STEPS,
    HORIZONS,
    MODEL_KEYS,
    MODELS_SAVED_ROOT,
    PIPELINE_ERROR_LOG,
    RESULTS_ROOT,
    SEASON_DIFFERENTIATION,
    SEASON_EXOG_COLS,
    SEASONS,
)
from pipeline_errors import PipelineErrorLogger, run_step  # noqa: E402
from data.features import add_cyclical_features  # noqa: E402
from data.loader import extract_target_exog, load_and_preprocess  # noqa: E402
from data.splitter import chronological_split, print_split_sizes, split_by_season  # noqa: E402
from evaluation.backtesting import (  # noqa: E402
    clone_forecaster,
    differentiation_for_forecaster,
    evaluate_on_split,
    extract_oof_residuals,
    make_cv_test,
    make_cv_validation,
    run_backtesting,
    select_best_model,
)
from evaluation.metrics import aggregate_metrics  # noqa: E402
from evaluation.probabilistic import (  # noqa: E402
    build_interval_metrics,
    run_bootstrap_intervals,
    run_conformal_intervals,
    run_quantile_intervals,
    save_probabilistic_outputs,
)
from explainability.shap_analysis import run_shap_analysis  # noqa: E402
from models.forecasters import build_all_forecasters  # noqa: E402
from models.stacking import (  # noqa: E402
    StackingArtifacts,
    fit_stacking_meta_learner,
    predict_stacking,
    save_stacking_artifacts,
)
from models.tuning import tune_forecasters  # noqa: E402
from plots.visualizations import (  # noqa: E402
    generate_all_season_plots,
    plot_master_metrics_heatmap,
)


def _season_paths(season: str) -> dict[str, Path]:
    return {
        "results": RESULTS_ROOT / season,
        "models": MODELS_SAVED_ROOT / season,
        "tuning": RESULTS_ROOT / season / "tuning",
        "validation": RESULTS_ROOT / season / "validation",
        "test": RESULTS_ROOT / season / "test",
        "probabilistic": RESULTS_ROOT / season / "probabilistic",
        "shap": RESULTS_ROOT / season / "shap",
        "plots": RESULTS_ROOT / season / "plots",
        "production": RESULTS_ROOT / season / "production",
    }


def _save_forecasters(
    forecasters: dict,
    stacking: StackingArtifacts | None,
    model_dir: Path,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "forecaster_recursive_lgbm.pkl": forecasters["A"],
        "forecaster_direct_lgbm.pkl": forecasters["B"],
        "forecaster_baseline.pkl": forecasters["C"],
        "forecaster_quantile_lower.pkl": forecasters["D_lower"],
        "forecaster_quantile_upper.pkl": forecasters["D_upper"],
        "forecaster_recursive_xgb.pkl": forecasters["E"],
        "forecaster_recursive_ridge.pkl": forecasters["F"],
        "forecaster_recursive_rf.pkl": forecasters["G"],
    }
    for fname, obj in mapping.items():
        path = model_dir / fname
        joblib.dump(obj, path)
        loaded = joblib.load(path)
        print(f"  Saved & verified: {fname}")
        assert hasattr(loaded, "predict")
    if stacking is not None:
        joblib.dump(stacking, model_dir / "stacking_meta_learner.pkl")
        print("  Saved & verified: stacking_meta_learner.pkl")


def _verify_loaded_models(
    model_dir: Path,
    exog_future: pd.DataFrame,
    y_history: pd.Series,
) -> None:
    exog_future = exog_future.iloc[:FORECAST_STEPS]
    for pkl in model_dir.glob("*.pkl"):
        if "stacking" in pkl.name:
            continue
        loaded = joblib.load(pkl)
        window_len = getattr(loaded, "window_size", None) or 200
        if pkl.name == "forecaster_baseline.pkl":
            last_window = y_history
        else:
            last_window = y_history.iloc[-window_len:]
        kwargs = {"steps": FORECAST_STEPS, "last_window": last_window}
        if pkl.name != "forecaster_baseline.pkl":
            kwargs["exog"] = exog_future
        pred = loaded.predict(**kwargs)
        assert len(pred) == FORECAST_STEPS, f"{pkl.name} prediction length mismatch"


def _validation_one_model(
    label: str,
    season: str,
    tuned: dict,
    stacking: StackingArtifacts | None,
    y_train: pd.Series,
    y_val: pd.Series,
    y_train_val: pd.Series,
    exog_train_val: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.Series | None]:
    if label == "H":
        if stacking is None:
            return None, None
        meta_preds = stacking.meta_learner.predict(
            stacking.oof_val_predictions.values
        )
        stack_preds = pd.Series(meta_preds, index=y_val.index, name="pred")
        horizon_df = _metrics_from_series(
            y_val, stack_preds, label, season, "validation"
        )
        return horizon_df, stack_preds

    fc = tuned[label]
    cv_val = make_cv_validation(
        len(y_train),
        differentiation=differentiation_for_forecaster(fc),
    )
    m_df = evaluate_on_split(
        fc, y_train_val, exog_train_val, cv_val, label, season, "validation"
    )
    _, preds = run_backtesting(fc, y_train_val, exog_train_val, cv_val)
    return m_df, preds["pred"]


def run_season(
    season: str,
    season_df: pd.DataFrame,
    master_rows: list,
    logger: PipelineErrorLogger,
    *,
    strict: bool = False,
) -> bool:
    """
    Run the full season pipeline. Returns True if the season completed without errors.

    On failure (unless strict=True), logs to pipeline_errors.log and continues
  wherever possible within the season; caller continues to the next season.
    """
    paths = _season_paths(season)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    def _fail(step: str, exc: BaseException) -> bool:
        logger.log(step, exc, season=season)
        if strict:
            raise exc
        return False

    exog_cols = SEASON_EXOG_COLS[season]
    try:
        splits = chronological_split(season_df)
        print_split_sizes(season, splits)
        y_train, exog_train = extract_target_exog(splits.train, exog_cols)
        y_val, exog_val = extract_target_exog(splits.validation, exog_cols)
        y_test, exog_test = extract_target_exog(splits.test, exog_cols)
        y_full = pd.concat([y_train, y_val, y_test])
        exog_full = pd.concat([exog_train, exog_val, exog_test])
    except Exception as exc:
        _fail("data_split", exc)
        return False

    ok, tuned = run_step(
        logger,
        season,
        "tuning",
        tune_forecasters,
        season=season,
        y_train=y_train,
        exog_train=exog_train,
        results_dir=paths["tuning"],
    )
    if not ok or tuned is None:
        logger.log_message(
            f"Skipping remaining steps for {season} (tuning failed).",
            season=season,
        )
        return False

    stacking: StackingArtifacts | None = None
    ok, stacking = run_step(
        logger,
        season,
        "stacking",
        fit_stacking_meta_learner,
        base_forecasters={k: clone_forecaster(tuned[k]) for k in ("A", "E", "F")},
        y_train=y_train,
        exog_train=exog_train,
        y_val=y_val,
        exog_val=exog_val,
    )
    if ok and stacking is not None:
        run_step(
            logger,
            season,
            "stacking_save",
            save_stacking_artifacts,
            paths["models"] / "stacking_meta_learner.pkl",
            stacking,
        )

    y_train_val = pd.concat([y_train, y_val])
    exog_train_val = pd.concat([exog_train, exog_val])
    val_records: list[pd.DataFrame] = []
    val_predictions_by_model: dict[str, pd.Series] = {}

    for label in ALL_MODEL_LABELS:
        if label == "D":
            continue
        ok, result = run_step(
            logger,
            season,
            f"validation_model_{label}",
            _validation_one_model,
            label,
            season,
            tuned,
            stacking,
            y_train,
            y_val,
            y_train_val,
            exog_train_val,
            default=(None, None),
        )
        if not ok or result is None:
            continue
        horizon_df, preds = result
        if horizon_df is None:
            continue
        val_records.append(horizon_df)
        if preds is not None:
            val_predictions_by_model[label] = preds
        print(f"\n[{season}] Validation metrics — model {label}:")
        print(horizon_df.to_string(index=False))

    if not val_records:
        logger.log_message(
            f"No validation metrics for {season}; skipping test and later steps.",
            season=season,
        )
        return False

    validation_metrics = pd.concat(val_records, ignore_index=True)
    validation_metrics.to_csv(
        paths["validation"] / "metrics_validation.csv", index=False
    )

    ok, best_label = run_step(
        logger,
        season,
        "select_best_model",
        select_best_model,
        validation_metrics,
        default="A",
    )
    if not ok or best_label is None:
        best_label = "A"
        logger.log_message(
            f"Using fallback best model A for {season}.", season=season
        )

    best_key = best_label if best_label != "H" else "H"
    if best_key == "H" and stacking is None:
        best_key = "A"
        best_label = "A"
        logger.log_message(
            f"Stacking unavailable; using model A for {season} test/production.",
            season=season,
        )

    if best_key == "H":
        best_model = stacking.meta_learner
    else:
        best_model = tuned[best_key]

    y_train_val_test = y_full
    exog_train_val_test = exog_full
    test_metrics: pd.DataFrame | None = None
    test_preds_series: pd.Series | None = None

    if best_key == "H" and stacking is not None:
        ok, test_preds_series = run_step(
            logger,
            season,
            "test_stacking",
            predict_stacking,
            stacking,
            {k: clone_forecaster(tuned[k]) for k in ("A", "E", "F")},
            y_train,
            y_val,
            exog_train,
            exog_val,
            exog_test,
            steps=len(y_test),
            default=None,
        )
        if ok and test_preds_series is not None:
            ok, test_metrics = run_step(
                logger,
                season,
                "test_metrics_stacking",
                _metrics_from_series,
                y_test,
                test_preds_series,
                "H",
                season,
                "test",
                default=None,
            )
    else:
        cv_test = make_cv_test(
            len(y_train) + len(y_val),
            differentiation=differentiation_for_forecaster(best_model),
        )

        def _test_backtest():
            metric_test, predictions_test = run_backtesting(
                best_model,
                y_train_val_test,
                exog_train_val_test,
                cv_test,
                metrics=[
                    "mean_absolute_error",
                    "mean_squared_error",
                    "mean_absolute_percentage_error",
                ],
            )
            print(f"\n[{season}] Test backtesting metric summary:")
            print(metric_test)
            return evaluate_on_split(
                best_model,
                y_train_val_test,
                exog_train_val_test,
                cv_test,
                best_label,
                season,
                "test",
            ), predictions_test["pred"]

        ok, result = run_step(
            logger,
            season,
            "test_backtesting",
            _test_backtest,
            default=(None, None),
        )
        if ok and result is not None:
            test_metrics, test_preds_series = result

    run_step(
        logger,
        season,
        "test_baseline_C",
        evaluate_on_split,
        tuned["C"],
        y_train_val_test,
        None,
        make_cv_test(
            len(y_train) + len(y_val),
            differentiation=differentiation_for_forecaster(tuned["C"]),
        ),
        "C",
        season,
        "test",
    )

    if test_metrics is not None and test_preds_series is not None:
        test_metrics.to_csv(paths["test"] / "metrics_test.csv", index=False)
        pd.DataFrame({"pred": test_preds_series}).to_csv(
            paths["test"] / "predictions_test.csv"
        )
        print(f"\n[{season}] Test metrics (best={best_label}):")
        print(test_metrics.to_string(index=False))
    else:
        logger.log_message(
            f"Test step skipped for {season} (no predictions).",
            season=season,
        )

    y_fit = pd.concat([y_train, y_val])
    exog_fit = pd.concat([exog_train, exog_val])
    exog_future_24h = exog_test.iloc[:FORECAST_STEPS]
    prob_forecaster = tuned["A"] if best_key in ("H", "C") else best_model

    run_step(
        logger,
        season,
        "refit_best",
        lambda: (
            [tuned[k].fit(y=y_fit, exog=exog_fit) for k in ("A", "E", "F")]
            if best_key == "H"
            else best_model.fit(y=y_fit, exog=exog_fit)
        ),
    )

    bootstrap_df = pd.DataFrame()
    conformal_df = pd.DataFrame()
    quantile_df = pd.DataFrame()
    interval_metrics = pd.DataFrame()

    ok, _ = run_step(
        logger,
        season,
        "probabilistic_fit",
        lambda: prob_forecaster.fit(
            y=y_fit, exog=exog_fit, store_in_sample_residuals=True
        ),
    )
    if ok:
        ok, bootstrap_df = run_step(
            logger,
            season,
            "probabilistic_bootstrap",
            run_bootstrap_intervals,
            prob_forecaster,
            FORECAST_STEPS,
            exog_future_24h,
            default=pd.DataFrame(),
        )
        if bootstrap_df is None:
            bootstrap_df = pd.DataFrame()

        def _conformal_pipeline():
            cv_prob = make_cv_validation(
                len(y_train),
                differentiation=differentiation_for_forecaster(prob_forecaster),
            )
            _, val_preds_bt = run_backtesting(
                prob_forecaster,
                y_train_val,
                exog_train_val,
                cv_prob,
            )
            val_eval_start = y_train_val.index[cv_prob.initial_train_size]
            y_val_oof = y_train_val.loc[val_eval_start:]
            conformal_fc = clone_forecaster(prob_forecaster)
            conformal_fc.fit(
                y=y_fit, exog=exog_fit, store_in_sample_residuals=True
            )
            return run_conformal_intervals(
                conformal_fc,
                y_true=y_val_oof,
                y_pred=val_preds_bt["pred"].reindex(y_val_oof.index),
                steps=FORECAST_STEPS,
                exog_future=exog_future_24h,
            )

        ok, conformal_df = run_step(
            logger,
            season,
            "probabilistic_conformal",
            _conformal_pipeline,
            default=pd.DataFrame(),
        )
        if conformal_df is None:
            conformal_df = pd.DataFrame()

        ok, quantile_df = run_step(
            logger,
            season,
            "probabilistic_quantile",
            run_quantile_intervals,
            clone_forecaster(tuned["D_lower"]),
            clone_forecaster(tuned["D_upper"]),
            y_fit,
            exog_fit,
            FORECAST_STEPS,
            exog_future_24h,
            default=pd.DataFrame(),
        )
        if quantile_df is None:
            quantile_df = pd.DataFrame()

        if not bootstrap_df.empty or not conformal_df.empty or not quantile_df.empty:
            y_interval_true = y_test.iloc[:FORECAST_STEPS]

            def _save_prob():
                im = build_interval_metrics(
                    y_interval_true, bootstrap_df, conformal_df, quantile_df
                )
                save_probabilistic_outputs(
                    paths["results"],
                    bootstrap_df,
                    conformal_df,
                    quantile_df,
                    y_interval_true,
                    im,
                )
                return im

            ok, interval_metrics = run_step(
                logger,
                season,
                "probabilistic_save",
                _save_prob,
                default=pd.DataFrame(),
            )
            if interval_metrics is None:
                interval_metrics = pd.DataFrame()

    run_step(
        logger,
        season,
        "shap",
        run_shap_analysis,
        tuned["A"],
        y_train,
        exog_train,
        paths["shap"],
    )

    if test_preds_series is not None:
        val_best_preds = val_predictions_by_model.get(
            best_label, val_predictions_by_model.get("A", test_preds_series)
        )
        run_step(
            logger,
            season,
            "plots",
            generate_all_season_plots,
            season=season,
            plots_dir=paths["plots"],
            y_full=y_full,
            splits=splits,
            validation_metrics=validation_metrics,
            val_predictions=val_best_preds,
            test_predictions=test_preds_series,
            val_intervals=bootstrap_df if not bootstrap_df.empty else None,
            test_intervals=bootstrap_df if not bootstrap_df.empty else None,
            bootstrap_df=bootstrap_df if not bootstrap_df.empty else quantile_df,
            conformal_df=conformal_df if not conformal_df.empty else quantile_df,
            quantile_df=quantile_df,
            best_forecaster=tuned["A"],
            cv_val=make_cv_validation(
                len(y_train), differentiation=SEASON_DIFFERENTIATION[season]
            ),
            y_val=y_val,
            y_test=y_test,
            y_train=y_train,
            exog_train=exog_train,
        )

    def _save_all_models():
        tuned["C"].fit(y=y_fit)
        for key in ("A", "B", "D_lower", "D_upper", "E", "F", "G"):
            fc = tuned[key]
            fc.fit(y=y_fit, exog=exog_fit)
        _save_forecasters(tuned, stacking, paths["models"])
        _verify_loaded_models(paths["models"], exog_future_24h, y_fit)

    run_step(logger, season, "save_models", _save_all_models)

    def _production_forecast():
        if best_key == "H" and stacking is not None:
            return predict_stacking(
                stacking,
                tuned,
                y_train,
                y_val,
                exog_train,
                exog_val,
                exog_future_24h,
                steps=FORECAST_STEPS,
            )
        last_window = y_fit.iloc[-200:]
        if best_key == "C":
            return best_model.predict(
                steps=FORECAST_STEPS,
                last_window=last_window,
            )
        return best_model.predict(
            steps=FORECAST_STEPS,
            exog=exog_future_24h,
            last_window=last_window,
        )

    ok, next_24h = run_step(
        logger,
        season,
        "production",
        _production_forecast,
        default=None,
    )
    if ok and next_24h is not None:
        prod_path = paths["production"] / "next_24h_forecast.csv"
        next_24h.to_frame("WS_forecast").to_csv(prod_path)
        print(f"[{season}] Production 24h forecast saved to {prod_path}")

    if test_metrics is not None:
        for _, row in test_metrics.iterrows():
            master_rows.append(
                {
                    "Season": season,
                    "Model": row["Model"],
                    "Split": "test",
                    "Horizon": row["Horizon"],
                    "MAE": row["MAE"],
                    "RMSE": row["RMSE"],
                    "MAPE": row["MAPE"],
                    "MedAE": row["MedAE"],
                    "IntervalCoverage": interval_metrics["IntervalCoverage"].mean()
                    if not interval_metrics.empty
                    else None,
                    "IntervalWidth": interval_metrics["IntervalWidth"].mean()
                    if not interval_metrics.empty
                    else None,
                }
            )

    print(f"[{season}] Season pipeline finished (see {logger.log_path} for any errors).")
    return True


def _metrics_from_series(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_label: str,
    season: str,
    split_name: str,
) -> pd.DataFrame:
    from evaluation.metrics import compute_horizon_metrics

    preds = pd.DataFrame({"pred": y_pred, "fold": 0}, index=y_true.index)
    horizon = compute_horizon_metrics(
        y_true=y_true,
        predictions=preds,
        horizons=HORIZONS,
        steps=FORECAST_STEPS,
    )
    return aggregate_metrics(horizon, model_label, season, split_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wind forecasting pipeline")
    parser.add_argument(
        "--season",
        choices=list(SEASONS),
        nargs="*",
        default=None,
        help="Run only selected season(s); default runs all four.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Reduce Bayesian trials for smoke testing.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on first error (default: log errors and continue).",
    )
    args = parser.parse_args()

    if args.fast or os.environ.get("WIND_FAST"):
        import config as cfg

        cfg.N_TRIALS_TUNING = 2
        print("FAST mode: N_TRIALS_TUNING=2")

    seasons_to_run = tuple(args.season) if args.season else SEASONS
    logger = PipelineErrorLogger(PIPELINE_ERROR_LOG)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    logger.log_message(
        f"Pipeline run started at {started} | seasons={list(seasons_to_run)} "
        f"| strict={args.strict} | fast={bool(args.fast or os.environ.get('WIND_FAST'))}"
    )

    print("=" * 72)
    print("Wind Power Forecasting Pipeline — skforecast")
    print(f"Error log: {PIPELINE_ERROR_LOG}")
    if not args.strict:
        print("Continue-on-error: ON (use --strict to stop on first failure)")
    print("=" * 72)

    try:
        df = load_and_preprocess()
        df = add_cyclical_features(df)
        assert df.isna().sum().sum() == 0, "NaNs present after feature engineering."
    except Exception as exc:
        logger.log("load_data", exc)
        if args.strict:
            raise
        print("Cannot continue without data. Exiting.")
        return

    season_frames = split_by_season(df)
    master_rows: list[dict] = []
    season_results: dict[str, bool] = {}

    for season in seasons_to_run:
        print("\n" + "#" * 72)
        print(f"SEASON: {season.upper()}")
        print("#" * 72)
        try:
            season_results[season] = run_season(
                season,
                season_frames[season],
                master_rows,
                logger,
                strict=args.strict,
            )
        except Exception as exc:
            logger.log("season", exc, season=season)
            season_results[season] = False
            if args.strict:
                raise
            print(f"[{season}] Season aborted; continuing to next season.")

    if master_rows:
        master_df = pd.DataFrame(master_rows)
        master_csv = RESULTS_ROOT / "MASTER_METRICS_ALL_SEASONS.csv"
        master_df.to_csv(master_csv, index=False)
        ok, _ = run_step(
            logger,
            "all",
            "master_heatmap",
            plot_master_metrics_heatmap,
            master_df,
            RESULTS_ROOT / "MASTER_METRICS_ALL_SEASONS.png",
        )
        if ok:
            print(f"\nMaster metrics saved to {master_csv}")
            print(master_df.groupby(["Season", "Horizon"])["RMSE"].mean())
    else:
        logger.log_message("No test metrics collected; master CSV not written.")

    finished = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    summary = ", ".join(
        f"{s}={'OK' if season_results.get(s) else 'FAILED/SKIPPED'}"
        for s in seasons_to_run
    )
    logger.log_message(f"Pipeline run finished at {finished} | {summary}")
    print(f"\nRun summary: {summary}")
    print(f"Full error details (if any): {PIPELINE_ERROR_LOG}")


if __name__ == "__main__":
    main()
