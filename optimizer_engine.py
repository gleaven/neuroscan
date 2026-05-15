"""NEUROSCAN — Automated Abliteration Optimizer.

Uses Optuna TPE (Tree-structured Parzen Estimator) for multi-objective
optimization of abliteration parameters. Dual objective:
  1. Minimize refusal rate (want 0% refusal after abliteration)
  2. Minimize KL divergence (want minimal capability damage)

The optimizer searches over:
  - Method (standard, norm_preserving, biprojected)
  - Layer range (start, end) for applying abliteration
  - Tent-shaped weight profile (peak position, max/min weight)
  - Activation layer type

The result is a Pareto front of non-dominated solutions that trade off
refusal removal effectiveness against capability preservation.

Inspired by Heretic (p-e-w/heretic) which uses Optuna for weight
orthogonalization optimization with similar dual objectives.
"""

import logging
import math
import threading
import time

logger = logging.getLogger("neuroscan.optimizer")


class OptimizerEngine:
    """Multi-objective abliteration parameter optimizer.

    Runs Optuna trials in a background thread, broadcasting progress
    via a callback. Each trial configures the abliteration engine with
    different parameters and measures refusal rate + KL divergence on
    a test set.
    """

    def __init__(self, abliteration_engine):
        self._abl = abliteration_engine
        self._study = None
        self._running = False
        self._should_stop = False
        self._thread = None
        self._lock = threading.Lock()
        self._trials_log = []       # all completed trial summaries
        self._pareto_front = []     # non-dominated solutions
        self._best_params = None    # params from best balanced trial

    @property
    def is_running(self):
        return self._running

    # ── Tent-Shaped Weight Kernel ────────────────────────────────

    @staticmethod
    def _build_tent_kernel(
        n_layers: int,
        layer_start: int,
        layer_end: int,
        max_weight: float,
        peak_position: float,
        min_weight: float,
    ) -> dict[int, float]:
        """Build a tent-shaped weight profile across layers.

        The peak is at `peak_position` (0.0 = start, 1.0 = end) with
        `max_weight` at the peak tapering linearly to `min_weight` at
        the edges of the active range.

        Returns {layer_idx: weight} for layers in [layer_start, layer_end].
        """
        weights = {}
        span = max(layer_end - layer_start, 1)
        peak_layer = layer_start + peak_position * span

        for l in range(layer_start, min(layer_end + 1, n_layers)):
            dist = abs(l - peak_layer) / max(span, 1)
            w = max_weight - (max_weight - min_weight) * dist
            weights[l] = round(max(min_weight, min(w, max_weight)), 3)

        return weights

    # ── Start/Stop ──────────────────────────────────────────────

    def start_optimization(
        self,
        n_trials: int = 50,
        n_test_prompts: int = 10,
        test_prompts: list[str] | None = None,
        activation_layer: str = "resid_post",
        progress_callback=None,
    ):
        """Start multi-objective optimization in background thread.

        Args:
            n_trials: Number of Optuna trials to run
            n_test_prompts: How many prompts to test per trial
            test_prompts: Custom test prompts (defaults to builtin harmful)
            activation_layer: Which layer type to optimize
            progress_callback: Called with progress dict after each trial
        """
        if self._running:
            raise RuntimeError("Optimization already running")

        if not self._abl._directions:
            raise RuntimeError("Compute refusal directions first")

        self._should_stop = False
        self._running = True
        self._trials_log = []
        self._pareto_front = []
        self._best_params = None

        def _run():
            try:
                self._run_optimization(
                    n_trials, n_test_prompts, test_prompts,
                    activation_layer, progress_callback,
                )
            except Exception as e:
                logger.error(f"Optimization failed: {e}")
                if progress_callback:
                    progress_callback({"phase": "error", "error": str(e)})
            finally:
                self._running = False
                if progress_callback:
                    progress_callback({
                        "phase": "complete",
                        "n_trials": len(self._trials_log),
                        "pareto_front": self._pareto_front,
                        "best_params": self._best_params,
                    })

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop_optimization(self):
        """Signal the optimizer to stop after current trial."""
        self._should_stop = True

    # ── Core Optimization Loop ──────────────────────────────────

    def _run_optimization(
        self,
        n_trials: int,
        n_test_prompts: int,
        test_prompts: list[str] | None,
        activation_layer: str,
        progress_callback,
    ):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        from activation_engine import MODEL_REGISTRY
        reg = MODEL_REGISTRY[self._abl._model_mgr._model_name]
        n_layers = reg["n_layers"]

        # Get test prompts
        if test_prompts:
            prompts = test_prompts[:n_test_prompts]
        else:
            from abliteration_engine import BUILTIN_HARMFUL
            prompts = BUILTIN_HARMFUL[:n_test_prompts]

        # Create multi-objective study
        self._study = optuna.create_study(
            directions=["minimize", "minimize"],  # refusal_rate, kl_divergence
            sampler=optuna.samplers.TPESampler(
                multivariate=True,
                n_startup_trials=min(5, n_trials // 3),
            ),
        )

        start_time = time.time()

        for trial_idx in range(n_trials):
            if self._should_stop:
                logger.info(f"Optimization stopped at trial {trial_idx}")
                break

            trial = self._study.ask()

            # ── Search Space ──
            method = trial.suggest_categorical(
                "method", ["standard", "norm_preserving", "biprojected"],
            )
            layer_start = trial.suggest_int("layer_start", 0, n_layers - 1)
            layer_end = trial.suggest_int(
                "layer_end", layer_start, n_layers - 1,
            )
            max_weight = trial.suggest_float("max_weight", 0.2, 2.5)
            min_weight = trial.suggest_float("min_weight", 0.0, max_weight)
            peak_position = trial.suggest_float("peak_position", 0.0, 1.0)

            # Build tent kernel
            layer_weights = self._build_tent_kernel(
                n_layers, layer_start, layer_end,
                max_weight, peak_position, min_weight,
            )

            # ── Evaluate ──
            refusal_count = 0
            kl_sum = 0.0
            kl_count = 0
            tested = 0

            for prompt in prompts:
                try:
                    result = self._abl.abliterate_generate(
                        prompt=prompt,
                        max_tokens=100,
                        layers=list(layer_weights.keys()),
                        layer_weights=layer_weights,
                        method=method,
                        activation_layer=activation_layer,
                    )
                    if result["abliterated_is_refusal"]:
                        refusal_count += 1
                    if result["kl_divergence"] >= 0:
                        kl_sum += result["kl_divergence"]
                        kl_count += 1
                    tested += 1
                except Exception as e:
                    logger.warning(f"Trial {trial_idx} prompt failed: {e}")

            if tested == 0:
                trial_refusal_rate = 1.0
                trial_kl = 10.0
            else:
                trial_refusal_rate = refusal_count / tested
                trial_kl = kl_sum / max(kl_count, 1)

            self._study.tell(trial, [trial_refusal_rate, trial_kl])

            # Log trial
            trial_summary = {
                "trial": trial_idx,
                "method": method,
                "layer_start": layer_start,
                "layer_end": layer_end,
                "max_weight": round(max_weight, 3),
                "min_weight": round(min_weight, 3),
                "peak_position": round(peak_position, 3),
                "refusal_rate": round(trial_refusal_rate, 3),
                "kl_divergence": round(trial_kl, 6),
                "n_layers_active": len(layer_weights),
                "layer_weights": layer_weights,
            }
            self._trials_log.append(trial_summary)

            # Update Pareto front
            self._pareto_front = self._compute_pareto_front()

            # Find best balanced trial (lowest sum of normalized objectives)
            self._best_params = self._find_best_balanced()

            elapsed = time.time() - start_time
            if progress_callback:
                progress_callback({
                    "phase": "trial",
                    "trial": trial_idx,
                    "total_trials": n_trials,
                    "refusal_rate": trial_summary["refusal_rate"],
                    "kl_divergence": trial_summary["kl_divergence"],
                    "method": method,
                    "best_params": self._best_params,
                    "pareto_size": len(self._pareto_front),
                    "pareto_front": self._pareto_front,
                    "elapsed": round(elapsed, 1),
                })

        logger.info(
            f"Optimization finished: {len(self._trials_log)} trials, "
            f"Pareto front size: {len(self._pareto_front)}"
        )

    # ── Pareto Front ────────────────────────────────────────────

    def _compute_pareto_front(self) -> list[dict]:
        """Extract Pareto-optimal trials (non-dominated solutions).

        A trial is Pareto-optimal if no other trial is better in BOTH
        objectives (refusal rate AND KL divergence).
        """
        front = []
        for t in self._trials_log:
            dominated = False
            for other in self._trials_log:
                if other is t:
                    continue
                # Other dominates t if it's <= on both and < on at least one
                if (other["refusal_rate"] <= t["refusal_rate"] and
                    other["kl_divergence"] <= t["kl_divergence"] and
                    (other["refusal_rate"] < t["refusal_rate"] or
                     other["kl_divergence"] < t["kl_divergence"])):
                    dominated = True
                    break
            if not dominated:
                front.append(t)

        front.sort(key=lambda x: x["refusal_rate"])
        return front

    def _find_best_balanced(self) -> dict | None:
        """Find the trial closest to the ideal point (0, 0).

        Uses normalized Euclidean distance: sqrt((refusal/max_r)^2 + (kl/max_kl)^2)
        """
        if not self._trials_log:
            return None

        max_r = max(t["refusal_rate"] for t in self._trials_log) or 1.0
        max_kl = max(t["kl_divergence"] for t in self._trials_log) or 1.0

        best = None
        best_dist = float("inf")
        for t in self._trials_log:
            dist = math.sqrt(
                (t["refusal_rate"] / max_r) ** 2 +
                (t["kl_divergence"] / max_kl) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best = t

        return best

    # ── Apply Best ──────────────────────────────────────────────

    def get_results(self) -> dict:
        """Get current optimization results."""
        return {
            "n_trials": len(self._trials_log),
            "running": self._running,
            "pareto_front": self._pareto_front,
            "best_params": self._best_params,
            "all_trials": self._trials_log,
        }

    def get_pareto_front(self) -> list[dict]:
        """Get the Pareto-optimal trial set."""
        return self._pareto_front
