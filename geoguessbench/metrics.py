"""All-round denominators, explicit failure handling and clustered uncertainty."""
import math
from collections import defaultdict

import numpy as np

from .geo import EARTH_RADIUS_KM, RADII_KM


def cluster_ci(rows, values, samples=2000, seed=42):
    groups = defaultdict(list)
    for row, value in zip(rows, values):
        groups[row["cluster"]].append(float(value))
    if not groups:
        return None
    sums = np.array([sum(v) for v in groups.values()])
    sizes = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for i in range(samples):
        selected = rng.integers(0, len(sums), len(sums))
        estimates[i] = sums[selected].sum() / sizes[selected].sum()
    return {"low": float(np.quantile(estimates, .025)), "high": float(np.quantile(estimates, .975)),
            "method": "percentile_cluster_bootstrap", "clusters": len(groups), "resamples": samples}


def summarize(rows, *, samples=2000, seed=42):
    if not rows:
        raise ValueError("cannot summarize an empty evaluation")
    valid = [r for r in rows if r["status"] == "ok"]
    distances = [r["distance_km"] for r in valid]
    # Undefined predictions count as failures, never disappear from accuracy/score denominators.
    penalized = [r["distance_km"] if r["status"] == "ok" else math.pi * EARTH_RADIUS_KM for r in rows]
    scores = [r["score"] for r in rows]
    accuracy = {str(k): sum(r["status"] == "ok" and r["distance_km"] < k for r in rows) / len(rows)
                for k in RADII_KM}
    confidence_rows = [r for r in valid if r.get("confidence_25km") is not None]
    strata = {}
    for country in sorted({r["country"] for r in rows if r.get("country")}):
        subset = [r for r in rows if r.get("country") == country]
        strata[country] = {"n": len(subset), "mean_score": float(np.mean([r["score"] for r in subset])),
                           "accuracy_25km": sum(r["status"] == "ok" and r["distance_km"] < 25
                                                for r in subset) / len(subset)}
    statuses = {status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})}
    return {
        "n": len(rows), "valid": len(valid), "completion_rate": len(valid) / len(rows),
        "status_counts": statuses, "mean_score": float(np.mean(scores)),
        "mean_score_ci95": cluster_ci(rows, scores, samples, seed),
        "accuracy_at_km": accuracy,
        "accuracy_25km_ci95": cluster_ci(rows, [r["status"] == "ok" and r["distance_km"] < 25
                                                for r in rows], samples, seed),
        "mean_error_km_valid_only": float(np.mean(distances)) if distances else None,
        "median_error_km_valid_only": float(np.median(distances)) if distances else None,
        "mean_error_km_failure_penalty": float(np.mean(penalized)),
        "median_error_km_failure_penalty": float(np.median(penalized)),
        "failure_distance_penalty_km": math.pi * EARTH_RADIUS_KM,
        "mean_actions": float(np.mean([r["actions"] for r in rows])),
        "mean_action_cost": float(np.mean([r["cost"] for r in rows])),
        "mean_elapsed_s": float(np.mean([r["elapsed_s"] for r in rows])),
        "brier_25km": float(np.mean([(r["confidence_25km"] - (r["distance_km"] < 25)) ** 2
                                      for r in confidence_rows])) if confidence_rows else None,
        "confidence_coverage": len(confidence_rows) / len(rows), "by_country": strata,
        "macro_country_accuracy_25km": float(np.mean([v["accuracy_25km"] for v in strata.values()]))
        if strata else None,
        "country_label_coverage": sum(bool(r.get("country")) for r in rows) / len(rows),
    }


def compare(left, right, *, samples=2000, seed=42):
    """Paired differences; exports must describe exactly the same protocol and rounds."""
    for key in ("dataset_sha256", "protocol_sha256"):
        if left[key] != right[key]:
            raise ValueError(f"incomparable runs: {key} differs")
    a = {r["round_id"]: r for r in left["rows"]}
    b = {r["round_id"]: r for r in right["rows"]}
    if not a or a.keys() != b.keys() or len(a) != len(left["rows"]) or len(b) != len(right["rows"]):
        raise ValueError("paired comparison requires the same unique completed rounds")
    ordered = [a[key] for key in sorted(a)]
    delta = [a[key]["score"] - b[key]["score"] for key in sorted(a)]
    return {"direction": "left_minus_right", "n": len(delta), "mean_score_difference": float(np.mean(delta)),
            "ci95": cluster_ci(ordered, delta, samples, seed)}


def telemetry_summary(traces):
    """Self-reported provider usage, separate from evaluator-measured exploration cost."""
    calls = [event["telemetry_self_reported"] for trace in traces for event in trace["events"]
             if event.get("telemetry_self_reported")]
    result = {"source": "self_reported_not_billing_verified", "calls": len(calls),
              "provider_errors": sum(bool(call.get("error")) for call in calls)}
    for field in ("prompt_tokens", "completion_tokens"):
        known = [call[field] for call in calls
                 if type(call.get(field)) is int and call[field] >= 0]
        result[field] = sum(known) if calls and len(known) == len(calls) else None
        result[field + "_known_sum"] = sum(known)
        result[field + "_coverage"] = len(known) / len(calls) if calls else 0.0
    return result
