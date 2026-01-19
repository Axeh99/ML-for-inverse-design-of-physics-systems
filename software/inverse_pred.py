from itertools import product
import numpy as np
import pandas as pd


def _linspace(low, high, n):
    return np.linspace(float(low), float(high), int(n))


def _make_feature_candidates(search_space, n_numeric=8):
    keys = []
    values_lists = []

    for feat, spec in search_space.items():
        keys.append(feat)

        if isinstance(spec, (list, tuple, np.ndarray)) and len(spec) > 0 and not isinstance(spec[0], (int, float, np.number)):
            values_lists.append(list(spec))
            continue

        if isinstance(spec, (list, tuple, np.ndarray)) and len(spec) > 0 and isinstance(spec[0], (int, float, np.number)):
            values_lists.append([float(v) for v in spec])
            continue

        if isinstance(spec, dict):
            if spec.get("type") == "categorical":
                values_lists.append(list(spec["values"]))
            elif spec.get("type") == "numeric_range":
                values_lists.append(list(_linspace(spec["min"], spec["max"], spec.get("n", n_numeric))))
            else:
                raise ValueError(f"Unknown search space spec for '{feat}': {spec}")
            continue

        raise ValueError(f"Invalid search space entry for '{feat}': {spec}")

    for combo in product(*values_lists):
        yield dict(zip(keys, combo))


def _requirement_penalty(value, req):
    rtype = req.get("type")
    weight = float(req.get("weight", 1.0))

    if rtype == "eq":
        target = float(req["value"])
        scale = float(req.get("scale", 1.0))
        return weight * abs(float(value) - target) / scale

    if rtype == "range":
        lo = float(req["min"])
        hi = float(req["max"])
        scale = float(req.get("scale", 1.0))
        v = float(value)
        if lo <= v <= hi:
            return 0.0
        return weight * min(abs(v - lo), abs(v - hi)) / scale

    if rtype == "min":
        lo = float(req["value"])
        scale = float(req.get("scale", 1.0))
        v = float(value)
        if v >= lo:
            return 0.0
        return weight * (lo - v) / scale

    if rtype == "max":
        hi = float(req["value"])
        scale = float(req.get("scale", 1.0))
        v = float(value)
        if v <= hi:
            return 0.0
        return weight * (v - hi) / scale

    raise ValueError(f"Unknown requirement type: {rtype}")


def recommend_candidates(
    model,
    base_features,
    search_space,
    target_columns,
    requirements,
    top_n=30,
    max_candidates=50000,
    n_numeric=8,
    predicted_target_name=None,
):
    """
    General inverse-prediction recommender.

    Parameters
    ----------
    model : fitted estimator or Pipeline
        Must support model.predict(DataFrame) -> array-like (n_samples, n_targets).
    base_features : dict
        Fixed feature values provided by the user.
    search_space : dict
        Features to vary. For each feature:
          - categorical: {"type":"categorical","values":[...]}
          - numeric range: {"type":"numeric_range","min":a,"max":b,"n":k}
          - or directly a list of values.
    target_columns : list[str]
        Names of targets in the same order as model outputs.
    requirements : dict
        Constraints on targets. Example:
          {
            "DTm": {"type":"eq","value":10,"scale":1,"weight":2},
            "%gel": {"type":"min","value":80,"scale":10,"weight":1},
            "Tc": {"type":"range","min":50,"max":70,"scale":10}
          }
    top_n : int
        Number of best candidates to return.
    max_candidates : int
        Safety cap to avoid exploding cartesian product.
    n_numeric : int
        Default number of points for numeric ranges.

    Returns
    -------
    pd.DataFrame
        Candidate features + predicted targets + penalties + total score, sorted ascending (best first).
    """
    candidates = []
    for i, delta in enumerate(_make_feature_candidates(search_space, n_numeric=n_numeric)):
        if i >= max_candidates:
            break
        x = dict(base_features)
        x.update(delta)
        candidates.append(x)

    if not candidates:
        raise ValueError("No candidates generated. Provide a non-empty search_space.")

    X = pd.DataFrame(candidates)
    y_pred = np.asarray(model.predict(X))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    if y_pred.shape[1] == 1 and len(target_columns) != 1:
        if predicted_target_name is None:
            raise ValueError(
                "Model returned 1 output but multiple target_columns were provided. "
                "Pass target_columns=[<the single trained target>] or set predicted_target_name."
            )
        target_columns = [predicted_target_name]

    if y_pred.shape[1] != len(target_columns):
        raise ValueError(
            f"Model returned {y_pred.shape[1]} outputs, but target_columns has {len(target_columns)}."
        )

    P = pd.DataFrame(y_pred, columns=target_columns)
    out = pd.concat([X.reset_index(drop=True), P], axis=1)

    penalties = {}
    total = np.zeros(len(out), dtype=float)

    for tname, req in requirements.items():
        if tname not in out.columns:
            raise ValueError(f"Requirement target '{tname}' not found in prediction columns.")
        pen = out[tname].apply(lambda v: _requirement_penalty(v, req)).to_numpy()
        penalties[f"penalty_{tname}"] = pen
        total += pen

    for k, v in penalties.items():
        out[k] = v

    out["score"] = total
    out = out.sort_values("score", ascending=True).head(int(top_n)).reset_index(drop=True)
    return out
