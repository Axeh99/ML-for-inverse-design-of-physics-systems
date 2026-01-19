from itertools import product
from typing_extensions import Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd


Objective = Literal["minimize_db_300L", "minimize_gamma_mag_300L"]


def rank_candidates(
    model,
    base_features: Dict[str, float],
    sweep_1_name: str,
    sweep_1_values: Sequence[float],
    sweep_2_name: str,
    sweep_2_values: Sequence[float],
    prediction_columns: List[str],
    objective: Objective = "minimize_target",
    objective_column: Optional[str] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    rows = []
    for v1, v2 in product(sweep_1_values, sweep_2_values):
        x = {**base_features, sweep_1_name: float(v1), sweep_2_name: float(v2)}
        rows.append(x)

    X = pd.DataFrame(rows)
    y_pred = np.asarray(model.predict(X))

    P = pd.DataFrame(y_pred, columns=prediction_columns)
    out = pd.concat([X[[sweep_1_name, sweep_2_name]], P], axis=1)

    if objective_column is None:
        if len(prediction_columns) != 1:
            raise ValueError("objective_column must be provided when there are multiple prediction columns.")
        objective_column = prediction_columns[0]

    if objective == "minimize_target":
        out["score"] = out[objective_column]
        out = out.sort_values("score", ascending=True)
    elif objective == "maximize_target":
        out["score"] = -out[objective_column]
        out = out.sort_values("score", ascending=True)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    return out.head(top_n).reset_index(drop=True)
