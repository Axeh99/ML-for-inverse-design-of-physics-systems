import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


def _get_feature_names_from_column_transformer(ct):
    feature_names = []
    for name, transformer, cols in ct.transformers_:
        if transformer == "drop":
            continue
        if transformer == "passthrough":
            feature_names.extend(list(cols))
        elif hasattr(transformer, "get_feature_names_out"):
            feature_names.extend(list(transformer.get_feature_names_out(cols)))
        else:
            feature_names.extend(list(cols))
    return feature_names


def plot_feature_importances_combined(
    fitted_pipeline,
    X,
    y,
    target_columns,
    seeds=10,
    n_repeats=10,
    test_size=0.1,
    scoring="neg_mean_squared_error",
):
    if "preprocessor" not in fitted_pipeline.named_steps or "model" not in fitted_pipeline.named_steps:
        raise ValueError("Expected a Pipeline with named steps: 'preprocessor' and 'model'.")

    preprocessor = fitted_pipeline.named_steps["preprocessor"]

    preprocessor.fit(X)
    full_feature_names = _get_feature_names_from_column_transformer(preprocessor)

    importances_accum = []

    for seed in range(seeds):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        Xt_train = preprocessor.transform(X_train)
        Xt_test = preprocessor.transform(X_test)

        if hasattr(Xt_test, "toarray"):
            Xt_train = Xt_train.toarray()
            Xt_test = Xt_test.toarray()

        model = fitted_pipeline.named_steps["model"]

        model.fit(Xt_train, y_train)

        result = permutation_importance(
            model,
            Xt_test,
            y_test,
            n_repeats=n_repeats,
            random_state=seed,
            scoring=scoring,
        )

        imp = np.asarray(result.importances_mean)
        if imp.shape[0] != len(full_feature_names):
            continue

        importances_accum.append(imp)

    if not importances_accum:
        raise RuntimeError("No valid permutation importance runs were collected.")

    imp_df = pd.DataFrame(importances_accum, columns=full_feature_names)
    mean_importances = imp_df.mean(axis=0).sort_values(ascending=False)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharey=False)

    mean_importances.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Feature Importances per Encoded Feature (Avg. over Seeds)")
    axes[0].set_ylabel("Importance (mean Δ MSE)")
    axes[0].tick_params(axis="x", rotation=90)

    aggregated = {}
    for feat, val in mean_importances.items():
        base = feat.split("_")[0]
        aggregated[base] = aggregated.get(base, 0.0) + float(val)

    aggregated_s = pd.Series(aggregated).sort_values(ascending=False)

    aggregated_s.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Aggregated Importances per Original Feature (Avg. over Seeds)")
    axes[1].set_ylabel("Aggregated Importance")
    axes[1].tick_params(axis="x", rotation=45)

    plt.suptitle(f"Permutation Feature Importances Averaged over {seeds} Random States", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return mean_importances, aggregated_s
