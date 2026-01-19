import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def plot_predictions(
    fitted_models,
    X_test,
    y_test,
    target_name,
):
    y_test_arr = np.asarray(y_test)
    if y_test_arr.ndim > 1 and y_test_arr.shape[1] == 1:
        y_test_arr = y_test_arr.ravel()

    n_models = len(fitted_models)
    cols = 2
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, (name, model) in enumerate(fitted_models.items()):
        y_pred = model.predict(X_test)
        y_pred = np.asarray(y_pred)
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()

        mse = mean_squared_error(y_test_arr, y_pred)

        ax = axes[idx]
        ax.scatter(y_test_arr, y_pred, alpha=0.7)
        lo = min(float(y_test_arr.min()), float(y_pred.min()))
        hi = max(float(y_test_arr.max()), float(y_pred.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(f"{name} (MSE={mse:.2f})")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

    for i in range(n_models, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(f"Predicted vs Actual – Target: {target_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

