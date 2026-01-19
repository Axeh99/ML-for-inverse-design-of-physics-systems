from typing import Dict, Any, Optional, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, cross_val_score, KFold


class ModelRunner:
    """
    Class for managing, training, and evaluating multiple machine-learning models.

    Parameters
    ----------
    dataset : DataSet
        An instance of the DataSet class containing train/test data arrays.
    problem_type : {'regression', 'classification'}, optional
        Specifies the type of problem for metric selection. Default is 'regression'.
    """

    def __init__(self, dataset, problem_type: str = "regression"):
        if problem_type not in {"regression", "classification"}:
            raise ValueError("problem_type must be 'regression' or 'classification'.")

        self.dataset = dataset
        self.problem_type = problem_type

        self.untrained_models: Dict[str, BaseEstimator] = {}
        self.trained_models: Dict[str, BaseEstimator] = {}
        self.results: Dict[str, Dict[str, float]] = {}

    def add_model(self, name: str, model: BaseEstimator) -> None:
        """
        Register a model by name.

        Parameters
        ----------
        name : str
            Identifier for the model.
        model : BaseEstimator
            A scikit-learn estimator.
        """
        self.untrained_models[name] = model

    def train(self) -> None:
        """
        Train all registered models on X_train and y_train.
        """
        for name, model in self.untrained_models.items():
            if name in self.trained_models:
                continue
            self.trained_models[name] = clone(model).fit(self.dataset.X_train, self.dataset.y_train)

    def evaluate(self) -> None:
        """
        Evaluate all trained models on the test set and store metrics.
        """
        if self.dataset.X_test is None or self.dataset.y_test is None:
            raise ValueError("Test data not found. Call dataset.create_train_test_split().")
        if not self.trained_models:
            raise ValueError("No trained models found. Call train() first.")

        y_true = self.dataset.y_test

        for name, model in self.trained_models.items():
            y_pred = model.predict(self.dataset.X_test)
            self.results[name] = self._compute_metrics(y_true, np.asarray(y_pred))

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics based on the selected problem type.

        Returns
        -------
        metrics : dict
            Dictionary containing metrics.
        """
        if self.problem_type != "regression":
            raise NotImplementedError("Classification metrics are not implemented.")

        return {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def grid_search(
        self,
        name: str,
        seed: int,
        param_grid: Dict[str, Any],
        cv_folds: int = 5,
        scoring: Optional[str] = None,
    ):
        """
        Perform GridSearchCV on a registered model.

        Parameters
        ----------
        name : str
            Name of the model in the registry.
        param_grid : dict
            Dictionary with the parameters to try.
        cv_folds : int, optional
            Number of folds in k-fold cross validation.
        scoring : str, optional
            Scikit-learn scoring string. If None, default scoring is used.

        Returns
        -------
        gs : GridSearchCV
            The fitted grid search object.
        """
        if name not in self.untrained_models:
            raise ValueError(f"Model '{name}' is not registered.")

        if self.dataset.X_train is None or self.dataset.y_train is None:
            raise ValueError("Training data not found. Call dataset.create_train_test_split().")

        base_model = self.untrained_models[name]
        cv=KFold(n_splits=20, shuffle=True, random_state=seed)

        grid = list(ParameterGrid(param_grid))
        best_score = -np.inf
        best_params = None
        best_model = None

        print(f"Running manual grid search for '{name}'")
        print(f"Total parameter sets: {len(grid)}")
        print(f"CV folds: {cv_folds}")
        print("-" * 60)

        for params in tqdm(grid, desc="Grid search", ncols=80):
            model = clone(base_model).set_params(**params)
            scores = cross_val_score(
                model,
                self.dataset.X_train,
                self.dataset.y_train,
                cv=cv,
                scoring=scoring,
            )
            mean_score = float(np.mean(scores))

            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_model = clone(model)

        if best_model is None or best_params is None:
            raise RuntimeError("Grid search failed to evaluate any parameter set.")

        print("-" * 60)
        print("Grid search finished")
        print("Best CV score:", best_score)
        print("Best parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

        best_model.fit(self.dataset.X_train, self.dataset.y_train)
        self.trained_models[name] = best_model

        return best_model, best_params, best_score
    
    def get_best_model(self, metric: str = "mse") -> Tuple[str, BaseEstimator]:
        if not self.results:
            raise ValueError("No results available. Call evaluate() first.")

        if metric == "mse":
            best_name = min(self.results, key=lambda n: self.results[n][metric])
        else:
            best_name = max(self.results, key=lambda n: self.results[n][metric])

        return best_name, self.trained_models[best_name]
    
    def refit_best_on_full_data(self, best_name):
        best_model = self.trained_models[best_name]
        fitted = clone(best_model).fit(self.dataset.X_full, self.dataset.y_full)
        return fitted

    def summary(self) -> None:
        if not self.results:
            print("No evaluation results available.")
            return

        print("Model performance summary:\n")
        for name, metrics in self.results.items():
            print(f"Model: {name}")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.6f}")
            print()