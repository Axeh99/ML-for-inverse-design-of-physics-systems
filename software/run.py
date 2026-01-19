import random

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from dataset import DataSet
from modelrunner import ModelRunner
from sklearn.pipeline import Pipeline
from config import load_config
from inverse_pred import recommend_candidates
from prediction_plots import plot_predictions
from importances import plot_feature_importances_combined

cfg = load_config("config.yaml")

TARGET_COLUMNS = cfg.target_columns
FEATURE_COLUMNS = cfg.feature_columns
BASE_FEATURES = cfg.base_features
SEARCH_SPACE = cfg.search_space
REQUIREMENTS = cfg.requirements
COLS = cfg.cols

seed = random.randint(0, 1000)
# seed = 42

data = DataSet(
    "data - copia.xlsx",
    seed,
    feature_columns=FEATURE_COLUMNS,
    target_columns=TARGET_COLUMNS,
)
data.load()
data.drop_missing()           
data.create_train_test_split()

runner = ModelRunner(data, problem_type="regression")
preprocessor = data.build_preprocessor()

multi = len(TARGET_COLUMNS) > 1

svr_est = MultiOutputRegressor(SVR()) if multi else SVR()
gbr_est = MultiOutputRegressor(GradientBoostingRegressor(random_state=42)) if multi else GradientBoostingRegressor(random_state=42)
rf_est  = RandomForestRegressor(random_state=42)

svr = Pipeline([
    ("preprocessor", preprocessor),
    ("model", svr_est),
])

rf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf_est),
])

gbr = Pipeline([
    ("preprocessor", preprocessor),
    ("model", gbr_est),
])

runner.add_model("SVR", svr)
runner.add_model("RandomForest", rf)
runner.add_model("GradientBoosting", gbr)

param_grid_svr = {
    "model__kernel": ["rbf"],
    "model__C": [0.1, 1, 10, 30, 100],
    # "model__epsilon": [0.001, 0.01, 0.05, 0.1],
    # "model__gamma": ["scale", "auto"],
    # "model__tol": [1e-3, 1e-2],
    # "model__shrinking": [True, False],
}

param_grid_rf = {
    "model__n_estimators": [200, 400, 800],
    "model__max_depth": [None, 10, 20, 30],
    # "model__min_samples_leaf": [1, 2, 4],
    # "model__min_samples_split": [2, 5, 10],
    # "model__max_features": ["sqrt", None],
    # "model__bootstrap": [True, False],
}


param_grid_gbr = {
    "model__n_estimators": [100, 200, 400],
    "model__learning_rate": [0.01, 0.05, 0.1],
    # "model__max_depth": [2, 3, 4],
    # "model__min_samples_leaf": [1, 2, 5, 10],
    # "model__subsample": [0.6, 0.8, 1.0],
    # "model__max_features": [None, "sqrt"],
}

runner.train()
runner.evaluate()

plot_predictions(
    fitted_models=runner.trained_models,
    X_test=data.X_test,
    y_test=data.y_test,
    target_name=TARGET_COLUMNS[0],
)

# runner.grid_search(
#     name="SVR",
#     param_grid=param_grid_svr,
#     seed=seed,
#     cv_folds=10,
#     scoring="neg_mean_squared_error",
# )

# runner.grid_search(
#     name="RandomForest",
#     seed=seed,
#     param_grid=param_grid_rf,
#     cv_folds=10,
#     scoring="neg_mean_squared_error",
# )

# runner.grid_search(
#     name="GradientBoosting",
#     seed=seed,
#     param_grid=param_grid_gbr,
#     cv_folds=10,
#     scoring="neg_mean_squared_error",
# )

runner.evaluate()
runner.summary()

# plot_predictions(
#     fitted_models=runner.trained_models,
#     X_test=data.X_test,
#     y_test=data.y_test,
#     target_name=TARGET_COLUMNS[0],
# )

best_name, best_model = runner.get_best_model(metric="mse")
print("BEST MODEL: ", best_name)




deployed_model = runner.refit_best_on_full_data(best_name)

requirements = {
    "DTm": {"type": "eq", "value": 10.0, "scale": 1.0, "weight": 1.0},
}

ranked = recommend_candidates(
    model=best_model,
    base_features=BASE_FEATURES,
    search_space=SEARCH_SPACE,
    target_columns=TARGET_COLUMNS,
    predicted_target_name="DTm",
    requirements=REQUIREMENTS,
)
cols = [
        # "Reaction mechanism",
         "Functionalization",
         "Functionalization reagents",
         "Crosslinker",
         "TPO",
         "Tiempo de residencia total/min",
         "DTm",
         "score",
         ]
print(ranked[cols])


plot_feature_importances_combined(
    fitted_pipeline=deployed_model,
    X=data.df[FEATURE_COLUMNS],
    y=data.df[TARGET_COLUMNS].to_numpy() if len(TARGET_COLUMNS) > 1 else data.df[TARGET_COLUMNS[0]].to_numpy(),
    target_columns=TARGET_COLUMNS,
    seeds=10,
    n_repeats=10,
)

