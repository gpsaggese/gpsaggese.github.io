from tpot import TPOTRegressor
from sklearn.pipeline import Pipeline
import joblib

def run_tpot_preprocessing(X, y, random_state=42):
    tpot = TPOTRegressor(
        generations=5,
        population_size=20,
        scoring="r2",
        cv=3,
        random_state=random_state,
        n_jobs=-1,
        verbosity=2,
        config_dict="TPOT light",
        early_stop=3
    )
    tpot.fit(X, y)
    pipe = tpot.fitted_pipeline_

    # drop final estimator, keep preprocessing steps
    if isinstance(pipe, Pipeline) and len(pipe.steps) > 1:
        preprocess = Pipeline(pipe.steps[:-1])
    else:
        preprocess = pipe

    return preprocess, tpot

def save_preprocess(preprocess, path):
    joblib.dump(preprocess, path)
