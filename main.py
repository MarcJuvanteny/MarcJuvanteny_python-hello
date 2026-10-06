"""Baseline and hyperparameter tuning for malignant breast-cancer detection."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import (
	GridSearchCV,
	RandomizedSearchCV,
	StratifiedKFold,
	train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
REPORT_PATH = Path(__file__).with_name("tuning_report.md")


def make_pipeline(numeric_columns, categorical_columns, random_state=None):
	transformers = []
	if numeric_columns:
		transformers.append(
			(
				"numeric",
				Pipeline([("imputer", SimpleImputer(strategy="median"))]),
				numeric_columns,
			)
		)
	if categorical_columns:
		transformers.append(
			(
				"categorical",
				Pipeline(
					[
						("imputer", SimpleImputer(strategy="most_frequent")),
						("onehot", OneHotEncoder(handle_unknown="ignore")),
					]
				),
				categorical_columns,
			)
		)

	preprocessing = ColumnTransformer(transformers=transformers)
	return Pipeline(
		[
			("preprocessing", preprocessing),
			("classifier", RandomForestClassifier(random_state=random_state)),
		]
	)


def clean_features(features):
	features = features.copy()
	features = features.replace(r"^\s*$", np.nan, regex=True)
	return features.replace([np.inf, -np.inf], np.nan)


def main():
	dataset = load_breast_cancer(as_frame=True)
	features = clean_features(dataset.data)
	target = dataset.target

	# In this dataset, target 0 is malignant and target 1 is benign.
	X_train, X_test, y_train, y_test = train_test_split(
		features,
		target,
		test_size=0.2,
		stratify=target,
		random_state=RANDOM_STATE,
	)

	numeric_columns = X_train.select_dtypes(include=np.number).columns.tolist()
	categorical_columns = X_train.select_dtypes(exclude=np.number).columns.tolist()
	scoring = make_scorer(recall_score, pos_label=0)
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

	baseline = make_pipeline(numeric_columns, categorical_columns)
	baseline.fit(X_train, y_train)
	baseline_recall = recall_score(y_test, baseline.predict(X_test), pos_label=0)

	search_pipeline = make_pipeline(numeric_columns, categorical_columns, RANDOM_STATE)
	randomized_search = RandomizedSearchCV(
		estimator=search_pipeline,
		param_distributions={
			"classifier__n_estimators": randint(100, 401),
			"classifier__max_depth": [None, 4, 8, 12, 16, 24],
			"classifier__min_samples_leaf": randint(1, 9),
			"classifier__max_features": ["sqrt", None, 0.5],
			"classifier__class_weight": [None, "balanced"],
		},
		n_iter=20,
		scoring=scoring,
		cv=cv,
		n_jobs=-1,
		refit=True,
		random_state=RANDOM_STATE,
	)
	randomized_search.fit(X_train, y_train)

	random_best = randomized_search.best_params_
	best_depth = random_best["classifier__max_depth"]
	depth_grid = (
		[None, 20]
		if best_depth is None
		else sorted({max(2, best_depth - 2), best_depth, best_depth + 2})
	)
	best_leaf = random_best["classifier__min_samples_leaf"]
	leaf_grid = sorted({max(1, best_leaf - 1), best_leaf, best_leaf + 1})
	best_estimators = random_best["classifier__n_estimators"]
	estimator_grid = sorted(
		{max(100, best_estimators - 50), best_estimators, best_estimators + 50}
	)

	grid_search = GridSearchCV(
		estimator=search_pipeline,
		param_grid={
			"classifier__n_estimators": estimator_grid,
			"classifier__max_depth": depth_grid,
			"classifier__min_samples_leaf": leaf_grid,
			"classifier__max_features": [random_best["classifier__max_features"]],
			"classifier__class_weight": [random_best["classifier__class_weight"]],
		},
		scoring=scoring,
		cv=cv,
		n_jobs=-1,
		refit=True,
	)
	grid_search.fit(X_train, y_train)

	# GridSearchCV has already refit the selected pipeline on all training data.
	final_model = grid_search.best_estimator_
	final_recall = recall_score(y_test, final_model.predict(X_test), pos_label=0)

	results = grid_search.cv_results_
	top_indices = np.argsort(results["rank_test_score"])[:5]
	most_stable_index = min(
		top_indices, key=lambda index: results["std_test_score"][index]
	)
	best_index = grid_search.best_index_

	candidate_rows = []
	for index in top_indices:
		params = results["params"][index]
		candidate_rows.append(
			"| {rank} | {mean:.3f} | {std:.3f} | `{params}` |".format(
				rank=results["rank_test_score"][index],
				mean=results["mean_test_score"][index],
				std=results["std_test_score"][index],
				params=json.dumps(params, sort_keys=True),
			)
		)

	if most_stable_index == best_index:
		selection_note = (
			"S'ha seleccionat el candidat amb millor recall mitjà; també és el més "
			"estable entre els cinc millors candidats mostrats."
		)
	else:
		selection_note = (
			"S'ha seleccionat el candidat amb millor recall mitjà. Entre els cinc "
			"millors, el candidat més estable té un recall mitjà de "
			f"{results['mean_test_score'][most_stable_index]:.3f} i una desviació "
			f"estàndard de {results['std_test_score'][most_stable_index]:.3f}; "
			"s'accepta aquesta diferència de variabilitat a canvi del millor recall."
		)

	report = f"""# Informe de tuning

## Dataset i mètrica

S'ha utilitzat el dataset integrat Breast Cancer Wisconsin de scikit-learn (569 registres). La classe positiva per al negoci és la maligna (`target=0`); per això s'optimitza el recall de malignitat, que penalitza els falsos negatius. El dataset d'exemple només conté variables numèriques i no té valors buits, però el pipeline també inclou imputació i codificació per gestionar entrades mixtes i valors absents.

La partició estratificada és 80/20. Les dues cerques i la validació creuada s'han ajustat únicament amb train. Test s'ha consultat una vegada per a la línia base i una vegada per al model final.

## Resultats

| Model | Recall de malignitat en test |
|---|---:|
| Línia base (RandomForestClassifier per defecte) | {baseline_recall:.3f} |
| Model ajustat | {final_recall:.3f} |

La cerca aleatòria ha obtingut un recall mitjà de CV de {randomized_search.best_score_:.3f}. El grid search ha obtingut {grid_search.best_score_:.3f}.

## Candidats del grid search

| Rang | Recall mitjà CV | Desviació estàndard CV | Hiperparàmetres |
|---:|---:|---:|---|
{chr(10).join(candidate_rows)}

## Selecció final

{selection_note}

Hiperparàmetres seleccionats: `{json.dumps(grid_search.best_params_, sort_keys=True)}`. `GridSearchCV(refit=True)` ja ha ajustat el millor estimador amb tot el conjunt d'entrenament; no s'ha tornat a entrenar manualment.
"""
	REPORT_PATH.write_text(report, encoding="utf-8")

	print(f"Recall baseline (test): {baseline_recall:.3f}")
	print(f"Recall final (test): {final_recall:.3f}")
	print(f"Informe escrit a: {REPORT_PATH}")


if __name__ == "__main__":
	main()