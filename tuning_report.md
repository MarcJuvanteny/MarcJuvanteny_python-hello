# Informe de tuning

## Dataset i mètrica

S'ha utilitzat el dataset integrat Breast Cancer Wisconsin de scikit-learn (569 registres). La classe positiva per al negoci és la maligna (`target=0`); per això s'optimitza el recall de malignitat, que penalitza els falsos negatius. El dataset d'exemple només conté variables numèriques i no té valors buits, però el pipeline també inclou imputació i codificació per gestionar entrades mixtes i valors absents.

La partició estratificada és 80/20. Les dues cerques i la validació creuada s'han ajustat únicament amb train. Test s'ha consultat una vegada per a la línia base i una vegada per al model final.

## Resultats

| Model | Recall de malignitat en test |
|---|---:|
| Línia base (RandomForestClassifier per defecte) | 0.929 |
| Model ajustat | 0.952 |

La cerca aleatòria ha obtingut un recall mitjà de CV de 0.965. El grid search ha obtingut 0.965.

## Candidats del grid search

| Rang | Recall mitjà CV | Desviació estàndard CV | Hiperparàmetres |
|---:|---:|---:|---|
| 1 | 0.965 | 0.022 | `{"classifier__class_weight": "balanced", "classifier__max_depth": 6, "classifier__max_features": null, "classifier__min_samples_leaf": 4, "classifier__n_estimators": 307}` |
| 1 | 0.965 | 0.022 | `{"classifier__class_weight": "balanced", "classifier__max_depth": 6, "classifier__max_features": null, "classifier__min_samples_leaf": 4, "classifier__n_estimators": 357}` |
| 1 | 0.965 | 0.022 | `{"classifier__class_weight": "balanced", "classifier__max_depth": 6, "classifier__max_features": null, "classifier__min_samples_leaf": 4, "classifier__n_estimators": 407}` |
| 1 | 0.965 | 0.022 | `{"classifier__class_weight": "balanced", "classifier__max_depth": 6, "classifier__max_features": null, "classifier__min_samples_leaf": 5, "classifier__n_estimators": 307}` |
| 1 | 0.965 | 0.022 | `{"classifier__class_weight": "balanced", "classifier__max_depth": 6, "classifier__max_features": null, "classifier__min_samples_leaf": 5, "classifier__n_estimators": 357}` |

## Selecció final

S'ha seleccionat el candidat amb millor recall mitjà; també és el més estable entre els cinc millors candidats mostrats.

Hiperparàmetres seleccionats: `{"classifier__class_weight": "balanced", "classifier__max_depth": 6, "classifier__max_features": null, "classifier__min_samples_leaf": 4, "classifier__n_estimators": 307}`. `GridSearchCV(refit=True)` ja ha ajustat el millor estimador amb tot el conjunt d'entrenament; no s'ha tornat a entrenar manualment.
