from __future__ import annotations

"""
Module d'entraînement et d'enregistrement d'un modèle de churn.

Ce script :
1. Charge le jeu de données prétraité `data/processed.csv` ;
2. Sépare les variables explicatives (features) de la cible `churn` ;
3. Définit un pipeline scikit-learn :
   - prétraitement (StandardScaler pour les variables numériques,
     OneHotEncoder pour les catégorielles) ;
   - modèle de régression logistique ;
4. Coupe les données en train / test ;
5. Entraîne le modèle, évalue les métriques (accuracy, precision, recall, F1) ;
6. Compare la F1 à une baseline "bête" (prédire toujours 0) ;
7. Sauvegarde :
   - le modèle entraîné dans le dossier `models/` ;
   - les métadonnées d'entraînement (métriques, seed, version, etc.)
     dans `registry/metadata.json` ;
   - le fichier `registry/current_model.txt` si le modèle passe le gate.

Ce module illustre une étape typique "Train + Register" d'un pipeline MLOps
minimaliste, avec un premier niveau de gouvernance via une F1 minimale.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Final

import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Chemins et constantes globales
# ---------------------------------------------------------------------------

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_PATH: Final[Path] = ROOT / "data" / "processed.csv"
MODELS_DIR: Final[Path] = ROOT / "models"
REGISTRY_DIR: Final[Path] = ROOT / "registry"
CURRENT_MODEL_PATH: Final[Path] = REGISTRY_DIR / "current_model.txt"
METADATA_PATH: Final[Path] = REGISTRY_DIR / "metadata.json"
# ---------------------------------------------------------------------------

# Fonctions pour la gestion des métadonnées

# ---------------------------------------------------------------------------





def load_metadata() -> list[dict[str, Any]]:

    """

    Charge la liste des métadonnées de modèles depuis le fichier JSON.



    Si le fichier n'existe pas encore, retourne une liste vide.



    Retour

    ------

    list[dict[str, Any]]

        Liste des entrées de métadonnées, chacune décrivant un modèle

        déjà entraîné et enregistré.

    """

    if not METADATA_PATH.exists():

        return []



    with METADATA_PATH.open("r", encoding="utf-8") as file:

        return json.load(file)





def save_metadata(items: list[dict[str, Any]]) -> None:

    """

    Sauvegarde la liste des métadonnées de modèles dans un fichier JSON.



    Paramètres

    ----------

    items : list[dict[str, Any]]

        Liste de dictionnaires contenant les métadonnées à persister.

    """

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)



    with METADATA_PATH.open("w", encoding="utf-8") as file:

        json.dump(items, file, indent=2)





# ---------------------------------------------------------------------------

# Fonctions utilitaires (baseline, pipeline, etc.)

# ---------------------------------------------------------------------------





def compute_baseline_f1(y_true: pd.Series | list[int]) -> float:

    """

    Calcule une F1 de baseline en prédisant toujours 0 (pas de churn).



    Cette baseline représente un modèle trivial, qui sert de référence

    minimale : le modèle appris doit au moins faire mieux que "tout le

    monde reste".



    Paramètres

    ----------

    y_true : pd.Series | list[int]

        Valeurs réelles de la cible (0 ou 1).



    Retour

    ------

    float

        F1-score de cette baseline.

    """

    # baseline : prédire systématiquement 0

    y_pred = [0] * len(y_true)

    return float(f1_score(y_true, y_pred, zero_division=0))





def build_preprocessing_pipeline(

    numeric_cols: list[str],

    categorical_cols: list[str],

) -> ColumnTransformer:

    """

    Construit le préprocesseur scikit-learn pour les données tabulaires.



    - Variables numériques : StandardScaler ;

    - Variables catégorielles : OneHotEncoder (avec gestion des catégories

      inconnues en inférence).



    Paramètres

    ----------

    numeric_cols : list[str]

        Noms des colonnes numériques.

    categorical_cols : list[str]

        Noms des colonnes catégorielles.



    Retour

    ------

    ColumnTransformer

        Transformateur de colonnes prêt à être intégré dans un Pipeline.

    """

    numeric_transformer = Pipeline(

        steps=[

            ("scaler", StandardScaler()),

        ]

    )



    categorical_transformer = OneHotEncoder(handle_unknown="ignore")



    preprocessor = ColumnTransformer(

        transformers=[

            ("num", numeric_transformer, numeric_cols),

            ("cat", categorical_transformer, categorical_cols),

        ]

    )



    return preprocessor





def build_model_pipeline(

    preprocessor: ColumnTransformer,

    seed: int,

) -> Pipeline:

    """

    Construit le pipeline complet de modèle (prétraitement + classifieur).



    Le modèle utilisé est une régression logistique binaire.



    Paramètres

    ----------

    preprocessor : ColumnTransformer

        Préprocesseur de features (scaling + one-hot).

    seed : int

        Graine pseudo-aléatoire pour la régression logistique.



    Retour

    ------

    Pipeline

        Pipeline scikit-learn contenant prétraitement et modèle.

    """

    classifier = LogisticRegression(

        max_iter=200,

        random_state=42,

    )



    pipe = Pipeline(

        steps=[

            ("prep", preprocessor),

            ("clf", classifier),

        ]

    )



    return pipe





# ---------------------------------------------------------------------------

# Fonction principale d'entraînement

# ---------------------------------------------------------------------------





def main(version: str = "v1", seed: int = 42, gate_f1: float = 0.60) -> None:

    """

    Point d'entrée de l'entraînement du modèle de churn.



    Étapes réalisées :

    - chargement du dataset prétraité ;

    - séparation features / cible ;

    - split train / test stratifié ;

    - entraînement du modèle (pipeline) ;

    - calcul des métriques ;

    - comparaison avec un seuil de qualité (gate F1) et une baseline ;

    - sauvegarde du modèle et des métadonnées ;

    - mise à jour du modèle courant si le gate est passé.



    Paramètres

    ----------

    version : str, optionnel

        Identifiant de version logique du modèle (ex. "v1", "v2").

    seed : int, optionnel

        Graine pseudo-aléatoire pour la reproductibilité.

    gate_f1 : float, optionnel

        Seuil minimal de F1 pour autoriser le modèle à passer le gate.



    Exceptions

    ----------

    FileNotFoundError

        Si le fichier `processed.csv` n'existe pas.

    """

    if not DATA_PATH.exists():

        raise FileNotFoundError(

            "Fichier processed.csv introuvable. "

            "Veuillez exécuter d'abord le script de préparation des données."

        )



    # Chargement des données

    df = pd.read_csv(DATA_PATH)



    target_col = "churn"

    X = df.drop(columns=[target_col])

    y = df[target_col].astype(int)



    numeric_cols = ["tenure_months", "num_complaints", "avg_session_minutes"]

    categorical_cols = ["plan_type", "region"]



    # Construction du pipeline complet

    preprocessor = build_preprocessing_pipeline(

        numeric_cols=numeric_cols,

        categorical_cols=categorical_cols,

    )

    model_pipeline = build_model_pipeline(preprocessor=preprocessor, seed=seed)



    # Split train / test

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.25,

        random_state=42,

        stratify=y,

    )



    # Entraînement

    model_pipeline.fit(X_train, y_train)



    # Prédictions sur le test

    y_pred = model_pipeline.predict(X_test)



    # Calcul des métriques d'évaluation

    metrics = {

        "accuracy": float(accuracy_score(y_test, y_pred)),

        "precision": float(

            precision_score(y_test, y_pred, zero_division=0)

        ),

        "recall": float(

            recall_score(y_test, y_pred, zero_division=0)

        ),

        "f1": float(

            f1_score(y_test, y_pred, zero_division=0)

        ),

        "baseline_f1": compute_baseline_f1(y_test),

    }



    # Sauvegarde du modèle entraîné

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    model_filename = f"churn_model_{version}_{timestamp}.joblib"

    model_path = MODELS_DIR / model_filename

    joblib.dump(model_pipeline, model_path)



    # Création de l'entrée de métadonnées pour ce run

    entry: dict[str, Any] = {

        "model_file": model_filename,

        "version": version,

        "trained_at_utc": timestamp,

        "data_file": DATA_PATH.name,

        "seed": seed,

        "metrics": metrics,

        "gate_f1": gate_f1,

        "passed_gate": bool(

            metrics["f1"] >= gate_f1

            and metrics["f1"] >= metrics["baseline_f1"]

        ),

    }



    # Mise à jour du fichier de métadonnées

    items = load_metadata()

    items.append(entry)

    save_metadata(items)



    # Log des métriques

    print("[METRICS]", json.dumps(metrics, indent=2))

    print(f"[OK] Modèle sauvegardé : {model_path}")



    # Logique de "registry" minimal : mise à jour du modèle courant

    if entry["passed_gate"]:

        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

        CURRENT_MODEL_PATH.write_text(

            model_filename,

            encoding="utf-8",

        )

        print(f"[DEPLOY] Modèle activé (current): {model_filename}")

    else:

        print(

            "[DEPLOY] Refusé par le gate : F1 insuffisante "

            "ou baseline non battue."

        )





if __name__ == "__main__":

    main()