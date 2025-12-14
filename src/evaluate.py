from __future__ import annotations


"""
Module d'entraînement d’un modèle de churn avec tuning du seuil optimal (F1).


Ce script :
1. Charge le dataset prétraité `data/processed.csv` ;
2. Crée un pipeline scikit-learn :
   - Standardisation des variables numériques,
   - OneHotEncoding des variables catégorielles,
   - Régression logistique ;
3. Découpe train/test avec stratification ;
4. Entraîne le modèle ;
5. Calcule :
   - Les métriques standard avec seuil = 0.5,
   - Le seuil optimal maximisant la F1,
   - Une baseline triviale (prédire toujours 0) ;
6. Sauvegarde :
   - Le modèle dans `models/`,
   - Les métadonnées dans `registry/metadata.json`,
   - Le modèle courant dans `registry/current_model.txt` si le gate F1 est validé.


Ce script illustre une étape “Train + Eval + Register” comme dans un vrai pipeline MLOps,
avec un composant de tuning simple mais efficace : optimisation du seuil.
"""


import json
from datetime import datetime
from pathlib import Path
from typing import Any, Final


import joblib
import numpy as np
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
# Constantes globales
# ---------------------------------------------------------------------------


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_PATH: Final[Path] = ROOT / "data" / "processed.csv"
MODELS_DIR: Final[Path] = ROOT / "models"
REGISTRY_DIR: Final[Path] = ROOT / "registry"
CURRENT_MODEL_PATH: Final[Path] = REGISTRY_DIR / "current_model.txt"
METADATA_PATH: Final[Path] = REGISTRY_DIR / "metadata.json"



# ---------------------------------------------------------------------------
# Gestion des métadonnées
# ---------------------------------------------------------------------------



def load_metadata() -> list[dict[str, Any]]:
    """
    Charge la liste des métadonnées des modèles.


    Retourne une liste vide si aucun modèle n'a encore été enregistré.
    """
    if not METADATA_PATH.exists():
        return []


    with METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)



def save_metadata(items: list[dict[str, Any]]) -> None:
    """
    Sauvegarde les métadonnées dans un fichier JSON structuré.


    Paramètres
    ----------
    items : list[dict[str, Any]]
        Liste complète des entrées de métadonnées.
    """
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(items, file, indent=2)



# ---------------------------------------------------------------------------
# Métriques : baseline et tuning du seuil
# ---------------------------------------------------------------------------



def compute_baseline_f1(y_true: pd.Series | list[int]) -> float:
    """
    Baseline triviale : prédire toujours 0 (personne ne churn).


    Sert de seuil minimal. Un bon modèle doit battre ce baseline.


    Paramètres
    ----------
    y_true : array-like
        Vraies valeurs de churn.


    Retour
    ------
    float
        Score F1 de la baseline.
    """
    y_pred = [0] * len(y_true)
    return float(f1_score(y_true, y_pred, zero_division=0))



def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, float]:
    """
    Cherche le seuil de probabilité maximisant la F1.


    Le seuil utilisé par défaut (0.5) est rarement optimal,
    surtout sur des datasets déséquilibrés.


    On teste 81 seuils entre 0.10 et 0.90.


    Paramètres
    ----------
    y_true : np.ndarray
        Vraies valeurs de churn.
    y_proba : np.ndarray
        Probabilités prédites pour la classe positive (1).


    Retour
    ------
    (float, float)
        (seuil optimal, F1 maximale obtenue).
    """
    best_threshold = 0.5
    best_f1 = 0.0


    for t in np.linspace(0.1, 0.9, 81):
        y_hat = (y_proba >= t).astype(int)
        score = f1_score(y_true, y_hat, zero_division=0)


        if score > best_f1:
            best_f1 = score
            best_threshold = t


    return float(best_threshold), float(best_f1)



# ---------------------------------------------------------------------------
# Fonction principale : entraînement + registry
# ---------------------------------------------------------------------------



def main(version: str = "v1", seed: int = 42, gate_f1: float = 0.70) -> None:
    """
    Entraîne un modèle de churn, évalue ses performances, enregistre
    le modèle et ses métadonnées, et décide s’il passe le gate F1.


    Paramètres
    ----------
    version : str
        Version logique du modèle.
    seed : int
        Graine pseudo-aléatoire pour reproductibilité.
    gate_f1 : float
        Seuil minimal de F1 pour valider le modèle.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "processed.csv introuvable. Exécuter prepare_data.py d'abord."
        )


    df = pd.read_csv(DATA_PATH)


    target = "churn"
    X = df.drop(columns=[target])
    y = df[target].astype(int)


    num_cols = ["tenure_months", "num_complaints", "avg_session_minutes"]
    cat_cols = ["plan_type", "region"]


    # Prétraitement
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )


    model = LogisticRegression(max_iter=200, random_state=seed)


    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", model),
        ]
    )


    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )


    # Entraînement
    pipe.fit(X_train, y_train)


    # Probabilités pour la classe positive
    y_proba = pipe.predict_proba(X_test)[:, 1]


    # Prédiction standard (seuil 0.5)
    y_pred_default = (y_proba >= 0.5).astype(int)


    # Métriques de base
    metrics_default = {
        "accuracy": accuracy_score(y_test, y_pred_default),
        "precision": precision_score(y_test, y_pred_default, zero_division=0),
        "recall": recall_score(y_test, y_pred_default, zero_division=0),
        "f1_threshold_05": f1_score(y_test, y_pred_default, zero_division=0),
    }


    # Seuil optimal
    best_threshold, best_f1 = find_best_threshold(y_test.to_numpy(), y_proba)


    # Baseline triviale
    baseline = compute_baseline_f1(y_test)


    # Final metrics
    metrics = {
        **metrics_default,
        "f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "baseline_f1": float(baseline),
    }


    # Sauvegarde du modèle
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_filename = f"churn_model_{version}_{timestamp}.joblib"
    model_path = MODELS_DIR / model_filename
    joblib.dump(pipe, model_path)


    # Métadonnées pour cette version du modèle
    entry: dict[str, Any] = {
        "model_file": model_filename,
        "version": version,
        "trained_at_utc": timestamp,
        "data_file": DATA_PATH.name,
        "seed": seed,
        "metrics": metrics,
        "gate_f1": gate_f1,
        "passed_gate": bool(
            metrics["f1"] >= gate_f1 and metrics["f1"] >= metrics["baseline_f1"]
        ),
    }


    # Mise à jour du registry
    items = load_metadata()
    items.append(entry)
    save_metadata(items)


    # Logs
    print("[METRICS]", json.dumps(metrics, indent=2))
    print(f"[OK] Modèle sauvegardé : {model_path}")


    # Déploiement du modèle (registry minimaliste)
    if entry["passed_gate"]:
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        CURRENT_MODEL_PATH.write_text(model_filename, encoding="utf-8")
        print(f"[DEPLOY] Modèle activé : {model_filename}")
    else:
        print("[DEPLOY] Refusé : F1 insuffisante ou baseline non battue.")



if __name__ == "__main__":
    main()