from __future__ import annotations

"""
Module de prétraitement des données de churn.

Ce script :
1. Charge le fichier brut `data/raw.csv` ;
2. Applique des règles de nettoyage simples :
   - clip des valeurs négatives sur `avg_session_minutes` ;
   - normalisation des champs catégoriels (`plan_type`, `region`) ;
3. Exécute des contrôles qualité (schéma, taux de valeurs manquantes,
   type des colonnes numériques) ;
4. Sauvegarde :
   - un fichier nettoyé `data/processed.csv` ;
   - un fichier `registry/train_stats.json` contenant moyenne et
     écart-type des variables numériques (pour normalisation ultérieure).

Ce module est typiquement utilisé comme étape "prétraitement" d’un pipeline
MLOps avant l’entraînement du modèle.
"""

from pathlib import Path
from typing import Final

import json
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes de chemin
# ---------------------------------------------------------------------------

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = ROOT / "data"
RAW_PATH: Final[Path] = DATA_DIR / "raw.csv"
PROCESSED_PATH: Final[Path] = DATA_DIR / "processed.csv"
TRAIN_STATS_PATH: Final[Path] = ROOT / "registry" / "train_stats.json"
# ---------------------------------------------------------------------------

# Fonctions de validation / qualité des données

# ---------------------------------------------------------------------------





def data_quality_checks(df: pd.DataFrame) -> None:

    """

    Valide la qualité minimale du DataFrame d'entrée.



    Cette fonction lève une exception si :

    - des colonnes attendues sont manquantes ;

    - un champ présente plus de 5 % de valeurs nulles ;

    - certains champs supposés numériques ne le sont pas.



    Paramètres

    ----------

    df : pd.DataFrame

        Jeu de données à valider.



    Exceptions

    ----------

    ValueError

        Si le schéma ou les taux de valeurs manquantes ne sont pas conformes.

    """

    expected = {

        "tenure_months",

        "num_complaints",

        "avg_session_minutes",

        "plan_type",

        "region",

        "churn",

    }



    # Vérification de la présence de toutes les colonnes attendues

    missing = expected - set(df.columns)

    if missing:

        raise ValueError(f"Colonnes manquantes dans le dataset : {missing}")



    # Taux de valeurs manquantes (en proportion)

    null_rate = df.isna().mean().to_dict()

    too_null = {name: rate for name, rate in null_rate.items() if rate > 0.05}

    if too_null:

        raise ValueError(

            "Trop de valeurs manquantes (> 5 %) pour les colonnes : "

            f"{too_null}"

        )



    # Vérification du type numérique pour les variables quantitatives

    if not pd.api.types.is_numeric_dtype(df["tenure_months"]):

        raise ValueError("La colonne 'tenure_months' doit être numérique.")



    if not pd.api.types.is_numeric_dtype(df["num_complaints"]):

        raise ValueError("La colonne 'num_complaints' doit être numérique.")



    if not pd.api.types.is_numeric_dtype(df["avg_session_minutes"]):

        raise ValueError("La colonne 'avg_session_minutes' doit être numérique.")





# ---------------------------------------------------------------------------

# Fonctions utilitaires

# ---------------------------------------------------------------------------





def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:

    """

    Applique les transformations de nettoyage de base au dataset brut.



    Opérations réalisées :

    - clip des valeurs négatives de `avg_session_minutes` à 0.0 ;

    - mise en minuscules et strip des espaces pour `plan_type` ;

    - mise en majuscules et strip des espaces pour `region`.



    Paramètres

    ----------

    df : pd.DataFrame

        DataFrame issu de `raw.csv`.



    Retour

    ------

    pd.DataFrame

        DataFrame nettoyé.

    """

    df = df.copy()



    df["avg_session_minutes"] = df["avg_session_minutes"].clip(lower=0.0)



    df["plan_type"] = (

        df["plan_type"]

        .astype(str)

        .str.lower()

        .str.strip()

    )



    df["region"] = (

        df["region"]

        .astype(str)

        .str.upper()

        .str.strip()

    )



    return df





def compute_numeric_stats(

    df: pd.DataFrame,

    numeric_cols: list[str],

) -> dict[str, dict[str, float]]:

    """

    Calcule la moyenne et l'écart-type des colonnes numériques.



    L'écart-type est calculé avec `ddof=1` (écart-type de l'échantillon).

    Si l'écart-type vaut 0, on force à 1.0 pour éviter les divisions par 0

    lors d'une éventuelle normalisation (z-score).



    Paramètres

    ----------

    df : pd.DataFrame

        Jeu de données prétraité.

    numeric_cols : list[str]

        Liste des noms de colonnes numériques à analyser.



    Retour

    ------

    dict[str, dict[str, float]]

        Dictionnaire du type :

        {

            "col": {"mean": ..., "std": ...},

            ...

        }

    """

    stats: dict[str, dict[str, float]] = {}



    for col in numeric_cols:

        mean_value = float(df[col].mean())

        std_value = float(df[col].std(ddof=1) or 1.0)



        stats[col] = {

            "mean": mean_value,

            "std": std_value,

        }



    return stats





# ---------------------------------------------------------------------------

# Point d'entrée script

# ---------------------------------------------------------------------------





def main() -> None:

    """

    Point d'entrée du script de prétraitement.



    Étapes :

    1. Vérifie l'existence de `raw.csv` ;

    2. Charge les données brutes ;

    3. Applique les transformations de nettoyage ;

    4. Exécute les contrôles de qualité ;

    5. Sauvegarde le dataset prétraité dans `processed.csv` ;

    6. Calcule et enregistre les statistiques de base des variables

       numériques dans `train_stats.json`.



    Exceptions

    ----------

    FileNotFoundError

        Si le fichier `raw.csv` n'existe pas.

    ValueError

        Si les contrôles de qualité échouent.

    """

    if not RAW_PATH.exists():

        raise FileNotFoundError(

            f"Fichier brut introuvable : {RAW_PATH}. "

            "Assurez-vous d'avoir exécuté le script de génération."

        )



    # Chargement des données brutes

    df_raw = pd.read_csv(RAW_PATH)



    # Nettoyage des données

    df_clean = clean_raw_data(df_raw)



    # Contrôles qualité (lève des exceptions en cas de problème)

    data_quality_checks(df_clean)



    # Sauvegarde du dataset prétraité

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(PROCESSED_PATH, index=False)



    # Calcul et sauvegarde des statistiques d'entraînement

    numeric_cols = ["tenure_months", "num_complaints", "avg_session_minutes"]

    stats = compute_numeric_stats(df_clean, numeric_cols=numeric_cols)



    TRAIN_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with TRAIN_STATS_PATH.open("w", encoding="utf-8") as f:

        json.dump(stats, f, indent=2)



    print(f"[OK] Fichier prétraité généré : {PROCESSED_PATH}")

    print(f"[OK] Statistiques d'entraînement générées : {TRAIN_STATS_PATH}")





if __name__ == "__main__":

    main()