from pathlib import Path
from typing import Final


import numpy as np
import pandas as pd


"""
Module de génération d'un dataset synthétique de churn client.


Ce script génère un fichier CSV `data/raw.csv` contenant des données
synthétiques d'abonnement (tenure, plaintes, usage, etc.) et une variable
binaire `churn` indiquant si le client quitte le service.


Il est pensé comme point de départ pour un lab MLOps :
- jeu de données contrôlé et reproductible ;
- logique métier simple mais réaliste ;
- génération déterministe grâce à une graine pseudo-aléatoire.
"""


# ---------------------------------------------------------------------------
# Constantes de chemin
# ---------------------------------------------------------------------------


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = ROOT / "data"
RAW_PATH: Final[Path] = DATA_DIR / "raw.csv"



# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------



def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Calcule la fonction sigmoïde de manière vectorisée.


    La sigmoïde permet de transformer une valeur réelle (logit) en probabilité
    comprise entre 0 et 1.


    Paramètres
    ----------
    x : np.ndarray
        Tableau de valeurs réelles (logits).


    Retour
    ------
    np.ndarray
        Tableau de probabilités dans [0, 1], de même forme que `x`.
    """
    return 1.0 / (1.0 + np.exp(-x))



def generate_churn_dataset(n: int, seed: int = 42) -> pd.DataFrame:
    """
    Génère un dataset synthétique de churn client.


    La génération repose sur un modèle logistique simple :
    - plus le nombre de plaintes augmente, plus la probabilité de churn monte ;
    - plus l'ancienneté (tenure) et le temps de session moyen sont élevés,
      plus la probabilité de churn diminue ;
    - les clients premium ont un churn plus faible que les clients basic ;
    - l'effet de la région est modélisé comme des ajustements du logit.


    Paramètres
    ----------
    n : int
        Nombre de lignes (clients) à générer.
    seed : int, optionnel
        Graine pour le générateur pseudo-aléatoire (reproductibilité).


    Retour
    ------
    pd.DataFrame
        DataFrame contenant les colonnes :
        - tenure_months : ancienneté en mois ;
        - num_complaints : nombre de plaintes ;
        - avg_session_minutes : durée moyenne de session (minutes) ;
        - plan_type : type d'abonnement ("basic" ou "premium") ;
        - region : région ("NA", "EU", "AF", "AS") ;
        - churn : variable binaire (0 = reste, 1 = churn).
    """
    rng = np.random.default_rng(seed)


    # Variables explicatives numériques
    tenure_months = rng.integers(low=1, high=60, size=n)
    num_complaints = rng.poisson(lam=1.2, size=n)
    avg_session_minutes = rng.normal(loc=35, scale=12, size=n)
    avg_session_minutes = np.clip(avg_session_minutes, 1, 120)


    # Variables explicatives catégorielles
    plan_type = rng.choice(
        ["basic", "premium"],
        size=n,
        p=[0.72, 0.28],
    )
    region = rng.choice(
        ["NA", "EU", "AF", "AS"],
        size=n,
        p=[0.25, 0.30, 0.25, 0.20],
    )


    # Modèle logistique : combinaison linéaire des features
    base_logit = (
        1.2
        + 0.55 * num_complaints
        - 0.03 * tenure_months
        - 0.02 * avg_session_minutes
        + np.where(plan_type == "premium", -0.35, 0.0)
    )


    # Effet de la région sur le logit (ajustements additifs)
    base_logit += np.where(region == "EU", -0.05, 0.0)
    base_logit += np.where(region == "AF", 0.08, 0.0)


    # Passage en probabilité via la sigmoïde
    churn_proba = sigmoid(base_logit)


    # Échantillonnage de la variable binaire churn (0/1)
    churn = rng.binomial(n=1, p=churn_proba, size=n)


    df = pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "num_complaints": num_complaints,
            "avg_session_minutes": np.round(avg_session_minutes, 2),
            "plan_type": plan_type,
            "region": region,
            "churn": churn,
        }
    )


    return df



# ---------------------------------------------------------------------------
# Point d'entrée script
# ---------------------------------------------------------------------------



def main(n: int = 1200, seed: int = 42) -> None:
    """
    Point d'entrée pour générer et sauvegarder le dataset sur disque.


    Paramètres
    ----------
    n : int, optionnel
        Nombre de lignes à générer. Défaut : 1200.
    seed : int, optionnel
        Graine pseudo-aléatoire pour rendre la génération reproductible.
        Défaut : 42.


    Effets de bord
    --------------
    - Crée le dossier `data/` s'il n'existe pas.
    - Écrit un fichier CSV à l'emplacement défini par `RAW_PATH`.
    - Affiche un message de confirmation sur la sortie standard.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)


    df = generate_churn_dataset(n=n, seed=seed)
    df.to_csv(RAW_PATH, index=False)


    print(
        f"[OK] Dataset généré : {RAW_PATH} "
        f"(rows={len(df)}, seed={seed})"
    )



if __name__ == "__main__":
    main()
