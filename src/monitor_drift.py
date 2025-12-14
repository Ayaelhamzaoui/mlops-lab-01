from __future__ import annotations


"""
Script de détection simple de data drift sur les features d’entrée.


Ce script :
1. Charge les statistiques d'entraînement (moyenne / écart-type) depuis
   `registry/train_stats.json` (généré par `prepare_data.py`) ;
2. Charge les requêtes de prédiction récentes depuis `logs/predictions.log`
   (une prédiction par ligne au format JSON) ;
3. Compare la moyenne des features observées en production aux moyennes
   d'entraînement à l'aide d'un score Z :
      z = |mean_prod - mean_train| / std_train
4. Déclenche une alerte si z >= z_threshold pour au moins une feature ;
5. Loggue un message indiquant la possibilité d'envoyer l'alerte vers
   un outil de monitoring externe (si `MONITORING_TOKEN` est défini).


Ce script illustre un mécanisme de monitoring très simple mais pédagogique
pour un lab MLOps : détecter un drift sur les distributions des features
d'entrée, en se basant sur les moyennes.
"""


import json
import os
from pathlib import Path
from typing import Final


import pandas as pd


# ---------------------------------------------------------------------------
# Configuration & chemins
# ---------------------------------------------------------------------------


MONITORING_TOKEN: str | None = os.getenv("MONITORING_TOKEN")


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
TRAIN_STATS_PATH: Final[Path] = ROOT / "registry" / "train_stats.json"
LOG_PATH: Final[Path] = ROOT / "logs" / "predictions.log"


# ---------------------------------------------------------------------------
# Fonction principale de drift-check
# ---------------------------------------------------------------------------



def main(last_n: int = 200, z_threshold: float = 2.5) -> None:
    """
    Analyse les derniers logs de prédiction pour détecter un éventuel drift.


    Le principe est le suivant :
    - On récupère les N dernières prédictions dans le fichier de log ;
    - On en extrait les features numéri ques (tenure, plaintes, durée de
      session) ;
    - On calcule la moyenne observée sur ces requêtes ;
    - On compare à la moyenne d'entraînement via un score Z :
        z = |mean_prod - mean_train| / std_train
      (avec un fallback std_train = 1.0 si l'écart-type est très faible).


    Paramètres
    ----------
    last_n : int, optionnel
        Nombre maximum de requêtes récentes sur lesquelles faire l'analyse.
        Défaut : 200.
    z_threshold : float, optionnel
        Seuil à partir duquel on considère qu'il y a drift sur une feature.
        Défaut : 2.5.
    """
    # Vérification de l'existence des fichiers nécessaires
    if not TRAIN_STATS_PATH.exists():
        raise FileNotFoundError(
            "train_stats.json introuvable. "
            "Lancer d'abord prepare_data.py pour générer les statistiques."
        )


    if not LOG_PATH.exists():
        print("[INFO] Aucun log trouvé. Appeler l'endpoint /predict d'abord.")
        return


    # Chargement des stats d'entraînement (moyennes, std par feature)
    stats = json.loads(TRAIN_STATS_PATH.read_text(encoding="utf-8"))


    # Chargement des lignes de log (une prédiction par ligne JSON)
    rows: list[dict] = []
    with LOG_PATH.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))


    if not rows:
        print("[INFO] Fichier de logs vide.")
        return


    # On ne garde que les N dernières requêtes
    rows = rows[-last_n:]


    # Transformation en DataFrame à partir de JSON imbriqué
    df = pd.json_normalize(rows)


    # Colonnes de features à vérifier dans les logs
    log_feature_cols = [
        "features.tenure_months",
        "features.num_complaints",
        "features.avg_session_minutes",
    ]


    missing_cols = [col for col in log_feature_cols if col not in df.columns]
    if missing_cols:
        print(f"[WARN] Colonnes manquantes dans les logs : {missing_cols}")
        return


    print(f"=== Drift check sur {len(df)} requêtes récentes ===")
    alerts = 0


    # Mapping entre noms de colonnes dans les logs et clés du fichier de stats
    mapping: dict[str, str] = {
        "features.tenure_months": "tenure_months",
        "features.num_complaints": "num_complaints",
        "features.avg_session_minutes": "avg_session_minutes",
    }


    for log_col, stat_key in mapping.items():
        mean_prod = float(df[log_col].mean())
        mean_train = float(stats[stat_key]["mean"])
        std_train = float(stats[stat_key]["std"])


        # Protection contre un std trop petit (évite division par 0)
        if std_train <= 1e-9:
            std_train = 1.0


        z = abs(mean_prod - mean_train) / std_train


        print(
            f"- {stat_key}: "
            f"mean_prod={mean_prod:.3f} | "
            f"mean_train={mean_train:.3f} | "
            f"z={z:.3f}"
        )


        if z >= z_threshold:
            alerts += 1
            print(
                f"  ALERTE: drift probable sur {stat_key} "
                f"(z >= {z_threshold})"
            )


    # Simulation d'un hook vers un outil de monitoring externe
    if alerts > 0 and MONITORING_TOKEN:
        print(
            "[INFO] Drift détecté — envoi possible vers un outil de "
            f"monitoring externe (token={MONITORING_TOKEN[:3]}***)"
        )


    if alerts == 0:
        print("Résultat : aucun drift détecté.")
    else:
        print(
            f"Résultat : {alerts} alerte(s) de drift. "
            "Analyse recommandée + retraining possible."
        )



if __name__ == "__main__":
    main()