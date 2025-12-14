from __future__ import annotations


"""
Script utilitaire de gestion du registry de modèles.


Objectif principal :
- Lister les modèles connus dans `registry/metadata.json` ;
- Permettre de changer le modèle courant (`current_model.txt`) vers :
  - un modèle spécifique (passé via `target`),
  - ou, par défaut, le modèle précédent (rollback d'une version).


Typiquement utilisé comme :
- un outil de rollback simple après un déploiement raté ;
- un mécanisme pédagogique pour illustrer le "model registry" en MLOps.
"""


import json
from pathlib import Path
from typing import Final, Optional


# ---------------------------------------------------------------------------
# Chemins et constantes
# ---------------------------------------------------------------------------


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REGISTRY_DIR: Final[Path] = ROOT / "registry"
CURRENT_MODEL_PATH: Final[Path] = REGISTRY_DIR / "current_model.txt"
METADATA_PATH: Final[Path] = REGISTRY_DIR / "metadata.json"



# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------



def list_models() -> list[str]:
    """
    Retourne la liste des fichiers modèles connus dans le registry.


    La fonction lit le fichier `metadata.json` qui contient une liste
    d'entrées (une par modèle entraîné). Chaque entrée est supposée
    contenir au moins la clé "model_file".


    Retour
    ------
    list[str]
        Liste des noms de fichiers modèles (dans l'ordre d'enregistrement).
        Retourne une liste vide si le fichier de métadonnées n'existe pas.
    """
    if not METADATA_PATH.exists():
        return []


    raw_text = METADATA_PATH.read_text(encoding="utf-8")
    items = json.loads(raw_text)


    return [item["model_file"] for item in items]



def set_current(model_file: str) -> None:
    """
    Met à jour le modèle courant dans le registry.


    Cette fonction écrit simplement le nom du fichier modèle dans
    `current_model.txt`. Elle ne vérifie pas que le fichier modèle existe
    réellement dans le dossier `models/` : cette responsabilité est laissée
    au script appelant / à l'orchestrateur.


    Paramètres
    ----------
    model_file : str
        Nom du fichier modèle à activer (ex. "churn_model_v1_20251212.joblib").
    """
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    CURRENT_MODEL_PATH.write_text(model_file, encoding="utf-8")



# ---------------------------------------------------------------------------
# Point d'entrée : rollback / activation de modèle
# ---------------------------------------------------------------------------



def main(target: Optional[str] = None) -> None:
    """
    Active un modèle spécifique ou effectue un rollback vers le modèle précédent.


    Comportement :
    - Lit la liste des modèles connus via `metadata.json` ;
    - Si `target` est fourni :
        - vérifie qu'il existe dans la liste ;
        - l'active comme modèle courant ;
    - Si `target` est None :
        - prend l'avant-dernier modèle de la liste (`models[-2]`) :
          c'est un rollback simple vers la version précédente ;
        - échoue si un seul modèle est disponible.


    Paramètres
    ----------
    target : str, optionnel
        Nom du fichier modèle à activer explicitement.
        Si None, on fait un rollback vers le modèle précédent.


    Exceptions
    ----------
    FileNotFoundError
        Si `metadata.json` n'existe pas ou ne contient aucun modèle.
    ValueError
        Si le rollback est impossible (un seul modèle) ou si
        `target` ne correspond à aucun modèle connu.
    """
    models = list_models()
    if not models:
        raise FileNotFoundError(
            "Aucun modèle dans metadata.json. "
            "Lancer train.py au moins une fois."
        )


    # Cas rollback automatique : on prend l'avant-dernier modèle
    if target is None:
        if len(models) < 2:
            raise ValueError(
                "Impossible de faire un rollback : un seul modèle existe."
            )
        target = models[-2]


    if target not in models:
        raise ValueError(f"Modèle inconnu : {target}")


    set_current(target)
    print(f"[OK] rollback / activation => current_model = {target}")



if __name__ == "__main__":
    main()
    