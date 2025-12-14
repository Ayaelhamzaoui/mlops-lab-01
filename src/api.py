from __future__ import annotations


"""
API FastAPI de prédiction de churn pour le lab MLOps.


Ce service :
- charge dynamiquement le modèle courant indiqué dans `registry/current_model.txt` ;
- expose un endpoint `/health` pour vérifier l'état de l'API et du modèle ;
- expose un endpoint `/predict` pour faire une prédiction de churn à partir
  de features simples (tenure, plaintes, durée de session, type d’abonnement,
  région) ;
- journalise chaque requête de prédiction dans `logs/predictions.log` au
  format JSON (une ligne par prédiction).


Cette API illustre une étape "Serve" dans un pipeline MLOps minimal :
un modèle versionné est promu côté registry, puis utilisé par un service
d’inférence léger.
"""


import json
import time
from pathlib import Path
from typing import Any, Optional


import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Constantes de chemin
# ---------------------------------------------------------------------------


ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = ROOT / "models"
REGISTRY_DIR: Path = ROOT / "registry"
CURRENT_MODEL_PATH: Path = REGISTRY_DIR / "current_model.txt"
LOG_PATH: Path = ROOT / "logs" / "predictions.log"


# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------


app = FastAPI(title="MLOps Lab 01 - Churn API")


# ---------------------------------------------------------------------------
# Schéma d'entrée (Pydantic)
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """
    Modèle de requête pour l'endpoint /predict.


    Champs
    ------
    tenure_months : int
        Ancienneté du client en mois (>= 0).
    num_complaints : int
        Nombre de plaintes enregistrées (>= 0).
    avg_session_minutes : float
        Durée moyenne de session en minutes (>= 0).
    plan_type : str
        Type d’abonnement ("basic", "premium", etc.). Normalisé en minuscules.
    region : str
        Région ("NA", "EU", "AF", "AS", ...). Normalisée en majuscules.
    request_id : Optional[str]
        Identifiant de corrélation optionnel fourni par le client.
    """


    tenure_months: int = Field(..., ge=0, le=200)
    num_complaints: int = Field(..., ge=0, le=50)
    avg_session_minutes: float = Field(..., ge=0.0, le=500.0)
    plan_type: str
    region: str
    request_id: Optional[str] = None



# ---------------------------------------------------------------------------
# Cache de modèle en mémoire
# ---------------------------------------------------------------------------


_model_cache: dict[str, Any] = {"name": None, "model": None}


# ---------------------------------------------------------------------------
# Fonctions utilitaires (chargement modèle, logging)
# ---------------------------------------------------------------------------



def get_current_model_name() -> str:
    """
    Lit le nom du modèle courant dans le fichier de registry.


    Retour
    ------
    str
        Nom du fichier modèle (ex. "churn_model_v1_20251210_120000.joblib").


    Exceptions
    ----------
    FileNotFoundError
        Si le fichier n'existe pas ou est vide.
    """
    if not CURRENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Aucun modèle courant. Lancer train.py (avec gate) d'abord."
        )


    name = CURRENT_MODEL_PATH.read_text(encoding="utf-8").strip()
    if not name:
        raise FileNotFoundError(
            "Fichier current_model.txt vide. Aucun modèle activé."
        )


    return name



def load_model_if_needed() -> tuple[str, Any]:
    """
    Charge le modèle courant en mémoire si nécessaire.


    Utilise un cache simple en mémoire (_model_cache) pour éviter
    de recharger le même modèle à chaque requête.


    Retour
    ------
    (str, Any)
        Tuple (nom_du_modèle, objet_modèle_scikit_learn).


    Exceptions
    ----------
    FileNotFoundError
        Si le modèle indiqué dans le registry n'existe pas sur disque.
    """
    name = get_current_model_name()


    # Si le modèle courant est déjà en cache, on le réutilise.
    if _model_cache["name"] == name and _model_cache["model"] is not None:
        return name, _model_cache["model"]


    # Sinon, chargement depuis le disque
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable sur disque : {path}")


    model = joblib.load(path)
    _model_cache["name"] = name
    _model_cache["model"] = model


    return name, model



def log_prediction(payload: dict[str, Any]) -> None:
    """
    Ajoute une ligne de log JSON dans le fichier de prédictions.


    Chaque prédiction est stockée sur une seule ligne, au format JSON,
    ce qui permet une ingestion facile par un outil de log / monitoring.


    Paramètres
    ----------
    payload : dict[str, Any]
        Dictionnaire représentant la prédiction et son contexte.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")



# ---------------------------------------------------------------------------
# Endpoints FastAPI
# ---------------------------------------------------------------------------



@app.get("/health")
def health() -> dict[str, Any]:
    """
    Endpoint de santé de l'API.


    Vérifie simplement qu'un modèle courant est bien configuré.


    Retour
    ------
    dict
        - status : "ok" ou "error"
        - current_model : nom du modèle courant (si OK)
        - detail : message d'erreur (si error)
    """
    try:
        model_name = get_current_model_name()
        return {"status": "ok", "current_model": model_name}
    except Exception as exc:  # pragma: no cover - simple endpoint de debug
        return {"status": "error", "detail": str(exc)}



@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    """
    Endpoint de prédiction de churn.


    Reçoit les features d’un client, applique le modèle courant et retourne :
    - la prédiction binaire (0 = reste, 1 = churn),
    - la probabilité de churn,
    - la latence d'inférence en millisecondes,
    - le nom du modèle utilisé,
    - l'identifiant de requête (s'il est fourni).


    Paramètres
    ----------
    req : PredictRequest
        Données d'entrée validées par Pydantic.


    Retour
    ------
    dict
        Résultat de la prédiction et métadonnées associées.


    Exceptions
    ----------
    HTTPException
        - 500 si le modèle ne peut pas être chargé,
        - 400 si une erreur survient lors de la prédiction.
    """
    try:
        model_name, model = load_model_if_needed()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


    # Normalisation minimale côté API (cohérente avec le prétraitement)
    features = {
        "tenure_months": req.tenure_months,
        "num_complaints": req.num_complaints,
        "avg_session_minutes": req.avg_session_minutes,
        "plan_type": req.plan_type.strip().lower(),
        "region": req.region.strip().upper(),
    }


    X_df = pd.DataFrame([features])


    start = time.perf_counter()
    try:
        proba = float(model.predict_proba(X_df)[0][1])
        pred = int(proba >= 0.5)  # seuil fixe 0.5 (peut être tuné)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Erreur de prédiction : {exc}",
        ) from exc
    latency_ms = (time.perf_counter() - start) * 1000.0


    out: dict[str, Any] = {
        "request_id": req.request_id,
        "model_version": model_name,
        "prediction": pred,
        "probability": round(proba, 6),
        "latency_ms": round(latency_ms, 3),
        "features": features,
        "ts": int(time.time()),
    }


    log_prediction(out)
    return out