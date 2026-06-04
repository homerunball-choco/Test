"""
요청 간 재사용되는 무거운 리소스(DB 엔진, S3, ML 모델, STT 클라이언트) 캐시.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from delivery_learning.config import settings
from delivery_learning.ml_models import TrainedModelBundle

_whisper_models: dict[str, Any] = {}


@lru_cache(maxsize=1)
def get_db_engine() -> Engine:
    return create_engine(settings.db_connection_string, fast_executemany=True, pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
    )


@lru_cache(maxsize=4)
def get_model_bundle(model_dir: str) -> TrainedModelBundle:
    return TrainedModelBundle.load(str(Path(model_dir).resolve()))


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 필요합니다(동작: OpenAI STT).")
    return OpenAI(api_key=api_key)


def get_local_whisper_model(model_name: str) -> Any:
    name = model_name or "base"
    cached = _whisper_models.get(name)
    if cached is not None:
        return cached
    import whisper

    model = whisper.load_model(name)
    _whisper_models[name] = model
    return model
