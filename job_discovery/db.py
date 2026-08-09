"""
MongoDB connection helper.
Returns a cached PyMongo MongoClient and database handle.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

from pymongo import MongoClient
from pymongo.database import Database

from config import get_settings


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    settings = get_settings()
    return MongoClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=5000,  # fail fast instead of hanging 30s
        connectTimeoutMS=5000,
        socketTimeoutMS=10000,
    )


def get_db() -> Database:
    settings = get_settings()
    return get_client()[settings.mongodb_database]
