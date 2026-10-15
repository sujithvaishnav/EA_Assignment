"""
api.services.feedback_store
---------------------------
Persistent storage for user feedback on RAG answers.
Stores entries in JSON format with thread-safe file operations and aggregate statistics.
"""

import os
import json
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Any, List, Optional

from api.schemas import FeedbackStatsResponse, FeedbackItem

FEEDBACK_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Data_ingestion",
    "feedback.json",
)

_LOCK = Lock()


def _ensure_file():
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)


def save_feedback(query_id: str, rating: str, comment: Optional[str] = None) -> str:
    """Saves a new feedback record and returns the assigned feedback_id."""
    with _LOCK:
        _ensure_file()
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []

        feedback_id = str(uuid.uuid4())
        record = {
            "feedback_id": feedback_id,
            "query_id": query_id,
            "rating": rating,
            "comment": comment,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        data.append(record)

        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return feedback_id


def get_feedback_stats() -> FeedbackStatsResponse:
    """Returns total feedback count, thumbs up/down distribution, and satisfaction rate."""
    with _LOCK:
        _ensure_file()
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                data: List[Dict[str, Any]] = json.load(f)
        except Exception:
            data = []

        total = len(data)
        thumbs_up = sum(1 for item in data if item.get("rating") == "thumbs_up")
        thumbs_down = sum(1 for item in data if item.get("rating") == "thumbs_down")
        rate = round((thumbs_up / total * 100.0), 1) if total > 0 else 0.0

        items = [
            FeedbackItem(
                feedback_id=item["feedback_id"],
                query_id=item["query_id"],
                rating=item["rating"],
                comment=item.get("comment"),
                created_at=item["created_at"],
            )
            for item in reversed(data)  # newest first
        ]

        return FeedbackStatsResponse(
            total_feedback=total,
            thumbs_up=thumbs_up,
            thumbs_down=thumbs_down,
            satisfaction_rate_percent=rate,
            feedback_entries=items,
        )
