"""
api.routers.feedback
--------------------
Endpoints for collecting user feedback and tracking answer quality.
"""

from fastapi import APIRouter, status
from api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStatsResponse,
)
from api.services.feedback_store import save_feedback, get_feedback_stats

router = APIRouter(tags=["User Feedback & Monitoring"])


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit Answer Feedback",
)
def submit_answer_feedback(payload: FeedbackRequest):
    """
    Submits user feedback on a generated answer by referencing its `query_id`.
    - **rating**: `thumbs_up` (helpful/accurate) or `thumbs_down` (unhelpful/hallucinated).
    - **comment**: Optional constructive feedback or correction.
    """
    feedback_id = save_feedback(
        query_id=payload.query_id,
        rating=payload.rating,
        comment=payload.comment,
    )
    return FeedbackResponse(
        status="success",
        feedback_id=feedback_id,
        message=f"Feedback recorded successfully for query '{payload.query_id}'.",
    )


@router.get(
    "/feedback",
    response_model=FeedbackStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Feedback Statistics & Audit Log",
)
def get_feedback_report():
    """
    Returns aggregate feedback metrics:
    - Total feedback submissions
    - Thumbs up vs Thumbs down breakdown
    - Overall satisfaction percentage
    - Reverse-chronological audit log of recent submissions
    """
    return get_feedback_stats()
