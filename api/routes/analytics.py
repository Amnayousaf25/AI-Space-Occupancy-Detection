from fastapi import APIRouter, Query
from typing import Optional
from src.analytics import compute_occupancy_analytics

router = APIRouter(prefix="/analytics", tags=["Analytics"])

@router.get("/summary")
def get_analytics_summary(space: Optional[str] = Query(None, description="Filter by space: computer_lab, lecture_room, auditorium")):
    return compute_occupancy_analytics(space_type=space)
