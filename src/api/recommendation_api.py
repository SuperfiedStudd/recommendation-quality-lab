"""
recommendation_api.py — Lightweight FastAPI layer for the Recommendation Loop
Author: Jasjyot Singh

Provides a REST endpoint to serve recommendations from the pipeline state.
Usage: uvicorn src.api.recommendation_api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sys
import os

# Ensure src/ is importable
sys.path.insert(0, os.path.abspath('src'))

from data.event_replay import load_interactions, generate_events
from features.state_manager import StateManager
from serving.recommender_service import RecommenderService


app = FastAPI(title="DiscoveryRank Recommendation API")

# Global state
master_df = None
state_mgr = None
recommender = None


class RecommendationItem(BaseModel):
    item_id: str
    score: float


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: list[RecommendationItem]


@app.on_event("startup")
def startup_event():
    """Load data and warm up state on server startup."""
    global master_df, state_mgr, recommender
    
    print("Initializing Recommendation API...")
    try:
        master_df = load_interactions(os.path.abspath("outputs/pipeline_sample.csv"))
    except FileNotFoundError:
        print("WARNING: pipeline_sample.csv not found. API will fail until data is available.")
        return

    # Initialize components
    state_mgr = StateManager()
    
    # Warm up state with the first 10000 events
    print("Warming up state with 10000 events...")
    warmup_df = master_df.sort_values("time_ms").head(10000)
    events = generate_events(warmup_df)
    for evt in events:
        state_mgr.process_event(evt)
        
    print(f"State ready: {state_mgr.summary()}")
    
    # default to hybrid policy (diversity aware)
    recommender = RecommenderService(master_df=master_df, state_manager=state_mgr, strategy="diversity_aware")
    print("Recommendation service initialized.")


@app.get("/recommend", response_model=RecommendationResponse)
def get_recommendations(user_id: int, session_id: str, k: int = 10):
    """
    Generate Top-K recommendations for a specific user and session.
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Service not initialized. Check data availability.")
        
    try:
        # RecommenderService expects integer user_id
        recs_df = recommender.recommend(user_id=user_id, session_id=session_id, k=k)
        
        items = []
        if not recs_df.empty:
            for _, row in recs_df.iterrows():
                items.append(RecommendationItem(
                    item_id=str(int(row["video_id"])),
                    score=float(row["score"])
                ))
            
        return RecommendationResponse(user_id=str(user_id), recommendations=items)
        
    except Exception as e:
        import traceback
        import logging
        logging.error(f"Error serving recommendations for user_id={user_id}, session_id={session_id}: {e}")
        logging.error(traceback.format_exc())
        # Return empty list instead of HTTP 500/400 to degrade gracefully
        return RecommendationResponse(user_id=str(user_id), recommendations=[])
