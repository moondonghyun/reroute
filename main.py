# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from model import run_pipeline, PipelineResult  # 기존 코드 파일명
import logging
from dotenv import load_dotenv

app = FastAPI(title="Safe Routing API")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

load_dotenv()

class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    hour: str = "now"


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/calculate-route")
def calculate_route(req: RouteRequest):
    try:
        result: PipelineResult = run_pipeline(
            start_lat=req.start_lat,
            start_lon=req.start_lon,
            end_lat=req.end_lat,
            end_lon=req.end_lon,
            hour=req.hour,
            app_key=os.getenv("TMAP_APP_KEY"),  # 환경변수에서 로드
            cctv_path="./cctv_data.xlsx",  # 컨테이너 내부 경로
            model_path="./edge_pref_model_dataset.json"  # 컨테이너 내부 경로
        )

        return {
            "base_route": result.base_route,
            "rerouted": result.rerouted,
            "base_weight": result.base_weight,
            "rerouted_weight": result.rerouted_weight
        }
    except Exception as e:
        logger.error(f"Error processing route: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))