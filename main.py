from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from pydantic import BaseModel
import os
import logging
from dotenv import load_dotenv
import boto3
import json
import time
from datetime import datetime
from decimal import Decimal
import uuid
import uvicorn

# ★ [중요] ai_dynamic_routing 대신 model 로 변경됨
from model import run_pipeline, PipelineResult, graph_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")
load_dotenv()

# ---------------------------------------------------------
# [1] 전역 그래프 로딩 (Lifespan)
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌍 [System] 서버 시작: 서울/인천 지도 로딩 중... (3~5분 소요)")
    
    # 여기서 서울과 인천을 모두 메모리에 올립니다.
    graph_manager.load_all_cities()
    
    if not graph_manager.graphs:
        logger.error("🔥 [System] 로딩된 지도가 하나도 없습니다! 기능이 제한됩니다.")
    
    yield
    logger.info("👋 [System] 서버 종료: 메모리 해제")

app = FastAPI(title="Safe Routing API", lifespan=lifespan)

# ---------------------------------------------------------
# [2] 정적 데이터 및 AWS 설정
# ---------------------------------------------------------
def load_json_data(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f: return json.load(f)
    except: return []

STREETLIGHTS = load_json_data("streetlight.json")
CCTVS = load_json_data("cctv.json")
POLICE_STATIONS = load_json_data("police_station.json")

route_table = None
try:
    dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-2')
    route_table = dynamodb.Table('inha-capstone-11-nosql')
except: pass

def float_to_decimal(data):
    return json.loads(json.dumps(data), parse_float=Decimal)

class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    hour: str = "now"

@app.get("/health")
def health_check():
    # 로딩된 도시 목록 확인 가능
    loaded_cities = list(graph_manager.graphs.keys())
    return {"status": "ok", "loaded_cities": loaded_cities}

# ---------------------------------------------------------
# [3] 메인 API
# ---------------------------------------------------------
def filter_features_in_bbox(features, min_lat, max_lat, min_lon, max_lon):
    result = []
    for item in features:
        try:
            lon, lat = item["coordinate"]
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                result.append(item)
        except: continue
    return result

def save_route_history(item: dict):
    if route_table:
        try: route_table.put_item(Item=item)
        except Exception as e: logger.error(f"DB Error: {e}")

@app.post("/calculate-route")
def calculate_route(req: RouteRequest, background_tasks: BackgroundTasks):
    # 1. 사용자 위치에 맞는 그래프 가져오기 (서울 or 인천)
    target_graph = graph_manager.get_graph(req.start_lat, req.start_lon)

    if target_graph is None:
        # 로딩이 안 됐거나 지원하지 않는 지역
        if not graph_manager.graphs:
            raise HTTPException(status_code=503, detail="Maps are still loading. Please wait.")
        else:
            raise HTTPException(status_code=404, detail="Service not available in this area (Only Seoul/Incheon).")

    try:
        # 2. 경로 계산 (메모리 그래프 사용 -> 0.1초)
        result = run_pipeline(
            req.start_lat, req.start_lon, req.end_lat, req.end_lon,
            app_key=os.getenv("TMAP_APP_KEY"),
            preloaded_graph=target_graph  # <--- ★ 선택된 도시 그래프 전달
        )

        # 3. 주변 시설물 필터링
        pad = 0.002
        min_lat, max_lat = min(req.start_lat, req.end_lat) - pad, max(req.start_lat, req.end_lat) + pad
        min_lon, max_lon = min(req.start_lon, req.end_lon) - pad, max(req.start_lon, req.end_lon) + pad
        
        response_data = {
            "base_route": result.base_route,
            "rerouted": result.rerouted,
            "base_weight": result.base_weight,
            "rerouted_weight": result.rerouted_weight,
            "safety_features": {
                "cctvs": filter_features_in_bbox(CCTVS, min_lat, max_lat, min_lon, max_lon),
                "streetlights": filter_features_in_bbox(STREETLIGHTS, min_lat, max_lat, min_lon, max_lon),
                "police_stations": filter_features_in_bbox(POLICE_STATIONS, min_lat, max_lat, min_lon, max_lon)
            },
            "grid_visualization": result.visual_segments
        }

        # 4. DB 저장
        # if route_table:
        #     item = {
        #         "route_id": str(uuid.uuid4()),
        #         "user_id": "99999",
        #         "timestamp": int(time.time()),
        #         "created_at": datetime.now().isoformat(),
        #         "start_point": {"lat": Decimal(str(req.start_lat)), "lon": Decimal(str(req.start_lon))},
        #         "end_point": {"lat": Decimal(str(req.end_lat)), "lon": Decimal(str(req.end_lon))},
        #         "route_data": float_to_decimal(response_data)
        #     }
        #     background_tasks.add_task(save_route_history, item)

        return response_data

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
