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

# 수정된 모듈 임포트
from ai_dynamic_routing import run_pipeline, PipelineResult, load_static_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")
load_dotenv()

# ---------------------------------------------------------
# [1] 전역 그래프 로딩 (Lifespan)
# ---------------------------------------------------------
GLOBAL_GRAPH = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_GRAPH
    logger.info("🌍 [System] 서버 시작: 인천 전체 지도 메모리 로딩 중... (약 1~2분 소요)")
    
    # 인천 시청 기준 반경 12km (인천 서구, 남동구, 부평구, 연수구 대부분 커버)
    # t3.xlarge (16GB RAM) 사용 시 약 2~4GB 소모 예상
    try:
        GLOBAL_GRAPH = load_static_graph(center_lat=37.4563, center_lon=126.7052, dist_m=12000)
        logger.info(f"✅ [System] 지도 로딩 완료! (Nodes: {len(GLOBAL_GRAPH.nodes)}, Edges: {len(GLOBAL_GRAPH.edges)})")
    except Exception as e:
        logger.error(f"🔥 [System] 지도 로딩 실패: {e}")
        GLOBAL_GRAPH = None
    
    yield
    GLOBAL_GRAPH = None
    logger.info("👋 [System] 서버 종료: 메모리 해제 완료")

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
    # 그래프가 로딩되었는지 상태 확인 가능
    status = "ok" if GLOBAL_GRAPH else "loading_map"
    return {"status": status}

# ---------------------------------------------------------
# [3] 메인 API
# ---------------------------------------------------------
def filter_features_in_bbox(features, min_lat, max_lat, min_lon, max_lon):
    """BBox 내 시설물 필터링"""
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
    # 전역 그래프가 로딩 중이면 503 에러 반환 (Service Unavailable)
    if GLOBAL_GRAPH is None:
        raise HTTPException(status_code=503, detail="Server is initializing the map. Please try again in a minute.")

    try:
        # 1. 경로 계산 (메모리에 있는 그래프 사용 -> 0.1초 컷)
        result = run_pipeline(
            req.start_lat, req.start_lon, req.end_lat, req.end_lon,
            app_key=os.getenv("TMAP_APP_KEY"),
            preloaded_graph=GLOBAL_GRAPH  # <--- ★ 핵심: 미리 만든 그래프 전달
        )

        # 2. 주변 시설물 필터링 (Bounding Box)
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

        # 3. DB 저장
        # if route_table:
        #     item = {
        #         "route_id": str(uuid.uuid4()),
        #         "user_id": "99999", # Test ID
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
