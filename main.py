from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
from model import run_pipeline, PipelineResult
import logging
from dotenv import load_dotenv
from fastapi import Depends
from sqlalchemy.orm import Session
import boto3
import json
import time
from datetime import datetime
from decimal import Decimal
from auth import get_current_user_sub
from database import get_db, User
import uuid

# ---------------------------------------------------------
# [1] 정적 데이터 로드 (전역 변수)
# ---------------------------------------------------------
STREETLIGHTS = []
CCTVS = []
POLICE_STATIONS = []

def load_json_data(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Warning: {filename} not found. Returning empty list.")
        return []

# 서버 시작 시 한 번만 로드
STREETLIGHTS = load_json_data("streetlight.json")
CCTVS = load_json_data("cctv.json")
POLICE_STATIONS = load_json_data("police_station.json")


# ---------------------------------------------------------
# AWS 및 앱 설정
# ---------------------------------------------------------
dynamodb = boto3.resource('dynamodb', region_name='us-west-2')
route_table = dynamodb.Table('inha-capstone-11-nosql')

def float_to_decimal(data):
    return json.loads(json.dumps(data), parse_float=Decimal)

app = FastAPI(title="Safe Routing API")

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


# ---------------------------------------------------------
# [2] 필터링 헬퍼 함수
# ---------------------------------------------------------
def filter_features_in_bbox(features, min_lat, max_lat, min_lon, max_lon):
    """
    주어진 범위(BBox) 안에 있는 피처만 필터링해서 반환
    feature format: {"coordinate": [lon, lat]}
    """
    result = []
    for item in features:
        # JSON 포맷이 [lon, lat] 순서라고 가정 (GeoJSON 표준)
        lon, lat = item["coordinate"]
        
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            result.append(item)
    return result


def save_route_history(item: dict):
    try:
        logger.info(f"Saving route history for user {item['user_id']}...")
        route_table.put_item(Item=item)
        logger.info("Successfully saved to DynamoDB.")
    except Exception as e:
        logger.error(f"Failed to save history to DynamoDB: {str(e)}")


@app.post("/calculate-route")
def calculate_route(
        req: RouteRequest,
        background_tasks: BackgroundTasks
        # sub: str = Depends(get_current_user_sub),
        # db: Session = Depends(get_db)
):
    # ---------------------------------------------------------
    # [테스트 모드]
    internal_user_id = 99999
    print(f"⚠️ [TEST MODE] 인증 없이 테스트 ID({internal_user_id})로 실행합니다.")
    # ---------------------------------------------------------

    try:
        # 1. 경로 계산 (AI 모델)
        result: PipelineResult = run_pipeline(
            start_lat=req.start_lat,
            start_lon=req.start_lon,
            end_lat=req.end_lat,
            end_lon=req.end_lon,
            hour=req.hour,
            app_key=os.getenv("TMAP_APP_KEY"),
            cctv_path="./cctv_data.xlsx",
            model_path="./edge_pref_model_dataset.json"
        )

        # 2. 직사각형 범위(Bounding Box) 계산
        # 출발지와 도착지 중 작은 값이 min, 큰 값이 max가 되어야 함
        min_lat = min(req.start_lat, req.end_lat)
        max_lat = max(req.start_lat, req.end_lat)
        min_lon = min(req.start_lon, req.end_lon)
        max_lon = max(req.start_lon, req.end_lon)

        # (옵션) 너무 딱 맞으면 경계선에 있는 게 안 보일 수 있으니 약간의 여유(Padding)를 줄 수도 있음
        padding = 0.002 # 약 200m 정도 여유
        min_lat -= padding
        max_lat += padding
        min_lon -= padding
        max_lon += padding

        # 3. 범위 내 시설물 필터링
        nearby_cctvs = filter_features_in_bbox(CCTVS, min_lat, max_lat, min_lon, max_lon)
        nearby_streetlights = filter_features_in_bbox(STREETLIGHTS, min_lat, max_lat, min_lon, max_lon)
        nearby_police = filter_features_in_bbox(POLICE_STATIONS, min_lat, max_lat, min_lon, max_lon)

        # 4. 응답 데이터 구성
        response_data = {
            "base_route": result.base_route,
            "rerouted": result.rerouted,
            "base_weight": result.base_weight,
            "rerouted_weight": result.rerouted_weight,
            # [추가됨] 주변 안전 시설물 정보
            "safety_features": {
                "cctvs": nearby_cctvs,
                "streetlights": nearby_streetlights,
                "police_stations": nearby_police
            }
        }

        # 5. DynamoDB 저장 (시설물 정보까지 포함할지 여부는 선택, 여기선 포함함)
        item = {
            "route_id": str(uuid.uuid4()),
            "user_id": str(internal_user_id),
            "timestamp": int(time.time()),
            "created_at": datetime.now().isoformat(),
            "start_point": {"lat": Decimal(str(req.start_lat)), "lon": Decimal(str(req.start_lon))},
            "end_point": {"lat": Decimal(str(req.end_lat)), "lon": Decimal(str(req.end_lon))},
            "route_data": float_to_decimal(response_data)
        }

        background_tasks.add_task(save_route_history, item)

        return response_data

    except Exception as e:
        logger.error(f"Error processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
