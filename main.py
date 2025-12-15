from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
from jose import jwt

# model.py에서 가져옴
from model import run_pipeline, PipelineResult, load_cctv_points, safe_load_generic_points, load_police_points

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")
load_dotenv()

# ---------------------------------------------------------
# [1] 설정 (S3 버킷 이름 설정 필수!)
# ---------------------------------------------------------
BUCKET_NAME = "inha-capstone-11-bucket" 
REGION_NAME = "us-west-2"           


# AWS 리소스 연결
try:
    dynamodb = boto3.resource('dynamodb', region_name=REGION_NAME)
    route_table = dynamodb.Table('inha-capstone-11-nosql')
    s3_client = boto3.client('s3', region_name=REGION_NAME)
    logger.info("✅ AWS DynamoDB & S3 Connected.")
except Exception as e:
    logger.error(f"⚠️ AWS Connection Error: {e}")
    route_table = None
    s3_client = None

# ---------------------------------------------------------
# [2] 전역 그래프 로딩
# ---------------------------------------------------------

GDF_CCTV = None
GDF_LIGHT = None
GDF_POLICE = None

try:
    GDF_CCTV = load_cctv_points("cctv_data.xlsx")
    GDF_LIGHT = safe_load_generic_points("nationwide_streetlight.xlsx", "streetlight")
    GDF_POLICE = load_police_points("Police_station.csv")
    logger.info("✅ [Master] 데이터 로딩 완료!")
except Exception as e:
    logger.error(f"❌ 데이터 로딩 실패: {e}")
    GDF_CCTV, GDF_LIGHT, GDF_POLICE = None, None, None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 [Worker] 워커 프로세스 시작 (데이터는 이미 로드됨)")
    yield
    logger.info("👋 [Worker] 워커 프로세스 종료")

app = FastAPI(title="Safe Routing API", lifespan=lifespan)

# ---------------------------------------------------------
# [3] 인증 및 유틸리티
# ---------------------------------------------------------
security = HTTPBearer()

def get_current_user_sub(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        # 서명 검증 없이 페이로드의 sub만 추출 (프로덕션에선 verify=True 권장)
        payload = jwt.get_unverified_claims(token)
        user_sub = payload.get("sub")
        if not user_sub:
            raise HTTPException(status_code=401, detail="Token missing 'sub'")
        return user_sub
    except Exception as e:
        logger.error(f"Token error: {e}")
        raise HTTPException(status_code=401, detail="Invalid Token")

def load_json_data(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f: return json.load(f)
    except: return []

STREETLIGHTS = load_json_data("streetlight.json")
CCTVS = load_json_data("cctv.json")
POLICE_STATIONS = load_json_data("police_station.json")

def float_to_decimal(data):
    # DynamoDB 저장을 위해 float -> Decimal 변환
    return json.loads(json.dumps(data), parse_float=Decimal)

class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    start_name: str
    end_name: str
    hour: str = "now"

# ---------------------------------------------------------
# [4] 저장 로직 (S3 + DynamoDB)
# ---------------------------------------------------------
def save_route_to_s3_and_db(metadata: dict, heavy_data: dict):
    """
    백그라운드에서 실행되는 함수:
    1. 무거운 데이터(경로, 시각화)는 S3에 JSON으로 업로드
    2. 메타데이터(ID, 시간, 좌표, S3링크)는 DynamoDB에 저장
    """
    if not route_table or not s3_client:
        return

    user_id = metadata['user_id']
    route_id = metadata['route_id']
    
    try:
        # [Step 1] S3 업로드
        s3_key = f"routes/{user_id}/{route_id}.json"
        
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=json.dumps(heavy_data, ensure_ascii=False), # 한글 깨짐 방지
            ContentType='application/json'
        )
        logger.info(f"☁️ S3 Upload Success: {s3_key}")

        # [Step 2] DynamoDB 저장 (S3 키 포함)
        metadata['s3_key'] = s3_key
        
        # float -> Decimal 변환 후 저장
        route_table.put_item(Item=float_to_decimal(metadata))
        logger.info(f"💾 DynamoDB Save Success: {route_id}")

    except Exception as e:
        logger.error(f"🔥 Save Failed: {str(e)}")


# ---------------------------------------------------------
# [5] 메인 API
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

@app.post("/calculate-route")
def calculate_route(
    req: RouteRequest, 
    background_tasks: BackgroundTasks,
    user_sub: str = Depends(get_current_user_sub)
):

    try:
        # 1. 경로 계산
        result = run_pipeline(
            req.start_lat, req.start_lon, req.end_lat, req.end_lon,
            app_key=os.getenv("TMAP_APP_KEY"),
            gdf_cctv=GDF_CCTV,   
            gdf_light=GDF_LIGHT, 
            gdf_police=GDF_POLICE,
            model_path="edge_pref_model_dataset.json"
        )
        
        # 2. 주변 시설물 필터링
        all_coords = result.base_route + result.rerouted
        
        if not all_coords:
            # 만약 경로가 아예 없다면 출발/도착지만 사용 (예외 처리)
            all_coords = [(req.start_lat, req.start_lon), (req.end_lat, req.end_lon)]

        # 모든 좌표에서 min/max 추출
        lats = [p[0] for p in all_coords]
        lons = [p[1] for p in all_coords]

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # 여유 공간 (Padding) 추가 (약 200m 정도)
        pad = 0.002
        min_lat -= pad
        max_lat += pad
        min_lon -= pad
        max_lon += pad
        # [Client 응답용 전체 데이터]
        route_id = str(uuid.uuid4())
        response_data = {
            "route_id": route_id,
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

        # 3. 백그라운드 저장 요청 (S3 Offloading)
        if route_table and s3_client:
            # (A) DynamoDB에 들어갈 가벼운 메타데이터
            meta_data = {
                "route_id": route_id,
                "user_id": user_sub,      # Partition Key
                "isSaved": False,         # Boolean (False)
                "start_name": req.start_name,
                "end_name": req.end_name,
                "timestamp": int(time.time()),
                "created_at": datetime.now().isoformat(),
                "start_point": {"lat": req.start_lat, "lon": req.start_lon}, # Decimal 변환 전
                "end_point": {"lat": req.end_lat, "lon": req.end_lon}        # Decimal 변환 전
            }
            
            # (B) S3에 들어갈 무거운 데이터 (전체)
            heavy_data = response_data
            
            # 백그라운드 작업 추가
            background_tasks.add_task(save_route_to_s3_and_db, meta_data, heavy_data)

        # Client에게는 데이터 바로 반환
        return response_data

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
