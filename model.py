"""
model.py
Unified dynamic routing pipeline (WGS84 Native Mode).
Fixes: Floating point precision errors, unnecessary detours, coordinate transformation artifacts.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from shapely.geometry import LineString, Point

# ------------------------ Settings ------------------------ #
TMAP_API_URL = "https://apis.openapi.sk.com/tmap/routes/pedestrian"
TMAP_APP_KEY = os.getenv("TMAP_APP_KEY")
TMAP_TIMEOUT = 15

# [설정] 미리 메모리에 올릴 도시 (반경 축소 권장: 로딩 속도 및 정밀도 향상)
CITIES_CONFIG = {
    "incheon": {"lat": 37.4563, "lon": 126.7052, "dist": 12000}, 
    "seoul":   {"lat": 37.5665, "lon": 126.9780, "dist": 15000},
    # "suwon":   {"lat": 37.2636, "lon": 127.0286, "dist": 15000},
}

NETWORK_TYPE = "walk"
CCTV_XLSX = "cctv_data.xlsx"
STREETLIGHT_PATH = "nationwide_streetlight.xlsx"
POLICE_PATH = "Police_station.csv"
ALPHA = 6.0
HOUR_DEFAULT = "now"
MODEL_PATH = "edge_pref_model_dataset.json"


# ------------------------ Utilities ------------------------ #
def log(msg: str) -> None:
    print(msg, flush=True)

def resolve_hour(val: Any) -> int:
    if val is None: return time.localtime().tm_hour
    try:
        if isinstance(val, str) and val.lower() in {"now", "auto"}:
            return time.localtime().tm_hour
        return int(float(val)) % 24
    except:
        return time.localtime().tm_hour

def utm_epsg_from_latlon(lat: float, lon: float) -> int:
    """위경도에 맞는 UTM 좌표계 코드를 반환"""
    zone = int(math.floor((lon + 180) / 6) + 1)
    return 32600 + zone if lat >= 0 else 32700 + zone

# ------------------------ Data Loading ------------------------ #
def load_cctv_points(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        return gpd.GeoDataFrame(columns=["camera_count", "geometry"], geometry=[], crs="EPSG:4326")
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    
    def pick(candidates):
        for c in candidates:
            if c in df.columns: return c
        return None

    lat_col = pick(["위도", "lat", "latitude"])
    lon_col = pick(["경도", "lon", "longitude"])
    cnt_col = pick(["카메라대수", "camera_count", "count"]) or "camera_count"
    
    if not lat_col or not lon_col:
        return gpd.GeoDataFrame(columns=["camera_count", "geometry"], geometry=[], crs="EPSG:4326")

    if cnt_col not in df.columns: df[cnt_col] = 1
    df[cnt_col] = pd.to_numeric(df[cnt_col], errors="coerce").fillna(1)
    
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
    gdf = gdf.rename(columns={cnt_col: "camera_count"})
    if "count" not in gdf.columns: gdf["count"] = gdf["camera_count"]
    return gdf

def load_generic_points(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path): return gpd.GeoDataFrame(columns=["count", "geometry"], geometry=[], crs="EPSG:4326")
    
    if path.endswith(".csv"): df = pd.read_csv(path)
    else: df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    def pick(candidates):
        for c in candidates:
            if c in df.columns: return c
        return None

    lat_col = pick(["위도", "lat", "latitude", "A2"])
    lon_col = pick(["경도", "lon", "longitude", "A1"])
    
    if not lat_col or not lon_col:
        return gpd.GeoDataFrame(columns=["count", "geometry"], geometry=[], crs="EPSG:4326")

    df["count"] = 1
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
    return gdf

# ------------------------ Graph & Weighting ------------------------ #
def ensure_line_geoms(edges_gdf, nodes_gdf):
    """지오메트리가 없는 엣지에 직선(LineString) 지오메트리 부여"""
    # 노드 인덱스가 정수형인지 문자열인지 확인 필요 없이 인덱스로 접근
    # nodes_gdf.index should be 'osmid'
    
    # 좌표 딕셔너리 생성 (빠른 조회를 위해)
    node_geometry = nodes_gdf["geometry"].to_dict()
    
    missing = edges_gdf["geometry"].isna()
    if missing.sum() > 0:
        # 벡터화된 연산은 아니지만 안전하게 처리
        for idx, row in edges_gdf[missing].iterrows():
            u, v = row["u"], row["v"]
            if u in node_geometry and v in node_geometry:
                u_pt = node_geometry[u]
                v_pt = node_geometry[v]
                edges_gdf.at[idx, "geometry"] = LineString([(u_pt.x, u_pt.y), (v_pt.x, v_pt.y)])
    return edges_gdf

def apply_weights_to_graph(G: nx.MultiDiGraph, alpha: float = ALPHA) -> None:
    """
    [핵심 수정] 원본 G는 Lat/Lon을 유지합니다.
    계산 시에만 투영된 복사본을 만들어 데이터를 집계하고, 결과값만 원본 G에 업데이트합니다.
    """
    # 1. 그래프 -> GeoDataFrame 변환
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    
    # 인덱스 리셋하여 u, v, key 컬럼 확보
    edges = edges.reset_index()
    edges = ensure_line_geoms(edges, nodes)

    # 2. UTM 투영 (거리/버퍼 계산용)
    cent_lat = nodes["y"].mean()
    cent_lon = nodes["x"].mean()
    epsg = utm_epsg_from_latlon(cent_lat, cent_lon)
    
    edges_utm = edges.to_crs(epsg=epsg)
    
    # 길이 재계산 (미터 단위 정확도 확보)
    edges_utm["length_m"] = edges_utm.length

    # POI 데이터 로드 및 투영
    cctv = load_cctv_points(CCTV_XLSX).to_crs(epsg=epsg)
    street = load_generic_points(STREETLIGHT_PATH).to_crs(epsg=epsg)
    police = load_generic_points(POLICE_PATH).to_crs(epsg=epsg)

    # 3. 버퍼 및 Spatial Join
    edges_buf = edges_utm[["u", "v", "key", "geometry"]].copy()
    edges_buf["geometry"] = edges_buf.buffer(80.0) # 80m 반경

    def spatial_join_count(points, buffers, col_name):
        try:
            # predicate='within' : 점이 버퍼 안에 있는지
            joined = gpd.sjoin(points, buffers, predicate="within", how="left")
        except:
            joined = gpd.sjoin(points, buffers, op="within", how="left")
            
        # u, v, key 별로 그룹핑하여 개수 세기
        counts = joined.groupby(["u", "v", "key"])["count"].sum().rename(col_name)
        return counts

    counts_cctv = spatial_join_count(cctv, edges_buf, "cctv_sum")
    counts_st = spatial_join_count(street, edges_buf, "light_sum")
    counts_po = spatial_join_count(police, edges_buf, "police_sum")

    # 4. 결과 병합
    edges_utm = edges_utm.join(counts_cctv, on=["u", "v", "key"])
    edges_utm = edges_utm.join(counts_st, on=["u", "v", "key"])
    edges_utm = edges_utm.join(counts_po, on=["u", "v", "key"])
    
    edges_utm = edges_utm.fillna({"cctv_sum": 0, "light_sum": 0, "police_sum": 0})

    # 5. 밀집도 및 점수 계산
    # 짧은 도로(0m 근처)로 인한 무한대 방지 (clip)
    edges_utm["edge_km"] = edges_utm["length_m"].clip(lower=1.0) / 1000.0
    
    edges_utm["density_per_km"] = edges_utm["cctv_sum"] / edges_utm["edge_km"]
    edges_utm["light_per_km"] = edges_utm["light_sum"] / edges_utm["edge_km"]
    edges_utm["police_per_km"] = edges_utm["police_sum"] / edges_utm["edge_km"]

    def normalize(s):
        lower = s.quantile(0.05)
        upper = s.quantile(0.95)
        if upper <= lower: return s * 0
        return ((s - lower) / (upper - lower)).clip(0, 1)

    edges_utm["dens_norm"] = normalize(edges_utm["density_per_km"])
    edges_utm["light_norm"] = normalize(edges_utm["light_per_km"])
    edges_utm["police_norm"] = normalize(edges_utm["police_per_km"])

    # 6. 원본 그래프(G, Lat/Lon)에 속성 업데이트
    # edges_utm은 순서가 섞였을 수 있으므로 iterrows 사용
    for _, row in edges_utm.iterrows():
        u, v, k = row["u"], row["v"], row["key"]
        if G.has_edge(u, v, k):
            data = G[u][v][k]
            # float 변환하여 저장 (JSON 직렬화 및 오차 방지)
            data.update({
                "length_m": float(row["length_m"]), # 미터 단위 길이
                "cctv_sum_num": float(row["cctv_sum"]),
                "light_sum_num": float(row["light_sum"]),
                "police_sum_num": float(row["police_sum"]),
                "dens_norm_num": float(row["dens_norm"]),
                "light_norm_num": float(row["light_norm"]),
                "police_norm_num": float(row["police_norm"]),
                "len_m_num": float(row["length_m"]),
                "density_per_km": float(row["density_per_km"])
            })

# ------------------------ AI Model Logic ------------------------ #
def edge_feats_ext(d: Dict[str, Any], hour: int) -> np.ndarray:
    L = d.get("len_m_num", 10.0) # 기본값 10m
    dn = d.get("dens_norm_num", 0.0)
    
    # 0으로 나누기 방지
    km = max(0.001, L / 1000.0)
    
    cctv_pk = d.get("cctv_sum_num", 0) / km
    light_pk = d.get("light_sum_num", 0) / km
    police_pk = d.get("police_sum_num", 0) / km
    
    hw = str(d.get("highway", "")).lower()
    def has(tag): return tag in hw

    return np.array([
        1.0, math.log1p(L), dn, cctv_pk, light_pk, police_pk,
        d.get("light_norm_num", 0), d.get("police_norm_num", 0),
        float(has("primary")), float(has("secondary")), float(has("tertiary")),
        float(has("unclassified")), float(has("residential")), float(has("service")),
        float(has("footway")), float(has("path")), float(has("cycleway")),
        float(has("steps")), float(has("track")), float(has("living_street")),
        float(has("pedestrian"))
    ], dtype=float)

def sigmoid(z): 
    # 오버플로우 방지
    z = max(-100, min(100, z))
    return 1.0 / (1.0 + math.exp(-z))

def update_graph_with_model(G, model_path, hour, alpha):
    try:
        with open(model_path, "r") as f:
            weights = np.array(json.load(f)["weights"])
    except:
        log("⚠️ 모델 파일 로드 실패. 기본 가중치를 사용합니다.")
        weights = np.zeros(21)

    for u, v, k, d in G.edges(keys=True, data=True):
        x = edge_feats_ext(d, hour)
        score = sigmoid(np.dot(weights, x))
        
        # [중요] 길이(비용) = 실제거리 / (1 + 알파 * 안전점수)
        # 안전점수가 높을수록(1.0에 가까울수록) 비용이 작아짐 -> 선택 확률 증가
        base_len = d.get("length_m", d.get("length", 10.0))
        d["weight_runtime"] = base_len / (1.0 + alpha * score)

# ------------------------ Main Logic ------------------------ #
@dataclass
class PipelineResult:
    tmap_raw: Dict[str, Any]
    base_route: List[Tuple[float, float]]
    rerouted: List[Tuple[float, float]]
    base_weight: float
    rerouted_weight: float
    visual_segments: List[Dict[str, Any]] | None = None

def load_static_graph(center_lat, center_lon, dist_m):
    log(f"🚀 Building Graph (r={dist_m}m)...")
    
    # 1. Lat/Lon 그래프 생성 (투영하지 않음!)
    # simplify=True: 중간 노드를 없애고 직선화 -> 데이터 가볍게
    G = ox.graph_from_point((center_lat, center_lon), dist=dist_m, network_type="walk", simplify=True)
    
    # 2. 가중치 계산 (내부적으로 투영해서 계산 후 결과만 G에 반영)
    apply_weights_to_graph(G)
    update_graph_with_model(G, MODEL_PATH, resolve_hour("now"), ALPHA)
    
    # 3. 투영 없이 반환 (Lat/Lon 상태)
    return G

class GraphManager:
    def __init__(self):
        self.graphs = {}

    def load_all_cities(self):
        for name, info in CITIES_CONFIG.items():
            log(f"🏙️ [System] '{name.upper()}' 지도 생성 중...")
            try:
                start_t = time.time()
                G = load_static_graph(info["lat"], info["lon"], info["dist"])
                self.graphs[name] = G
                elapsed = time.time() - start_t
                log(f"✅ [System] '{name.upper()}' 완료! ({elapsed:.1f}초)")
            except Exception as e:
                log(f"🔥 [System] '{name.upper()}' 실패: {e}")

    def get_graph(self, lat, lon):
        limit_dist_sq = (0.2) ** 2 # 약 20km 범위
        
        best_city = None
        min_dist = float('inf')

        for name, info in CITIES_CONFIG.items():
            dist = (lat - info["lat"])**2 + (lon - info["lon"])**2
            if dist < min_dist:
                min_dist = dist
                best_city = name
        
        if min_dist > limit_dist_sq:
            return None
        
        return self.graphs.get(best_city)

graph_manager = GraphManager()

def nearest_node(G, lat, lon):
    # G가 Lat/Lon이므로 입력값 그대로 사용하여 최근접 노드 찾기
    return ox.distance.nearest_nodes(G, lon, lat)

def run_pipeline(
    start_lat, start_lon, end_lat, end_lon,
    app_key, 
    preloaded_graph=None,
    **kwargs
) -> PipelineResult:
    
    # 1. Tmap 호출
    params = {
        "version": "1", "startX": str(start_lon), "startY": str(start_lat),
        "endX": str(end_lon), "endY": str(end_lat), "startName": "S", "endName": "E", "appKey": app_key
    }
    try:
        raw = requests.get(TMAP_API_URL, params=params, timeout=5).json()
        features = raw.get("features", [])
        base_route = []
        for f in features:
            if f["geometry"]["type"] == "LineString":
                for lon, lat in f["geometry"]["coordinates"]:
                    base_route.append((lat, lon))
    except:
        raw = {}
        base_route = []

    # 2. 그래프 준비
    if preloaded_graph:
        G = preloaded_graph
    else:
        log("🐢 [Fallback] 지원하지 않는 지역. 실시간 생성 시작...")
        center_lat = (start_lat + end_lat) / 2
        center_lon = (start_lon + end_lon) / 2
        # 대략적인 거리 계산 (1도 = 약 111km)
        dist_deg = ((start_lat - end_lat)**2 + (start_lon - end_lon)**2)**0.5
        dist_m = max(1000, dist_deg * 111000 * 1.5)
        G = load_static_graph(center_lat, center_lon, dist_m=int(dist_m))

    # 3. 길 찾기
    orig = nearest_node(G, start_lat, start_lon)
    dest = nearest_node(G, end_lat, end_lon)
    
    rerouted = []
    # 시작점 연결
    rerouted.append((start_lat, start_lon))

    try:
        # Lat/Lon 그래프에서 직접 경로 탐색
        path_nodes = nx.shortest_path(G, orig, dest, weight="weight_runtime")
        
        for i in range(len(path_nodes)-1):
            u, v = path_nodes[i], path_nodes[i+1]
            
            edges = G.get_edge_data(u, v)
            if not edges: continue
            
            # 가장 가중치가 낮은(좋은) 엣지 선택
            best_key = min(edges, key=lambda k: edges[k].get("weight_runtime", 1e9))
            data = edges[best_key]
            
            if "geometry" in data:
                # 이미 Lat/Lon 좌표이므로 변환 없이 바로 사용
                # LineString.coords는 (lon, lat) 순서
                seg_coords = [(y, x) for x, y in data["geometry"].coords]
                rerouted.extend(seg_coords)
            else:
                # 지오메트리가 없으면 직선 연결
                uy, ux = G.nodes[u]['y'], G.nodes[u]['x']
                vy, vx = G.nodes[v]['y'], G.nodes[v]['x']
                rerouted.append((uy, ux))
                rerouted.append((vy, vx))

    except nx.NetworkXNoPath:
        log("❌ 경로를 찾을 수 없음 (NetworkXNoPath)")
        rerouted = []
    except Exception as e:
        log(f"Error in shortest_path: {e}")
        rerouted = []

    # 도착점 연결
    rerouted.append((end_lat, end_lon))

    # 4. 시각화 데이터 추출 (변환기 필요 없음)
    visual_segments = extract_visual_segments_bbox(G, start_lat, start_lon, end_lat, end_lon)

    return PipelineResult(
        tmap_raw=raw,
        base_route=base_route,
        rerouted=rerouted,
        base_weight=0, 
        rerouted_weight=0,
        visual_segments=visual_segments
    )

def extract_visual_segments_bbox(G, slat, slon, elat, elon, padding=0.005):
    """
    BBox 내 엣지 추출 (Lat/Lon 그래프 사용)
    """
    min_lat, max_lat = min(slat, elat) - padding, max(slat, elat) + padding
    min_lon, max_lon = min(slon, elon) - padding, max(slon, elon) + padding
    
    segments = []
    
    # 그래프가 Lat/Lon이므로 바로 좌표 비교 가능
    # (최적화를 위해 nodes를 먼저 필터링하는 방법도 있으나 일단 전체 순회)
    for u, v, d in G.edges(data=True):
        # u 노드의 좌표
        uy = G.nodes[u]['y']
        ux = G.nodes[u]['x']
        
        if min_lat <= uy <= max_lat and min_lon <= ux <= max_lon:
            coords = []
            if "geometry" in d:
                # (lon, lat) -> (lat, lon)
                coords = [(y, x) for x, y in d["geometry"].coords]
            else:
                vy, vx = G.nodes[v]['y'], G.nodes[v]['x']
                coords = [(uy, ux), (vy, vx)]
            
            dens = d.get("density_per_km", 0.0)
            color = "#1a9641" # Green
            if dens < 5: color = "#d7191c" # Red
            elif dens < 15: color = "#fdae61" # Orange
            
            segments.append({
                "geometry": coords,
                "color": color,
                "properties": {"density": dens}
            })
            
    return segments

