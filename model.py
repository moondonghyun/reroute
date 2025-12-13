"""
Unified dynamic routing pipeline (Optimized for Pre-loading).
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
from shapely.geometry import LineString, MultiLineString, Point, box
import branca.colormap as cm

# ------------------------ Settings ------------------------ #
TMAP_API_URL = "https://apis.openapi.sk.com/tmap/routes/pedestrian"
TMAP_APP_KEY = os.getenv("TMAP_APP_KEY", "IqFRypKZ8h81kp9xXLyKY5OfY9PwYSxi8K2pHLkb")
TMAP_TIMEOUT = 15

# 기본 설정 (인천 중심)
START_LAT = 37.4451
START_LON = 126.6942
END_LAT = 37.4166
END_LON = 126.6863
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

def latlon_to_graph_xy(Gp, lat: float, lon: float) -> Tuple[float, float]:
    crs = Gp.graph.get("crs", "EPSG:3857")
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs)
    geom = pt.geometry.iloc[0]
    return float(geom.x), float(geom.y)

def utm_epsg_from_latlon(lat: float, lon: float) -> int:
    zone = int(math.floor((lon + 180) / 6) + 1)
    return 32600 + zone if lat >= 0 else 32700 + zone

# ------------------------ Data Loading ------------------------ #
def load_cctv_points(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        log(f"⚠️ {path} not found. Using empty data.")
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
    node_y = nodes_gdf["y"].to_dict()
    node_x = nodes_gdf["x"].to_dict()
    missing = edges_gdf["geometry"].isna()
    for idx, row in edges_gdf[missing].iterrows():
        uy, ux = node_y[row["u"]], node_x[row["u"]]
        vy, vx = node_y[row["v"]], node_x[row["v"]]
        edges_gdf.at[idx, "geometry"] = LineString([(ux, uy), (vx, vy)])
    return edges_gdf

def apply_weights_to_graph(G: nx.MultiDiGraph, alpha: float = ALPHA) -> None:
    """
    미리 로딩된 파일들을 사용하여 그래프에 가중치를 입힙니다.
    """
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    edges = ensure_line_geoms(edges, nodes)

    # 좌표계 변환 (UTMK)
    cent_lat = nodes["y"].mean()
    cent_lon = nodes["x"].mean()
    epsg = utm_epsg_from_latlon(cent_lat, cent_lon)
    edges_utm = edges.to_crs(epsg=epsg)
    
    # 길이 계산
    edges_utm["length_m"] = edges_utm.length

    # 데이터 로드 (파일 경로가 있으면 로드)
    cctv = load_cctv_points(CCTV_XLSX).to_crs(epsg=epsg)
    street = load_generic_points(STREETLIGHT_PATH).to_crs(epsg=epsg)
    police = load_generic_points(POLICE_PATH).to_crs(epsg=epsg) # Police도 generic 사용

    # 버퍼 생성 (도로 주변 80m)
    edges_buf = edges_utm[["u", "v", "key", "geometry"]].copy()
    edges_buf["geometry"] = edges_buf.buffer(80.0)

    # 공간 조인 (Spatial Join) - 여기가 가장 무거운 작업 (Startup에서 한 번만 수행)
    try:
        joined_cctv = gpd.sjoin(cctv, edges_buf, predicate="within", how="left")
        joined_st = gpd.sjoin(street, edges_buf, predicate="within", how="left")
        joined_po = gpd.sjoin(police, edges_buf, predicate="within", how="left")
    except:
        # 구버전 geopandas 호환
        joined_cctv = gpd.sjoin(cctv, edges_buf, op="within", how="left")
        joined_st = gpd.sjoin(street, edges_buf, op="within", how="left")
        joined_po = gpd.sjoin(police, edges_buf, op="within", how="left")

    # 집계
    def agg_count(joined, col):
        return joined.groupby(["u", "v", "key"])["count"].sum().rename(col)

    counts_cctv = agg_count(joined_cctv, "cctv_sum")
    counts_st = agg_count(joined_st, "light_sum")
    counts_po = agg_count(joined_po, "police_sum")

    edges_utm = edges_utm.join(counts_cctv, on=["u", "v", "key"])
    edges_utm = edges_utm.join(counts_st, on=["u", "v", "key"])
    edges_utm = edges_utm.join(counts_po, on=["u", "v", "key"])
    
    edges_utm = edges_utm.fillna({"cctv_sum": 0, "light_sum": 0, "police_sum": 0})

    # 밀집도 계산
    edges_utm["edge_km"] = edges_utm["length_m"].clip(lower=1e-6) / 1000.0
    edges_utm["density_per_km"] = edges_utm["cctv_sum"] / edges_utm["edge_km"]
    edges_utm["light_per_km"] = edges_utm["light_sum"] / edges_utm["edge_km"]
    edges_utm["police_per_km"] = edges_utm["police_sum"] / edges_utm["edge_km"]

    # 정규화
    def normalize(s):
        lower = s.quantile(0.05)
        upper = s.quantile(0.95)
        if upper == lower: return s * 0
        return ((s - lower) / (upper - lower)).clip(0, 1)

    edges_utm["dens_norm"] = normalize(edges_utm["density_per_km"])
    edges_utm["light_norm"] = normalize(edges_utm["light_per_km"])
    edges_utm["police_norm"] = normalize(edges_utm["police_per_km"])

    # 가중치 계산 (기본)
    # score = dens_norm + 1.5*light + 3.0*police
    combined_score = edges_utm["dens_norm"] + 1.5 * edges_utm["light_norm"] + 3.0 * edges_utm["police_norm"]
    edges_utm["weight_cctv"] = edges_utm["length_m"] / (1.0 + alpha * combined_score)

    # 그래프에 속성 업데이트
    for _, r in edges_utm.iterrows():
        if G.has_edge(r["u"], r["v"], r["key"]):
            d = G[r["u"]][r["v"]][r["key"]]
            d.update({
                "length_m": r["length_m"],
                "cctv_sum_num": r["cctv_sum"],
                "light_sum_num": r["light_sum"],
                "police_sum_num": r["police_sum"],
                "dens_norm_num": r["dens_norm"],
                "light_norm_num": r["light_norm"],
                "police_norm_num": r["police_norm"],
                "len_m_num": r["length_m"],
                "density_per_km": r["density_per_km"]
            })

# ------------------------ AI Model Logic ------------------------ #
def edge_feats_ext(d: Dict[str, Any], hour: int) -> np.ndarray:
    L = d.get("len_m_num", 1.0)
    dn = d.get("dens_norm_num", 0.0)
    cctv_pk = d.get("cctv_sum_num", 0) / max(1e-6, L/1000)
    light_pk = d.get("light_sum_num", 0) / max(1e-6, L/1000)
    police_pk = d.get("police_sum_num", 0) / max(1e-6, L/1000)
    
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
    # overflow 방지
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

def update_graph_with_model(G, model_path, hour, alpha):
    try:
        with open(model_path, "r") as f:
            weights = np.array(json.load(f)["weights"])
    except:
        log("⚠️ 모델 파일 로드 실패. 기본 가중치를 사용합니다.")
        weights = np.zeros(21) # fallback

    for _, _, d in G.edges(data=True):
        x = edge_feats_ext(d, hour)
        score = sigmoid(np.dot(weights, x))
        # weight_runtime이 최종 Dijkstra에 사용될 가중치입니다.
        d["weight_runtime"] = d.get("len_m_num", 1.0) / (1.0 + alpha * score)

# ------------------------ Main Logic ------------------------ #
@dataclass
class PipelineResult:
    tmap_raw: Dict[str, Any]
    base_route: List[Tuple[float, float]]
    rerouted: List[Tuple[float, float]]
    base_weight: float
    rerouted_weight: float
    visual_segments: List[Dict[str, Any]] | None = None

# ★ [핵심] 정적 그래프 로딩 함수 (서버 켤 때 한 번만 호출)
def load_static_graph(center_lat=37.4563, center_lon=126.7052, dist_m=10000):
    log(f"🚀 [Startup] Building Graph (radius={dist_m}m)... This may take a while.")
    # 1. 그래프 다운로드
    # simplify=True로 노드 수를 줄여 메모리 절약
    G = ox.graph_from_point((center_lat, center_lon), dist=dist_m, network_type="walk", simplify=True)
    
    # 2. 투영 및 가중치 주입 (여기서 sjoin 등 무거운 작업 수행)
    log("🚀 [Startup] Injecting Weights (CCTV, Lights)...")
    apply_weights_to_graph(G)
    
    # 3. AI 모델 적용 (기본값 now)
    log("🚀 [Startup] Applying AI Model...")
    update_graph_with_model(G, MODEL_PATH, resolve_hour("now"), ALPHA)
    
    # 4. 투영된 그래프(m 단위)로 반환
    G_proj = ox.project_graph(G)
    log("✅ [Startup] Graph Ready! Loaded into Memory.")
    return G_proj

# 경로 찾기 (미리 로딩된 G 사용)
def nearest_node(G, lat, lon):
    x, y = latlon_to_graph_xy(G, lat, lon)
    return ox.distance.nearest_nodes(G, x, y)

def run_pipeline(
    start_lat, start_lon, end_lat, end_lon,
    app_key, 
    preloaded_graph=None, # <--- 여기가 핵심
    **kwargs
) -> PipelineResult:
    
    # 1. Tmap 호출 (비교용)
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

    # 2. 그래프 준비 (메모리 로딩된 것 사용)
    if preloaded_graph:
        G = preloaded_graph
    else:
        # fallback: 로컬 그래프 생성 (매우 느림 - 비상용)
        G = ox.graph_from_point(((start_lat+end_lat)/2, (start_lon+end_lon)/2), dist=500, network_type="walk")
        G = ox.project_graph(G)

    # 3. 길 찾기
    orig = nearest_node(G, start_lat, start_lon)
    dest = nearest_node(G, end_lat, end_lon)
    
    rerouted = []
    try:
        path_nodes = nx.shortest_path(G, orig, dest, weight="weight_runtime")
        
        for i in range(len(path_nodes)-1):
            u, v = path_nodes[i], path_nodes[i+1]
            # 엣지 중 길이가 가장 짧은 것 선택 (MultiGraph 대비)
            edges = G.get_edge_data(u, v)
            # edges가 dict 형태 {0: {attr...}, 1: {attr...}}
            # 가장 가중치 낮은 키 찾기
            best_key = min(edges, key=lambda k: edges[k].get("weight_runtime", 1e9))
            data = edges[best_key]
            
            if "geometry" in data:
                xs, ys = data["geometry"].xy
                rerouted.extend(list(zip(ys, xs)))
            else:
                rerouted.append((G.nodes[u]['y'], G.nodes[u]['x']))
        rerouted.append((G.nodes[dest]['y'], G.nodes[dest]['x']))
    except nx.NetworkXNoPath:
        log("❌ No path found between nodes.")
        rerouted = []
    except Exception as e:
        log(f"❌ Error finding path: {e}")
        rerouted = []

    # 4. 시각화 데이터 추출 (전체 그래프가 아니라 경로 주변만!)
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
    전체 그래프를 다 뒤지면 느리니까 BBox 내의 엣지만 필터링해서 시각화 데이터를 만듭니다.
    """
    min_lat, max_lat = min(slat, elat) - padding, max(slat, elat) + padding
    min_lon, max_lon = min(slon, elon) - padding, max(slon, elon) + padding
    
    segments = []
    
    # 노드들의 좌표 캐싱 (속도 향상)
    nodes = G.nodes
    
    for u, v, d in G.edges(data=True):
        uy, ux = nodes[u]['y'], nodes[u]['x']
        
        # 엣지 시작점이 범위 안에 있으면 추가 (대략적인 필터링)
        if min_lat <= uy <= max_lat and min_lon <= ux <= max_lon:
            coords = []
            if "geometry" in d:
                xs, ys = d["geometry"].xy
                coords = list(zip(ys, xs)) # (lat, lon)
            else:
                coords = [(uy, ux), (nodes[v]['y'], nodes[v]['x'])]
            
            # 색상 계산
            dens = d.get("density_per_km", 0.0)
            
            # 컬러맵 (초록 -> 빨강)
            color = "#1a9641" # Green
            if dens < 5: color = "#d7191c" # Red
            elif dens < 15: color = "#fdae61" # Orange
            
            segments.append({
                "geometry": coords,
                "color": color,
                "properties": {"density": dens}
            })
            
    return segments
