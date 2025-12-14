"""
model.py
Refined routing pipeline with Geometry Snapping.
Fixes: "Overshooting" near destination by trimming the path geometry.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from shapely.geometry import LineString, Point
from shapely.ops import substring

# ------------------------ Settings ------------------------ #
TMAP_API_URL = "https://apis.openapi.sk.com/tmap/routes/pedestrian"
TMAP_APP_KEY = os.getenv("TMAP_APP_KEY")
TMAP_TIMEOUT = 15

CITIES_CONFIG = {
    "incheon": {"lat": 37.4563, "lon": 126.7052, "dist": 5000}, 
    "seoul":   {"lat": 37.5665, "lon": 126.9780, "dist": 5000},
}

NETWORK_TYPE = "walk"
CCTV_XLSX = "cctv_data.xlsx"
STREETLIGHT_PATH = "nationwide_streetlight.xlsx"
POLICE_PATH = "Police_station.csv"

# [수정 1] Alpha 값 하향 조정 (6.0 -> 2.5)
# 너무 과도한 우회를 방지하기 위해 안전 가중치의 영향력을 조금 줄입니다.
ALPHA = 2.5 
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
    zone = int(math.floor((lon + 180) / 6) + 1)
    return 32600 + zone if lat >= 0 else 32700 + zone

# ------------------------ Data Loading ------------------------ #
# (기존 데이터 로딩 함수들 유지)
def load_cctv_points(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        # 빈 데이터프레임 반환 시에도 count 컬럼 명시
        return gpd.GeoDataFrame(columns=["camera_count", "count", "geometry"], geometry=[], crs="EPSG:4326")
    
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    
    def pick(cands):
        for c in cands:
            if c in df.columns: return c
        return None
    
    lat_col = pick(["위도", "lat", "latitude"])
    lon_col = pick(["경도", "lon", "longitude"])
    cnt_col = pick(["카메라대수", "camera_count", "count"]) or "camera_count"
    
    if not lat_col or not lon_col:
        return gpd.GeoDataFrame(columns=["camera_count", "count", "geometry"], geometry=[], crs="EPSG:4326")
    
    if cnt_col not in df.columns: 
        df[cnt_col] = 1
        
    df[cnt_col] = pd.to_numeric(df[cnt_col], errors="coerce").fillna(1)
    
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
    gdf = gdf.rename(columns={cnt_col: "camera_count"})
    
    gdf["count"] = gdf["camera_count"]
    
    return gdf

def load_generic_points(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path): return gpd.GeoDataFrame(columns=["count", "geometry"], geometry=[], crs="EPSG:4326")
    if path.endswith(".csv"): df = pd.read_csv(path)
    else: df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    def pick(cands):
        for c in cands:
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
    node_geometry = nodes_gdf["geometry"].to_dict()
    missing = edges_gdf["geometry"].isna()
    if missing.sum() > 0:
        for idx, row in edges_gdf[missing].iterrows():
            u, v = row["u"], row["v"]
            if u in node_geometry and v in node_geometry:
                u_pt = node_geometry[u]
                v_pt = node_geometry[v]
                edges_gdf.at[idx, "geometry"] = LineString([(u_pt.x, u_pt.y), (v_pt.x, v_pt.y)])
    return edges_gdf

def apply_weights_to_graph(G: nx.MultiDiGraph) -> None:
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    edges = ensure_line_geoms(edges, nodes)

    cent_lat, cent_lon = nodes["y"].mean(), nodes["x"].mean()
    epsg = utm_epsg_from_latlon(cent_lat, cent_lon)
    
    edges_utm = edges.to_crs(epsg=epsg)
    edges_utm["length_m"] = edges_utm.length

    cctv = load_cctv_points(CCTV_XLSX).to_crs(epsg=epsg)
    street = load_generic_points(STREETLIGHT_PATH).to_crs(epsg=epsg)
    police = load_generic_points(POLICE_PATH).to_crs(epsg=epsg)

    edges_buf = edges_utm[["u", "v", "key", "geometry"]].copy()
    edges_buf["geometry"] = edges_buf.buffer(80.0)

    def spatial_join_count(points, buffers, col_name):
        try: joined = gpd.sjoin(points, buffers, predicate="within", how="left")
        except: joined = gpd.sjoin(points, buffers, op="within", how="left")
        return joined.groupby(["u", "v", "key"])["count"].sum().rename(col_name)

    counts_cctv = spatial_join_count(cctv, edges_buf, "cctv_sum")
    counts_st = spatial_join_count(street, edges_buf, "light_sum")
    counts_po = spatial_join_count(police, edges_buf, "police_sum")

    edges_utm = edges_utm.join(counts_cctv, on=["u", "v", "key"]).join(counts_st, on=["u", "v", "key"]).join(counts_po, on=["u", "v", "key"])
    edges_utm = edges_utm.fillna({"cctv_sum": 0, "light_sum": 0, "police_sum": 0})

    edges_utm["edge_km"] = edges_utm["length_m"].clip(lower=1.0) / 1000.0
    edges_utm["density_per_km"] = edges_utm["cctv_sum"] / edges_utm["edge_km"]
    
    # Feature extraction for model
    for _, row in edges_utm.iterrows():
        u, v, k = row["u"], row["v"], row["key"]
        if G.has_edge(u, v, k):
            data = G[u][v][k]
            data.update({
                "length_m": float(row["length_m"]),
                "cctv_sum_num": float(row["cctv_sum"]),
                "light_sum_num": float(row["light_sum"]),
                "police_sum_num": float(row["police_sum"]),
                "density_per_km": float(row["density_per_km"]),
                "len_m_num": float(row["length_m"])
            })

# ------------------------ AI Model Logic ------------------------ #
def edge_feats_ext(d: Dict[str, Any], hour: int) -> np.ndarray:
    L = d.get("len_m_num", 10.0)
    km = max(0.001, L / 1000.0)
    cctv_pk = d.get("cctv_sum_num", 0) / km
    light_pk = d.get("light_sum_num", 0) / km
    police_pk = d.get("police_sum_num", 0) / km
    hw = str(d.get("highway", "")).lower()
    
    # Simple feature vector (aligned with your training logic)
    return np.array([
        1.0, math.log1p(L), 0.0, cctv_pk, light_pk, police_pk, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ], dtype=float)

def sigmoid(z): 
    z = max(-100, min(100, z))
    return 1.0 / (1.0 + math.exp(-z))

def update_graph_with_model(G, model_path, hour, alpha):
    try:
        with open(model_path, "r") as f:
            weights = np.array(json.load(f)["weights"])
    except:
        log("⚠️ 모델 로드 실패, 기본값 사용")
        weights = np.zeros(21)

    for u, v, k, d in G.edges(keys=True, data=True):
        x = edge_feats_ext(d, hour)
        # score: 0 (위험) ~ 1 (안전)
        score = sigmoid(np.dot(weights, x))
        
        base_len = d.get("length_m", d.get("length", 10.0))
        
        # [수정 1 관련] 안전할수록 거리가 짧게 느껴지게 함 (비용 감소)
        # alpha=2.5일 때 score=1이면 길이는 약 28%로 인식됨 (1 / 3.5)
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
    log(f"🚀 Graph Build (r={dist_m}m)...")
    G = ox.graph_from_point((center_lat, center_lon), dist=dist_m, network_type="walk", simplify=True)
    if len(G) > 0:
        try: G = ox.truncate.largest_component(G, strongly=True)
        except: pass
    
    apply_weights_to_graph(G)
    update_graph_with_model(G, MODEL_PATH, resolve_hour("now"), ALPHA)
    return G

class GraphManager:
    def __init__(self): self.graphs = {}
    def load_all_cities(self):
        for name, info in CITIES_CONFIG.items():
            try: self.graphs[name] = load_static_graph(info["lat"], info["lon"], info["dist"])
            except: pass
    def get_graph(self, lat, lon):
        for name, info in CITIES_CONFIG.items():
            if (lat - info["lat"])**2 + (lon - info["lon"])**2 < 0.04:
                return self.graphs.get(name)
        return None

graph_manager = GraphManager()

def nearest_node(G, lat, lon):
    return ox.distance.nearest_nodes(G, lon, lat)

def run_pipeline(
    start_lat, start_lon, end_lat, end_lon,
    app_key, 
    preloaded_graph=None,
    **kwargs
) -> PipelineResult:
    
    # 1. Tmap 호출 (Base Route)
    params = {
        "version": "1", "startX": str(start_lon), "startY": str(start_lat),
        "endX": str(end_lon), "endY": str(end_lat), "startName": "S", "endName": "E", "appKey": app_key
    }
    try:
        raw = requests.get(TMAP_API_URL, params=params, timeout=5).json()
        base_route = []
        for f in raw.get("features", []):
            if f["geometry"]["type"] == "LineString":
                for lon, lat in f["geometry"]["coordinates"]:
                    base_route.append((lat, lon))
    except:
        raw, base_route = {}, []

    # 2. Graph 준비
    if preloaded_graph: G = preloaded_graph
    else:
        G = load_static_graph((start_lat + end_lat)/2, (start_lon + end_lon)/2, 1500)

    # 3. 길 찾기 (안전 경로)
    orig = nearest_node(G, start_lat, start_lon)
    dest = nearest_node(G, end_lat, end_lon)
    
    rerouted_coords = []
    
    try:
        # A* or Dijkstra
        path_nodes = nx.shortest_path(G, orig, dest, weight="weight_runtime")
        
        # [수정 2] Geometry Cutting Logic (핵심 수정)
        # 전체 경로를 하나의 LineString으로 만듭니다.
        full_line_coords = []
        
        # 시작점 추가
        full_line_coords.append((start_lon, start_lat)) # (x, y) 순서 주의
        
        # 노드 경로를 따라 좌표 수집
        for i in range(len(path_nodes)-1):
            u, v = path_nodes[i], path_nodes[i+1]
            edges = G.get_edge_data(u, v)
            if not edges: continue
            # 가장 가중치가 낮은(선택된) 엣지 선택
            best_key = min(edges, key=lambda k: edges[k].get("weight_runtime", 1e9))
            data = edges[best_key]
            
            if "geometry" in data:
                # shapely geometry (x, y) = (lon, lat)
                full_line_coords.extend(list(data["geometry"].coords))
            else:
                full_line_coords.append((G.nodes[u]['x'], G.nodes[u]['y']))
                full_line_coords.append((G.nodes[v]['x'], G.nodes[v]['y']))
        
        # 전체 경로 라인 생성
        route_line = LineString(full_line_coords)
        
        # 도착지점(Point) 생성
        dest_point = Point(end_lon, end_lat)
        
        # 도착지점이 경로 선상 어디에 투영(Project)되는지 계산 (거리 값)
        projected_dist = route_line.project(dest_point)
        
        # 경로를 투영된 지점까지만 잘라냄 (Trim)
        # 이렇게 하면 도착지 노드를 지나쳐서 갔다가 되돌아오는 부분을 제거할 수 있습니다.
        trimmed_line = substring(route_line, 0, projected_dist)
        
        # Shapely 좌표(x, y) -> (lat, lon)으로 변환하여 저장
        rerouted_coords = [(y, x) for x, y in trimmed_line.coords]
        
        # 마지막으로 실제 도착지 좌표 추가
        rerouted_coords.append((end_lat, end_lon))

    except nx.NetworkXNoPath:
        log("❌ 경로 탐색 실패")
        rerouted_coords = [(start_lat, start_lon), (end_lat, end_lon)]
    except Exception as e:
        log(f"⚠️ Error: {e}")
        rerouted_coords = []

    visual_segments = extract_visual_segments_bbox(G, start_lat, start_lon, end_lat, end_lon)

    return PipelineResult(
        tmap_raw=raw,
        base_route=base_route,
        rerouted=rerouted_coords,
        base_weight=0, 
        rerouted_weight=0,
        visual_segments=visual_segments
    )

def extract_visual_segments_bbox(G, slat, slon, elat, elon, padding=0.005):
    min_lat, max_lat = min(slat, elat) - padding, max(slat, elat) + padding
    min_lon, max_lon = min(slon, elon) - padding, max(slon, elon) + padding
    
    segments = []
    for u, v, d in G.edges(data=True):
        uy, ux = G.nodes[u]['y'], G.nodes[u]['x']
        if min_lat <= uy <= max_lat and min_lon <= ux <= max_lon:
            coords = []
            if "geometry" in d:
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

