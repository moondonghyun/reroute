"""
model.py
Refined routing pipeline.
Fixes: "Overshooting" near destination and snapping to disconnected nodes.
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

# [설정] 반경을 너무 크게 잡으면 로딩이 오래 걸리므로 적절히 조절
CITIES_CONFIG = {
    "incheon": {"lat": 37.4563, "lon": 126.7052, "dist": 5000}, 
    "seoul":   {"lat": 37.5665, "lon": 126.9780, "dist": 5000},
    # "suwon":   {"lat": 37.2636, "lon": 127.0286, "dist": 5000},
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
    zone = int(math.floor((lon + 180) / 6) + 1)
    return 32600 + zone if lat >= 0 else 32700 + zone

def haversine_dist(lat1, lon1, lat2, lon2):
    """두 좌표 간의 대략적인 미터 거리 계산"""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# ------------------------ Data Loading ------------------------ #
# (이전과 동일: load_cctv_points, load_generic_points 생략 가능하지만 전체 코드를 위해 유지)
def load_cctv_points(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        return gpd.GeoDataFrame(columns=["camera_count", "geometry"], geometry=[], crs="EPSG:4326")
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

def apply_weights_to_graph(G: nx.MultiDiGraph, alpha: float = ALPHA) -> None:
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    edges = edges.reset_index()
    edges = ensure_line_geoms(edges, nodes)

    cent_lat = nodes["y"].mean()
    cent_lon = nodes["x"].mean()
    epsg = utm_epsg_from_latlon(cent_lat, cent_lon)
    
    edges_utm = edges.to_crs(epsg=epsg)
    edges_utm["length_m"] = edges_utm.length

    cctv = load_cctv_points(CCTV_XLSX).to_crs(epsg=epsg)
    street = load_generic_points(STREETLIGHT_PATH).to_crs(epsg=epsg)
    police = load_generic_points(POLICE_PATH).to_crs(epsg=epsg)

    edges_buf = edges_utm[["u", "v", "key", "geometry"]].copy()
    edges_buf["geometry"] = edges_buf.buffer(80.0)

    def spatial_join_count(points, buffers, col_name):
        try:
            joined = gpd.sjoin(points, buffers, predicate="within", how="left")
        except:
            joined = gpd.sjoin(points, buffers, op="within", how="left")
        return joined.groupby(["u", "v", "key"])["count"].sum().rename(col_name)

    counts_cctv = spatial_join_count(cctv, edges_buf, "cctv_sum")
    counts_st = spatial_join_count(street, edges_buf, "light_sum")
    counts_po = spatial_join_count(police, edges_buf, "police_sum")

    edges_utm = edges_utm.join(counts_cctv, on=["u", "v", "key"])
    edges_utm = edges_utm.join(counts_st, on=["u", "v", "key"])
    edges_utm = edges_utm.join(counts_po, on=["u", "v", "key"])
    edges_utm = edges_utm.fillna({"cctv_sum": 0, "light_sum": 0, "police_sum": 0})

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

    for _, row in edges_utm.iterrows():
        u, v, k = row["u"], row["v"], row["key"]
        if G.has_edge(u, v, k):
            data = G[u][v][k]
            data.update({
                "length_m": float(row["length_m"]),
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
    L = d.get("len_m_num", 10.0)
    dn = d.get("dens_norm_num", 0.0)
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
    
    # 1. Lat/Lon 그래프 생성
    G = ox.graph_from_point((center_lat, center_lon), dist=dist_m, network_type="walk", simplify=True)
    
    # [수정] OSMnx 버전 호환성 처리 (v2.0 vs v1.x)
    # 고립된 노드 제거 (가장 큰 연결 덩어리만 남김)
    if len(G) > 0:
        try:
            # OSMnx 2.0.0 이상 (새로운 방식)
            G = ox.truncate.largest_component(G, strongly=True)
        except AttributeError:
            # OSMnx 1.x 이하 (기존 방식)
            try:
                G = ox.utils_graph.get_largest_component(G, strongly=True)
            except AttributeError:
                # 혹시 모를 구버전 대비 (직접 접근 실패 시 패스하거나 다른 alias 시도)
                pass

    # 2. 가중치 계산
    apply_weights_to_graph(G)
    update_graph_with_model(G, MODEL_PATH, resolve_hour("now"), ALPHA)
    
    return G

class GraphManager:
    def __init__(self):
        self.graphs = {}

    def load_all_cities(self):
        for name, info in CITIES_CONFIG.items():
            log(f"🏙️ [System] '{name.upper()}' 지도 생성 중...")
            try:
                G = load_static_graph(info["lat"], info["lon"], info["dist"])
                self.graphs[name] = G
                log(f"✅ [System] '{name.upper()}' 완료!")
            except Exception as e:
                log(f"🔥 [System] '{name.upper()}' 실패: {e}")

    def get_graph(self, lat, lon):
        limit_dist_sq = (0.2) ** 2 
        best_city = None
        min_dist = float('inf')
        for name, info in CITIES_CONFIG.items():
            dist = (lat - info["lat"])**2 + (lon - info["lon"])**2
            if dist < min_dist:
                min_dist = dist
                best_city = name
        if min_dist > limit_dist_sq: return None
        return self.graphs.get(best_city)

graph_manager = GraphManager()

def nearest_node(G, lat, lon):
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
        log("🐢 [Fallback] 실시간 생성...")
        center_lat = (start_lat + end_lat) / 2
        center_lon = (start_lon + end_lon) / 2
        dist_deg = ((start_lat - end_lat)**2 + (start_lon - end_lon)**2)**0.5
        dist_m = max(1000, dist_deg * 111000 * 1.5)
        G = load_static_graph(center_lat, center_lon, dist_m=int(dist_m))

    # 3. 길 찾기
    orig = nearest_node(G, start_lat, start_lon)
    dest = nearest_node(G, end_lat, end_lon)
    
    rerouted = []
    # 시작점
    rerouted.append((start_lat, start_lon))

    try:
        path_nodes = nx.shortest_path(G, orig, dest, weight="weight_runtime")
        
        # [수정 2] 경로 끝부분 가지치기 (Pruning)
        # 마지막 노드(교차로)가 도착지보다 오히려 멀다면, 마지막 노드를 방문하지 않고 그 전에서 끊음
        if len(path_nodes) >= 2:
            last_node = path_nodes[-1]
            prev_node = path_nodes[-2]
            
            last_y, last_x = G.nodes[last_node]['y'], G.nodes[last_node]['x']
            prev_y, prev_x = G.nodes[prev_node]['y'], G.nodes[prev_node]['x']
            
            dist_prev_to_end = haversine_dist(prev_y, prev_x, end_lat, end_lon)
            dist_last_to_end = haversine_dist(last_y, last_x, end_lat, end_lon)
            dist_prev_to_last = haversine_dist(prev_y, prev_x, last_y, last_x)
            
            # 조건: (이전노드->도착지) 거리가 (마지막노드->도착지) 보다 가깝고,
            # (이전노드->마지막노드) 거리의 절반보다 도착지가 가까우면, "지나친 것"으로 간주
            if dist_prev_to_end < dist_last_to_end and dist_prev_to_end < dist_prev_to_last:
                # 마지막 노드 제거
                path_nodes.pop()
        
        # 경로 좌표 변환
        for i in range(len(path_nodes)-1):
            u, v = path_nodes[i], path_nodes[i+1]
            edges = G.get_edge_data(u, v)
            if not edges: continue
            best_key = min(edges, key=lambda k: edges[k].get("weight_runtime", 1e9))
            data = edges[best_key]
            
            if "geometry" in data:
                seg_coords = [(y, x) for x, y in data["geometry"].coords]
                rerouted.extend(seg_coords)
            else:
                uy, ux = G.nodes[u]['y'], G.nodes[u]['x']
                vy, vx = G.nodes[v]['y'], G.nodes[v]['x']
                rerouted.append((uy, ux))
                rerouted.append((vy, vx))

    except nx.NetworkXNoPath:
        log("❌ 경로를 찾을 수 없음")
        rerouted = []
    except Exception as e:
        log(f"Error: {e}")
        rerouted = []

    # 도착점
    rerouted.append((end_lat, end_lon))

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

