"""
model.py
Unified dynamic routing pipeline (Optimized for Pre-loading Multi-City).
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

# [설정] 지원할 도시 목록 (중심좌표, 반경m)
# t3.xlarge (16GB) 기준: 서울(15km) + 인천(12km) 동시 로딩 가능
CITIES_CONFIG = {
    "incheon": {"lat": 37.4563, "lon": 126.7052, "dist": 12000}, # 인천 반경 12km
    "seoul":   {"lat": 37.5665, "lon": 126.9780, "dist": 15000}, # 서울 반경 15km
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

    # 데이터 로드
    cctv = load_cctv_points(CCTV_XLSX).to_crs(epsg=epsg)
    street = load_generic_points(STREETLIGHT_PATH).to_crs(epsg=epsg)
    police = load_generic_points(POLICE_PATH).to_crs(epsg=epsg)

    # 버퍼 생성 (도로 주변 80m)
    edges_buf = edges_utm[["u", "v", "key", "geometry"]].copy()
    edges_buf["geometry"] = edges_buf.buffer(80.0)

    # 공간 조인 (Spatial Join)
    try:
        joined_cctv = gpd.sjoin(cctv, edges_buf, predicate="within", how="left")
        joined_st = gpd.sjoin(street, edges_buf, predicate="within", how="left")
        joined_po = gpd.sjoin(police, edges_buf, predicate="within", how="left")
    except:
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

    # 가중치 계산
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
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

def update_graph_with_model(G, model_path, hour, alpha):
    try:
        with open(model_path, "r") as f:
            weights = np.array(json.load(f)["weights"])
    except:
        log("⚠️ 모델 파일 로드 실패. 기본 가중치를 사용합니다.")
        weights = np.zeros(21)

    for _, _, d in G.edges(data=True):
        x = edge_feats_ext(d, hour)
        score = sigmoid(np.dot(weights, x))
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

# [★] 정적 그래프 로딩 함수
def load_static_graph(center_lat, center_lon, dist_m):
    log(f"🚀 Building Graph (r={dist_m}m)...")
    G = ox.graph_from_point((center_lat, center_lon), dist=dist_m, network_type="walk", simplify=True)
    apply_weights_to_graph(G)
    update_graph_with_model(G, MODEL_PATH, resolve_hour("now"), ALPHA)
    G_proj = ox.project_graph(G)
    return G_proj

# [★] 그래프 매니저: 여러 도시를 관리
class GraphManager:
    def __init__(self):
        self.graphs = {}

    def load_all_cities(self):
        """서버 시작 시 정의된 모든 도시 로딩"""
        for name, info in CITIES_CONFIG.items():
            log(f"🏙️ [System] '{name.upper()}' 지도 생성 및 데이터 주입 중... (오래 걸림)")
            try:
                start_t = time.time()
                G = load_static_graph(info["lat"], info["lon"], info["dist"])
                self.graphs[name] = G
                elapsed = time.time() - start_t
                log(f"✅ [System] '{name.upper()}' 완료! ({elapsed:.1f}초)")
            except Exception as e:
                log(f"🔥 [System] '{name.upper()}' 실패: {e}")

    def get_graph(self, lat, lon):
        """좌표와 가장 가까운 도시의 그래프 반환"""
        best_city = None
        min_dist = float('inf')

        for name, info in CITIES_CONFIG.items():
            dist = (lat - info["lat"])**2 + (lon - info["lon"])**2
            if dist < min_dist:
                min_dist = dist
                best_city = name
        
        return self.graphs.get(best_city)

# 전역 인스턴스
graph_manager = GraphManager()

# 경로 찾기
def nearest_node(G, lat, lon):
    x, y = latlon_to_graph_xy(G, lat, lon)
    return ox.distance.nearest_nodes(G, x, y)

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
        # 비상용 (느림)
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
            edges = G.get_edge_data(u, v)
            best_key = min(edges, key=lambda k: edges[k].get("weight_runtime", 1e9))
            data = edges[best_key]
            
            if "geometry" in data:
                xs, ys = data["geometry"].xy
                rerouted.extend(list(zip(ys, xs)))
            else:
                rerouted.append((G.nodes[u]['y'], G.nodes[u]['x']))
        rerouted.append((G.nodes[dest]['y'], G.nodes[dest]['x']))
    except nx.NetworkXNoPath:
        rerouted = []
    except Exception:
        rerouted = []

    # 4. 시각화 데이터 추출
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
    nodes = G.nodes
    
    for u, v, d in G.edges(data=True):
        uy, ux = nodes[u]['y'], nodes[u]['x']
        if min_lat <= uy <= max_lat and min_lon <= ux <= max_lon:
            coords = []
            if "geometry" in d:
                xs, ys = d["geometry"].xy
                coords = list(zip(ys, xs))
            else:
                coords = [(uy, ux), (nodes[v]['y'], nodes[v]['x'])]
            
            dens = d.get("density_per_km", 0.0)
            color = "#1a9641"
            if dens < 5: color = "#d7191c"
            elif dens < 15: color = "#fdae61"
            
            segments.append({
                "geometry": coords,
                "color": color,
                "properties": {"density": dens}
            })
            
    return segments
