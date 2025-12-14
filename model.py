"""
model.py
Unified dynamic routing pipeline (Hybrid Mode + Coordinate Transformation).
Fixes: Coordinate transformation precision, path geometry reconstruction, start/end point snapping.
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
from pyproj import Transformer 

# ------------------------ Settings ------------------------ #
TMAP_API_URL = "https://apis.openapi.sk.com/tmap/routes/pedestrian"
TMAP_APP_KEY = os.getenv("TMAP_APP_KEY")
TMAP_TIMEOUT = 15

# [설정] 미리 메모리에 올릴 도시
CITIES_CONFIG = {
    "incheon": {"lat": 37.4563, "lon": 126.7052, "dist": 12000}, 
    "seoul":   {"lat": 37.5665, "lon": 126.9780, "dist": 15000},
    "suwon":   {"lat": 37.2636, "lon": 127.0286, "dist": 10000},
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
    """
    Lat/Lon(4326) 좌표를 그래프의 투영 좌표계(UTM 등)로 변환
    """
    crs = Gp.graph.get("crs", "EPSG:3857")
    # Point(lon, lat) 순서 주의
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs)
    geom = pt.geometry.iloc[0]
    return float(geom.x), float(geom.y)

def utm_epsg_from_latlon(lat: float, lon: float) -> int:
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
    """지오메트리가 없는 엣지에 직선 지오메트리 부여 (Lat/Lon 상태에서 수행)"""
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
    원본 G(Lat/Lon)에 가중치 속성을 부여합니다.
    계산 과정에서만 일시적으로 투영(UTM)하여 거리/면적 계산을 수행합니다.
    """
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    edges = ensure_line_geoms(edges, nodes)

    # 중심점 기준으로 적절한 UTM Zone 찾기
    cent_lat = nodes["y"].mean()
    cent_lon = nodes["x"].mean()
    epsg = utm_epsg_from_latlon(cent_lat, cent_lon)
    
    # 계산용 투영 (Meters)
    edges_utm = edges.to_crs(epsg=epsg)
    edges_utm["length_m"] = edges_utm.length # 미터 단위 길이 재계산

    cctv = load_cctv_points(CCTV_XLSX).to_crs(epsg=epsg)
    street = load_generic_points(STREETLIGHT_PATH).to_crs(epsg=epsg)
    police = load_generic_points(POLICE_PATH).to_crs(epsg=epsg)

    # 버퍼 80m
    edges_buf = edges_utm[["u", "v", "key", "geometry"]].copy()
    edges_buf["geometry"] = edges_buf.buffer(80.0)

    # Spatial Join
    try:
        joined_cctv = gpd.sjoin(cctv, edges_buf, predicate="within", how="left")
        joined_st = gpd.sjoin(street, edges_buf, predicate="within", how="left")
        joined_po = gpd.sjoin(police, edges_buf, predicate="within", how="left")
    except:
        joined_cctv = gpd.sjoin(cctv, edges_buf, op="within", how="left")
        joined_st = gpd.sjoin(street, edges_buf, op="within", how="left")
        joined_po = gpd.sjoin(police, edges_buf, op="within", how="left")

    def agg_count(joined, col):
        return joined.groupby(["u", "v", "key"])["count"].sum().rename(col)

    counts_cctv = agg_count(joined_cctv, "cctv_sum")
    counts_st = agg_count(joined_st, "light_sum")
    counts_po = agg_count(joined_po, "police_sum")

    edges_utm = edges_utm.join(counts_cctv, on=["u", "v", "key"])
    edges_utm = edges_utm.join(counts_st, on=["u", "v", "key"])
    edges_utm = edges_utm.join(counts_po, on=["u", "v", "key"])
    
    edges_utm = edges_utm.fillna({"cctv_sum": 0, "light_sum": 0, "police_sum": 0})

    edges_utm["edge_km"] = edges_utm["length_m"].clip(lower=1e-6) / 1000.0
    edges_utm["density_per_km"] = edges_utm["cctv_sum"] / edges_utm["edge_km"]
    edges_utm["light_per_km"] = edges_utm["light_sum"] / edges_utm["edge_km"]
    edges_utm["police_per_km"] = edges_utm["police_sum"] / edges_utm["edge_km"]

    def normalize(s):
        lower = s.quantile(0.05)
        upper = s.quantile(0.95)
        if upper == lower: return s * 0
        return ((s - lower) / (upper - lower)).clip(0, 1)

    edges_utm["dens_norm"] = normalize(edges_utm["density_per_km"])
    edges_utm["light_norm"] = normalize(edges_utm["light_per_km"])
    edges_utm["police_norm"] = normalize(edges_utm["police_per_km"])

    # 원본 그래프 G 업데이트 (속성만 복사)
    for _, r in edges_utm.iterrows():
        if G.has_edge(r["u"], r["v"], r["key"]):
            d = G[r["u"]][r["v"]][r["key"]]
            d.update({
                "length_m": float(r["length_m"]),
                "cctv_sum_num": float(r["cctv_sum"]),
                "light_sum_num": float(r["light_sum"]),
                "police_sum_num": float(r["police_sum"]),
                "dens_norm_num": float(r["dens_norm"]),
                "light_norm_num": float(r["light_norm"]),
                "police_norm_num": float(r["police_norm"]),
                "len_m_num": float(r["length_m"]),
                "density_per_km": float(r["density_per_km"])
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
        # 안전할수록(점수 높을수록) weight_runtime(비용) 감소
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

def load_static_graph(center_lat, center_lon, dist_m):
    log(f"🚀 Building Graph (r={dist_m}m)...")
    # 1. Lat/Lon 그래프 생성
    G = ox.graph_from_point((center_lat, center_lon), dist=dist_m, network_type="walk", simplify=True)
    
    # 2. 가중치 계산 (Lat/Lon 상태에서 속성 주입)
    apply_weights_to_graph(G)
    update_graph_with_model(G, MODEL_PATH, resolve_hour("now"), ALPHA)
    
    # 3. 그래프 투영 (Lat/Lon -> Meters)
    # 이 단계 이후 G_proj의 노드는 (x, y) 미터 좌표를 가짐
    G_proj = ox.project_graph(G)
    return G_proj

class GraphManager:
    def __init__(self):
        self.graphs = {}

    def load_all_cities(self):
        for name, info in CITIES_CONFIG.items():
            log(f"🏙️ [System] '{name.upper()}' 지도 생성 중... (메모리 로딩)")
            try:
                start_t = time.time()
                G = load_static_graph(info["lat"], info["lon"], info["dist"])
                self.graphs[name] = G
                elapsed = time.time() - start_t
                log(f"✅ [System] '{name.upper()}' 완료! ({elapsed:.1f}초)")
            except Exception as e:
                log(f"🔥 [System] '{name.upper()}' 실패: {e}")

    def get_graph(self, lat, lon):
        limit_dist_sq = (0.2) ** 2 # 약 20km
        
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
    # G는 이미 투영된 그래프이므로, 입력 lat/lon도 투영해서 찾아야 함
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
        log("🐢 [Fallback] 지원하지 않는 지역. 실시간 생성 시작...")
        center_lat = (start_lat + end_lat) / 2
        center_lon = (start_lon + end_lon) / 2
        dist_deg = ((start_lat - end_lat)**2 + (start_lon - end_lon)**2)**0.5
        dist_m = max(1000, dist_deg * 111000 * 1.5)
        G = load_static_graph(center_lat, center_lon, dist_m=int(dist_m))

    # 3. 길 찾기
    orig = nearest_node(G, start_lat, start_lon)
    dest = nearest_node(G, end_lat, end_lon)
    
    # 좌표 변환기 (Graph CRS -> WGS84)
    # always_xy=True: 입력(x,y) -> 출력(lon, lat) 순서 보장
    graph_crs = G.graph.get("crs", "EPSG:3857")
    transformer = Transformer.from_crs(graph_crs, "EPSG:4326", always_xy=True)

    rerouted = []
    
    # [수정] 시작점 연결 (User Start -> Nearest Node)
    rerouted.append((start_lat, start_lon))

    try:
        path_nodes = nx.shortest_path(G, orig, dest, weight="weight_runtime")
        
        # 경로 재구성 (Geometry 복원)
        path_points = []
        
        for i in range(len(path_nodes)-1):
            u, v = path_nodes[i], path_nodes[i+1]
            
            # 엣지 데이터 가져오기 (MultiDiGraph이므로 최소 가중치 엣지 선택)
            edges = G.get_edge_data(u, v)
            if not edges: continue
            
            best_key = min(edges, key=lambda k: edges[k].get("weight_runtime", 1e9))
            data = edges[best_key]
            
            if "geometry" in data:
                # 1) 원래 형상이 있는 경우 (곡선 도로)
                xs, ys = data["geometry"].xy
                # transformer returns (lon, lat) due to always_xy=True
                lonlats = [transformer.transform(x, y) for x, y in zip(xs, ys)]
                # append as (lat, lon)
                path_points.extend([(lat, lon) for lon, lat in lonlats])
            else:
                # 2) 형상이 없는 경우 (직선 도로 - ox.project_graph 시 누락될 수 있음)
                # 시작 노드 u
                ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
                ulon, ulat = transformer.transform(ux, uy)
                
                # 끝 노드 v
                vx, vy = G.nodes[v]['x'], G.nodes[v]['y']
                vlon, vlat = transformer.transform(vx, vy)
                
                path_points.append((ulat, ulon))
                path_points.append((vlat, vlon))

        rerouted.extend(path_points)

    except nx.NetworkXNoPath:
        rerouted = []
    except Exception as e:
        log(f"Error in shortest_path: {e}")
        rerouted = []

    rerouted.append((end_lat, end_lon))

    # 4. 시각화 데이터 추출
    visual_segments = extract_visual_segments_bbox(G, start_lat, start_lon, end_lat, end_lon, transformer)

    return PipelineResult(
        tmap_raw=raw,
        base_route=base_route,
        rerouted=rerouted,
        base_weight=0, 
        rerouted_weight=0,
        visual_segments=visual_segments
    )

def extract_visual_segments_bbox(G, slat, slon, elat, elon, transformer, padding=0.005):
    """
    BBox 내 엣지 추출 (Transformer 전달받아 사용)
    """
    min_lat, max_lat = min(slat, elat) - padding, max(slat, elat) + padding
    min_lon, max_lon = min(slon, elon) - padding, max(slon, elon) + padding
    
    segments = []
    nodes = G.nodes
    
    # 너무 많은 엣지 탐색 방지 (필요시 최적화 가능)
    # 현재는 전체 엣지 순회 (메모리 상 큰 도시라면 Spatial Index 필요하지만 일단 단순 루프)
    for u, v, d in G.edges(data=True):
        # 노드 좌표 가져오기 (Meters)
        ux, uy = nodes[u]['x'], nodes[u]['y']
        
        # WGS84 변환 (Lon, Lat)
        ulon, ulat = transformer.transform(ux, uy)
        
        # BBox 체크
        if min_lat <= ulat <= max_lat and min_lon <= ulon <= max_lon:
            coords = []
            if "geometry" in d:
                xs, ys = d["geometry"].xy
                lonlats = [transformer.transform(x, y) for x, y in zip(xs, ys)]
                coords = [(lat, lon) for lon, lat in lonlats]
            else:
                vx, vy = nodes[v]['x'], nodes[v]['y']
                vlon, vlat = transformer.transform(vx, vy)
                coords = [(ulat, ulon), (vlat, vlon)]
            
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
