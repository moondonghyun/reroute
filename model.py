"""
Unified dynamic routing pipeline.

Steps:
1) Fetch base route from Tmap API using start/end lat/lon.
2) Build a local OSM walk graph around that route (margin) with osmnx.
3) Inject CCTV density into the graph and derive CCTV-aware edge weights.
4) Load a trained preference model and apply rerouting weights.
5) Return rerouted path as JSON; optional test mode visualizes base vs rerouted.

References combined: route.py, build_graph.py, build_cctv_graph.py, ai_dynamic_routing.py.
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

# ------------------------ Tmap API ------------------------ #
TMAP_API_URL = "https://apis.openapi.sk.com/tmap/routes/pedestrian"
TMAP_APP_KEY = os.getenv("TMAP_APP_KEY", "IqFRypKZ8h81kp9xXLyKY5OfY9PwYSxi8K2pHLkb")
TMAP_TIMEOUT = 15

# ------------------------ Defaults ------------------------ #
START_LAT = 37.4451
START_LON = 126.6942
END_LAT = 37.4166
END_LON = 126.6863
NETWORK_TYPE = "walk"
MARGIN_M = 400
CCTV_XLSX = "cctv_data.xlsx"
STREETLIGHT_PATH = "nationwide_streetlight.xlsx"
POLICE_PATH = "Police_station.csv"
ALPHA = 6.0
HOUR_DEFAULT = "now"
OUT_JSON = "model_test/result.json"
OUT_HTML = "model_test/compare.html"
MODEL_PATH = "edge_pref_model_dataset.json"


# ------------------------ Utilities ------------------------ #
def log(msg: str) -> None:
    print(msg, flush=True)


def resolve_hour(val: Any) -> int:
    if val is None:
        return time.localtime().tm_hour
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"now", "auto", "current", "local"}:
            return time.localtime().tm_hour
        try:
            h = int(float(s))
        except Exception:
            h = time.localtime().tm_hour
    elif isinstance(val, (int, float)):
        h = int(val)
    else:
        h = time.localtime().tm_hour
    if h < 0:
        h = 0
    if h > 23:
        h = h % 24
    return int(h)


def as_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None or val == "":
            return float(default)
        return float(val)
    except Exception:
        return float(default)


def latlon_margin_deg(lat_center: float, margin_m: float) -> Tuple[float, float]:
    dlat = margin_m / 111320.0
    dlon = margin_m / (111320.0 * max(1e-6, math.cos(math.radians(lat_center))))
    return dlat, dlon


def make_bbox(route_coords: Sequence[Tuple[float, float]], margin_m: float) -> gpd.GeoDataFrame:
    lats = [p[0] for p in route_coords]
    lons = [p[1] for p in route_coords]
    lat_c = sum(lats) / len(lats)
    dlat, dlon = latlon_margin_deg(lat_c, margin_m)
    min_lat = min(lats) - dlat
    max_lat = max(lats) + dlat
    min_lon = min(lons) - dlon
    max_lon = max(lons) + dlon
    return gpd.GeoDataFrame(geometry=[box(min_lon, min_lat, max_lon, max_lat)], crs="EPSG:4326")


def graph_from_polygon_compat(poly, network_type: str = NETWORK_TYPE):
    attempts = [
        dict(network_type=network_type, simplify=True, retain_all=False),
        dict(network_type=network_type, simplify=True),
        dict(network_type=network_type),
    ]
    last_err = None
    for kw in attempts:
        try:
            return ox.graph_from_polygon(poly, **kw)
        except TypeError as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("graph_from_polygon failed")


def graph_from_point_compat(center_latlon: Tuple[float, float], dist_m: float, network_type: str = NETWORK_TYPE):
    attempts = [
        dict(center_point=center_latlon, dist=dist_m, network_type=network_type, simplify=True, retain_all=False),
        dict(center_point=center_latlon, dist=dist_m, network_type=network_type, simplify=True),
        dict(center_point=center_latlon, dist=dist_m, network_type=network_type),
    ]
    last_err = None
    for kw in attempts:
        try:
            return ox.graph_from_point(**kw)
        except TypeError as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("graph_from_point failed")


def project_graph_compat(G, to_crs=None):
    try:
        return ox.project_graph(G, to_crs=to_crs)
    except Exception:
        from osmnx.projection import project_graph as pj
        return pj(G, to_crs=to_crs)


def latlon_to_graph_xy(Gp, lat: float, lon: float) -> Tuple[float, float]:
    crs = Gp.graph.get("crs", "EPSG:3857")
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs)
    geom = pt.geometry.iloc[0]
    return float(geom.x), float(geom.y)


def nearest_node(G, lat, lon):
    Gp = project_graph_compat(G)
    x, y = latlon_to_graph_xy(Gp, lat, lon)
    
    # 1. 일단 가장 가까운 노드(기존 방식)를 찾음 (후보 1)
    nearest_node = ox.distance.nearest_nodes(Gp, x, y)
    
    # 2. 그 노드가 어떤 도로에 붙어있는지 확인
    # (해당 노드에 연결된 엣지들의 highway 태그를 검사)
    hw_types = []
    try:
        # 이 노드에서 나가는 도로들을 확인
        for _, _, data in G.edges(nearest_node, data=True):
            hw = data.get("highway", "")
            if isinstance(hw, list): hw = hw[0]
            hw_types.append(str(hw))
    except:
        pass
    
    # 3. 만약 뒷골목(service, track, path)이나 좁은길이라면, 더 넓은 길을 찾아봄
    # (예: service 로드라면 반경 50m 내의 다른 노드를 탐색)
    BAD_ROADS = ["service", "track", "path", "footway", "steps", "corridor"]
    
    is_bad_node = False
    if not hw_types: 
        is_bad_node = True # 연결된 도로 정보가 없으면 나쁜 노드 취급
    else:
        # 연결된 도로 중 하나라도 '좋은 도로'가 없으면 나쁜 노드
        # 즉, 모든 도로가 bad_roads에 속하면 True
        if all(any(bad in h for bad in BAD_ROADS) for h in hw_types):
            is_bad_node = True

    if is_bad_node:
        # log(f"  [스마트 스냅] '{nearest_node}'는 뒷골목({hw_types})입니다. 큰 길을 찾습니다...")
        
        # 현재 노드의 '이웃 노드'들을 뒤져서 더 좋은 도로(residential, tertiary 등)가 있는지 확인
        # (BFS 탐색: 1단계 이웃만 확인)
        best_node = nearest_node
        min_penalty_dist = float('inf')
        
        neighbors = list(G.neighbors(nearest_node))
        
        # 현재 노드의 좌표
        nx_val = G.nodes[nearest_node]['x']
        ny_val = G.nodes[nearest_node]['y']
        base_dist = ((x - nx_val)**2 + (y - ny_val)**2)**0.5 # 현재 노드까지의 거리
        
        # 이웃 노드 평가
        for n in neighbors:
            # 이웃 노드의 도로 타입 확인
            n_hw_types = []
            for _, _, d in G.edges(n, data=True):
                h = d.get("highway", "")
                if isinstance(h, list): h = h[0]
                n_hw_types.append(str(h))
            
            # 이웃이 '좋은 도로'를 포함하고 있는지?
            is_good_neighbor = False
            if n_hw_types and not all(any(bad in h for bad in BAD_ROADS) for h in n_hw_types):
                is_good_neighbor = True
            
            if is_good_neighbor:
                # 거리 계산
                nx_n = G.nodes[n]['x']
                ny_n = G.nodes[n]['y']
                dist_n = ((x - nx_n)**2 + (y - ny_n)**2)**0.5
                
                # [핵심 로직]
                # 좋은 도로라면 거리에 '보너스(할인)'를 줌 (예: 30m 정도는 더 멀어도 봐줌)
                # 즉, 뒷골목 5m보다 큰길 30m를 더 선호하게 만듦
                score = dist_n - 30.0 
                
                if score < base_dist: # 원래 노드보다 점수가 좋으면 교체
                    return n
                    
    # 더 좋은 대안이 없거나, 원래 노드가 이미 큰 길이라면 그대로 반환
    return nearest_node


def extract_linestring_coords_from_features(features: Iterable[Dict[str, Any]]) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    for feat in features:
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        if gtype == "LineString":
            for lon, lat in geom.get("coordinates", []):
                coords.append((lat, lon))
        elif gtype == "MultiLineString":
            for line in geom.get("coordinates", []):
                for lon, lat in line:
                    coords.append((lat, lon))
    return coords


def parse_route_coords(raw: Dict[str, Any]) -> List[Tuple[float, float]]:
    if "response" in raw:
        data = raw["response"]
    else:
        data = raw
    if isinstance(data, dict) and "features" in data:
        coords = extract_linestring_coords_from_features(data["features"])
        if len(coords) >= 2:
            return coords
    if isinstance(data, dict):
        for key in ["geojson", "result", "route", "data"]:
            val = data.get(key)
            if isinstance(val, dict) and "features" in val:
                coords = extract_linestring_coords_from_features(val["features"])
                if len(coords) >= 2:
                    return coords
    if isinstance(raw, dict) and "features" in raw:
        coords = extract_linestring_coords_from_features(raw["features"])
        if len(coords) >= 2:
            return coords
    raise ValueError("No LineString coordinates found in Tmap response")


def extract_visual_segments(G: nx.MultiDiGraph) -> List[Dict[str, Any]]:
    # 1. 그래프에서 엣지(도로) 데이터 추출
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()

    # 2. 색상 계산을 위한 준비 (CCTV 밀집도 기준)
    if "density_per_km" not in edges.columns:
        edges["density_per_km"] = 0.0

    vals = edges["density_per_km"]
    vmin = float(vals.quantile(0.05))
    vmax = float(vals.quantile(0.95))
    if vmin == vmax: vmax = vmin + 0.0001

    # Python에서 색상 코드를 계산해주는 도구 (기존과 동일한 로직)
    cmap = cm.linear.RdYlGn_11.scale(vmin, vmax)

    segments = []

    for _, r in edges.iterrows():
        geom = r["geometry"]
        dens = float(r.get("density_per_km", 0.0))

        # [핵심] 점수(dens)를 넣으면 Hex Color Code (#RRGGBB)를 뱉어줌
        color_hex = cmap(dens)

        # 좌표 추출 ([lat, lon] 순서로 변환)
        coords = []
        if isinstance(geom, LineString):
            # (lon, lat) -> (lat, lon)
            coords = [(y, x) for x, y in list(geom.coords)]
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                coords += [(y, x) for x, y in list(line.coords)]

        # 결과 리스트에 추가
        segments.append({
            "geometry": coords,  # [[37.xx, 126.xx], [37.xx, 126.xx], ...]
            "color": color_hex,  # "#d9f0a3" (안전도에 따른 색상)
            "properties": {  # 상세 정보 (앱에서 클릭 시 보여줄 용도)
                "cctv_count": int(r.get('cctv_sum', 0)),
                "light_count": int(r.get('light_sum', 0)),
                "police_count": int(r.get('police_sum', 0)),
                "density": float(f"{dens:.2f}")
            }
        })

    return segments


# ------------------------ Tmap fetch ------------------------ #
def fetch_tmap_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    app_key: str = TMAP_APP_KEY,
    start_name: str = "start",
    end_name: str = "end",
) -> Tuple[Dict[str, Any], List[Tuple[float, float]]]:
    params = {
        "version": "1",
        "startX": str(start_lon),
        "startY": str(start_lat),
        "endX": str(end_lon),
        "endY": str(end_lat),
        "startName": start_name,
        "endName": end_name,
        "appKey": app_key,
    }
    resp = requests.get(TMAP_API_URL, params=params, timeout=TMAP_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    coords = parse_route_coords(data)
    return data, coords


# ------------------------ CCTV weighting ------------------------ #
def utm_epsg_from_latlon(lat: float, lon: float) -> int:
    zone = int(math.floor((lon + 180) / 6) + 1)
    return 32600 + zone if lat >= 0 else 32700 + zone


def ensure_line_geoms(edges_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    node_y = nodes_gdf["y"].to_dict()
    node_x = nodes_gdf["x"].to_dict()
    missing = edges_gdf["geometry"].isna()
    for idx, row in edges_gdf[missing].iterrows():
        uy, ux = node_y[row["u"]], node_x[row["u"]]
        vy, vx = node_y[row["v"]], node_x[row["v"]]
        edges_gdf.at[idx, "geometry"] = LineString([(ux, uy), (vx, vy)])
    return edges_gdf


def load_cctv_points(path: str) -> gpd.GeoDataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    def pick(col_names: Sequence[str]) -> str:
        for cand in col_names:
            if cand in df.columns:
                return cand
        raise KeyError(f"Columns {col_names} not found in CCTV file {path}")

    lat_col = pick(["위도", "lat", "latitude"])
    lon_col = pick(["경도", "lon", "longitude", "lng"])
    count_col = None
    for cand in ["카메라대수", "카메라 수", "camera_count", "count"]:
        if cand in df.columns:
            count_col = cand
            break
    if count_col is None:
        df["camera_count"] = 1
        count_col = "camera_count"

    df[count_col] = pd.to_numeric(df[count_col], errors="coerce").fillna(1).clip(lower=1)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
    gdf = gdf.rename(columns={count_col: "camera_count"})
    if "count" not in gdf.columns:
        gdf["count"] = gdf["camera_count"]
    return gdf


def load_generic_points(path: str, lat_keys=None, lon_keys=None, count_keys=None) -> gpd.GeoDataFrame:
    lat_keys = lat_keys or ["lat", "latitude", "위도", "Y", "y"]
    lon_keys = lon_keys or ["lon", "longitude", "lng", "경도", "X", "x"]
    count_keys = count_keys or ["count", "cnt", "num", "value"]

    if path.lower().endswith((".csv", ".txt")):
        df = pd.read_csv(path)
    else:
        try:
            df = gpd.read_file(path)
        except Exception:
            df = pd.read_excel(path)
    if isinstance(df, gpd.GeoDataFrame) and "geometry" in df.columns:
        gdf = df
    else:
        df.columns = df.columns.str.strip()
        def pick(keys):
            for k in keys:
                if k in df.columns:
                    return k
            raise KeyError(f"{keys} not found in {path}")
        lat_col = pick(lat_keys)
        lon_col = pick(lon_keys)
        cnt_col = None
        for k in count_keys:
            if k in df.columns:
                cnt_col = k
                break
        if cnt_col is None:
            df["count"] = 1
            cnt_col = "count"
        df[cnt_col] = pd.to_numeric(df[cnt_col], errors="coerce").fillna(1).clip(lower=1)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
        gdf = gdf.rename(columns={cnt_col: "count"})
    if "count" not in gdf.columns:
        gdf["count"] = 1
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf, geometry=gdf["geometry"], crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notna()]
    return gdf



def safe_load_generic_points(path: str, name: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        log(f"  warning: {name} file not found at {path}; using empty points.")
        return gpd.GeoDataFrame(columns=["count", "geometry"], geometry=[], crs="EPSG:4326")
    return load_generic_points(path)


def load_police_points(path: str) -> gpd.GeoDataFrame:
    """Police_station.csv: now uses A1(=lon), A2(=lat) columns; falls back to first two cols."""
    if not os.path.exists(path):
        log(f"  warning: police file not found at {path}; using empty points.")
        return gpd.GeoDataFrame(columns=["count", "geometry"], geometry=[], crs="EPSG:4326")
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        log(f"  warning: police file lacks lon/lat columns at {path}; using empty points.")
        return gpd.GeoDataFrame(columns=["count", "geometry"], geometry=[], crs="EPSG:4326")

    lon_series = df["A1"] if "A1" in df.columns else df.iloc[:, 0]
    lat_series = df["A2"] if "A2" in df.columns else df.iloc[:, 1]
    lon = pd.to_numeric(lon_series, errors="coerce")
    lat = pd.to_numeric(lat_series, errors="coerce")

    mask = lon.notna() & lat.notna()
    if mask.sum() == 0:
        log(f"  warning: police file has no valid coordinates at {path}; using empty points.")
        return gpd.GeoDataFrame(columns=["count", "geometry"], geometry=[], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(
        {"count": [1] * mask.sum()},
        geometry=gpd.points_from_xy(lon[mask], lat[mask]),
        crs="EPSG:4326",
    )
    return gdf[gdf.geometry.notna()]

def apply_cctv_weights(
    G: nx.MultiDiGraph,
    cctv_path: str = CCTV_XLSX,
    streetlight_path: str = STREETLIGHT_PATH,
    police_path: str = POLICE_PATH,
    radius_m: float = 80.0,
    alpha: float = ALPHA,
) -> None:
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    edges = ensure_line_geoms(edges, nodes)

    cent_lat, cent_lon = nodes["y"].mean(), nodes["x"].mean()
    epsg = utm_epsg_from_latlon(cent_lat, cent_lon)
    edges_utm = edges.to_crs(epsg=epsg)
    edges_utm["length_m"] = edges_utm["length"] if "length" in edges_utm.columns else edges_utm.length

    cctv = load_cctv_points(cctv_path).to_crs(epsg=epsg)
    street = safe_load_generic_points(streetlight_path, "streetlight").to_crs(epsg=epsg)
    police = load_police_points(police_path).to_crs(epsg=epsg)
    edges_buf = edges_utm[["u", "v", "key", "geometry"]].copy()
    edges_buf["geometry"] = edges_buf.buffer(radius_m)

    try:
        joined_cctv = gpd.sjoin(cctv, edges_buf, predicate="within", how="left")
        joined_st = gpd.sjoin(street, edges_buf, predicate="within", how="left")
        joined_po = gpd.sjoin(police, edges_buf, predicate="within", how="left")
    except TypeError:
        joined_cctv = gpd.sjoin(cctv, edges_buf, op="within", how="left")
        joined_st = gpd.sjoin(street, edges_buf, op="within", how="left")
        joined_po = gpd.sjoin(police, edges_buf, op="within", how="left")

    def agg_joined(joined_df, col_name):
        grp = joined_df.dropna(subset=["u"]).groupby(["u", "v", "key"])["count"].sum().rename(col_name)
        return grp

    counts_cctv = agg_joined(joined_cctv, "cctv_sum")
    counts_st = agg_joined(joined_st, "light_sum")
    counts_po = agg_joined(joined_po, "police_sum")

    edges_utm = edges_utm.merge(counts_cctv.reset_index(), on=["u", "v", "key"], how="left")
    edges_utm = edges_utm.merge(counts_st.reset_index(), on=["u", "v", "key"], how="left")
    edges_utm = edges_utm.merge(counts_po.reset_index(), on=["u", "v", "key"], how="left")
    edges_utm["cctv_sum"] = edges_utm["cctv_sum"].fillna(0)
    edges_utm["light_sum"] = edges_utm["light_sum"].fillna(0)
    edges_utm["police_sum"] = edges_utm["police_sum"].fillna(0)

    edges_utm["edge_km"] = edges_utm["length_m"].clip(lower=1e-6) / 1000.0
    edges_utm["density_per_km"] = edges_utm["cctv_sum"] / edges_utm["edge_km"]
    edges_utm["light_per_km"] = edges_utm["light_sum"] / edges_utm["edge_km"]
    edges_utm["police_per_km"] = edges_utm["police_sum"] / edges_utm["edge_km"]

    def norm_col(df, src, q_low=0.05, q_high=0.95, out="norm"):
        low = float(df[src].quantile(q_low))
        high = float(df[src].quantile(q_high))
        rng = max(1e-6, high - low)
        df[out] = ((df[src] - low) / rng).clip(0, 1)

    norm_col(edges_utm, "density_per_km", out="dens_norm")
    norm_col(edges_utm, "light_per_km", out="light_norm")
    norm_col(edges_utm, "police_per_km", out="police_norm")

    combined_score = edges_utm["dens_norm"] + 1.5 * edges_utm["light_norm"] + 3.0 * edges_utm["police_norm"]
    edges_utm["weight_cctv"] = edges_utm["length_m"] / (1.0 + alpha * combined_score)

    for _, r in edges_utm.iterrows():
        u, v, k = r["u"], r["v"], r["key"]
        if G.has_edge(u, v, k):
            d = G[u][v][k]
            d["length_m"] = float(r["length_m"])
            d["cctv_sum"] = float(r["cctv_sum"])
            d["density_per_km"] = float(r["density_per_km"])
            d["dens_norm"] = float(r["dens_norm"])
            d["light_sum"] = float(r["light_sum"])
            d["light_per_km"] = float(r["light_per_km"])
            d["light_norm"] = float(r["light_norm"])
            d["police_sum"] = float(r["police_sum"])
            d["police_per_km"] = float(r["police_per_km"])
            d["police_norm"] = float(r["police_norm"])
            d["weight_cctv"] = float(r["weight_cctv"])


# ------------------------ Model helpers ------------------------ #
def sanitize_graph_edge_attrs(G: nx.MultiDiGraph) -> None:
    for _, _, _, d in G.edges(keys=True, data=True):
        length_m = d.get("length_m", d.get("length", 1.0))
        d["len_m_num"] = as_float(length_m, default=1.0)
        d["dens_norm_num"] = as_float(d.get("dens_norm", 0.0), default=0.0)
        d["cctv_sum_num"] = as_float(d.get("cctv_sum", 0.0), default=0.0)
        d["light_norm_num"] = as_float(d.get("light_norm", 0.0), default=0.0)
        d["light_sum_num"] = as_float(d.get("light_sum", 0.0), default=0.0)
        d["police_norm_num"] = as_float(d.get("police_norm", 0.0), default=0.0)
        d["police_sum_num"] = as_float(d.get("police_sum", 0.0), default=0.0)
        if "weight_runtime" in d:
            d["weight_runtime"] = as_float(d["weight_runtime"], default=d["len_m_num"])
        else:
            d["weight_runtime"] = d["len_m_num"]


def edge_feats_ext(d: Dict[str, Any], hour: int) -> np.ndarray:
    L = d["len_m_num"]
    dn = d["dens_norm_num"]
    cctv = d["cctv_sum_num"]
    light = d.get("light_sum_num", 0.0)
    police = d.get("police_sum_num", 0.0)
    km = max(1e-6, L / 1000.0)
    cctv_per_km = cctv / km
    light_per_km = light / km
    police_per_km = police / km
    light_norm = d.get("light_norm_num", 0.0)
    police_norm = d.get("police_norm_num", 0.0)

    hw_val = d.get("highway", "")
    hw_list = [str(hw_val).lower()]
    if isinstance(hw_val, list) and hw_val:
        hw_list = [str(x).lower() for x in hw_val]

    def has(tag: str) -> bool:
        return any(tag in h for h in hw_list)

    return np.array(
        [
            1.0,
            math.log1p(L),
            dn,
            cctv_per_km,
            light_per_km,
            police_per_km,
            light_norm,
            police_norm,
            float(has("primary")),
            float(has("secondary")),
            float(has("tertiary")),
            float(has("unclassified")),
            float(has("residential")),
            float(has("service")),
            float(has("footway")),
            float(has("path")),
            float(has("cycleway")),
            float(has("steps")),
            float(has("track")),
            float(has("living_street")),
            float(has("pedestrian")),
        ],
        dtype=float,
    )


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def load_model(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data.get("weights", []), dtype=float)


def apply_model_scores_to_graph(G: nx.MultiDiGraph, w: np.ndarray, hour: int, alpha: float) -> None:
    for _, _, _, d in G.edges(keys=True, data=True):
        x = edge_feats_ext(d, hour)
        score = sigmoid(float(np.dot(w, x)))
        L = d["len_m_num"]
        d["weight_runtime"] = L / (1.0 + alpha * score)


# ------------------------ Routing helpers ------------------------ #
def path_edges(G: nx.MultiDiGraph, nodes_path: Sequence) -> List[Tuple[Any, Any, Any, Dict[str, Any]]]:
    out: List[Tuple[Any, Any, Any, Dict[str, Any]]] = []
    for i in range(1, len(nodes_path)):
        u, v = nodes_path[i - 1], nodes_path[i]
        dd = G.get_edge_data(u, v)
        if not dd:
            continue
        k = min(dd, key=lambda kk: dd[kk].get("len_m_num", 1e12))
        out.append((u, v, k, dd[k]))
    return out


def path_to_latlons(G: nx.MultiDiGraph, nodes_path: Sequence, weight_attr: str = "weight_runtime") -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    for i in range(1, len(nodes_path)):
        u, v = nodes_path[i - 1], nodes_path[i]
        dd = G.get_edge_data(u, v)
        if not dd:
            continue
        k = min(dd, key=lambda kk: dd[kk].get(weight_attr, dd[kk].get("len_m_num", 1e12)))
        geom = dd[k].get("geometry")
        if isinstance(geom, LineString):
            seg = [(y, x) for x, y in list(geom.coords)]
        elif isinstance(geom, MultiLineString):
            seg = []
            for line in geom.geoms:
                seg += [(y, x) for x, y in list(line.coords)]
        else:
            seg = [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]
        if coords and coords[-1] == seg[0]:
            coords += seg[1:]
        else:
            coords += seg
    return coords


def path_weight_sum(G: nx.MultiDiGraph, nodes_path: Sequence, weight_attr: str = "weight_runtime") -> float:
    total = 0.0
    for u, v, k, d in path_edges(G, nodes_path):
        total += float(d.get(weight_attr, d.get("len_m_num", 0.0)))
    return total


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def polyline_length_m(coords: Sequence[Tuple[float, float]]) -> float:
    if len(coords) < 2:
        return 0.0
    dist = 0.0
    for i in range(1, len(coords)):
        lat1, lon1 = coords[i - 1]
        lat2, lon2 = coords[i]
        dist += haversine_m(lat1, lon1, lat2, lon2)
    return dist


# ------------------------ Core pipeline ------------------------ #
@dataclass
class PipelineResult:
    tmap_raw: Dict[str, Any]
    base_route: List[Tuple[float, float]]
    rerouted: List[Tuple[float, float]]
    base_weight: float
    rerouted_weight: float
    html_path: str | None = None
    visual_segments: List[Dict[str, Any]] | None = None


def build_graph_from_route(route_coords: Sequence[Tuple[float, float]], margin_m: float = MARGIN_M) -> nx.MultiDiGraph:
    bbox = make_bbox(route_coords, margin_m)
    poly = bbox.geometry.iloc[0]
    try:
        G = graph_from_polygon_compat(poly, NETWORK_TYPE)
    except Exception:
        lat_c = sum(p[0] for p in route_coords) / len(route_coords)
        lon_c = sum(p[1] for p in route_coords) / len(route_coords)
        G = graph_from_point_compat((lat_c, lon_c), int(margin_m + 50), NETWORK_TYPE)
    try:
        G = ox.add_edge_lengths(G)
    except Exception:
        pass
    return G


def run_pipeline(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    *,
    app_key: str = TMAP_APP_KEY,
    margin_m: float = MARGIN_M,
    cctv_path: str = CCTV_XLSX,
    streetlight_path: str = STREETLIGHT_PATH,
    police_path: str = POLICE_PATH,
    model_path: str = "",
    alpha: float = ALPHA,
    hour: Any = HOUR_DEFAULT,
    html_out: str | None = None,
) -> PipelineResult:
    log("[1/5] Fetch base route from Tmap")
    raw, base_route = fetch_tmap_route(start_lat, start_lon, end_lat, end_lon, app_key=app_key)

    log("[2/5] Build OSM walk graph around route")
    G = build_graph_from_route(base_route, margin_m=margin_m)

    log("[3/5] Inject CCTV density")
    apply_cctv_weights(
        G,
        cctv_path=cctv_path,
        streetlight_path=streetlight_path,
        police_path=police_path,
        radius_m=80.0,
        alpha=alpha,
    )

    log("[4/5] Apply model weights")
    sanitize_graph_edge_attrs(G)
    for _, _, _, d in G.edges(keys=True, data=True):
        dn = d["dens_norm_num"]
        ln = d.get("light_norm_num", 0.0)
        pn = d.get("police_norm_num", 0.0)
        L = d["len_m_num"]
        score = dn + 1.5 * ln + 3.0 * pn
        d["weight_runtime"] = L / (1.0 + alpha * score)
    if not model_path:
        raise FileNotFoundError("Model weights path is empty; provide --model.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    w = load_model(model_path)
    apply_model_scores_to_graph(G, w, hour=resolve_hour(hour), alpha=alpha)

    log("[5/5] Reroute with AI weights")
    o = nearest_node(G, start_lat, start_lon)
    dnode = nearest_node(G, end_lat, end_lon)
    reroute_nodes = nx.shortest_path(G, o, dnode, weight="weight_runtime")
    
    # Base route는 Tmap 응답을 그대로 사용
    base_latlons = base_route
    reroute_latlons = path_to_latlons(G, reroute_nodes, weight_attr="weight_runtime")

    if reroute_latlons:
        start_pt = (start_lat, start_lon)
        end_pt = (end_lat, end_lon)
        reroute_latlons = [start_pt] + reroute_latlons + [end_pt]
    visual_segments = extract_visual_segments(G)
    html_path = None
    if html_out:
        html_path = visualize_routes(G, base_latlons, reroute_latlons, html_out)

    return PipelineResult(
        tmap_raw=raw,
        base_route=base_latlons,
        rerouted=reroute_latlons,
        base_weight=polyline_length_m(base_latlons),
        rerouted_weight=path_weight_sum(G, reroute_nodes, weight_attr="weight_runtime"),
        html_path=html_path,
        visual_segments=visual_segments
    )


# ------------------------ Visualization ------------------------ #
def visualize_routes(
    G: nx.MultiDiGraph,
    base_route: Sequence[Tuple[float, float]],
    rerouted: Sequence[Tuple[float, float]],
    html_out: str = OUT_HTML,
) -> str:
    nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)
    cy, cx = nodes_gdf["y"].mean(), nodes_gdf["x"].mean()
    
    import folium
    m = folium.Map(location=(cy, cx), zoom_start=15, tiles="cartodbpositron")
    
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()

    # [1] 컬러맵 생성 (CCTV 밀집도 기준)
    # 값이 없으면 0으로 처리
    if "density_per_km" not in edges.columns:
        edges["density_per_km"] = 0.0
        
    vals = edges["density_per_km"]
    # 하위 5% ~ 상위 5% 기준으로 색상 범위 설정 (너무 튀는 값 제외)
    vmin = float(vals.quantile(0.05))
    vmax = float(vals.quantile(0.95))
    if vmin == vmax: vmax = vmin + 0.0001
    
    cmap = cm.linear.RdYlGn_11.scale(vmin, vmax)
    cmap.caption = "Safety Score (CCTV Density)"
    
    # [2] 도로 그리기 (가중치에 따라 색칠)
    for _, r in edges.iterrows():
        geom = r["geometry"]
        dens = float(r.get("density_per_km", 0.0))
        
        # 색상 결정
        color = cmap(dens)
        
        # 툴팁에 상세 정보 표시
        tooltip_txt = (
            f"CCTV: {r.get('cctv_sum', 0)}대<br>"
            f"보안등: {r.get('light_sum', 0)}개<br>"
            f"경찰서: {r.get('police_sum', 0)}개<br>"
            f"밀집도: {dens:.2f}"
        )

        if isinstance(geom, LineString):
            xs, ys = list(geom.coords.xy[0]), list(geom.coords.xy[1])
            folium.PolyLine(
                [(ys[i], xs[i]) for i in range(len(xs))],
                color=color, weight=4, opacity=0.7, tooltip=tooltip_txt
            ).add_to(m)
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                xs, ys = list(line.coords.xy[0]), list(line.coords.xy[1])
                folium.PolyLine(
                    [(ys[i], xs[i]) for i in range(len(xs))],
                    color=color, weight=4, opacity=0.7, tooltip=tooltip_txt
                ).add_to(m)

    # [3] 경로 그리기 (기존 로직 유지)
    if base_route:
        folium.PolyLine(base_route, color="blue", weight=4, opacity=0.5, dash_array='5, 10', tooltip="Tmap 추천 (최단)").add_to(m)
    
    if rerouted:
        folium.PolyLine(rerouted, color="magenta", weight=7, opacity=0.95, tooltip="AI reroute").add_to(m)
        folium.Marker(rerouted[0], icon=folium.Icon(color="green", icon="play"), tooltip="start").add_to(m)
        folium.Marker(rerouted[-1], icon=folium.Icon(color="red", icon="stop"), tooltip="end").add_to(m)
    
    # 컬러바 추가
    cmap.add_to(m)
    
    os.makedirs(os.path.dirname(html_out), exist_ok=True)
    m.save(html_out)
    log(f"[TEST] HTML saved: {html_out}")
    return html_out


# ------------------------ CLI ------------------------ #
def to_jsonable_coords(coords: Sequence[Tuple[float, float]]) -> List[Dict[str, float]]:
    return [{"lat": float(lat), "lon": float(lon)} for lat, lon in coords]


def run_cli(
    start_lat: float = START_LAT,
    start_lon: float = START_LON,
    end_lat: float = END_LAT,
    end_lon: float = END_LON,
    *,
    app_key: str = TMAP_APP_KEY,
    margin_m: float = MARGIN_M,
    cctv_xlsx: str = CCTV_XLSX,
    streetlight_path: str = STREETLIGHT_PATH,
    police_path: str = POLICE_PATH,
    model: str | None = MODEL_PATH,
    alpha: float = ALPHA,
    hour: Any = HOUR_DEFAULT,
    out_json: str = OUT_JSON,
    out_html: str = OUT_HTML,
    visualize: bool = True,
) -> PipelineResult:
    """
    코드 내부에서 바로 호출할 수 있는 진입점.
    예) run_cli(START_LAT=..., START_LON=..., END_LAT=..., END_LON=..., model="edge_pref_model.json")
    """
    if not model:
        raise FileNotFoundError("모델 경로(model)가 비었습니다.")

    ox.settings.log_console = False
    ox.settings.use_cache = False
    ox.settings.cache_folder = "osm_cache"
    ox.settings.timeout = 45

    res = run_pipeline(
        start_lat=start_lat,
        start_lon=start_lon,
        end_lat=end_lat,
        end_lon=end_lon,
        app_key=app_key,
        margin_m=margin_m,
        cctv_path=cctv_xlsx,
        streetlight_path=streetlight_path,
        police_path=police_path,
        model_path=model,
        alpha=alpha,
        hour=hour,
        html_out=out_html if visualize else None,
    )

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tmap_raw": res.tmap_raw,
                "base_route": to_jsonable_coords(res.base_route),
                "rerouted": to_jsonable_coords(res.rerouted),
                "base_weight": res.base_weight,
                "rerouted_weight": res.rerouted_weight,
                "html_path": res.html_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    log(f"[DONE] JSON saved: {out_json}")
    if res.html_path:
        log(f"[DONE] HTML saved: {res.html_path}")
    return res


if __name__ == "__main__":
    # 아래 값들만 수정해도 바로 실행 흐름을 제어할 수 있습니다.
    MAIN_START_LAT = START_LAT
    MAIN_START_LON = START_LON
    MAIN_END_LAT = END_LAT
    MAIN_END_LON = END_LON
    MAIN_MODEL = MODEL_PATH          # 필수: 모델 가중치 경로
    MAIN_OUT_JSON = OUT_JSON
    MAIN_OUT_HTML = OUT_HTML
    MAIN_ALPHA = ALPHA
    MAIN_HOUR = HOUR_DEFAULT
    MAIN_MARGIN_M = MARGIN_M
    MAIN_STREETLIGHT_PATH = STREETLIGHT_PATH
    MAIN_POLICE_PATH = POLICE_PATH
    MAIN_VISUALIZE = True

    run_cli(
        start_lat=MAIN_START_LAT,
        start_lon=MAIN_START_LON,
        end_lat=MAIN_END_LAT,
        end_lon=MAIN_END_LON,
        app_key=TMAP_APP_KEY,
        margin_m=MAIN_MARGIN_M,
        cctv_xlsx=CCTV_XLSX,
        streetlight_path=MAIN_STREETLIGHT_PATH,
        police_path=MAIN_POLICE_PATH,
        model=MAIN_MODEL,
        alpha=MAIN_ALPHA,
        hour=MAIN_HOUR,
        out_json=MAIN_OUT_JSON,
        out_html=MAIN_OUT_HTML,
        visualize=MAIN_VISUALIZE,
    )
