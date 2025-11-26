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


def nearest_node(G, lat: float, lon: float):
    Gp = project_graph_compat(G)
    x, y = latlon_to_graph_xy(Gp, lat, lon)
    return ox.distance.nearest_nodes(Gp, x, y)


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
    return gdf.rename(columns={count_col: "camera_count"})


def apply_cctv_weights(
    G: nx.MultiDiGraph,
    cctv_path: str = CCTV_XLSX,
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
    edges_buf = edges_utm[["u", "v", "key", "geometry"]].copy()
    edges_buf["geometry"] = edges_buf.buffer(radius_m)

    try:
        joined = gpd.sjoin(cctv, edges_buf, predicate="within", how="left")
    except TypeError:
        joined = gpd.sjoin(cctv, edges_buf, op="within", how="left")

    counts = joined.dropna(subset=["u"]).groupby(["u", "v", "key"])["camera_count"].sum().rename("cctv_sum")
    edges_utm = edges_utm.merge(counts.reset_index(), on=["u", "v", "key"], how="left")
    edges_utm["cctv_sum"] = edges_utm["cctv_sum"].fillna(0)

    edges_utm["edge_km"] = edges_utm["length_m"].clip(lower=1e-6) / 1000.0
    edges_utm["density_per_km"] = edges_utm["cctv_sum"] / edges_utm["edge_km"]

    low = float(edges_utm["density_per_km"].quantile(0.05))
    high = float(edges_utm["density_per_km"].quantile(0.95))
    rng = max(1e-6, high - low)
    edges_utm["dens_norm"] = ((edges_utm["density_per_km"] - low) / rng).clip(0, 1)
    edges_utm["weight_cctv"] = edges_utm["length_m"] / (1.0 + alpha * edges_utm["dens_norm"])

    for _, r in edges_utm.iterrows():
        u, v, k = r["u"], r["v"], r["key"]
        if G.has_edge(u, v, k):
            d = G[u][v][k]
            d["length_m"] = float(r["length_m"])
            d["cctv_sum"] = float(r["cctv_sum"])
            d["density_per_km"] = float(r["density_per_km"])
            d["dens_norm"] = float(r["dens_norm"])
            d["weight_cctv"] = float(r["weight_cctv"])


# ------------------------ Model helpers ------------------------ #
def sanitize_graph_edge_attrs(G: nx.MultiDiGraph) -> None:
    for _, _, _, d in G.edges(keys=True, data=True):
        length_m = d.get("length_m", d.get("length", 1.0))
        d["len_m_num"] = as_float(length_m, default=1.0)
        d["dens_norm_num"] = as_float(d.get("dens_norm", 0.0), default=0.0)
        d["cctv_sum_num"] = as_float(d.get("cctv_sum", 0.0), default=0.0)
        if "weight_runtime" in d:
            d["weight_runtime"] = as_float(d["weight_runtime"], default=d["len_m_num"])
        else:
            d["weight_runtime"] = d["len_m_num"]


def edge_feats_ext(d: Dict[str, Any], hour: int) -> np.ndarray:
    L = d["len_m_num"]
    dn = d["dens_norm_num"]
    cctv = d["cctv_sum_num"]
    km = max(1e-6, L / 1000.0)
    cctv_per_km = cctv / km

    hw_val = d.get("highway", "")
    hw_list = [str(hw_val).lower()]
    if isinstance(hw_val, list) and hw_val:
        hw_list = [str(x).lower() for x in hw_val]

    def has(tag: str) -> bool:
        return any(tag in h for h in hw_list)

    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    return np.array(
        [
            1.0,
            math.log1p(L),
            dn,
            cctv_per_km,
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
            hour_sin,
            hour_cos,
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


# ------------------------ Core pipeline ------------------------ #
@dataclass
class PipelineResult:
    tmap_raw: Dict[str, Any]
    base_route: List[Tuple[float, float]]
    rerouted: List[Tuple[float, float]]
    base_weight: float
    rerouted_weight: float
    html_path: str | None = None


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
    apply_cctv_weights(G, cctv_path=cctv_path, radius_m=80.0, alpha=alpha)

    log("[4/5] Apply model weights")
    sanitize_graph_edge_attrs(G)
    for _, _, _, d in G.edges(keys=True, data=True):
        dn = d["dens_norm_num"]
        L = d["len_m_num"]
        d["weight_runtime"] = L / (1.0 + alpha * dn)
    if not model_path:
        raise FileNotFoundError("Model weights path is empty; provide --model.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    w = load_model(model_path)
    apply_model_scores_to_graph(G, w, hour=resolve_hour(hour), alpha=alpha)

    log("[5/5] Reroute with AI weights")
    o = nearest_node(G, start_lat, start_lon)
    dnode = nearest_node(G, end_lat, end_lon)
    base_nodes = nx.shortest_path(G, o, dnode, weight="len_m_num")
    reroute_nodes = nx.shortest_path(G, o, dnode, weight="weight_runtime")

    base_latlons = path_to_latlons(G, base_nodes, weight_attr="len_m_num")
    reroute_latlons = path_to_latlons(G, reroute_nodes, weight_attr="weight_runtime")

    html_path = None
    if html_out:
        html_path = visualize_routes(G, base_latlons, reroute_latlons, html_out)

    return PipelineResult(
        tmap_raw=raw,
        base_route=base_latlons,
        rerouted=reroute_latlons,
        base_weight=path_weight_sum(G, base_nodes, weight_attr="len_m_num"),
        rerouted_weight=path_weight_sum(G, reroute_nodes, weight_attr="weight_runtime"),
        html_path=html_path,
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
    for _, r in edges.iterrows():
        geom = r["geometry"]
        if isinstance(geom, LineString):
            xs, ys = list(geom.coords.xy[0]), list(geom.coords.xy[1])
            folium.PolyLine([(ys[i], xs[i]) for i in range(len(xs))], color="#b8b8b8", weight=3, opacity=0.55).add_to(m)
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                xs, ys = list(line.coords.xy[0]), list(line.coords.xy[1])
                folium.PolyLine([(ys[i], xs[i]) for i in range(len(xs))], color="#b8b8b8", weight=3, opacity=0.55).add_to(m)
    if base_route:
        folium.PolyLine(base_route, color="blue", weight=6, opacity=0.75, tooltip="Tmap base").add_to(m)
    if rerouted:
        folium.PolyLine(rerouted, color="magenta", weight=7, opacity=0.95, tooltip="AI reroute").add_to(m)
        folium.Marker(rerouted[0], icon=folium.Icon(color="green", icon="play"), tooltip="start").add_to(m)
        folium.Marker(rerouted[-1], icon=folium.Icon(color="red", icon="stop"), tooltip="end").add_to(m)
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
    model: str | None = MODEL_PATH,
    alpha: float = ALPHA,
    hour: Any = HOUR_DEFAULT,
    out_json: str = OUT_JSON,
    out_html: str = OUT_HTML,
    visualize: bool = False,
) -> PipelineResult:
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
    MAIN_START_LAT = START_LAT
    MAIN_START_LON = START_LON
    MAIN_END_LAT = END_LAT
    MAIN_END_LON = END_LON
    MAIN_MODEL = MODEL_PATH         
    MAIN_OUT_JSON = OUT_JSON
    MAIN_OUT_HTML = OUT_HTML
    MAIN_ALPHA = ALPHA
    MAIN_HOUR = HOUR_DEFAULT
    MAIN_MARGIN_M = MARGIN_M
    MAIN_VISUALIZE = False

    run_cli(
        start_lat=MAIN_START_LAT,
        start_lon=MAIN_START_LON,
        end_lat=MAIN_END_LAT,
        end_lon=MAIN_END_LON,
        app_key=TMAP_APP_KEY,
        margin_m=MAIN_MARGIN_M,
        cctv_xlsx=CCTV_XLSX,
        model=MAIN_MODEL,
        alpha=MAIN_ALPHA,
        hour=MAIN_HOUR,
        out_json=MAIN_OUT_JSON,
        out_html=MAIN_OUT_HTML,
        visualize=MAIN_VISUALIZE,
    )
