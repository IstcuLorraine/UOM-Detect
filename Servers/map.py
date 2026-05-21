#!/usr/bin/env python3 
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pyrosm import OSM
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from rapidfuzz import fuzz
from math import cos, pi
import yaml
from datetime import datetime

# ================== Configuration ==================
HOST = "0.0.0.0"
PORT = 8118

CITY_GEOJSON = "/MapShop/常州市_市.json"              #可选   
OSM_PBF_PATH = "/MapShop/jiangsu.osm.pbf"            #可选
MANUAL_MAP_FILE = "/MapShop/extra_locations.yml"     #自建地址群
FLIGHT_AREA_FILE = "/MapShop/fly_zones_all.geojson"  #UOM适飞区文件

TOP_K = 5
SCORE_THRESHOLD = 50.0
CITY_BBOX_MARGIN = 0.05

WEIGHT_STATION = 30.0
WEIGHT_POLYGON = 10.0
WEIGHT_POI = 0.0

# whitelist for incoming requests
_WHITELIST = {"127.0.0.1", "::1",""}

# ================== Logging ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("map")

# ================== Manual mappings container ==================
_MANUAL_MAPPINGS: Dict[str, Tuple[float, float]] = {}

def _normalize_name(name: str) -> str:
    if not name:
        return ""
    return " ".join(name.strip().lower().split())

def load_manual_mappings_from_yaml(path: str) -> Dict[str, Tuple[float, float]]:
    if not os.path.exists(path):
        raise RuntimeError(f"Manual mapping YAML not found at: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read YAML {path}: {e}") from e

    if not isinstance(doc, dict):
        raise RuntimeError(f"Manual mapping YAML {path} must contain a mapping (dict) at top level")

    out: Dict[str, Tuple[float, float]] = {}
    for raw_key, raw_val in doc.items():
        if raw_key is None:
            continue
        key = _normalize_name(str(raw_key))
        if raw_val is None:
            raise RuntimeError(f"Manual mapping for key '{raw_key}' has empty value")

        parsed = None
        if isinstance(raw_val, str):
            if "," in raw_val:
                try:
                    lon_s, lat_s = raw_val.split(",", 1)
                    parsed = (float(lon_s.strip()), float(lat_s.strip()))
                except Exception:
                    parsed = None
        if parsed is None and isinstance(raw_val, (list, tuple)) and len(raw_val) >= 2:
            try:
                parsed = (float(raw_val[0]), float(raw_val[1]))
            except Exception:
                parsed = None
        if parsed is None and isinstance(raw_val, dict):
            try:
                lon = raw_val.get("lon") or raw_val.get("longitude") or raw_val.get("lng")
                lat = raw_val.get("lat") or raw_val.get("latitude")
                parsed = (float(lon), float(lat))
            except Exception:
                parsed = None
        if parsed is None:
            try:
                s = str(raw_val)
                if "," in s:
                    lon_s, lat_s = s.split(",", 1)
                    parsed = (float(lon_s.strip()), float(lat_s.strip()))
            except Exception:
                parsed = None
        if parsed is None:
            raise RuntimeError(f"Cannot parse manual mapping value for key '{raw_key}': {raw_val!r}")
        out[key] = parsed
    return out

def _reload_manual_mappings():
    global _MANUAL_MAPPINGS
    loaded = load_manual_mappings_from_yaml(MANUAL_MAP_FILE)
    _MANUAL_MAPPINGS = loaded
    logger.info("Loaded %d manual mappings from %s", len(_MANUAL_MAPPINGS), MANUAL_MAP_FILE)

# ================== Flight area ==================
_flight_area_geom = None  # shapely geometry (Polygon or MultiPolygon)

def _load_flight_area(path: str):
    global _flight_area_geom
    if not os.path.exists(path):
        raise RuntimeError(f"Flight area file not found: {path}")

    features = []
    try:
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    obj = json.loads(ln)
                    if obj.get("type") == "FeatureCollection":
                        features.extend(obj.get("features", []))
                    elif obj.get("type") == "Feature":
                        features.append(obj)
                    else:
                        if "geometry" in obj:
                            features.append(obj)
        else:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            if isinstance(doc, dict) and doc.get("type") == "FeatureCollection":
                features = doc.get("features", []) or []
            elif isinstance(doc, dict) and doc.get("type") == "Feature":
                features = [doc]
            elif isinstance(doc, dict) and "features" in doc:
                features = doc.get("features", []) or []
            else:
                raise RuntimeError("Unsupported flight area GeoJSON structure")
    except Exception as e:
        raise RuntimeError(f"Failed to parse flight area file {path}: {e}") from e

    geoms = []
    for feat in features:
        geom = feat.get("geometry") if isinstance(feat, dict) else None
        if not geom:
            continue
        try:
            g = shape(geom)
            if g.geom_type in ("Polygon", "MultiPolygon"):
                geoms.append(g)
        except Exception:
            continue

    if not geoms:
        raise RuntimeError("No polygon geometries found in flight area file")

    try:
        _flight_area_geom = unary_union(geoms)
    except Exception:
        # fallback: try to create MultiPolygon
        from shapely.geometry import MultiPolygon
        polys = []
        for g in geoms:
            if g.geom_type == "Polygon":
                polys.append(g)
            elif g.geom_type == "MultiPolygon":
                polys.extend(list(g))
        _flight_area_geom = MultiPolygon(polys)
    logger.info("Loaded flight area from %s -> %s", path, getattr(_flight_area_geom, "geom_type", None))
    return _flight_area_geom

def flight_area_contains(lon: float, lat: float, buffer_m: float = 0.0) -> Dict[str, Any]:
    global _flight_area_geom
    if _flight_area_geom is None:
        return {"ok": False, "contains": False, "distance_m": None, "note": "flight_area_not_loaded"}

    pt = Point(float(lon), float(lat))
    geom = _flight_area_geom
    try:
        if buffer_m and buffer_m > 0:
            dlat = buffer_m / 111320.0
            dlon = buffer_m / (111320.0 * max(0.000001, cos(lat * pi / 180.0)))
            geom_buf = geom.buffer(max(dlat, dlon))
        else:
            geom_buf = geom

        contains = bool(geom_buf.contains(pt) or geom_buf.touches(pt))
        if contains:
            return {"ok": True, "contains": True, "distance_m": 0.0, "note": None}
        else:
            deg_dist = geom_buf.distance(pt)
            approx_m = deg_dist * 111320.0
            return {"ok": True, "contains": False, "distance_m": approx_m, "note": None}
    except Exception as e:
        logger.exception("flight_area_contains error: %s", e)
        return {"ok": False, "contains": False, "distance_m": None, "note": str(e)}

# ================== Module caches (PBF) ==================
_city_geom = None
_city_bbox: Optional[Tuple[float, float, float, float]] = None
_osm = None
_cache_stations: List[Dict[str, Any]] = []
_cache_polygons: List[Dict[str, Any]] = []
_cache_pois: List[Dict[str, Any]] = []
_cache_loaded = False

# ================== FastAPI app & lifespan ==================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: load manual mappings, flight area, init pbf cache
    _reload_manual_mappings()   # will raise if missing/invalid
    logger.info("Manual mappings loaded.")
    _load_flight_area(FLIGHT_AREA_FILE)  # will raise if missing/invalid
    logger.info("Flight area loaded.")
    _init_from_pbf()
    logger.info("PBF cache initialized.")
    yield
    logger.info("Map service shutting down.")

app = FastAPI(title="Map Geocoding Service", version="1.0", lifespan=lifespan)

# minimal CORS middleware to add Access-Control-Allow-Origin (fixes browser CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # change to specific origins if you want stricter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== Request / Response models ==================
class LookupRequest(BaseModel):
    place: str
    top_k: Optional[int] = 1

class QueryRequest(BaseModel):
    # accept "1"/"2" or numeric
    class_id: str = Field(..., alias="class")  # client sends {"class": "2"} or {"class": 2}
    place: Optional[str] = None
    lon: Optional[float] = None
    lat: Optional[float] = None
    buffer_m: Optional[float] = 0.0
    sn: Optional[str] = None  # serial number for drone (required for class=2)

# ================== Utilities ==================
def fuzzy_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(fuzz.token_set_ratio(a, b))

def load_city_boundary() -> Optional[object]:
    global _city_geom, _city_bbox
    if _city_geom is not None:
        return _city_geom
    if not os.path.exists(CITY_GEOJSON):
        logger.warning("CITY_GEOJSON not found: %s (city boundary disabled)", CITY_GEOJSON)
        _city_geom = None
        _city_bbox = None
        return None
    with open(CITY_GEOJSON, "r", encoding="utf-8") as f:
        gj = json.load(f)
    geoms = []
    if gj.get("type") == "FeatureCollection":
        for feat in gj.get("features", []):
            geom = feat.get("geometry")
            if geom:
                try:
                    geoms.append(shape(geom))
                except Exception:
                    pass
    else:
        try:
            if "geometry" in gj:
                geoms.append(shape(gj["geometry"]))
            else:
                geoms.append(shape(gj))
        except Exception:
            pass
    if not geoms:
        _city_geom = None
        _city_bbox = None
        logger.warning("No valid geometry found in CITY_GEOJSON")
        return None
    _city_geom = unary_union(geoms)
    minx, miny, maxx, maxy = _city_geom.bounds
    _city_bbox = (minx - CITY_BBOX_MARGIN, miny - CITY_BBOX_MARGIN, maxx + CITY_BBOX_MARGIN, maxy + CITY_BBOX_MARGIN)
    logger.info("Loaded city boundary from %s ; bbox=%s", CITY_GEOJSON, _city_bbox)
    return _city_geom

def point_in_city(lon: float, lat: float) -> bool:
    city = load_city_boundary()
    if city is None:
        return True
    try:
        return city.contains(Point(lon, lat))
    except Exception:
        return False

def _centroid_xy(geom):
    try:
        c = geom.centroid
        return float(c.x), float(c.y)
    except Exception:
        try:
            return float(geom.x), float(geom.y)
        except Exception:
            return None, None

# ================== PBF init (cache geometries) ==================
def _load_local_landmarks_into_cache(files_or_dir: str, bbox: Optional[Tuple[float,float,float,float]] = None):
    global _cache_pois
    files = []
    if os.path.isdir(files_or_dir):
        for fn in os.listdir(files_or_dir):
            if fn.lower().endswith(".geojson") or fn.lower().endswith(".json"):
                files.append(os.path.join(files_or_dir, fn))
    elif os.path.isfile(files_or_dir):
        files = [files_or_dir]
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                gj = json.load(f)
            features = gj.get("features") or []
            for feat in features:
                geom = feat.get("geometry")
                props = feat.get("properties", {})
                if not geom:
                    continue
                if geom.get("type") != "Point":
                    try:
                        pt = shape(geom).centroid
                        lon, lat = float(pt.x), float(pt.y)
                    except Exception:
                        continue
                else:
                    coords = geom.get("coordinates", [])
                    if len(coords) < 2:
                        continue
                    lon, lat = float(coords[0]), float(coords[1])
                if bbox:
                    minx, miny, maxx, maxy = bbox
                    if not (minx <= lon <= maxx and miny <= lat <= maxy):
                        continue
                name = props.get("name") or props.get("NAME") or props.get("Name") or props.get("title")
                if not name:
                    continue
                _cache_pois.append({"name": str(name), "lon": lon, "lat": lat, "tags": props, "source": "local_geojson", "geometry": None})
        except Exception as e:
            logger.warning("Failed to load local geojson %s: %s", fp, e)

def _init_from_pbf():
    global _osm, _cache_pois, _cache_polygons, _cache_stations, _cache_loaded, _city_bbox

    if _cache_loaded:
        return

    load_city_boundary()  # ensure _city_bbox populated

    if not OSM_PBF_PATH or not os.path.exists(OSM_PBF_PATH):
        logger.warning("OSM_PBF_PATH not set or not found: %s. Skipping PBF load.", OSM_PBF_PATH)
        _cache_loaded = True
        return

    logger.info("Initializing OSM from PBF: %s", OSM_PBF_PATH)
    _osm = OSM(OSM_PBF_PATH)
    bbox = _city_bbox

    # POIs
    try:
        pois = _osm.get_pois()
    except Exception:
        pois = None

    if pois is not None:
        logger.info("Processing POIs from PBF (total rows: %d)", len(pois))
        for idx, row in pois.iterrows():
            name = None
            try:
                name = row.get("name") or row.get("NAME") or None
            except Exception:
                name = None
            if not name:
                continue
            geom = None
            try:
                geom = row.geometry
            except Exception:
                geom = None
            if geom is None:
                lon = row.get("lon") or row.get("longitude") or None
                lat = row.get("lat") or row.get("latitude") or None
                if lon is None or lat is None:
                    continue
            else:
                lon, lat = _centroid_xy(geom)
                if lon is None or lat is None:
                    continue
            if bbox:
                minx, miny, maxx, maxy = bbox
                if not (minx <= lon <= maxx and miny <= lat <= maxy):
                    continue
            tags = {}
            try:
                keys = ["amenity", "shop", "tourism", "railway", "public_transport", "building"]
                for k in keys:
                    v = row.get(k)
                    if v is not None:
                        tags[k] = v
            except Exception:
                pass
            entry = {"name": str(name), "lon": float(lon), "lat": float(lat), "tags": tags, "source": "poi", "geometry": None}
            # preserve geometry if available
            try:
                if geom is not None:
                    entry["geometry"] = geom
            except Exception:
                entry["geometry"] = None
            if ("railway" in tags and tags.get("railway") == "station") or ("public_transport" in tags and tags.get("public_transport") == "station"):
                entry["source"] = "station"
                _cache_stations.append(entry)
            else:
                _cache_pois.append(entry)

    # Buildings / polygons
    try:
        buildings = _osm.get_buildings()
    except Exception:
        buildings = None

    if buildings is not None:
        logger.info("Processing buildings/polygons from PBF (total rows: %d)", len(buildings))
        for idx, row in buildings.iterrows():
            name = row.get("name") or row.get("NAME") or None
            if not name:
                continue
            geom = None
            try:
                geom = row.geometry
            except Exception:
                geom = None
            if geom is None:
                continue
            lon, lat = _centroid_xy(geom)
            if lon is None or lat is None:
                continue
            if bbox:
                minx, miny, maxx, maxy = bbox
                if not (minx <= lon <= maxx and miny <= lat <= maxy):
                    continue
            tags = {}
            try:
                keys = ["amenity", "shop", "tourism", "building"]
                for k in keys:
                    v = row.get(k)
                    if v is not None:
                        tags[k] = v
            except Exception:
                pass
            entry = {"name": str(name), "lon": float(lon), "lat": float(lat), "tags": tags, "source": "polygon", "geometry": None}
            try:
                entry["geometry"] = geom
            except Exception:
                entry["geometry"] = None
            _cache_polygons.append(entry)

    # local geojsons optionally
    if os.path.exists(os.path.dirname(MANUAL_MAP_FILE)):
        try:
            _load_local_landmarks_into_cache(os.path.dirname(MANUAL_MAP_FILE), bbox)
        except Exception:
            pass

    _cache_loaded = True
    logger.info("Cache built: stations=%d, polygons=%d, pois=%d", len(_cache_stations), len(_cache_polygons), len(_cache_pois))

# ================== Search and geocode ==================
def adjusted_score(base_score: float, source: str) -> float:
    if source == "station":
        return base_score + WEIGHT_STATION
    if source == "polygon":
        return base_score + WEIGHT_POLYGON
    return base_score + WEIGHT_POI

def _search_candidates(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    q = query.strip()
    if not q:
        return []
    candidates: List[Dict[str, Any]] = []

    for entry in _cache_stations:
        name = entry.get("name")
        if not name:
            continue
        if name == q:
            score = 100.0
        else:
            score = fuzzy_score(q.lower(), _normalize_name(name))
        adj = adjusted_score(score, entry.get("source", "poi"))
        if adj >= SCORE_THRESHOLD:
            candidates.append({**entry, "score": score, "adjusted_score": adj})

    for entry in _cache_polygons:
        name = entry.get("name")
        if not name:
            continue
        if name == q:
            score = 98.0
        else:
            score = fuzzy_score(q.lower(), _normalize_name(name))
        adj = adjusted_score(score, entry.get("source", "polygon"))
        if adj >= SCORE_THRESHOLD:
            candidates.append({**entry, "score": score, "adjusted_score": adj})

    for entry in _cache_pois:
        name = entry.get("name")
        if not name:
            continue
        if name == q:
            score = 95.0
        else:
            score = fuzzy_score(q.lower(), _normalize_name(name))
        adj = adjusted_score(score, entry.get("source", "poi"))
        if adj >= SCORE_THRESHOLD:
            candidates.append({**entry, "score": score, "adjusted_score": adj})

    if not candidates:
        pool: List[Dict[str, Any]] = []
        for entry in _cache_stations + _cache_polygons + _cache_pois:
            name = entry.get("name")
            if not name:
                continue
            score = fuzzy_score(q.lower(), _normalize_name(name))
            adj = adjusted_score(score, entry.get("source", "poi"))
            pool.append({**entry, "score": score, "adjusted_score": adj})
        pool.sort(key=lambda x: x["adjusted_score"], reverse=True)
        candidates = pool[:top_k]

    candidates.sort(key=lambda x: x["adjusted_score"], reverse=True)
    return candidates[:top_k]

def geocode_best(q: str) -> Optional[Dict[str, Any]]:
    if not q or not q.strip():
        return None
    q = q.strip()
    if not _cache_loaded:
        try:
            _init_from_pbf()
        except Exception:
            pass
    candidates = _search_candidates(q, top_k=TOP_K)
    if not candidates:
        return None
    best = candidates[0]
    result = {
        "query": q,
        # keep previous scaling factors if you relied on them
        "name": best.get("name"),
        "lon": float(best.get("lon") * 1.0000385),
        "lat": float(best.get("lat") * 0.99993237),
        "source": best.get("source"),
        "score": float(best.get("score", 0.0)),
        "adjusted_score": float(best.get("adjusted_score", 0.0)),
    }
    if _city_bbox:
        result["city_bbox"] = _city_bbox
        result["inside_city"] = (_city_bbox[0] <= result["lon"] <= _city_bbox[2] and _city_bbox[1] <= result["lat"] <= _city_bbox[3])
    else:
        result["city_bbox"] = None
        result["inside_city"] = point_in_city(result["lon"], result["lat"])
    return result

# ================== Place geometry finder (used by contains etc) ==================
def find_place_geometry(place: str):
    if not place:
        return None, None, None
    norm = _normalize_name(place)

    # manual mapping (point)
    try:
        if norm in _MANUAL_MAPPINGS:
            lon, lat = _MANUAL_MAPPINGS[norm]
            return Point(float(lon), float(lat)), "manual", place
    except Exception:
        pass

    # polygons (exact first)
    best = None
    best_score = -1.0
    for entry in _cache_polygons:
        name = entry.get("name")
        if not name:
            continue
        if _normalize_name(name) == norm:
            geom = entry.get("geometry")
            if geom is not None:
                return geom, "polygon", entry.get("name")
            else:
                return Point(entry.get("lon"), entry.get("lat")), "polygon_centroid", entry.get("name")
        score = fuzzy_score(norm, _normalize_name(name))
        if score > best_score:
            best_score = score
            best = entry
    if best and best_score >= 70.0:
        geom = best.get("geometry")
        if geom is not None:
            return geom, "polygon", best.get("name")
        else:
            return Point(best.get("lon"), best.get("lat")), "polygon_centroid", best.get("name")

    # POIs / stations (points)
    best = None
    best_score = -1.0
    for entry in _cache_pois + _cache_stations:
        name = entry.get("name")
        if not name:
            continue
        if _normalize_name(name) == norm:
            geom = entry.get("geometry")
            if geom is not None:
                return geom, "poi", entry.get("name")
            else:
                return Point(entry.get("lon"), entry.get("lat")), "poi_point", entry.get("name")
        score = fuzzy_score(norm, _normalize_name(name))
        if score > best_score:
            best_score = score
            best = entry
    if best and best_score >= 70.0:
        geom = best.get("geometry")
        if geom is not None:
            return geom, "poi", best.get("name")
        else:
            return Point(best.get("lon"), best.get("lat")), "poi_point", best.get("name")

    return None, None, None

def point_in_place(place: str, lon: float, lat: float, buffer_m: float = 0.0) -> Dict[str, Any]:
    geom, source, matched_name = find_place_geometry(place)
    if geom is None:
        return {"ok": False, "contains": False, "place_name": None, "source": None, "distance_m": None, "note": "no_match"}

    pt = Point(float(lon), float(lat))

    try:
        if buffer_m and buffer_m > 0:
            dlat = buffer_m / 111320.0
            dlon = buffer_m / (111320.0 * max(0.000001, cos(lat * pi / 180.0)))
            geom_buf = geom.buffer(max(dlat, dlon))
        else:
            geom_buf = geom

        if geom_buf.geom_type in ("Polygon", "MultiPolygon"):
            contains = bool(geom_buf.contains(pt) or geom_buf.touches(pt))
            if contains:
                return {"ok": True, "contains": True, "place_name": matched_name, "source": source, "distance_m": 0.0, "note": None}
            else:
                deg_dist = geom_buf.distance(pt)
                approx_m = deg_dist * 111320.0
                return {"ok": True, "contains": False, "place_name": matched_name, "source": source, "distance_m": approx_m, "note": None}
        else:
            # point-like geometry
            place_pt = geom if geom.geom_type == "Point" else Point(float(geom.x), float(geom.y))
            deg_dx = abs(place_pt.x - pt.x)
            deg_dy = abs(place_pt.y - pt.y)
            m_x = deg_dx * (111320.0 * cos(pt.y * pi / 180.0))
            m_y = deg_dy * 111320.0
            dist_m = (m_x ** 2 + m_y ** 2) ** 0.5
            contains = dist_m <= (buffer_m if buffer_m and buffer_m > 0 else 1.0)
            return {"ok": True, "contains": contains, "place_name": matched_name, "source": source, "distance_m": dist_m, "note": None}
    except Exception as e:
        logger.exception("point_in_place error: %s", e)
        return {"ok": False, "contains": False, "place_name": matched_name, "source": source, "distance_m": None, "note": str(e)}

# ================== Whitelist check and Print logs==================
def _check_whitelist(request: Request):
    client = request.client
    if not client:
        raise HTTPException(status_code=403, detail="forbidden")
    host = client.host
    if host not in _WHITELIST:
        raise HTTPException(status_code=403, detail="forbidden")

def _log_request(request: Request, status_code: int, result_value: Any):
    try:
        ip = request.client.host if request.client else "unknown"
        method = request.method
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"Time:{now}, IP:{ip}, Method:{method}->{status_code}OK, Result:{result_value}"
        )
    except Exception:
        pass
# ================== Endpoints ==================
@app.get("/health")
def health(request: Request):
    _check_whitelist(request)
    return {
        "ok": True,
        "stations": len(_cache_stations),
        "polygons": len(_cache_polygons),
        "pois": len(_cache_pois),
        "manual_mappings": len(_MANUAL_MAPPINGS),
        "flight_area_loaded": _flight_area_geom is not None
    }

@app.post("/lookup")
def lookup(request: Request, payload: LookupRequest):
    _check_whitelist(request)
    place = payload.place
    top_k = max(1, int(payload.top_k or 1))

    norm = _normalize_name(place)
    if norm in _MANUAL_MAPPINGS:
        lon, lat = _MANUAL_MAPPINGS[norm]
        manual_result = {
            "name": place,
            "lon": float(lon),
            "lat": float(lat),
            "source": "manual",
            "score": 100.0,
            "adjusted_score": 100.0
        }
        if top_k == 1:
            resp = {"ok": True, "query": place, "result": manual_result}
            _log_request(request, 200, True)
            return resp
        else:
            extra_candidates = _search_candidates(place, top_k=TOP_K)
            results = [manual_result]
            def is_dup(candidate, existing_list):
                try:
                    cl = candidate.get("name")
                    for e in existing_list:
                        if e.get("name") == cl:
                            return True
                        if abs(float(candidate.get("lon")) - float(e.get("lon"))) < 1e-7 and abs(float(candidate.get("lat")) - float(e.get("lat"))) < 1e-7:
                            return True
                except Exception:
                    pass
                return False
            for c in extra_candidates:
                cand = {
                    "name": c.get("name"),
                    "lon": float(c.get("lon")),
                    "lat": float(c.get("lat")),
                    "source": c.get("source"),
                    "score": float(c.get("score", 0.0)),
                    "adjusted_score": float(c.get("adjusted_score", 0.0))
                }
                if not is_dup(cand, results):
                    results.append(cand)
                if len(results) >= top_k:
                    break
            resp = {"ok": True, "query": place, "results": results}
            _log_request(request, 200, True)
            return resp

    # fallback to pbf search
    try:
        if top_k == 1:
            res = geocode_best(place)
            if not res:
                resp = {"ok": False, "query": place, "error": "no_match"}
                _log_request(request, 200, resp)
                return resp
            resp = {"ok": True, "query": place, "result": res}
            _log_request(request, 200, resp)
            return resp
        else:
            candidates = _search_candidates(place, top_k=top_k)
            results = []
            for best in candidates:
                inside = True
                if _city_geom:
                    try:
                        inside = _city_geom.contains(Point(best["lon"], best["lat"]))
                    except Exception:
                        inside = False
                results.append({
                    "name": best.get("name"),
                    "lon": float(best.get("lon")),
                    "lat": float(best.get("lat")),
                    "source": best.get("source"),
                    "score": float(best.get("score", 0.0)),
                    "adjusted_score": float(best.get("adjusted_score", 0.0)),
                    "inside_city": inside
                })
            return {"ok": True, "query": place, "results": results}
    except Exception as e:
        logger.exception("geocode error: %s", e)
        raise HTTPException(status_code=500, detail="internal_error")

@app.post("/reload_manual_mappings")
def reload_manual_mappings(request: Request):
    _check_whitelist(request)
    _reload_manual_mappings()
    return {"ok": True, "entries": len(_MANUAL_MAPPINGS)}

@app.post("/query")
def query_handler(request: Request, payload: QueryRequest):
    _check_whitelist(request)
    # accept class as string or number
    try:
        cls = int(str(payload.class_id))
    except Exception:
        raise HTTPException(status_code=400, detail="invalid class (must be 1 or 2)")

    if cls == 1:
        if not payload.place:
            raise HTTPException(status_code=400, detail="class=1 requires 'place' field")
        norm = _normalize_name(payload.place)
        if norm in _MANUAL_MAPPINGS:
            lon, lat = _MANUAL_MAPPINGS[norm]
            manual_result = {"name": payload.place, "lon": float(lon), "lat": float(lat), "source": "manual", "score": 100.0}
            resp =  {"ok": True, "class": 1, "query": payload.place, "result": manual_result}
            _log_request(request, 200, resp)
            return resp
        res = geocode_best(payload.place)
        if not res:
            resp = {"ok": False, "class": 1, "query": payload.place, "error": "no_match"}
            _log_request(request, 200, resp)
            return resp
        return {"ok": True, "class": 1, "query": payload.place, "result": res}

    elif cls == 2:
        # require sn, lon, lat
        if not payload.sn:
            raise HTTPException(status_code=400, detail="class=2 requires 'sn' field")
        if payload.lon is None or payload.lat is None:
            raise HTTPException(status_code=400, detail="class=2 requires 'lon' and 'lat'")

        # parse lon/lat robustly
        try:
            lon = float(payload.lon)
            lat = float(payload.lat)
        except Exception:
            # if parsing fails, still return a deterministic response for the given sn
            resp = {"ok": False, "class": 1, "query": payload.place, "error": "no_match"}
            _log_request(request, 200, resp)
            return resp


        # do the check
        try:
            check = flight_area_contains(lon, lat, buffer_m=float(payload.buffer_m or 0.0))
        except Exception:
            # internal error -> map service fail: still return sn + false so frontend can correlate
            resp = {"ok": False, "class": 1, "query": payload.place, "error": "no_match"}
            _log_request(request, 200, resp)
            return resp


        # if flight_area_contains indicates failure, return false
        if not check.get("ok", False):
            resp = {"ok": False, "class": 1, "query": payload.place, "error": "no_match"}
            _log_request(request, 200, resp)
            return resp


        contains_flag = bool(check.get("contains", False))
        final_result = "true" if contains_flag else "false"
        resp = {"sn": str(payload.sn), "result": final_result}
        _log_request(request, 200, resp)
        return resp


    else:
        raise HTTPException(status_code=400, detail="unsupported class (must be 1 or 2)")

# ================== Main ==================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("map:app", host=HOST, port=PORT, workers=1, log_level="info",access_log=False)
