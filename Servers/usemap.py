#!/usr/bin/env python3
import requests

MAP_URL = "http://127.0.0.1:8118/query"
TIMEOUT = 5


def lookup_place(place: str):
    """
    class = 1
    地名 → 经纬度
    """
    payload = {
        "class": "1",
        "place": place
    }
    resp = requests.post(MAP_URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("ok"):
        return None

    r = data["result"]
    return {
        "name": r.get("name"),
        "lon": r.get("lon"),
        "lat": r.get("lat"),
        "source": r.get("source"),
        "score": r.get("score"),
    }


def check_flight_area(lon: float, lat: float, sn: str):
    """
    class = 2
    经纬度 → 是否在适飞区
    """
    payload = {
        "class": "2",
        "sn": sn,
        "lon": lon,
        "lat": lat
    }
    resp = requests.post(MAP_URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    if "result" not in data:
        return None

    return {
        "sn": data.get("sn"),
        "contains": True if data.get("result") == "true" else False
    }


if __name__ == "__main__":
    # ===== 示例 1：地名查坐标 =====
    # res = lookup_place("xxx")
    # if res:
    #     print(f"{res['name']} 的经纬度为：{res['lon']:.6f}, {res['lat']:.6f}")
    # else:
    #     print("未找到地名")

    # ===== 示例 2：坐标是否在适飞区 =====
    lon = 119.000000
    lat = 31.000000
    sn = "x"

    check = check_flight_area(lon, lat, sn)
    if check:
        print(f"无人机{sn},地址为{lon},{lat}是否在适飞区：{check['contains']}")
        print(f"raw_json:{check}")
    else:
        print("适飞区判断失败")
