# -*- coding: utf-8 -*-

"""
main.py

功能：
- 抓取一組台灣火車站的經緯度
- 計算兩兩站點之間的地理距離矩陣
- 使用 classical MDS 將距離矩陣降到 2D
- 使用 Plotly Express 畫互動式散佈圖，並用 Dash 包裝成網頁
- 在側邊顯示框選後的站名清單與兩兩距離
"""

from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import plotly.express as px
from dash import Dash, dcc, html, Input, Output


def fetch_coordinates(names: list[str]) -> list[tuple[float, float]]:
    geolocator = Nominatim(user_agent="mds-demo")
    coords = []
    for n in names:
        loc = geolocator.geocode(n)
        if loc is None:
            raise RuntimeError(f"無法取得站名「{n}」的經緯度")
        coords.append((loc.latitude, loc.longitude))
    return coords


def compute_distance_matrix(coords: list[tuple[float, float]]) -> np.ndarray:
    n = len(coords)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = geodesic(coords[i], coords[j]).km
            mat[i, j] = mat[j, i] = d
    return mat


def run_mds(dist_matrix: np.ndarray) -> np.ndarray:
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
    embedding = mds.fit_transform(dist_matrix)
    return embedding


def prepare_dataframe(
    points: np.ndarray, station_names: list[str], coords: list[tuple[float, float]]
) -> pd.DataFrame:
    df = pd.DataFrame(points, columns=["x", "y"])
    df["station"] = station_names
    df["lat"] = [c[0] for c in coords]
    df["lon"] = [c[1] for c in coords]
    return df


def create_figure(df: pd.DataFrame) -> px.scatter:
    fig = px.scatter(
        df,
        x="x",
        y="y",
        text="station",
        custom_data=["station", "lat", "lon"],
        title="MDS of Taiwanese Train Stations",
    )
    fig.update_traces(marker_size=10, selector=dict(mode="markers"))
    fig.update_layout(dragmode="lasso")
    return fig


def compute_pairwise_distances(selected_points: list[dict]) -> str:
    stations = []
    for pt in selected_points:
        try:
            name = pt["customdata"][0]
            lat = float(pt["customdata"][1])
            lon = float(pt["customdata"][2])
            stations.append((name, (lat, lon)))
        except (KeyError, ValueError, TypeError):
            continue

    if len(stations) < 2:
        return "無法計算距離：請至少選取兩個有效車站"

    n = len(stations)
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            name1, coord1 = stations[i]
            name2, coord2 = stations[j]
            try:
                dist = geodesic(coord1, coord2).km
                results.append(f"{name1} ↔ {name2} = {dist:.2f} km")
            except Exception:
                results.append(f"{name1} ↔ {name2} = 無法計算")
    return "\n".join(results)


def serve_app(df: pd.DataFrame, fig) -> None:
    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H2("互動式 MDS 車站圈選"),
            dcc.Graph(id="mds-plot", figure=fig),
            html.H4("已圈選站名與距離："),
            html.Pre(id="sel-out", style={"fontSize": "16px", "color": "#333"}),
        ],
        style={"width": "60%", "margin": "auto", "textAlign": "center"},
    )

    @app.callback(Output("sel-out", "children"), Input("mds-plot", "selectedData"))
    def show_selected(selectedData):
        if not selectedData or "points" not in selectedData:
            return "請用滑鼠框選或套索選取資料點"
        lines = []
        points = selectedData["points"]
        station_names = [pt["customdata"][0] for pt in points if "customdata" in pt]
        lines.append("站名清單：" + ", ".join(station_names))
        lines.append("\n兩兩距離：")
        lines.append(compute_pairwise_distances(points))
        return "\n".join(lines)

    app.run(debug=True)


def main():
    station_names = [
        "Taipei Main Station",
        "Hsinchu Station",
        "Taichung Station",
        "Douliu Station",
        "Kaohsiung Station",
        "Yuli, Hualien",
        "Zhiben, Taitung",
    ]

    print("Fetching coordinates")
    coords = fetch_coordinates(station_names)
    for name, (lat, lon) in zip(station_names, coords):
        print(f"{name}: ({lat:.5f}, {lon:.5f})")

    print("Computing distance matrix")
    dist_matrix = compute_distance_matrix(coords)

    print("Running MDS embedding")
    points = run_mds(dist_matrix)

    df = prepare_dataframe(points, station_names, coords)

    fig = create_figure(df)

    serve_app(df, fig)


if __name__ == "__main__":
    main()
