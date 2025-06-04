# -*- coding: utf-8 -*-
"""Script for computing distances between Taiwanese train stations and
visualizing them in 2D using Multidimensional Scaling (MDS).

The script will:
1. Use the OpenStreetMap Nominatim API (via geopy) to fetch the
   latitude and longitude for each station.
2. Compute the great circle distance between every pair of stations.
3. Run classical MDS (from scikit-learn) on the distance matrix.
4. Plot the resulting 2D embedding and save it as ``mds_plot.png``.
5. Generate ``stations_map.html`` using gmplot to show the station
   markers on a Google Map.

This file assumes that the ``geopy``, ``scikit-learn``, ``matplotlib``
and ``gmplot`` packages are installed. If they are not, install them via
``pip install geopy scikit-learn matplotlib gmplot``.
"""

from __future__ import annotations

import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import gmplot


def fetch_coordinates(names: list[str]) -> list[tuple[float, float]]:
    """Return (latitude, longitude) for each location name."""
    geolocator = Nominatim(user_agent="mds-demo")
    coords = []
    for n in names:
        loc = geolocator.geocode(n)
        if not loc:
            raise RuntimeError(f"Failed to geocode {n}")
        coords.append((loc.latitude, loc.longitude))
    return coords


def compute_distance_matrix(coords: list[tuple[float, float]]) -> np.ndarray:
    """Compute pairwise geodesic distances (km) for coordinates."""
    n = len(coords)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = geodesic(coords[i], coords[j]).km
            mat[i, j] = mat[j, i] = d
    return mat


def run_mds(dist_matrix: np.ndarray) -> np.ndarray:
    """Return 2D embedding from MDS given a precomputed distance matrix."""
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
    return mds.fit_transform(dist_matrix)


def plot_embedding(points: np.ndarray, labels: list[str], outfile: str) -> None:
    """Plot the 2D embedding and save to *outfile*."""
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1])
    for (x, y), label in zip(points, labels):
        plt.annotate(label, (x, y))
    plt.title("MDS of Taiwanese Train Stations")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def generate_google_map(coords: list[tuple[float, float]], names: list[str], outfile: str) -> None:
    """Create an HTML map with markers using gmplot."""
    latitudes, longitudes = zip(*coords)
    center_lat = sum(latitudes) / len(latitudes)
    center_lon = sum(longitudes) / len(longitudes)
    gmap = gmplot.GoogleMapPlotter(center_lat, center_lon, zoom=7)
    for lat, lon, name in zip(latitudes, longitudes, names):
        gmap.marker(lat, lon, title=name)
    gmap.draw(outfile)


def main() -> None:
    station_names = [
        "Taipei Main Station",
        "Hsinchu Station",
        "Taichung Station",
        "Douliu Station",
        "Kaohsiung Station",
        "Yuli, Hualien",
        "Zhiben, Taitung",
    ]

    coords = fetch_coordinates(station_names)
    dist_matrix = compute_distance_matrix(coords)
    embedding = run_mds(dist_matrix)

    plot_embedding(embedding, station_names, "mds_plot.png")
    generate_google_map(coords, station_names, "stations_map.html")
    print("Results saved: mds_plot.png and stations_map.html")


if __name__ == "__main__":
    main()
