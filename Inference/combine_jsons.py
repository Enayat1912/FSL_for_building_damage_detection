






"""
merge_inference_with_geometry

Combines classification results (from inference JSON) with spatial geometry (from xBD format JSON)
to produce a GeoJSON file for geospatial analysis or visualization.

"""

import json
import argparse
from shapely import wkt
from shapely.geometry import mapping


def add_geometry_to_inference(inference_file, geometry_file, output_geojson):
    """
    Combines classification predictions with building geometries and saves as a GeoJSON.

    Args:
        inference_file (str): Path to the JSON file containing UID-to-damage predictions.
        geometry_file (str): Path to the JSON file with WKT geometries (from xBD).
        output_geojson (str): Path to save the final GeoJSON file.
    """
    with open(inference_file, 'r') as inf_file:
        inference_data = json.load(inf_file)

    with open(geometry_file, 'r') as geom_file:
        geometry_data = json.load(geom_file)

    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    # Index geometry data by UID
    geometry_dict = {
        feature["properties"]["uid"]: feature
        for feature in geometry_data.get("features", {}).get("lng_lat", [])
    }

    for uid, damage_type in inference_data.items():
        if uid in geometry_dict:
            geom_feature = geometry_dict[uid]
            geom = mapping(wkt.loads(geom_feature["wkt"]))

            geojson["features"].append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "uid": uid,
                    "subtype": damage_type
                }
            })
        else:
            print(f"Warning: Geometry not found for UID: {uid}")

    with open(output_geojson, 'w') as out_file:
        json.dump(geojson, out_file, indent=4)
    print(f"GeoJSON saved to {output_geojson}")


def main():
    parser = argparse.ArgumentParser(description="Merge classification results with geometries to create GeoJSON.")
    parser.add_argument('--inference_file', required=True, help='Path to inference results JSON (uid: damage_type)')
    parser.add_argument('--geometry_file', required=True, help='Path to JSON with geometry WKT per uid')
    parser.add_argument('--output_geojson', required=True, help='Output path for the GeoJSON file')

    args = parser.parse_args()
    add_geometry_to_inference(args.inference_file, args.geometry_file, args.output_geojson)


if __name__ == "__main__":
    main()

