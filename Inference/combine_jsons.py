import json
from shapely import wkt
from shapely.geometry import mapping

def add_geometry_to_inference(inference_file, geometry_file, output_geojson):
    """
    Adds geometries from a geometry file to an inference file and saves as GeoJSON.

    :param inference_file: Path to the JSON inference file (classification results).
    :param geometry_file: Path to the JSON file containing geometries.
    :param output_geojson: Path to save the combined GeoJSON file.
    """
    # Load inference file
    with open(inference_file, 'r') as inf_file:
        inference_data = json.load(inf_file)

    # Load geometry file
    with open(geometry_file, 'r') as geom_file:
        geometry_data = json.load(geom_file)

    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    # Match `uid` and add geometries
    geometry_dict = {feature["properties"]["uid"]: feature for feature in geometry_data["features"]["lng_lat"]}
    
    for uid, damage_type in inference_data.items():
        if uid in geometry_dict:
            geometry_feature = geometry_dict[uid]
            geojson["features"].append({
                "type": "Feature",
                "properties": {
                    "uid": uid,
                    "subtype": damage_type
                },
                "geometry": mapping(wkt.loads(geometry_feature["wkt"]))
            })
        else:
            print(f"Geometry not found for UID: {uid}")

    # Save the combined GeoJSON
    with open(output_geojson, 'w') as out_file:
        json.dump(geojson, out_file, indent=4)
    print(f"GeoJSON saved to {output_geojson}")

if __name__ == "__main__":
    # Paths to input files and output GeoJSON
    inference_file = "/content/drive/MyDrive/Thesis/protonet_inference_output/inference_results.json"
    geometry_file = "/content/drive/MyDrive/Thesis/protonet_inference_output/combined_results.json"
    output_geojson = "/content/drive/MyDrive/Thesis/protonet_inference_output/geo_results.json"

    # Add geometries and export
    add_geometry_to_inference(inference_file, geometry_file, output_geojson)
