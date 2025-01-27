import os
import shutil
import json

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def normalize_path(path):
    return path.replace("\\", "/")

def move_verified_diagrams(src_dir="static/verified_crops/Diagrams", dest_dir="static/permanent_diagrams"):
    os.makedirs(dest_dir, exist_ok=True)
    for file in os.listdir(src_dir):
        src_path = os.path.join(src_dir, file)
        dest_path = os.path.join(dest_dir, file)
        if os.path.isfile(src_path):
            shutil.move(src_path, dest_path)

def update_diagram_paths_in_json(json_file="questions_with_details.json", permanent_dir="static/permanent_diagrams"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        if entry.get("diagram_image_path"):
            diagram_filename = os.path.basename(entry["diagram_image_path"])
            entry["diagram_image_path"] = os.path.join(permanent_dir, diagram_filename)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

