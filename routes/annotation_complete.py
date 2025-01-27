from flask import Blueprint, request, render_template
import json
import os
from utils.file_utils import move_verified_diagrams, update_diagram_paths_in_json
from utils.json_utils import extract_text_and_save_to_json

# Create a Blueprint for annotation_complete
bp = Blueprint('annotation_complete', __name__)

@bp.route('/annotation_complete', methods=['GET', 'POST'])
def annotation_complete():
    json_file = "questions_with_details.json"  # Path to the output JSON file
    input_file = "question_diagram_links.json"  # Path to the input JSON file

    try:
        # Call the extract function to generate the JSON file
        move_verified_diagrams()
        extract_text_and_save_to_json(input_file, json_file)
        update_diagram_paths_in_json(json_file)
        # Load the newly generated JSON file
        with open(json_file, "r", encoding="utf-8") as f:
            extracted_data = json.load(f)

        # Render the page with the extracted data
        return render_template(
            "annotation_complete.html",
            extracted_data=extracted_data,
            output_json=json_file
        )

    except Exception as e:
        print(f"Error during annotation completion: {e}")
        return "An error occurred during annotation completion. Please check the logs.", 500
