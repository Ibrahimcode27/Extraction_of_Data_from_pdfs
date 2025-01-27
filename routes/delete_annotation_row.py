from flask import Blueprint, request, redirect, url_for
import os
import json

# Create a Blueprint for delete_annotation_row
bp = Blueprint('delete_annotation_row', __name__)

@bp.route('/delete_annotation_row', methods=['POST'])
def delete_annotation_row():
    """
    Deletes a row from the JSON file based on the provided index.
    """
    row_index = request.form.get('row_index')  # Extract the row index from the form
    json_file = "questions_with_details.json"

    if not row_index:
        print("Row index not provided in the form.")
        return "Row index not provided. Please try again.", 400

    try:
        # Convert the row index to an integer
        row_index = int(row_index)

        # Check if the JSON file exists
        if not os.path.exists(json_file):
            print("JSON file not found.")
            return "JSON file not found. Please reload the page.", 400

        # Load the existing JSON data
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate the row index
        if 0 <= row_index < len(data):
            # Remove the specific row
            del data[row_index]

            # Save the updated JSON data back to the file
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            print(f"Deleted row {row_index} successfully.")
        else:
            print(f"Invalid row index: {row_index}.")
            return f"Invalid row index: {row_index}", 400

    except ValueError as e:
        print(f"Invalid row index value: {e}.")
        return "Invalid row index format. Please provide a valid number.", 400
    except Exception as e:
        print(f"Unexpected error: {e}.")
        return "An error occurred while deleting the row. Please try again.", 500

    # Redirect back to the annotation_complete route to refresh the page
    return redirect(url_for('annotation_complete.annotation_complete'))
