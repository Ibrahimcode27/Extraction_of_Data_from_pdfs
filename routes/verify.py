from flask import Blueprint, request, session, redirect, url_for, render_template
import os
import json
import cv2
from utils.json_utils import save_question_diagram_links

# Create a Blueprint for verify
bp = Blueprint('verify', __name__)

@bp.route('/verify', methods=['GET', 'POST'])
def verify_crops():
    if request.method == 'GET':
        # Get the file paths from the session
        temp_file_path = session.get('predictions_file')  # Path to the JSON file
        processed_images = session.get('processed_images', [])  # List of processed image paths

        # Check if the prediction file exists
        if not temp_file_path or not os.path.exists(temp_file_path):
            return "Predictions file not found. Please process the images again.", 400

        # Load predictions from the JSON file
        with open(temp_file_path, "r") as f:
            predictions = json.load(f)

        if not processed_images or not predictions:
            return "No images or predictions found to verify.", 400

        # Render the verification page
        return render_template('verify.html', image_paths=processed_images, predictions=predictions)

    elif request.method == 'POST':
        # Get the list of verified images from the form
        verified_images = request.form.getlist('verified_crops')
        verified_images = [os.path.basename(image) for image in verified_images]

        # Load predictions from the session's JSON file
        temp_file_path = session.get('predictions_file')
        if not temp_file_path or not os.path.exists(temp_file_path):
            return "Predictions file not found. Please process the images again.", 400

        with open(temp_file_path, "r") as f:
            predictions = json.load(f)

        # Directory to save verified crops
        output_dir = "./static/verified_crops"
        os.makedirs(output_dir, exist_ok=True)

        verified_predictions = {}

        # Iterate through the predictions and save verified crops
        for page_id, prediction_list in predictions.items():
            verified_predictions[page_id] = []
            for prediction in prediction_list:
                if os.path.basename(prediction['path']) in verified_images:
                    verified_predictions[page_id].append(prediction)

                    # Extract bounding box coordinates
                    x_min, y_min, x_max, y_max = (
                        int(prediction['x_min']),
                        int(prediction['y_min']),
                        int(prediction['x_max']),
                        int(prediction['y_max']),
                    )
                    class_name = prediction['class']
                    img = cv2.imread(prediction['path'])

                    if img is None:
                        print(f"Failed to load image: {prediction['path']}")
                        continue

                    # Crop the image
                    cropped = img[y_min:y_max, x_min:x_max]
                    if cropped.size == 0:
                        print(f"Invalid crop for: {prediction}")
                        continue

                    # Save the cropped image in the respective class directory
                    class_dir = os.path.join(output_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)
                    crop_filename = os.path.join(
                        class_dir, f"{os.path.basename(prediction['path'])}_{x_min}_{y_min}.png"
                    )
                    cv2.imwrite(crop_filename, cropped)

        # Save the verified predictions to a JSON file
        save_question_diagram_links(verified_predictions, "question_diagram_links.json")

        # Update the session with unverified images
        session['unverified_images'] = [
            prediction for page_id, preds in predictions.items()
            for prediction in preds if prediction not in verified_predictions[page_id]
        ]

        # Redirect based on remaining unverified images
        if not session['unverified_images']:
            return redirect(url_for('annotation_complete.annotation_complete'))
        return redirect(url_for('annotate.annotate'))
