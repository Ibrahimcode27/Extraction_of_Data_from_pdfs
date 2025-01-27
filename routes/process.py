from flask import Blueprint, request, session, redirect, url_for
import os
from utils.yolo_utils import convert_pdf_to_images
from utils.yolo_utils import run_yolo_and_save_with_boxes

# Create a Blueprint for the process route
bp = Blueprint('process', __name__)

@bp.route('/process', methods=['POST'])
def process_pdf():
    """
    Processes an uploaded PDF: Converts it to images, runs YOLO on the images, and prepares for verification.
    """
    pdf_path = session.get('uploaded_pdf')
    if not pdf_path:
        return "No uploaded PDF found in session.", 400

    try:
        # Step 1: Convert PDF to images
        images = convert_pdf_to_images(pdf_path)

        # Step 2: Run YOLO and save predictions
        predictions_file_path = run_yolo_and_save_with_boxes(images)

        if not predictions_file_path:
            return "Failed to save YOLO predictions.", 500

        # Step 3: Save the path to the predictions file in the session
        session['predictions_file'] = predictions_file_path

        # Step 4: Save processed image paths for verification
        processed_images = [os.path.join('static/output_with_boxes', os.path.basename(image)) for image in images]
        session['processed_images'] = [path.replace('\\', '/') for path in processed_images]

        # Redirect to the verification page
        return redirect(url_for('verify.verify_crops'))

    except Exception as e:
        print(f"Error processing PDF: {e}")
        session.clear()
        return "An error occurred while processing the PDF. Please try again.", 500
