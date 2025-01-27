from flask import Blueprint, request, session, redirect, url_for, render_template
import os
import json
import cv2
from utils.json_utils import save_question_diagram_links

# Create a Blueprint for annotate
bp = Blueprint('annotate', __name__)

@bp.route('/annotate', methods=['GET', 'POST'])
def annotate():
    if request.method == 'GET':
        # Retrieve unverified images from session
        unverified_images = session.get('unverified_images', [])
        if not unverified_images:
            return "No more images to annotate.", 200  # Return when no images are left

        # Extract the current image and its annotations
        current_image = unverified_images[0]  # Get the first unverified image
        current_image_path = current_image['path']
        current_annotations = [anno for anno in unverified_images if anno['path'] == current_image_path]

        # Debug: Print the current image and annotations
        print(f"Annotating image: {current_image_path}")

        # Save current annotations to the session for reference
        session['current_annotations'] = current_annotations

        # Render the annotation page
        return render_template(
            'annotate.html',
            image_path=current_image_path,
            page_number=current_image_path.split('/')[-1].split('.')[0],  # Deduce page number
        )

    elif request.method == 'POST':
        try:
            # Retrieve annotations and image path from the form
            image_path = request.form.get('image_path', '')
            annotations = request.form.get('annotations', '[]')

            # Decode JSON
            annotations = json.loads(annotations)

            # Validate annotations
            if not annotations:
                return "No annotations submitted. Please try again.", 400

            # Prepare a list to store cropped images
            new_annotations = []

            # Save each annotation as a cropped image
            for annotation in annotations:
                x1, y1, x2, y2 = annotation['x1'], annotation['y1'], annotation['x2'], annotation['y2']
                print(f"Cropping coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                # Map received class names to valid class names
                valid_class_mapping = {
                    'option': 'Options',
                    'options': 'Options',
                    'question': 'Questions',
                    'questions': 'Questions',
                    'solution': 'Solutions',
                    'solutions': 'Solutions',
                    'diagram': 'Diagrams',
                    'diagrams': 'Diagrams',
                }
                
                # Normalize class name using the mapping
                class_name = valid_class_mapping.get(annotation['class_name'].lower(), None)
                # Debugging: Print the normalized class name
                print(f"Normalized class name: {class_name}")
                # Check if the class name is valid
                if class_name not in ['Options', 'Questions', 'Solutions', 'Diagrams']:
                    print(f"Invalid class name: {class_name}")
                    continue

                # Load the image
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Failed to load image: {image_path}")
                    return f"Image {image_path} not found.", 400

                # Crop the annotation
                cropped = img[y1:y2, x1:x2]
                print(f"Cropped image size: {cropped.shape if cropped.size > 0 else 'Invalid crop'}")
                if cropped.size == 0:
                    print(f"Invalid crop for: {annotation}")
                    continue
                # Save the cropped annotation in the respective class directory
                output_dir = os.path.join('static', 'verified_crops', class_name)
                os.makedirs(output_dir, exist_ok=True)

                crop_filename = os.path.join(output_dir, f"{os.path.basename(image_path)}_{x1}_{y1}.png")
                print(f"Saving cropped image to: {crop_filename}")
                cv2.imwrite(crop_filename, cropped)

                # Add the annotation to new_annotations
                new_annotations.append({
                    "class": class_name,
                    "path": image_path,
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                })

            # Update unverified images to remove the current image
            unverified_images = session.get('unverified_images', [])
            unverified_images = [anno for anno in unverified_images if anno['path'] != image_path]
            session['unverified_images'] = unverified_images

            # Process only the new manual annotations for the current image
            page_predictions = {image_path: new_annotations}
            print(page_predictions)
            save_question_diagram_links(page_predictions)
            # Debugging: Print the page predictions sent to the clustering function
            print(f"Page predictions sent for clustering: {page_predictions}")

            # Move to the next image or finish annotation
            if unverified_images:
                return redirect(url_for('annotate.annotate'))
            else:
                return redirect(url_for('annotation_complete.annotation_complete'))

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return "Invalid JSON format in annotations.", 400
        except Exception as e:
            print(f"Unexpected error: {e}")
            return f"An error occurred: {str(e)}", 500
