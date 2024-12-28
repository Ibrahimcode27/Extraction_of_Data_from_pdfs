import os
import json
from ultralytics import YOLO
from flask import Flask, render_template, request, redirect, url_for, session
from pdf2image import convert_from_path
import cv2
import sqlite3
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# Directories
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
DATABASE = 'database.db'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Utility to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_pdf_to_images(pdf_path, output_folder="static/temp_images"):
    """
    Converts a PDF into images, one per page.
    Args-pdf_path (str): Path to the PDF file,output_folder (str): Folder to save the converted images.
    Returns-list: A list of file paths to the generated images.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=300, fmt='png')  # 300 DPI for good quality
    
    # Save images and collect paths
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)

    return image_paths

def run_yolo_and_save_with_boxes(image_paths, model_path="best.pt", output_dir="static/output_with_boxes"):
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)
    predictions = {}

    for image_path in image_paths:
        original_image = cv2.imread(image_path)
        orig_height, orig_width = original_image.shape[:2]

        results = model(image_path)

        # Adjust predictions to use static/temp_images paths
        temp_image_path = os.path.join("static/temp_images", os.path.basename(image_path)).replace("\\", "/")

        for box in results[0].boxes:
            bbox = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Adjust bounding box coordinates
            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[2])
            y_max = int(bbox[3])

            # Store predictions using temp_image_path
            predictions.setdefault(temp_image_path, []).append({
                "class": class_name,
                "bbox": [x_min, y_min, x_max, y_max],
                "confidence": confidence
            })

            # Draw bounding boxes on the output image
            cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 15)
            cv2.putText(original_image, f"{class_name} {confidence:.2f}",
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.9, (255, 0, 0), 10)

        # Save the processed image with bounding boxes
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, original_image)

    return predictions

# Initialize database
def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdfs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            exam_type TEXT NOT NULL,
            subject TEXT NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()

@app.route('/')
def upload_page():
    # Optional: Clear specific session keys if needed
    session.pop('uploaded_pdf', None)
    return render_template('index.html')  # Same upload UI as before

# Route: Handle form submission
@app.route('/success', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part in request", 400

    file = request.files['file']

    if file.filename == '':
        return "No file selected", 400

    if file and allowed_file(file.filename):
        # Secure the file name and save it
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Get form data
        exam_type = request.form['exam_type']
        subject = request.form['subject']

        # Insert into database
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO pdfs (filename, exam_type, subject)
            VALUES (?, ?, ?)
            ''', (filename, exam_type, subject))
            conn.commit()

        session['uploaded_pdf'] = file_path

        return render_template('success.html', filename=filename, exam_type=exam_type, subject=subject)
    else:
        return "Invalid file type", 400

@app.route('/delete/<int:id>', methods=['POST'])
def delete_row(id):
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM pdfs WHERE id = ?', (id,))
            conn.commit()
        return redirect(url_for('view_data'))
    except Exception as e:
        print(f"Error deleting row: {e}")
        return "An error occurred while deleting the row.", 500

# Route: View uploaded data
@app.route('/view_data')
def view_data():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM pdfs')
        rows = cursor.fetchall()
    return render_template('view_data.html', rows=rows)

#Route: Process
@app.route('/process', methods=['POST'])
def process_pdf():
    pdf_path = session.get('uploaded_pdf')
    if not pdf_path:
        return "No uploaded PDF found in session.", 400

    try:
        # Step 1: Convert PDF to images
        images = convert_pdf_to_images(pdf_path)

        # Step 2: Run YOLO and process predictions
        predictions = run_yolo_and_save_with_boxes(images)

        # Step 3: Save processed image paths
        processed_images = [os.path.join('static/output_with_boxes', os.path.basename(image)) for image in images]
        session['processed_images'] = [path.replace('\\', '/') for path in processed_images]  # Normalize paths
        session['predictions'] = predictions

        # Redirect to the verification page
        return redirect(url_for('verify_crops'))
    except Exception as e:
        print(f"Error processing PDF: {e}")
        session.clear()
        return "An error occurred while processing the PDF. Please try again.", 500

@app.route('/verify', methods=['GET', 'POST'])
def verify_crops():
    if request.method == 'GET':
        # Retrieve processed image paths from the session
        processed_images = session.get('processed_images', [])
        if not processed_images:
            return "No images found to verify.", 400

        # Render the verification page with image paths
        return render_template('verify.html', image_paths=processed_images)

    elif request.method == 'POST':
        # Retrieve verified image paths from the form submission
        verified_images = request.form.getlist('verified_crops')
        verified_images = [os.path.basename(image) for image in verified_images]

        # Retrieve predictions from the session
        predictions = session.get('predictions', {})
        if not predictions:
            return "Predictions not found. Please process the images again.", 400

        # Define output directory for verified crops
        output_dir = "./static/verified_crops"
        os.makedirs(output_dir, exist_ok=True)

        unverified_images = []

        # Process and save verified images
        for image_path, prediction_list in predictions.items():
            img_path = image_path.replace("\\", "/")  # Ensure consistent path format
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue

            if os.path.basename(image_path) in verified_images:
                # Save verified crops
                for prediction in prediction_list:
                    class_name = prediction['class']
                    x_min, y_min, x_max, y_max = map(int, prediction['bbox'])

                    cropped = img[y_min:y_max, x_min:x_max]
                    if cropped.size == 0:
                        print(f"Invalid crop for bbox: {prediction['bbox']}")
                        continue

                    class_dir = os.path.join(output_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)
                    crop_filename = os.path.join(class_dir, f"{os.path.basename(image_path)}_{x_min}_{y_min}.png")
                    cv2.imwrite(crop_filename, cropped)
            else:
                # Add to unverified images
                unverified_images.append(img_path)

        # Save unverified images to session
        session['unverified_images'] = unverified_images

        # Redirect to /annotate after processing
        return redirect(url_for('annotate'))

@app.route('/annotate', methods=['GET', 'POST'])
def annotate():
    if request.method == 'GET':
        # Retrieve unverified images from session
        unverified_images = session.get('unverified_images', [])
        if not unverified_images:
            return "No more images to annotate.", 200

        # Display the first unverified image
        current_image = unverified_images[0]
        original_image = cv2.imread(current_image)

        if original_image is None:
            # Handle missing or inaccessible image
            unverified_images.pop(0)  # Remove the broken image
            session['unverified_images'] = unverified_images
            return "The current image could not be loaded. Skipping to the next image.", 400

        # Save canvas size in session for scaling annotations
        canvas_width, canvas_height = 800, 600  # Example default dimensions
        session['canvas_size'] = (canvas_width, canvas_height)

        return render_template(
            'annotate.html',
            image_path=current_image,
            canvas_size=(canvas_width, canvas_height),
        )

    elif request.method == 'POST':
        try:
            # Retrieve JSON strings from the form
            image_path = request.form.get('image_path', '')
            annotations = request.form.get('annotations', '[]')

            # Decode JSON
            annotations = json.loads(annotations)

            # Validate annotations
            if not annotations:
                return "No annotations submitted. Please try again.", 400

            # Process each annotation
            for annotation in annotations:
                x1, y1, x2, y2 = annotation['x1'], annotation['y1'], annotation['x2'], annotation['y2']
                class_name = annotation['class_name']

                # Load image
                img = cv2.imread(image_path)
                if img is None:
                    return f"Image {image_path} not found.", 400

                # Crop and save
                cropped = img[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue

                output_dir = os.path.join('static', 'verified_crops', class_name)
                os.makedirs(output_dir, exist_ok=True)

                crop_filename = os.path.join(output_dir, f"{os.path.basename(image_path)}_{x1}_{y1}.png")
                cv2.imwrite(crop_filename, cropped)

            # Update session for remaining unverified images
            unverified_images = session.get('unverified_images', [])
            if unverified_images:
                unverified_images.pop(0)  # Remove the processed image
                session['unverified_images'] = unverified_images

            # Redirect to the next image or finish annotation
            if unverified_images:
                return redirect(url_for('annotate'))
            else:
                return redirect(url_for('annotation_complete'))

        except json.JSONDecodeError:
            return "Invalid JSON format in annotations.", 400
        except Exception as e:
            return f"An error occurred: {str(e)}", 500

@app.route('/annotation_complete', methods=['GET'])
def annotation_complete():
    return "Annotation process completed successfully!"

if __name__ == "__main__":
    app.run(debug=True)
