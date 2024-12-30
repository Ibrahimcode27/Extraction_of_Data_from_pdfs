import os
import json
from ultralytics import YOLO
from paddleocr import PaddleOCR
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

def normalize_path(path):
    return path.replace("\\", "/")

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
        if original_image is None:
            print(f"Failed to load image: {image_path}")
            continue

        page_id = os.path.basename(image_path).split('.')[0]  # Use filename without extension as page ID
        predictions[page_id] = []  # Initialize page annotations

        results = model(image_path)

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

            # Store prediction data
            predictions[page_id].append({
                "class": class_name,
                "path": image_path,
                "y_min": y_min,
                "y_max": y_max,
                "x_min": x_min,
                "x_max": x_max,
                "confidence": confidence
            })

            # Draw bounding boxes on the image
            cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(original_image, f"{class_name} {confidence:.2f}",
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Save processed image with bounding boxes
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, original_image)

    return predictions

def save_question_diagram_links(annotation_data, output_dir="static/verified_crops", output_file="question_diagram_links.json"):
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            linked_data = json.load(f)
    else:
        linked_data = []

    new_links = []

    for page_id, annotations in annotation_data.items():
        questions = [a for a in annotations if a['class'] == 'quiz']
        diagrams = [a for a in annotations if a['class'] == 'diagram']

        for question in questions:
            question_cropped_path = os.path.join(output_dir, "quiz", f"{os.path.basename(question['path'])}_{question['x_min']}_{question['y_min']}.png")
            question_cropped_path = normalize_path(question_cropped_path)

            linked_diagram = None
            for diagram in diagrams:
                if (
                    diagram['x_min'] >= question['x_min'] and
                    diagram['x_max'] <= question['x_max'] and
                    diagram['y_min'] >= question['y_min'] and
                    diagram['y_max'] <= question['y_max']
                ):
                    linked_diagram = normalize_path(
                        os.path.join(output_dir, "diagram", f"{os.path.basename(diagram['path'])}_{diagram['x_min']}_{diagram['y_min']}.png")
                    )
                    break

            # Add the PDF filename to the link
            pdf_filename = os.path.basename(question['path']).split("_")[0] + ".pdf"
            pdf_path = session.get('uploaded_pdf')
            pdf_path = pdf_path.replace("static/uploads\\", "")
            link = {
                "page": page_id,
                "question": question_cropped_path,
                "diagram": linked_diagram,
                "pdf_filename": pdf_path
            }

            if link not in linked_data and link not in new_links:
                new_links.append(link)

    linked_data.extend(new_links)
    with open(output_file, "w") as f:
        json.dump(linked_data, f, indent=4)

def extract_text_and_link_to_diagrams(json_file, output_dir="static/verified_crops", database=DATABASE):
    ocr = PaddleOCR(use_angle_cls=True, lang="latin")

    with open(json_file, "r") as f:
        question_diagram_links = json.load(f)

    with sqlite3.connect(database) as conn:
        cursor = conn.cursor()
        for link in question_diagram_links:
            quiz_path = link["question"]
            diagram_path = link["diagram"]
            pdf_filename = link["pdf_filename"]

            if not os.path.exists(quiz_path):
                print(f"Quiz image not found: {quiz_path}")
                continue

            results = ocr.ocr(quiz_path, cls=True)
            extracted_text = " ".join([line[1][0] for line in results[0]]) if results else ""

            cursor.execute('''
            SELECT id, topic_tags, subject, exam_type FROM pdfs WHERE filename = ?
            ''', (pdf_filename,))
            pdf_data = cursor.fetchone()

            if pdf_data:
                pdf_id, topic, subject, exam_type = pdf_data
                cursor.execute('''
                INSERT INTO questions (pdf_id, question_text, diagram_path, topic, subject, exam_type)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (pdf_id, extracted_text, diagram_path, topic, subject, exam_type))
        conn.commit()

# Initialize database
def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        
        # Create the PDFs table (adding topic_tags field)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdfs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            exam_type TEXT NOT NULL,
            subject TEXT NOT NULL,
            topic_tags TEXT,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create the Questions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_id INTEGER,
            question_text TEXT,
            diagram_path TEXT,
            topic TEXT,
            subject TEXT,
            exam_type TEXT,
            FOREIGN KEY (pdf_id) REFERENCES pdfs (id)
        )
        ''')
        
        # Create the Topics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_type TEXT,
            subject TEXT,
            topic_name TEXT
        )
        ''')
        
        conn.commit()

@app.route('/')
def upload_page():
    # Optional: Clear specific session keys if needed
    session.pop('uploaded_pdf', None)
    return render_template('index.html')  # Same upload UI as before

@app.route('/success', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part in request", 400

    file = request.files['file']

    if file.filename == '':
        return "No file selected", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Get form data
        exam_type = request.form['exam_type']
        subject = request.form['subject']
        topic = request.form.get('topic', None)

        # Insert into database
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()

            # Insert PDF metadata
            cursor.execute('''
            INSERT INTO pdfs (filename, exam_type, subject, topic_tags)
            VALUES (?, ?, ?, ?)
            ''', (filename, exam_type, subject, topic))

            # Ensure topic exists in the topics table
            cursor.execute('''
            SELECT id FROM topics WHERE exam_type = ? AND subject = ? AND topic_name = ?
            ''', (exam_type, subject, topic))
            topic_exists = cursor.fetchone()

            if not topic_exists and topic:
                cursor.execute('''
                INSERT INTO topics (exam_type, subject, topic_name)
                VALUES (?, ?, ?)
                ''', (exam_type, subject, topic))

            conn.commit()

        session['uploaded_pdf'] = file_path

        return render_template(
            'success.html',
            filename=filename,
            exam_type=exam_type,
            subject=subject,
            topic=topic
        )
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

        # Step 3: Save predictions for linking
        session['predictions'] = predictions  # Store predictions in session
        save_question_diagram_links(predictions)  # Link questions and diagrams

        # Step 3: Save processed image paths
        processed_images = [os.path.join('static/output_with_boxes', os.path.basename(image)) for image in images]
        session['processed_images'] = [path.replace('\\', '/') for path in processed_images]  # Normalize paths
        
        # Redirect to the verification page
        return redirect(url_for('verify_crops'))
    except Exception as e:
        print(f"Error processing PDF: {e}")
        session.clear()
        return "An error occurred while processing the PDF. Please try again.", 500

@app.route('/verify', methods=['GET', 'POST'])
def verify_crops():
    if request.method == 'GET':
        processed_images = session.get('processed_images', [])
        predictions = session.get('predictions', {})

        if not processed_images or not predictions:
            return "No images or predictions found to verify.", 400

        # Pass predictions and processed images to the template for verification
        return render_template('verify.html', image_paths=processed_images, predictions=predictions)

    elif request.method == 'POST':
        verified_images = request.form.getlist('verified_crops')
        verified_images = [os.path.basename(image) for image in verified_images]

        predictions = session.get('predictions', {})
        if not predictions:
            return "Predictions not found. Please process the images again.", 400

        output_dir = "./static/verified_crops"
        os.makedirs(output_dir, exist_ok=True)

        verified_predictions = {}

        # Iterate through the predictions
        for page_id, prediction_list in predictions.items():
            verified_predictions[page_id] = []  # Initialize verified predictions for the page
            for prediction in prediction_list:
                if os.path.basename(prediction['path']) in verified_images:
                    # Add to verified predictions
                    verified_predictions[page_id].append(prediction)

                    # Save verified crops
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

        # Save links only for verified predictions
        save_question_diagram_links(verified_predictions)

        # Update session with unverified images
        session['unverified_images'] = [
            prediction for page_id, preds in predictions.items() for prediction in preds if prediction not in verified_predictions[page_id]
        ]
        return redirect(url_for('annotate'))

@app.route('/annotate', methods=['GET', 'POST'])
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
        print(f"Annotations: {current_annotations}")

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

            # Save each annotation as a cropped image
            for annotation in annotations:
                x1, y1, x2, y2 = annotation['x1'], annotation['y1'], annotation['x2'], annotation['y2']
                class_name = annotation['class_name']

                # Load the image
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Failed to load image: {image_path}")
                    return f"Image {image_path} not found.", 400

                # Crop the annotation
                cropped = img[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue

                # Save the cropped annotation
                output_dir = os.path.join('static', 'verified_crops', class_name)
                os.makedirs(output_dir, exist_ok=True)

                crop_filename = os.path.join(output_dir, f"{os.path.basename(image_path)}_{x1}_{y1}.png")
                cv2.imwrite(crop_filename, cropped)

            # Update session for remaining annotations in the current image
            current_annotations = session.get('current_annotations', [])
            unverified_images = session.get('unverified_images', [])

            # Remove the processed annotations from unverified_images
            unverified_images = [anno for anno in unverified_images if anno not in current_annotations]
            session['unverified_images'] = unverified_images

            # Update question-diagram links for the annotated page
            page_predictions = {current_annotations[0]['path']: current_annotations}
            save_question_diagram_links(page_predictions)

            # Move to the next image or finish annotation
            if unverified_images:
                return redirect(url_for('annotate'))
            else:
                return redirect(url_for('annotation_complete'))

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return "Invalid JSON format in annotations.", 400
        except Exception as e:
            print(f"Unexpected error: {e}")
            return f"An error occurred: {str(e)}", 500

@app.route('/annotation_complete', methods=['GET'])
def annotation_complete():
    # Path to the JSON file containing the links
    json_file = "question_diagram_links.json"

    # Call the function to extract text and link diagrams
    try:
        extract_text_and_link_to_diagrams(json_file)
    except Exception as e:
        print(f"Error during text extraction and linking: {e}")
        return "An error occurred during text extraction and linking. Please check the logs.", 500

    return "Annotation process completed successfully!"

if __name__ == "__main__":
    init_db()
    app.run(debug=True)