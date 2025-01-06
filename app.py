import os
import io
from google.cloud import vision
import json
from ultralytics import YOLO
from flask import Flask, render_template, request, redirect, url_for, session
from pdf2image import convert_from_path
import cv2
from werkzeug.utils import secure_filename
import tempfile
import mysql.connector

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# Directories
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

DATABASE_CONFIG = {
    'host': 'localhost',          # Your MySQL host
    'user': 'roots',               # Your MySQL username
    'password': '1234',  # Your MySQL password
    'database': 'extractor'   # Your database name
}

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
        print(f"page {i} done!")

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
            cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 12)
            cv2.putText(original_image, f"{class_name} {confidence:.2f}",
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 10)

        # Save processed image with bounding boxes
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, original_image)

    # Save predictions to a temporary JSON file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
    with temp_file as f:
        json.dump(predictions, f, indent=4)

    print(f"Predictions saved to temporary file: {temp_file.name}")
    print(temp_file.name)
    return temp_file.name

def save_question_diagram_links(annotation_data, output_dir="static/verified_crops", output_file="question_diagram_links.json"):
    """
    Links diagrams, options, and solutions to questions based on proximity and spatial relationships.
    """
    # Load existing links if the file exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            linked_data = json.load(f)
    else:
        linked_data = []

    new_links = []

    for page_id, annotations in annotation_data.items():
        # Separate annotations by class
        questions = [a for a in annotations if a['class'] == 'Questions']
        diagrams = [a for a in annotations if a['class'] == 'Diagrams']
        options = [a for a in annotations if a['class'] == 'Options']
        solutions = [a for a in annotations if a['class'] == 'Solutions']

        # To track used options and solutions
        used_options = set()
        used_solutions = set()

        for question in questions:
            question_cropped_path = os.path.join(
                output_dir, "Questions",
                f"{os.path.basename(question['path'])}_{question['x_min']}_{question['y_min']}.png"
            ).replace("\\", "/")

            # Get the vertical center of the question
            question_center_y = (question['y_min'] + question['y_max']) / 2

            # Find the closest diagram
            linked_diagram = None
            min_distance = float('inf')
            for diagram in diagrams:
                if diagram['y_min'] >= question['y_min']:  # Diagram is below or overlaps with the question
                    distance = abs(diagram['y_min'] - question['y_max'])
                    if distance < min_distance:
                        min_distance = distance
                        linked_diagram = os.path.join(
                            output_dir, "Diagrams",
                            f"{os.path.basename(diagram['path'])}_{diagram['x_min']}_{diagram['y_min']}.png"
                        ).replace("\\", "/")

            # Find the closest option (ensure option is not already linked and the question is above the option)
            linked_option = None
            min_distance = float('inf')
            for option in options:
                option_id = (option['path'], option['x_min'], option['y_min'])  # Use a unique identifier
                if option['y_min'] > question['y_max'] and option_id not in used_options:  # Option is strictly below
                    # Check if this question is the closest to the option
                    distance_to_question = abs(option['y_min'] - question['y_max'])
                    distance_to_any_other_question = float('inf')
                    for other_question in questions:
                        if other_question['y_max'] < option['y_min']:  # Ensure other question is above the option
                            distance_to_any_other_question = min(
                                distance_to_any_other_question,
                                abs(option['y_min'] - other_question['y_max'])
                            )

                    # Only link if this question is the closest to the option
                    if distance_to_question <= distance_to_any_other_question and distance_to_question < min_distance:
                        min_distance = distance_to_question
                        linked_option = os.path.join(
                            output_dir, "Options",
                            f"{os.path.basename(option['path'])}_{option['x_min']}_{option['y_min']}.png"
                        ).replace("\\", "/")
                        used_options.add(option_id)  # Mark this option as used

            # Find the closest solution (ensure solution is not already linked and the question is above the solution)
            linked_solution = None
            min_distance = float('inf')
            for solution in solutions:
                solution_id = (solution['path'], solution['x_min'], solution['y_min'])  # Use a unique identifier
                if solution['y_min'] > question['y_max'] and solution_id not in used_solutions:  # Solution is strictly below
                    # Check if this question is the closest to the solution
                    distance_to_question = abs(solution['y_min'] - question['y_max'])
                    distance_to_any_other_question = float('inf')
                    for other_question in questions:
                        if other_question['y_max'] < solution['y_min']:  # Ensure other question is above the solution
                            distance_to_any_other_question = min(
                                distance_to_any_other_question,
                                abs(solution['y_min'] - other_question['y_max'])
                            )

                    # Only link if this question is the closest to the solution
                    if distance_to_question <= distance_to_any_other_question and distance_to_question < min_distance:
                        min_distance = distance_to_question
                        linked_solution = os.path.join(
                            output_dir, "Solutions",
                            f"{os.path.basename(solution['path'])}_{solution['x_min']}_{solution['y_min']}.png"
                        ).replace("\\", "/")
                        used_solutions.add(solution_id)  # Mark this solution as used

            # Add the PDF filename to the link
            pdf_filename = os.path.basename(question['path']).split("_")[0] + ".pdf"
            pdf_path = session.get('uploaded_pdf', "").replace("static/uploads\\", "")

            # Prepare the link data
            link = {
                "page": page_id,
                "question": question_cropped_path,
                "diagram": linked_diagram,
                "options": linked_option,
                "solution": linked_solution,
                "pdf_filename": pdf_path,
            }

            # Avoid duplicates
            if link not in linked_data and link not in new_links:
                new_links.append(link)

    # Append new links to the existing data
    linked_data.extend(new_links)

    # Save the updated linked data back to the JSON file
    with open(output_file, "w") as f:
        json.dump(linked_data, f, indent=4)

    print(f"Updated linked data saved to {output_file}")

# Function to extract text using Google Vision API
def extract_text_google_vision(image_path):
    """
    Extract text from an image using Google Cloud Vision API.
    Args:
        image_path (str): Path to the image file.
    Returns:
        str: Extracted text or an empty string if no text is detected.
    """
    try:
        # Initialize the Vision API client
        client = vision.ImageAnnotatorClient()

        # Load the image from file
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()

        # Create an Image object from the image content
        image = vision.Image(content=content)

        # Perform text detection on the image
        response = client.document_text_detection(image=image)

        # Handle potential errors
        if response.error.message:
            raise Exception(f"Google Vision API Error: {response.error.message}")

        # Extract the detected text
        return response.full_text_annotation.text if response.full_text_annotation else ""

    except Exception as e:
        print(f"Error in text extraction: {e}")
        return ""

# Function to refine mathematical expressions
def refine_math_symbols(text):
    """
    Attempt to parse mathematical expressions and normalize them using SymPy.
    Args:
        text (str): Text extracted from OCR.
    Returns:
        str: Refined and simplified text if mathematical content is detected, otherwise the original text.
    """
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify
        math_expr = parse_latex(text)
        simplified_expr = simplify(math_expr)
        return str(simplified_expr)
    except Exception as e:
        print(f"Error refining math symbols: {e}")
        return text

# Main function to process data and save to JSON
def extract_text_and_refine_equations(json_file, output_json="questions_with_details.json"):
    """
    Extract text from images using Google Vision API and refine equations.
    Args:
        json_file (str): Path to input JSON file containing image links.
        output_json (str): Path to output JSON file for saving extracted data.
    """
    # Load question-diagram links from the input JSON file
    with open(json_file, "r") as f:
        question_diagram_links = json.load(f)

    # Prepare a list to hold the extracted data
    extracted_data = []

    # Process each entry in the JSON
    for entry in question_diagram_links:
        page = entry["page"]
        question_path = entry.get("question", None)
        diagram_path = entry.get("diagram", None)
        options_path = entry.get("options", None)
        solution_path = entry.get("solution", None)
        pdf_filename = entry.get("pdf_filename", "unknown_pdf")

        # Perform OCR for each type of image if paths exist
        question_text, options_text, solution_text, diagram_text = "", [], "", ""

        if question_path and os.path.exists(question_path):
            raw_text = extract_text_google_vision(question_path)
            print(f"raw Question: {raw_text}")
            question_text = refine_math_symbols(raw_text)  # Refine math symbols in the question

        if options_path and os.path.exists(options_path):
            raw_text = extract_text_google_vision(options_path)
            print(f"Raw Option: {raw_text}")
            options_text = [refine_math_symbols(line) for line in raw_text.split('\n')]

        if solution_path and os.path.exists(solution_path):
            raw_text = extract_text_google_vision(solution_path)
            print(f"Solution path: {raw_text}")
            solution_text = refine_math_symbols(raw_text)  # Refine math symbols in the solution

        if diagram_path and os.path.exists(diagram_path):
            diagram_text = extract_text_google_vision(diagram_path)
            print(f"diagram text: {diagram_text}")
            # Diagrams might not need refinement

        # Collect extracted data
        extracted_entry = {
            "page": page,
            "pdf_filename": pdf_filename,
            "question_text": question_text,
            "options_text": options_text,
            "solution_text": solution_text,
            "diagram_text": diagram_text,
            "quiz_image_path": question_path,
            "options_image_path": options_path,
            "solution_image_path": solution_path,
            "diagram_image_path": diagram_path,
        }
        extracted_data.append(extracted_entry)

    # Save the extracted data to a JSON file
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)
    print(f"Extracted data saved to {output_json}")

def init_db():
    conn = mysql.connector.connect(**DATABASE_CONFIG)  # Ensure conn is defined even if connection fails
    try:
        cursor = conn.cursor()

        # Create the PDFs table (with difficulty_level field)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdfs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            exam_type VARCHAR(50) NOT NULL,
            subject VARCHAR(50) NOT NULL,
            topic_tags TEXT,
            difficulty_level VARCHAR(50),  -- Added field
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        # Create the Questions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            pdf_id INT,
            question_text TEXT,
            diagram_path VARCHAR(255),
            topic VARCHAR(255),
            subject VARCHAR(255),
            exam_type VARCHAR(50),
            FOREIGN KEY (pdf_id) REFERENCES pdfs (id) ON DELETE CASCADE
        )
        ''')

        # Create the Options table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS options (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question_id INT NOT NULL,
            option_text TEXT NOT NULL,
            FOREIGN KEY (question_id) REFERENCES questions (id) ON DELETE CASCADE
        )
        ''')

        # Create the Solutions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS solutions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question_id INT NOT NULL,
            solution_text TEXT NOT NULL,
            FOREIGN KEY (question_id) REFERENCES questions (id) ON DELETE CASCADE
        )
        ''')

        # Create the Diagrams table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagrams (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question_id INT NOT NULL,
            diagram_path VARCHAR(255) NOT NULL,
            FOREIGN KEY (question_id) REFERENCES questions (id) ON DELETE CASCADE
        )
        ''')

        # Create the Topics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS topics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            exam_type VARCHAR(50),
            subject VARCHAR(255),
            topic_name VARCHAR(255)
        )
        ''')

        conn.commit()
        print("Database initialized successfully.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        # Ensure conn is checked before attempting to close it
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

#first route.
@app.route('/')
def upload_page():
    # Optional: Clear specific session keys if needed
    session.clear()
    session.pop('uploaded_pdf', None)
    return render_template('index.html')  # Same upload UI as before


#second route.
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
        difficulty_level = request.form.get('difficulty_level')  # New field

        try:
            # Connect to MySQL database
            conn = mysql.connector.connect(**DATABASE_CONFIG)
            cursor = conn.cursor()

            # Insert PDF metadata into `pdfs` table
            cursor.execute('''
            INSERT INTO pdfs (filename, exam_type, subject, topic_tags, difficulty_level)
            VALUES (%s, %s, %s, %s, %s)
            ''', (filename, exam_type, subject, topic, difficulty_level))

            # Ensure topic exists in the `topics` table
            cursor.execute('''
            SELECT id FROM topics WHERE exam_type = %s AND subject = %s AND topic_name = %s
            ''', (exam_type, subject, topic))
            topic_exists = cursor.fetchone()

            if not topic_exists and topic:
                cursor.execute('''
                INSERT INTO topics (exam_type, subject, topic_name)
                VALUES (%s, %s, %s)
                ''', (exam_type, subject, topic))

            # Commit the transaction
            conn.commit()

        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return "An error occurred while saving to the database.", 500

        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

        # Store the uploaded file path in the session
        session['uploaded_pdf'] = file_path

        return render_template(
            'success.html',
            filename=filename,
            exam_type=exam_type,
            subject=subject,
            topic=topic,
            difficulty_level=difficulty_level  # Pass the new field to the template
        )
    else:
        return "Invalid file type", 400


@app.route('/delete/<int:id>', methods=['POST'])
def delete_row(id):
    try:
        # Connect to MySQL database
        conn = mysql.connector.connect(**DATABASE_CONFIG)
        cursor = conn.cursor()

        # Execute the DELETE query
        cursor.execute('DELETE FROM pdfs WHERE id = %s', (id,))

        # Commit the transaction
        conn.commit()

        # Redirect to the data view page
        return redirect(url_for('view_data'))
    except mysql.connector.Error as err:
        print(f"Error deleting row: {err}")
        return "An error occurred while deleting the row.", 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/view_data')
def view_data():
    try:
        # Connect to MySQL database
        conn = mysql.connector.connect(**DATABASE_CONFIG)
        cursor = conn.cursor()

        # Execute the query to retrieve all rows from the pdfs table
        cursor.execute('SELECT * FROM pdfs')
        rows = cursor.fetchall()

        # Render the data in the view_data.html template
        return render_template('view_data.html', rows=rows)
    except mysql.connector.Error as err:
        print(f"Error fetching data: {err}")
        return "An error occurred while fetching data. Please try again.", 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


#third route.
@app.route('/process', methods=['POST'])
def process_pdf():
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
        return redirect(url_for('verify_crops'))
    except Exception as e:
        print(f"Error processing PDF: {e}")
        session.clear()
        return "An error occurred while processing the PDF. Please try again.", 500

#fourth route.
@app.route('/verify', methods=['GET', 'POST'])
def verify_crops():
    if request.method == 'GET':
        temp_file_path = session.get('predictions_file')  # Path to the JSON file
        processed_images = session.get('processed_images', [])

        if not temp_file_path or not os.path.exists(temp_file_path):
            return "Predictions file not found. Please process the images again.", 400

        with open(temp_file_path, "r") as f:
            predictions = json.load(f)

        if not processed_images or not predictions:
            return "No images or predictions found to verify.", 400

        # Pass predictions and processed images to the template for verification
        return render_template('verify.html', image_paths=processed_images, predictions=predictions)

    elif request.method == 'POST':
        verified_images = request.form.getlist('verified_crops')
        verified_images = [os.path.basename(image) for image in verified_images]

        # Load predictions from the JSON file
        temp_file_path = session.get('predictions_file')  # Path to the JSON file
        if not temp_file_path or not os.path.exists(temp_file_path):
            return "Predictions file not found. Please process the images again.", 400

        with open(temp_file_path, "r") as f:
            predictions = json.load(f)

        output_dir = "./static/verified_crops"
        os.makedirs(output_dir, exist_ok=True)

        verified_predictions = {}

        # Iterate through the predictions
        for page_id, prediction_list in predictions.items():
            verified_predictions[page_id] = []  
            for prediction in prediction_list:
                if os.path.basename(prediction['path']) in verified_images:
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
            save_question_diagram_links(page_predictions)

            # Debugging: Print the page predictions sent to the clustering function
            print(f"Page predictions sent for clustering: {page_predictions}")

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

#last route. (in process)
@app.route('/annotation_complete', methods=['GET'])
def annotation_complete():
    # Path to the JSON file containing the links
    json_file = "question_diagram_links.json"

    # Call the function to extract text and link diagrams
    try:
        extract_text_and_refine_equations(json_file)
    except Exception as e:
        print(f"Error during text extraction and linking: {e}")
        return "An error occurred during text extraction and linking. Please check the logs.", 500

    return "Annotation process completed successfully!"

#main app.py running.
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
    