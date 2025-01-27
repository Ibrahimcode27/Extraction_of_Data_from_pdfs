from flask import Blueprint, request, render_template, session
import os
import time
import json
import mysql.connector
from config import Config
from utils.file_utils import allowed_file

# Create a Blueprint for success route
bp = Blueprint('success', __name__)

@bp.route('/success', methods=['POST'])
def upload_file():
    """
    Handles file upload, saves metadata to the database, and prepares for further processing.
    """
    if 'file' not in request.files:
        return "No file part in request", 400

    file = request.files['file']

    if file.filename == '':
        return "No file selected", 400

    # Pass the allowed_extensions argument to allowed_file
    if file and allowed_file(file.filename, Config.ALLOWED_EXTENSIONS):
        # Generate a unique filename
        filename, ext = os.path.splitext(file.filename)
        unique_filename = f"{filename}_{int(time.time())}{ext}"  # Add timestamp
        file_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        file.save(file_path)

        # Get form data
        exam_type = request.form['exam_type']
        subject = request.form['subject']
        topic = request.form.get('topic', None)
        difficulty_level = request.form.get('difficulty_level')  # New field

        # Clear JSON file
        json_file = "question_diagram_links.json"
        try:
            if os.path.exists(json_file):
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=4)
                print(f"{json_file} cleared successfully.")
        except Exception as e:
            print(f"Error clearing JSON file: {e}")

        # Clear verified_crops directory
        verified_crops_dir = "./static/verified_crops"
        try:
            if os.path.exists(verified_crops_dir):
                for root, dirs, files in os.walk(verified_crops_dir):
                    for file in files:
                        os.remove(os.path.join(root, file))
                print(f"Cleared all images in {verified_crops_dir}.")
        except Exception as e:
            print(f"Error clearing verified_crops directory: {e}")

        # Save PDF metadata in the database
        try:
            conn = mysql.connector.connect(**Config.DATABASE_CONFIG)
            cursor = conn.cursor()

            # Insert PDF metadata into `pdfs` table
            cursor.execute('''
                INSERT INTO pdfs (filename, exam_type, subject, topic_tags, difficulty_level)
                VALUES (%s, %s, %s, %s, %s)
            ''', (unique_filename, exam_type, subject, topic, difficulty_level))

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

        # Render the success page
        return render_template(
            'success.html',
            filename=unique_filename,
            exam_type=exam_type,
            subject=subject,
            topic=topic,
            difficulty_level=difficulty_level  # Pass the new field to the template
        )
    else:
        return "Invalid file type", 400
    
    