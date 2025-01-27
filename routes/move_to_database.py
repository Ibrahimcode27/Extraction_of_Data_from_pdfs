from flask import Blueprint, request, redirect, url_for
import json
import mysql.connector
import os
from config import Config
from utils.gemini_utils import identify_correct_option

# Create a Blueprint for move_to_database
bp = Blueprint('move_to_database', __name__)

@bp.route('/move_to_database', methods=['POST'])
def move_to_database():
    output_json = request.form.get('output_json')
    if not output_json or not os.path.isfile(output_json):
        return "JSON file not found. Please try again.", 400

    conn = None
    try:
        # Load the verified data from the JSON file
        with open(output_json, "r", encoding="utf-8") as f:
            verified_data = json.load(f)

        if not isinstance(verified_data, list):
            raise ValueError("The JSON data must be a list of entries.")

        # Connect to the database
        conn = mysql.connector.connect(**Config.DATABASE_CONFIG)
        cursor = conn.cursor()

        for entry in verified_data:
            if not isinstance(entry, dict):
                print(f"Invalid entry: {entry}. Skipping...")
                continue

            # Retrieve and validate expected fields
            pdf_filename = entry.get("pdf_filename")
            question_text = entry.get("question_text")
            options_text = entry.get("options_text", [])
            solution_text = entry.get("solution_text", "")
            diagram_path = entry.get("diagram_image_path", "")
            topic_name = entry.get("topic", None)

            if not pdf_filename or not question_text:
                print(f"Incomplete entry: {entry}. Skipping...")
                continue

            # Query PDF metadata to fetch pdf_id, exam_type, and subject
            cursor.execute('SELECT id, exam_type, subject FROM pdfs WHERE filename = %s', (pdf_filename,))
            pdf_result = cursor.fetchone()
            cursor.nextset()  # Clear any unread results
            if not pdf_result:
                print(f"PDF metadata not found for filename: {pdf_filename}. Skipping entry.")
                continue

            pdf_id, exam_type, subject = pdf_result

            # Ensure topic exists and get its ID
            topic_id = None
            if topic_name:
                cursor.execute(
                    'SELECT id FROM topics WHERE exam_type = %s AND subject = %s AND topic_name = %s',
                    (exam_type, subject, topic_name)
                )
                topic_result = cursor.fetchone()
                cursor.nextset()  # Clear unread results
                if not topic_result:
                    cursor.execute(
                        'INSERT INTO topics (exam_type, subject, topic_name) VALUES (%s, %s, %s)',
                        (exam_type, subject, topic_name)
                    )
                    topic_id = cursor.lastrowid
                else:
                    topic_id = topic_result[0]

            # Check if the question_text already exists in the database
            cursor.execute('SELECT id FROM questions WHERE question_text = %s', (question_text,))
            existing_question = cursor.fetchone()
            cursor.nextset()  # Clear unread results
            if existing_question:
                print(f"Duplicate question found: {question_text}. Skipping insertion.")
                continue

            # Insert question data
            cursor.execute(
                '''
                INSERT INTO questions (pdf_id, question_text, diagram_id, topic_id)
                VALUES (%s, %s, %s, %s)
                ''',
                (pdf_id, question_text, None, topic_id)
            )
            question_id = cursor.lastrowid

            # Use Gemini API to identify the correct option and fetch the explanation
            correct_option_index, gemini_solution = identify_correct_option(question_text, options_text)

            # Insert options with is_correct field
            for idx, option in enumerate(options_text):
                is_correct = (idx == correct_option_index)
                cursor.execute(
                    'INSERT INTO options (question_id, option_text, is_correct) VALUES (%s, %s, %s)',
                    (question_id, option, is_correct)
                )

            # Insert solution (if available) or the detailed explanation from the Gemini API
            solution_to_save = solution_text if solution_text else gemini_solution
            if solution_to_save:
                cursor.execute(
                    'INSERT INTO solutions (question_id, solution_text) VALUES (%s, %s)',
                    (question_id, solution_to_save)
                )

        conn.commit()
        print("Data moved to the database successfully.")

    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
        return "A database error occurred. Please check the logs.", 500
    except Exception as e:
        print(f"Error moving data to the database: {e}")
        return "An error occurred while moving data to the database. Please check the logs.", 500
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

    # Adjust the route name as per your application's structure
    return redirect(url_for('data.view_data'))
