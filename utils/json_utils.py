import re
import json
import os
from flask import session
from utils.gemini_utils import extract_text_gemini

def clean_text(text):
    """
    Cleans a block of text by removing unwanted phrases, formatting, and misrecognized characters.
    Args:
        text (str): The text to clean.
    Returns:
        str: Cleaned text.
    """
    if not text:
        return ""

    # Extract the portion after "Extracted Text:" if it exists
    match = re.search(r"Extracted Text:\s*(.+)", text, flags=re.DOTALL)
    cleaned_text = match.group(1).strip() if match else text.strip()

    # Replace misrecognized characters (e.g., 'À' with 'Å')
    cleaned_text = cleaned_text.replace("À", "Å")

    # Remove any unwanted code block delimiters or extra whitespace
    cleaned_text = re.sub(r"```", "", cleaned_text).strip()

    return cleaned_text

def clean_options(options):
    """
    Cleans the options list to extract valid options in a consistent format.
    Args:
        options (str or list): Raw options text extracted from the image, either as a single string or a list of strings.
    Returns:
        list: Cleaned list of options.
    """
    cleaned_options = []

    # Ensure `options` is treated as a list of lines
    if isinstance(options, str):
        lines = options.split("\n")  # Split string into lines
    elif isinstance(options, list):
        lines = options  # Use the list directly
    else:
        raise ValueError("Unsupported input type for options. Expected str or list.")

    # Process each line to extract valid options
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        # Extract options in formats like "A. Text", "1. Text", "(1) Text", etc.
        match = re.match(r"(?:[A-D]\.|(?:\(\d+\)|\d+\.))\s*(.*)", line)
        if match:
            option_text = match.group(1).strip()
            # Retain mathematical symbols or special characters without modification
            cleaned_options.append(option_text)

    return cleaned_options

def clean_extracted_json(json_data):
    """
    Cleans the extracted JSON by removing unwanted text and formatting.
    Args:
        json_data (list): The extracted JSON data.
    Returns:
        list: Cleaned JSON data.
    """
    for entry in json_data:
        # Clean question text
        entry["question_text"] = clean_text(entry.get("question_text", ""))
        
        # Clean options text
        entry["options_text"] = clean_options(entry.get("options_text", []))
        
        # Clean solution text
        entry["solution_text"] = clean_text(entry.get("solution_text", ""))
        
        # Clean diagram text
        entry["diagram_text"] = clean_text(entry.get("diagram_text", ""))
    return json_data

def parse_extracted_text(text):
    """
    Parse the extracted text to handle both plain text and JSON-like outputs.
    Args:
        text (str): Extracted text.
    Returns:
        str: Cleaned and parsed text.
    """
    try:
        # Attempt to parse as JSON
        parsed = json.loads(text)
        return parsed.get("question", text)  # Default to original text if "question" key doesn't exist
    except json.JSONDecodeError:
        # Return plain text if it's not valid JSON
        return text

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
    used_diagrams = set()  # To ensure diagrams are linked to only one question

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

            # Find the closest diagram that is not already used
            linked_diagram = None
            min_distance = float('inf')
            for diagram in diagrams:
                diagram_id = (diagram['path'], diagram['x_min'], diagram['y_min'])
                if diagram_id in used_diagrams:  # Skip already used diagrams
                    continue

                if diagram['y_min'] >= question['y_min']:  # Diagram is below or overlaps with the question
                    distance = abs(diagram['y_min'] - question['y_max'])
                    if distance < min_distance:
                        min_distance = distance
                        linked_diagram = os.path.join(
                            output_dir, "Diagrams",
                            f"{os.path.basename(diagram['path'])}_{diagram['x_min']}_{diagram['y_min']}.png"
                        ).replace("\\", "/")
                        closest_diagram_id = diagram_id

            # Mark the diagram as used if linked
            if linked_diagram:
                used_diagrams.add(closest_diagram_id)

            # Find the closest option (ensure option is not already linked and the question is above the option)
            linked_option = None
            min_distance = float('inf')
            threshold = 50  # Allow small proximity overlap

            for option in options:
                option_id = (option['path'], option['x_min'], option['y_min'])  # Use a unique identifier
                if option['y_min'] >= question['y_max'] - threshold and option_id not in used_options:  # Relaxed condition
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

            print(f"Final linked option for question {question['path']}: {linked_option}")

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

def extract_text_and_save_to_json(json_file, output_json="questions_with_details.json"):
    """
    Extract text from images using Gemini API and save it to a JSON file.
    Args:
        json_file (str): Path to input JSON file containing image links.
        output_json (str): Path to output JSON file for saving extracted data.
    """
    try:
        # Load the input JSON file
        with open(json_file, "r") as f:
            question_diagram_links = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {json_file}: {e}")
        return

    # List to store extracted data
    extracted_data = []

    # Process each entry in the JSON file
    for entry in question_diagram_links:
        try:
            page = entry.get("page", "")
            question_path = entry.get("question", None)
            options_path = entry.get("options", None)
            solution_path = entry.get("solution", None)
            diagram_path = entry.get("diagram", None)
            pdf_filename = entry.get("pdf_filename", "unknown_pdf")

            # Extract and parse question text
            question_text = extract_text_gemini(question_path)
            question_text = parse_extracted_text(question_text) if question_text else ""

            # Extract and parse options text
            options_text = extract_text_gemini(options_path)
            print(f"Options extracted from {options_path}: {options_text}")  # Print options for debugging
            options_text = parse_extracted_text(options_text).split("\n") if options_text else []

            # Extract and parse solution text
            solution_text = extract_text_gemini(solution_path)
            print(f"Solution extracted from {solution_path}: {solution_text}")  # Print solution for debugging

            # Add extracted data to the list
            extracted_data.append({
                "page": page,
                "pdf_filename": pdf_filename,
                "question_text": question_text,
                "options_text": options_text,
                "quiz_image_path": question_path,
                "options_image_path": options_path,
                "solution_image_path": solution_path,
                "diagram_image_path": diagram_path,
                "solution_text": solution_text
            })

        except Exception as e:
            print(f"Error processing entry {entry}: {e}")

    try:
        # Clean the extracted data
        cleaned_data = clean_extracted_json(extracted_data)

        # Save the cleaned data to the output JSON file
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

        print(f"Cleaned and extracted data saved to {output_json}")
    except Exception as e:
        print(f"Error cleaning or saving data: {e}")

