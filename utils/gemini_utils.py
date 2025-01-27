import google.generativeai as genai
import re
from PIL import Image
import time
import config

def extract_text_gemini(image_path):
    """
    Extract text from an image using Google's Generative AI.
    Args:
        image_path (str): Path to the image file.
    Returns:
        str: Extracted text or an error message.
    """
    try:
        # Initialize Gemini API
        api_key = config.Config.GOOGLE_API_KEY
        genai.configure(api_key=api_key)

        # Load the image
        image = Image.open(image_path)

        # Use the model for text extraction with a specific format
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([
            "Extract text from the given image in the following format:\n"
            "Extracted Text: -----\n"
            "If no text is present, return: Extracted Text: None\n",
            image
        ])
        response.resolve()  # Ensure the response is fully resolved

        # Extract and return the text
        extracted_text = response.text.strip()
        print(f"Extracted text from {image_path}: {extracted_text}")  # Print the extracted text for debugging
        return extracted_text
    except Exception as e:
        print(f"Error extracting text: {e}")
        time.sleep(1)  # Add a delay before the next API call
        return None

def identify_correct_option(question_text, options_text):
    """
    Use the Gemini API to identify the correct option and provide a detailed solution.
    """
    try:
        # Combine question and options into a structured prompt
        prompt = f"""
        Question: {question_text}
        Options:
        """
        for i, option in enumerate(options_text, 1):
            prompt += f"{i}. {option}\n"
        prompt += "Identify the correct option and provide a detailed solution. Respond in the following format:\n"
        prompt += "Correct Option: <option number>\n"
        prompt += "Solution:\n<solution text>"

        # Use Gemini API to generate a response
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        answer = response.text.strip()

        # Log the API response
        print(f"Gemini API Response:\n{answer}")

        # Extract correct option and solution
        match = re.search(r"Correct Option:\s*(\d+)", answer, re.IGNORECASE)
        correct_index = int(match.group(1)) - 1 if match else None

        solution_match = re.search(r"Solution:\s*(.+)", answer, re.IGNORECASE | re.DOTALL)
        solution_text = solution_match.group(1).strip() if solution_match else ""

        if correct_index is not None and 0 <= correct_index < len(options_text):
            print(f"Correct Option Index: {correct_index + 1}")
            print(f"Selected Answer: {options_text[correct_index]}")
            print(f"Solution: {solution_text}")
            return correct_index, solution_text

    except Exception as e:
        print(f"Error identifying correct option: {e}")

    return None, None

