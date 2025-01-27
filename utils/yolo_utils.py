from ultralytics import YOLO
import cv2
import os 
from pdf2image import convert_from_path
import tempfile
import json

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

