﻿# Extraction_of_Data_from_pdfs

## Uploading PDF 
Using Flask and taking upload of files. 

## Converting to IMAGES. 
Using pdf2image library to convert every page to image. 

## Yolo Model Prediction.
Using Yolov8 we predict the location of Questions, Options, Diagrams and Solutions. We store predictions in /static/temp_files and the predicted images with bounding boxes of diffrent clasess are stored in /static/output_with_boxes folder. these output images are displayed in verify route of flask app. 

## Cropping and Storing relevent images. 
We save the relevent images in /verified_crops folder. 
in this folder we store all images as per there class names. like /verified_crops/Questions etc.......

## Extraction of text using Gemini API. 
We extract text from gemini and Save the output to file /questions_with_details.json file which then can be pushed to database.

## Database storage. 
We store data in Xampp server for now. 
phpMyAdmin app.

