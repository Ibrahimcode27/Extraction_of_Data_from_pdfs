# Extraction_of_Data_from_pdfs
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Annotate</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin: 0;
            padding: 0;
        }

        #canvas-container {
            position: relative;
            display: inline-block;
            border: 2px solid #ccc;
            margin-top: 20px;
        }

        #image {
            display: block;
            width: auto;
            height: auto;
        }

        #canvas {
            position: absolute;
            top: 0;
            left: 0;
        }

        #class-selector {
            margin-top: 20px;
            display: none;
        }

        .button {
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }

        .button-primary {
            background-color: #4CAF50;
            color: white;
        }

        .button-danger {
            background-color: #f44336;
            color: white;
        }

        .button-secondary {
            background-color: #008CBA;
            color: white;
        }
    </style>
</head>
<body>
    <h1>Annotate Image</h1>
    <div id="canvas-container">
        <img id="image" src="{{ image_path }}" alt="Image to Annotate">
        <canvas id="canvas"></canvas>
    </div>
    <div id="class-selector">
        <select id="class-dropdown">
            <option value="">Select Class</option>
            <option value="diagram">Diagram</option>
            <option value="quiz">Quiz</option>
            <option value="solution">Solution</option>
        </select>
        <button class="button button-primary" onclick="saveAnnotation()">Save</button>
        <button class="button button-danger" onclick="discardAnnotation()">Discard</button>
    </div>
    <form id="annotationForm" action="/annotate" method="POST">
        <input type="hidden" name="image_path" value="{{ image_path }}">
        <input type="hidden" id="annotations" name="annotations">
        <input type="hidden" id="class_names" name="class_names">
        <button class="button button-secondary" type="button" onclick="submitAnnotations()">Submit</button>
    </form>

    <script>
        const canvas = document.getElementById('canvas');
        const image = document.getElementById('image');
        const classSelector = document.getElementById('class-selector');
        const classDropdown = document.getElementById('class-dropdown');
        const annotationsInput = document.getElementById('annotations');
        const classNamesInput = document.getElementById('class_names');
        const ctx = canvas.getContext('2d');

        let annotations = [];
        let classNames = [];
        let startX, startY, isDrawing = false, currentAnnotation;

        // Set canvas size to match the original image dimensions
        function adjustCanvas() {
            canvas.width = image.naturalWidth;
            canvas.height = image.naturalHeight;
        }

        // Ensure canvas is ready after image load
        image.onload = adjustCanvas;

        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            startX = e.clientX - rect.left;
            startY = e.clientY - rect.top;
            isDrawing = true;
        });

        canvas.addEventListener('mousemove', (e) => {
            if (!isDrawing) return;

            const rect = canvas.getBoundingClientRect();
            const endX = e.clientX - rect.left;
            const endY = e.clientY - rect.top;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            annotations.forEach(([x1, y1, x2, y2]) => {
                ctx.strokeStyle = 'red';
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            });

            ctx.strokeStyle = 'blue';
            ctx.lineWidth = 2;
            ctx.strokeRect(startX, startY, endX - startX, endY - startY);
        });

        canvas.addEventListener('mouseup', (e) => {
            if (!isDrawing) return;

            const rect = canvas.getBoundingClientRect();
            const endX = e.clientX - rect.left;
            const endY = e.clientY - rect.top;

            currentAnnotation = [Math.round(startX), Math.round(startY), Math.round(endX), Math.round(endY)];
            annotations.push(currentAnnotation);

            classSelector.style.display = 'block';
            isDrawing = false;
        });

        function saveAnnotation() {
            const className = classDropdown.value;
            if (!className) {
                alert('Please select a class!');
                return;
            }

            classNames.push(className);
            classSelector.style.display = 'none';
            classDropdown.value = '';

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            annotations.forEach(([x1, y1, x2, y2]) => {
                ctx.strokeStyle = 'red';
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            });
        }

        function discardAnnotation() {
            if (annotations.length === 0) {
                alert('No annotation to discard!');
                return;
            }

            annotations.pop();
            classNames.pop();

            classSelector.style.display = 'none';
            classDropdown.value = '';

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            annotations.forEach(([x1, y1, x2, y2]) => {
                ctx.strokeStyle = 'red';
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            });
        }

        function submitAnnotations() {
            if (annotations.length === 0 || annotations.length !== classNames.length) {
                alert('Please complete all annotations with class assignments.');
                return;
            }

            annotationsInput.value = JSON.stringify(annotations);
            classNamesInput.value = JSON.stringify(classNames);
            document.getElementById('annotationForm').submit();
        }
    </script>
</body>
</html>