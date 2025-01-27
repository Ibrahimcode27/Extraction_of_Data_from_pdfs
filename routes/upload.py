from flask import Blueprint, render_template, session

# Create a Blueprint for upload
bp = Blueprint('upload', __name__)

@bp.route('/')
def upload_page():
    """
    Renders the upload page. Clears session data to reset the workflow.
    """
    # Optional: Clear specific session keys if needed
    session.clear()
    session.pop('uploaded_pdf', None)
    return render_template('index.html')  # Renders the upload page

