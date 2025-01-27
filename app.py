from flask import Flask
from config import Config
from utils.db_utils import init_db
from routes.upload import bp as upload_bp
from routes.success import bp as success_bp
from routes.process import bp as process_bp
from routes.verify import bp as verify_bp
from routes.annotate import bp as annotate_bp
from routes.annotation_complete import bp as annotation_complete_bp
from routes.delete_annotation_row import bp as delete_bp
from routes.view_data import bp as view_data_bp
from routes.move_to_database import bp as move_to_database_bp

app = Flask(__name__)
app.config.from_object(Config)

# Register Blueprints
app.register_blueprint(upload_bp, url_prefix='/')
app.register_blueprint(success_bp, url_prefix='/')
app.register_blueprint(process_bp, url_prefix='/')
app.register_blueprint(verify_bp, url_prefix='/')
app.register_blueprint(annotate_bp, url_prefix='/')
app.register_blueprint(annotation_complete_bp, url_prefix='/')
app.register_blueprint(delete_bp, url_prefix='/')
app.register_blueprint(view_data_bp, url_prefix='/')
app.register_blueprint(move_to_database_bp, url_prefix='/')

if __name__ == "__main__":
    app.run(debug=True)
    init_db()
