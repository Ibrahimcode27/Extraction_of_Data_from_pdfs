import mysql.connector
from config import Config

def init_db():
    conn = mysql.connector.connect(Config.DATABASE_CONFIG)  # Ensure conn is defined even if connection fails
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
            difficulty_level VARCHAR(50) NOT NULL,  -- Ensure NOT NULL for foreign key references
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

        # Create the Diagrams table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagrams (
            id INT AUTO_INCREMENT PRIMARY KEY,
            diagram_path VARCHAR(255) NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Create the Questions table without subject, exam_type, difficulty_level fields
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            pdf_id INT NOT NULL,
            question_text TEXT,
            diagram_id INT,
            topic_id INT,
            FOREIGN KEY (pdf_id) REFERENCES pdfs (id) ON DELETE CASCADE,
            FOREIGN KEY (diagram_id) REFERENCES diagrams (id) ON DELETE SET NULL,
            FOREIGN KEY (topic_id) REFERENCES topics (id) ON DELETE SET NULL
        )
        ''')

        # Create the Options table with the is_correct field
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS options (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question_id INT NOT NULL,
            option_text TEXT NOT NULL,
            is_correct BOOLEAN NOT NULL DEFAULT FALSE,
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

        conn.commit()
        print("Database initialized successfully.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        # Ensure conn is checked before attempting to close it
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

