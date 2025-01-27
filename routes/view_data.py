from flask import Blueprint, render_template, redirect, url_for
import mysql.connector
from config import Config

# Create a Blueprint for view_data routes
bp = Blueprint('view_data', __name__)

@bp.route('/view_data')
def view_data():
    """
    Fetches and displays all rows from the `pdfs` table in the database.
    """
    try:
        # Connect to MySQL database
        conn = mysql.connector.connect(**Config.DATABASE_CONFIG)
        cursor = conn.cursor()

        # Execute the query to retrieve all rows from the pdfs table
        cursor.execute('SELECT * FROM pdfs')
        rows = cursor.fetchall()

        # Render the data in the view_data.html template
        return render_template('view_data.html', rows=rows)

    except mysql.connector.Error as err:
        print(f"Error fetching data: {err}")
        return "An error occurred while fetching data. Please try again.", 500

    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


@bp.route('/delete/<int:id>', methods=['POST'])
def delete_row(id):
    """
    Deletes a row from the `pdfs` table based on the provided ID.
    """
    try:
        # Connect to MySQL database
        conn = mysql.connector.connect(**Config.DATABASE_CONFIG)
        cursor = conn.cursor()

        # Execute the DELETE query
        cursor.execute('DELETE FROM pdfs WHERE id = %s', (id,))

        # Commit the transaction
        conn.commit()

        # Redirect to the data view page
        return redirect(url_for('view_data.view_data'))

    except mysql.connector.Error as err:
        print(f"Error deleting row: {err}")
        return "An error occurred while deleting the row.", 500

    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
