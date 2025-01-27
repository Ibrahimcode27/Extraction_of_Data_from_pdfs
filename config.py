import os

class Config:
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXTENSIONS = {'pdf'}  # Allowed file extensions
    GOOGLE_API_KEY = "AIzaSyAhUVXdf65Q0_S8BSY8x6CN8lnkFx0KH_g"
    SECRET_KEY = 'your_secret_key'  # Replace with your actual secret key
    
    DATABASE_CONFIG = {
        'host': 'auth-db982.hstgr.io',          # Your MySQL host
        'user': 'u273147311_admin',            # Your MySQL username
        'password': 'Code4bharat@123',         # Your MySQL password
        'database': 'u273147311_examportal'    # Your database name
    }
    CLOUDINARY_CONFIG = {
        'cloud_name': "dpdqxdlz5",
        'api_key': "956864746811318",
        'api_secret': "U8vyHhuTQ52KH2-xUsQQJWZdH0I"
    }
