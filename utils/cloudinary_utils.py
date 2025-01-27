import cloudinary.uploader

def upload_to_cloudinary(file_path, folder="default_folder"):
    result = cloudinary.uploader.upload(file_path, folder=folder)
    return result.get("secure_url")
