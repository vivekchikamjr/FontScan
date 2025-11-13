from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from font_recognition import predict_font  # Your prediction function

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return render_template('index.html', 
                         error="File is too large. Maximum size is 10MB."), 413

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return render_template('index.html', 
                                     error="No file selected. Please choose an image file.")
            
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', 
                                     error="No file selected. Please choose an image file.")
            
            if not allowed_file(file.filename):
                return render_template('index.html', 
                                     error="Invalid file type. Please upload a PNG, JPG, JPEG, or WEBP image.")
            
            filename = secure_filename(file.filename)
            if not filename:
                return render_template('index.html', 
                                     error="Invalid filename. Please use a valid filename.")
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save file
            file.save(filepath)
            
            # Verify file was saved
            if not os.path.exists(filepath):
                return render_template('index.html', 
                                     error="Failed to save file. Please try again.")
            
            # Get prediction with error handling
            try:
                prediction = predict_font(filepath)
            except Exception as e:
                # Clean up uploaded file on error
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                return render_template('index.html', 
                                     error=f"Error analyzing image: {str(e)}. Please try with a different image.")
            
            return render_template('index.html', 
                                 prediction=prediction,
                                 uploaded_image=filename)
        
        except Exception as e:
            return render_template('index.html', 
                                 error=f"An unexpected error occurred: {str(e)}. Please try again.")
    
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")