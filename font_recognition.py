import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer

# Configure TensorFlow to use less memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Limit GPU memory growth (if GPU is available)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Define a custom InputLayer to handle 'batch_shape' parameter
class CustomInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        # Extract 'batch_shape' if present and convert to 'batch_input_shape'
        batch_shape = kwargs.pop('batch_shape', None)
        if batch_shape is not None:
            kwargs['batch_input_shape'] = batch_shape
        super().__init__(*args, **kwargs)

# Load the model with the custom InputLayer (lazy loading)
_model = None

def get_model():
    """Lazy load the model to reduce initial memory usage"""
    global _model
    if _model is None:
        try:
            model_path = 'model/font_classifier_model.h5'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            _model = load_model(model_path, custom_objects={'InputLayer': CustomInputLayer})
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return _model


# Load your pre-trained model
# model = load_model('model/font_classifier_model.h5') 

# Create your font label mapping
font_labels = {
    0: 'Agency',
    1: 'Akzidenz Grotesk',
    2: 'Algerian',
    3: 'Arial',
    4: 'Baskerville',
    5: 'Bell MT',
    6: 'Bembo',
    7: 'Bodoni',
    8: 'Book Antiqua',
    9: 'Brandish',
    10: 'Calibry',
    11: 'Californian FB',
    12: 'Calligraphy',
    13: 'Calvin',
    14: 'Cambria',
    15: 'Candara',
    16: 'Century',
    17: 'Comic Sans MS',
    18: 'Consolas',
    19: 'Corbel',
    20: 'Courier',
    21: 'Didot',
    22: 'Elephant',
    23: 'Fascinate',
    24: 'Franklin Gothic',
    25: 'Futigre',
    26: 'Futura',
    27: 'Garamond',
    28: 'Georgia',
    29: 'Gill Sans',
    30: 'Helvetica',
    31: 'Hombre',
    32: 'Lato',
    33: 'LCD Mono',
    34: 'Lucida Bright',
    35: 'Monotype Corsiva',
    36: 'Mrs Eaves',
    37: 'Myriad',
    38: 'Nasalization',
    39: 'News Gothic',
    40: 'Palatino linotype',
    41: 'Papyrus',
    42: 'Perpetua',
    43: 'Rockwell',
    44: 'Segoe UI',
    45: 'Tahoma',
    46: 'Times New Roman',
    47: 'Verdana'
}

def test_mappings():
    print("\nFont Class Mappings:")
    for idx, name in sorted(font_labels.items()):
        print(f"Class {idx}: {name}")

# Call the test function when the file loads
test_mappings()

def preprocess_image(image_path):
    """Preprocess image for model prediction with error handling"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))  # Critical dimension change
        img = img / 255.0
        img = img.astype(np.float32)  # Ensure correct dtype
        
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

def predict_font(image_path):
    """Predict font from image with error handling and memory optimization"""
    try:
        # Get model (lazy loading)
        model = get_model()
        
        # Preprocess image
        processed_img = preprocess_image(image_path)
        
        # Use verbose=0 to reduce output and batch_size=1 for memory efficiency
        predictions = model.predict(processed_img, verbose=0, batch_size=1)
        
        # Get predicted class
        predicted_class = np.argmax(predictions[0])
        font_name = font_labels[int(predicted_class)]
        
        # Clean up memory (let garbage collector handle the rest)
        del processed_img
        del predictions
        
        return font_name
    except Exception as e:
        print(f"Error predicting font: {e}")
        raise ValueError(f"Failed to predict font: {str(e)}")
