import torch.nn.functional as F
import torch
from torchvision import transforms
from PIL import Image
import cv2
import io
import numpy as np
import torchvision.models as models
import torch.nn as nn

def detect_faces(image_bytes):
    """
    Detect faces in the image using OpenCV and return a list of cropped faces as PIL images.
    
    Parameters:
    - image_bytes: Byte stream of the image
    
    Returns:
    - face_images: List of cropped face images as PIL images
    """
    # Convert image bytes to a NumPy array for OpenCV
    image_np = np.frombuffer(image_bytes, np.uint8)
    
    # Convert NumPy array to OpenCV image
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    
    # Load OpenCV's pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # If no face is detected, return an empty list
    if len(faces) == 0:
        print("No face detected")
        return []
    
    # Extract each face and convert to a PIL Image
    face_images = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        face_images.append(face_image)

    return face_images

def show_class_statistics(model, face_image, class_names, device=None):
    """
    Show the predicted probabilities or scores for each class for a given face image.
    
    Parameters:
    - model: Trained model
    - face_image: Cropped face as a PIL image
    - class_names: List of class names corresponding to the model's output
    - device: Device to run inference on, 'cuda' or 'cpu'
    
    Returns:
    - class_statistics: A dictionary with class names as keys and predicted scores as values
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model.to(device)
    model.eval()

    # Transformations to be applied to the face image before prediction
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Apply the transform to the face image
    face_image = transform(face_image).unsqueeze(0)  # Add batch dimension

    # Move face_image to the device
    face_image = face_image.to(device)
    
    # Make the prediction
    with torch.no_grad():
        outputs = model(face_image)

    # Apply softmax to get probabilities
    probabilities = F.softmax(outputs, dim=1)
    
    # Convert to list for easier handling
    probabilities = probabilities.cpu().numpy().flatten()
    
    # Create a dictionary of class statistics
    class_statistics = {int(value): probabilities[index] for index, value in enumerate(list(class_names.values()))}
    
    return class_statistics

def calculate_beauty_score(class_dict):
    """
    Calculate a beauty score based on a dictionary where keys are class numbers and values are probabilities.
    
    Parameters:
    - class_dict: A dictionary where the keys are class numbers and the values are probabilities
                  (e.g., {1: 0.0001, 2: 0.0253, 3: 0.4719, 4: 0.5027})
    
    Returns:
    - beauty_score: A float value between 0 and 100 representing the beauty score
    """
    # Ensure the dictionary is not empty
    assert len(class_dict) > 0, "The class dictionary must not be empty."
    
    # Sort the class_dict by class number to ensure weights are correct (if order matters)
    sorted_classes = sorted(class_dict.items())  # Sort by keys (class numbers)
    
    # Extract class weights and probabilities
    class_weights = [class_num for class_num, _ in sorted_classes]
    class_probabilities = [prob for _, prob in sorted_classes]
    
    # Calculate the weighted sum of the probabilities
    weighted_sum = sum(p * w for p, w in zip(class_probabilities, class_weights))
    
    # Normalize the score to be between 0 and 100
    max_weighted_sum = max(class_weights)
    beauty_score = (weighted_sum / max_weighted_sum) * 100
    
    return beauty_score

def load_model_weights(path='beauti_classificator.pth', device=None):
    """
    Load the model's weights from the specified path.
    
    Parameters:
    - path: The file path where the model weights are saved (default: 'model_weights.pth').
    - device: The device to load the model onto ('cuda' or 'cpu').
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(weights=False)

    # Only train the final layer
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

def get_beauti_scores(image_bytes, device=None):
    """
    Get the beauty scores for all faces detected in the image using a trained model.
    
    Parameters:
    - image_bytes: Byte stream of the image.
    - device: Device to use ('cuda' or 'cpu'). Default is auto-detect.
    
    Returns:
    - results: A list of tuples where each tuple contains the beauty score and the face image as bytes
    """
    model = load_model_weights()
    idx_to_class = {0: '1', 1: '2', 2: '3', 3: '4'}
    
    # Detect faces in the image
    face_images = detect_faces(image_bytes)
    
    # If no faces are found, return an empty result
    if not face_images:
        return "No face detected."
    
    results = []
    
    for face_image in face_images:
        # Calculate beauty score for each face
        result = show_class_statistics(model, face_image, idx_to_class, device)
        if result is None:
            continue
        
        beauty_score = calculate_beauty_score(result)
        
        # Convert the face image to bytes
        byte_io = io.BytesIO()
        face_image.save(byte_io, format='JPEG')
        face_bytes = byte_io.getvalue()
        
        # Append the beauty score and face bytes as a tuple to the results list
        results.append([beauty_score, face_bytes])
    
    return results
