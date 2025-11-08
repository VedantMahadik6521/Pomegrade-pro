import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# ========================
# CONFIGURATION
# ========================
IMG_SIZE = (224, 224)
MODEL_PATH = "fine_tuned_model.h5"   # your trained model path
CLASS_NAMES = ["healthy", "non_healthy"]  # Replace with your actual class names

# ========================
# FUNCTION TO PREDICT SINGLE IMAGE
# ========================
def predict_image(model, img_path):
    # Load image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    img_array = img_array / 255.0  # rescale

    # Predict
    preds = model.predict(img_array)
    
    if preds.shape[-1] == 1:  # binary sigmoid
        prob = preds[0][0]
        class_idx = int(prob > 0.5)
        confidence = prob if class_idx == 1 else 1 - prob
    else:  # multi-class softmax
        class_idx = np.argmax(preds[0])
        confidence = preds[0][class_idx]

    predicted_class = CLASS_NAMES[class_idx]
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")

    # Show image
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class} ({confidence:.2f})")
    plt.axis("off")
    plt.show()

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    # Load trained model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file '{MODEL_PATH}' not found! Make sure you trained and saved it first.")
        exit()

    model = load_model(MODEL_PATH)

    # Ask user for image path
    img_path = input("Enter the path of the image to test: ").strip()
    if not os.path.exists(img_path):
        print("❌ Image path does not exist!")
    else:
        predict_image(model, img_path)









































import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load your trained model
model = tf.keras.models.load_model("pomegranate_model.h5")

# Define your class labels in the same order as during training
# Example: ["Low Quality", "Medium Quality", "High Quality"]
class_labels = ["Low Quality", "Medium Quality", "High Quality"]

def predict_pomegranate(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to model input
    img_array = image.img_to_array(img) / 255.0             # Normalize
    img_array = np.expand_dims(img_array, axis=0)           # Add batch dimension

    # Get predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    return class_labels[predicted_class], confidence

if __name__ == "__main__":
    # Test with your uploaded image
    test_image = "8de98d57-5931-4a9c-a01b-c39489da8b2d.png"  # put your file path here
    label, confidence = predict_pomegranate(test_image)

    print(f"Prediction: {label} ({confidence:.2f}% confidence)")
