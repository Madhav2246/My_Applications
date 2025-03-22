import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import logging

# Set up logging for better debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the emotions (labels) based on your folder structure
emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
emotion_to_label = {emotion: idx for idx, emotion in enumerate(emotions)}

# Function to load images from a directory with error handling
def load_images_from_folder(base_path, folder_type="train"):
    images = []
    labels = []
    
    folder_path = os.path.join(base_path, folder_type)
    if not os.path.exists(folder_path):
        logger.error(f"Directory {folder_path} does not exist. Please check the path.")
        raise FileNotFoundError(f"Directory {folder_path} does not exist.")

    for emotion in emotions:
        emotion_path = os.path.join(folder_path, emotion)
        if not os.path.exists(emotion_path):
            logger.warning(f"Emotion directory {emotion_path} does not exist. Skipping...")
            continue
        
        logger.info(f"Loading images from {emotion_path}...")
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                if img is None:
                    logger.warning(f"Failed to load {img_path}. Skipping...")
                    continue
                
                # Resize to 48x48 (standard for FER-2013)
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(emotion_to_label[emotion])
            except Exception as e:
                logger.warning(f"Error loading {img_path}: {e}. Skipping...")
                continue
    
    if not images:
        logger.error("No valid images were loaded. Please check the dataset.")
        raise ValueError("No valid images were loaded from the dataset.")
    
    return np.array(images), np.array(labels)

# Main training function with error handling
def train_emotion_model():
    try:
        # Define paths
        base_path = "e:/ACM/My_Applications/AdvancedFaceBiometricSystem/emotion"  # Update this to your main folder path
        save_path = "e:/ACM/My_Applications/AdvancedFaceBiometricSystem/emotion_model.h5"

        # Load train and test data
        logger.info("Loading training data...")
        X_train, y_train = load_images_from_folder(base_path, "train")
        logger.info(f"Loaded {len(X_train)} training samples.")

        logger.info("Loading test data...")
        X_test, y_test = load_images_from_folder(base_path, "test")
        logger.info(f"Loaded {len(X_test)} test samples.")

        # Preprocess the data
        logger.info("Preprocessing data...")
        X_train = X_train / 255.0  # Normalize to [0, 1]
        X_test = X_test / 255.0
        X_train = X_train.reshape(-1, 48, 48, 1)  # Add channel dimension
        X_test = X_test.reshape(-1, 48, 48, 1)

        # Build the CNN model
        logger.info("Building the CNN model...")
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),  # Add dropout to prevent overfitting
            layers.Dense(len(emotions), activation='softmax')  # Output layer for 7 emotions
        ])

        # Compile the model
        logger.info("Compiling the model...")
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        logger.info("Training the model...")
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

        # Save the model
        logger.info(f"Saving the model to {save_path}...")
        model.save(save_path)
        logger.info("Model saved successfully as 'emotion_model.h5'")

    except FileNotFoundError as e:
        logger.error(f"File or directory error: {e}")
        raise
    except ValueError as e:
        logger.error(f"Data error: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}")
        raise

if __name__ == "__main__":
    try:
        train_emotion_model()
    except Exception as e:
        logger.critical(f"Training failed: {e}")
        exit(1)