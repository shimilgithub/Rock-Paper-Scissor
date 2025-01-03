'''rock-paper-scissor program. Test image is passed as command line argument.'''

#import packages
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse


def load_image(file_path):
    '''function to load thetst image using the path mentioned during compilation'''
    image = Image.open(file_path).convert("RGB")
    return image

def init():
    ''' Initializes global settings for NumPy.'''
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

def load_my_model():
    '''function to load the model and labels'''
    # Load the model
    model = load_model("model/keras_model.h5")

    # Load the labels
    class_names = open("model/labels.txt", "r").readlines()

    return model, class_names

def prep_input(image):
    '''Preprocesses the input image to prepare it for model prediction.'''

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    return data


def predict(model, class_names, data):
    '''function to predict the class and confidence score'''
    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)

def display_image(image):
    '''Function to display the test image'''
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def main():
    '''Main function that ties everything together.It handles argument parsing, loads the image, makes a prediction, and displays the image.'''
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Rock-paper-scissors image classifier.")
    parser.add_argument('--image-path', type=str, required=True, help='Path to the image to be classified.')
    
    # Parse the arguments from the command line
    args = parser.parse_args()

    # Initialize settings
    init()

    # Load the image from the provided file path
    image = load_image(args.image_path)

    # Load the pre-trained model and class labels
    model, class_names = load_my_model()

    # Preprocess the image
    data = prep_input(image)

    # Make a prediction
    # class_name, confidence_score = predict(model, class_names, data)
    predict(model, class_names, data)

    # Display the image with matplotlib
    display_image(image)

if __name__ == "__main__":
    # call main function
    main()
