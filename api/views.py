from rest_framework.response import Response
from rest_framework.decorators import api_view


from keras.models import load_model
import numpy as np
from PIL import Image
from pathlib import Path

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout



BASE_DIR = Path(__file__).resolve().parent.parent
# Load the trained model once when the server starts
MODEL_PATH = f'{BASE_DIR}/api/trainedModel.h5'

# MODEL_PATH = 'saved_model.h5'
model = load_model(MODEL_PATH)
class_names = [
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
               'U', 'V', 'W', 'X', 'Y', 'Z']

def preprocess_image(image):
    """
    Preprocess the image to match the input format of the model.
    """
    # Convert the image to grayscale
    image = image.convert('L')
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert the image to a numpy array and normalize it
    image_array = np.array(image) / 255.0
    # Reshape the array to (28, 28, 1)
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

@api_view(['GET','POST'])
def mymodel_list(request):
    if request.method == 'GET':
        test = {'data':'testing data','metadataasdfasdf':'asdfasdfasdfasd'}
        return Response(test)
    elif request.method == 'POST':
        image_file = request.FILES.get('image')
        if image_file:
            # Read the image file
            image = Image.open(image_file)
            # Preprocess the image
            preprocessed_image = preprocess_image(image)
            # Make a prediction using the model
            prediction = model.predict(preprocessed_image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = class_names[predicted_class]
            return Response({'predicted_label': predicted_label}, status=201)
        else:
            return Response({'error': 'No image file provided'}, status=400)
        
@api_view(['GET', 'POST'])
def train_and_save_model(request, epochs=10, batch_size=128):
    # Load the datasets
    newtrain = pd.read_csv(f'{BASE_DIR}/api/new_all/output_dataset.csv')
    train = pd.read_csv(f'{BASE_DIR}/api/project_sign/project_train/project_train.csv')
    test = pd.read_csv(f'{BASE_DIR}/api/project_sign/project_test/project_test.csv')
    model_save_path = f'{BASE_DIR}/api/trainedModel.h5'

    # Convert data to numpy arrays
    newtrain_data = np.array(newtrain, dtype='float32')
    train_data = np.array(train, dtype='float32')
    test_data = np.array(test, dtype='float32')

    # Extract class labels
    y_train = train_data[:, 0]
    y_test = test_data[:, 0]
    new_train = newtrain_data[:, 0]

    # Find the maximum class label in each dataset
    max_label_train = int(y_train.max())
    max_label_test = int(y_test.max())
    max_label_new_train = int(new_train.max())

    # Determine the maximum class label overall
    max_label = max(max_label_train, max_label_test, max_label_new_train)
    print(f"Maximum class label: {max_label}")

    # Set num_classes to be the maximum label value plus one
    num_classes = max_label + 1
    print(f"Number of classes: {num_classes}")

    # Preprocess data for training
    X_train = train_data[:, 1:] / 255.0
    X_test = test_data[:, 1:] / 255.0

    # Reshape input data
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # Convert labels to categorical
    y_train_category = to_categorical(y_train, num_classes=num_classes)
    y_test_category = to_categorical(y_test, num_classes=num_classes)
    new_train_category = to_categorical(new_train, num_classes=num_classes)

    # Define the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()

    history = model.fit(
        X_train, y_train_category,
        batch_size=batch_size, epochs=epochs, verbose=1,
        validation_data=(X_test, y_test_category)
    )

    model.save(model_save_path)
    return Response({'status': "Completed"})


    





    