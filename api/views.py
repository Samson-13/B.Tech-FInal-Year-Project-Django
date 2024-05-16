from rest_framework.response import Response
from rest_framework.decorators import api_view


from keras.models import load_model
import numpy as np
from PIL import Image
import io
from pathlib import Path

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score
import os

BASE_DIR = Path(__file__).resolve().parent.parent
# Load the trained model once when the server starts
MODEL_PATH = f'{BASE_DIR}/api/trainedModel.h5'

# MODEL_PATH = 'saved_model.h5'
model = load_model(MODEL_PATH)
class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y', 'Z']

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

def train_and_save_model(  epochs=10, batch_size=128):

    train = pd.read_csv(f'{BASE_DIR}/api/project_sign/project_train/project_train.csv')
    test = pd.read_csv(f'{BASE_DIR}/api/project_sign/project_test/project_test.csv')
    model_save_path = 'trainedModel.h5'
    # Load and preprocess data
    train_data = np.array(train , dtype = 'float32')
    test_data = np.array(test , dtype = 'float32')  
    class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y', 'Z']
    i = train.shape[0]
    fig1, ax1 = plt.subplots(figsize=(2, 2))
    j = train_data[2,1:]
    plt.imshow(j.reshape((28, 28)), cmap='gray')
    print("Label For The Image is:", class_names[int(train_data[11, 0])])


    i = random.randint(1, train.shape[0])
    fig1, ax1 = plt.subplots(figsize=(2, 2))
    plt.imshow(train_data[i, 1:].reshape((28, 28)), cmap='gray')
    print("Label For The Image is:", class_names[int(train_data[i, 0])])

    i = random.randint(1, train.shape[0])
    fig1, ax1 = plt.subplots(figsize=(2, 2))
    plt.imshow(train_data[i, 1:].reshape((28, 28)), cmap='gray')
    print("Label For The Image is:", class_names[int(train_data[i, 0])])

    fig = plt.figure(figsize=(18,18))
    ax1 = fig.add_subplot(221)
    train['label'].value_counts().plot(kind='bar',ax=ax1)
    ax1.set_ylabel('Count')
    ax1.set_title('label')

    X_train = train_data[:,1:]/255.
    X_test = test_data[:,1:]/255.

    y_train = train_data[:,0]
    y_train_category = to_categorical(y_train, num_classes=25)

    y_test = test_data[:,0]
    y_test_category = to_categorical(y_test, num_classes=25)

    X_train = X_train.reshape(X_train.shape[0],*(28,28,1))
    X_test = X_test.reshape(X_test.shape[0],*(28,28,1))
    model = Sequential()

    model.add(Conv2D(32,(3,3),input_shape = (28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128,activation = 'relu'))
    model.add(Dense(25, activation = 'softmax'))

    model.compile(loss= 'categorical_crossentropy',optimizer='adam', metrics = ['acc'])
    model.summary()

    history = model.fit(X_train , y_train_category , batch_size = 128 , epochs = 10 ,verbose =1,validation_data =(X_test ,y_test_category))
    model.save(model_save_path)

    # acc = history.history['acc']
    # val_acc = history.history['val_acc']

    # plt.plot(epochs,acc,'y',label='training acc')
    # plt.plot(epochs,val_acc,'r',label='validation acc')
    # plt.title('training and validation accuracy')
    # plt.xlabel('epochs')
    # plt.ylabel('accuracy')
    # plt.legend()
    # plt.show()

    # accuracy = accuracy_score(y_test, predicted_classes)
    # print('accuracy score =',accuracy)

    # i = random.randint(1,len(predicted_classes))
    # plt.imshow(X_test[i,:,:,0])
    # print("predicted label : ",class_names[int(predicted_classes[i])])
    # print("true label : ", class_names[int(y_test[i])])


    # # Make predictions
    # predictions = model.predict(X_test)

    # # Convert raw predictions to predicted classes
    # predicted_classes = np.argmax(predictions, axis=1)


    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1,len(loss)+1)
    # plt.plot(epochs,loss,'y',label='training loss')
    # plt.plot(epochs, val_loss,'r',label='validation loss')
    # plt.title('training and validation loss')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.show()

    # Check if the model already exists
    # if os.path.exists(model_save_path):
    #     model = load_model(model_save_path)
    #     print("Loaded existing model.")
    # else:
        # Build the model
        # model = Sequential()
        # model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))

        # model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))

        # model.add(Conv2D(128, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))

        # model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(25, activation='softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        # model.summary()

        # # Train the model
        # history = model.fit(X_train, y_train_category, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test_category))

        # # Save the model
        # model.save(model_save_path)
        # print(f"Model saved to {model_save_path}")

        # # Plot training and validation loss
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']
        # epochs_range = range(1, len(loss) + 1)
        # plt.plot(epochs_range, loss, 'y', label='training loss')
        # plt.plot(epochs_range, val_loss, 'r', label='validation loss')
        # plt.title('Training and Validation Loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()

        # # Plot training and validation accuracy
        # acc = history.history['acc']
        # val_acc = history.history['val_acc']
        # plt.plot(epochs_range, acc, 'y', label='training accuracy')
        # plt.plot(epochs_range, val_acc, 'r', label='validation accuracy')
        # plt.title('Training and Validation Accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # plt.show()

    # Evaluate the model
    # predictions = model.predict(X_test)
    # predicted_classes = np.argmax(predictions, axis=1)
    # accuracy = accuracy_score(y_test, predicted_classes)
    # print('Accuracy score =', accuracy)

    # # Display a random prediction
    # i = random.randint(0, len(predicted_classes) - 1)
    # plt.imshow(X_test[i, :, :, 0], cmap='gray')
    # plt.title(f"Predicted: {class_names[int(predicted_classes[i])]}, True: {class_names[int(y_test[i])]}")
    # plt.show()

    # return model

# Example usage:
# model = train_and_save_model('project_sign/project_train/project_train.csv', 'project_sign/project_test/project_test.csv', 'saved_model.h5', epochs=10, batch_size=128)
