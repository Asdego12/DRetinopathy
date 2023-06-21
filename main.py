import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
from sklearn.preprocessing import OneHotEncoder


encoder = OneHotEncoder()  # Encoder is used to translate an image into an array.
encoder.fit([[0], [1], [2], [3]])

data = []  # Data array- Contains images.
pathsN = []  # Path array Normal
pathsM = []  # Path array Mild
pathsO = []  # Path array Moderate
pathsS = []  # Path array Severe


label = []  # Label array.

# Training 1:
for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Retinopathy/Data/output-normal'):
    for file in f:
        pathsN.append(os.path.join(r, file))  # Full path.


# Image pre-processing, using an encoder to transform its data to array.
for path in pathsN:
    img = Image.open(path)
    img = img.resize((256, 256))
    img = np.array(img)
    if img.shape == (256, 256, 3):
        data.append(np.array(img))
        label.append(encoder.transform([[0]]).toarray())
print(len(label))
pathsN = []  # Path array


for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Retinopathy/Data/output-MildDR'):
    for file in f:
        pathsM.append(os.path.join(r, file))


for path in pathsM:
    img = Image.open(path)
    img = img.resize((256, 256))
    img = np.array(img)
    if img.shape == (256, 256, 3):
        data.append(np.array(img))
    label.append(encoder.transform([[1]]).toarray())
print(len(label))
pathsM = []

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Retinopathy/Data/output-ModerateDR'):
    for file in f:
        pathsO.append(os.path.join(r, file))


for path in pathsO:
    img = Image.open(path)
    img = img.resize((256, 256))
    img = np.array(img)
    if img.shape == (256, 256, 3):
        data.append(np.array(img))
    label.append(encoder.transform([[2]]).toarray())
print(len(label))
pathsO = []

for r, d, f in os.walk(r'C:/Users/Diego/Desktop/Retinopathy/Data/output-SevereDR'):
    for file in f:
        pathsS.append(os.path.join(r, file))


for path in pathsS:
    img = Image.open(path)
    img = img.resize((256, 256))
    img = np.array(img)
    if img.shape == (256, 256, 3):
        data.append(np.array(img))
    label.append(encoder.transform([[3]]).toarray())
print(len(label))
pathsS = []


# Filling data array.
data = np.array(data)
var = data.shape

# Filling label array.
label = np.array(label)
label = label.reshape(40986, 4)  # Returns an array with the same data, to the new shape (Classification).


# Separates the Test from Train data.
x_Train, x_Test, y_Train, y_Test = train_test_split(data, label, test_size=0.1, shuffle=True, random_state=0)


# Separates the Validation from Train data.
x_Train, x_Val, y_Train, y_Val = train_test_split(x_Train, y_Train, test_size=0.25, shuffle=True, random_state=0)

# Saving Test data
np.save('TestData', x_Test)
np.save('TestData2', y_Test)



# Creating a CNN model
model = Sequential()

# INPUT
model.add(Conv2D(16, kernel_size=(3, 3), input_shape=(256, 256, 3), padding='Same'))

# 1st CONV
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='Same'))
model.add(BatchNormalization())  # Normalizes data in batches
model.add(MaxPooling2D(pool_size=(2, 2)))  # Down-samples the data

# 2nd CONV
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 3rd CONV
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(4, activation='softmax'))

# Model settings.
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='Adam', metrics=['acc'])
history = model.fit(x_Train, y_Train, epochs=12, batch_size=32, verbose=1, validation_data=(x_Val, y_Val))



# Figure1
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Figure 1')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

# Figure2
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.title('Figure 2')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.plot(range(12), acc, label='train')
plt.plot(range(12), val_acc, label='test')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

# Saves the model results.
model.save('Retinopathy_Model')

# Labels
labels = ["Normal", "Mild-DR", "Moderate-DR", "Severe-DR"]
prediction = model.predict(x_Test)


# Figure3
figure = plt.figure(figsize=(12, 10))

for i, index in enumerate(np.random.choice(x_Test.shape[0], size=12, replace=False)):
    ax = figure.add_subplot(3, 6, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_Test[index]))
    predict_index = np.argmax(prediction[index])
    true_index = np.argmax(y_Test[index])
    ax.set_title("{}".format(labels[predict_index]), color=("green" if predict_index == true_index else "red"))

plt.show()