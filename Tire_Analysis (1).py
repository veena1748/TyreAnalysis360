#!/usr/bin/env python
# coding: utf-8

# MID REVIEW MODELLING
# 

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os

# Specify the destination folder name and path
new_folder_name = 'Tyre_Condition_Prediction'  # Change this to your desired folder name
destination_folder_path = f'/content/drive/My Drive/{new_folder_name}'

# Create the new folder
os.makedirs(destination_folder_path, exist_ok=True)
print(f"Folder '{destination_folder_path}' created.")


# In[ ]:


import os

# Specify the output directory name and path
output_dir = '/content/drive/My Drive/Tyre_Condition_Splits'  # Desired output directory

# Create the output directory
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory '{output_dir}' created.")


# In[ ]:


get_ipython().system('pip install split-folders')


# In[ ]:


import splitfolders

# Specify the input directory where your dataset is located
input_dir = '/content/drive/My Drive/Tyre_Condition_Prediction'  # Update this to your dataset path

# Use splitfolders to split the dataset
splitfolders.ratio(input_dir, output=output_dir, seed=1337, ratio=(.8, 0.1, 0.1))
# (train:val:test)

print("Dataset split into train, validation, and test sets.")


# In[ ]:


import tensorflow as tf


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/drive/My Drive/Tyre_Condition_Splits/train",
    seed=123,
    image_size=(128, 128),
    batch_size=64
)

# Load testing dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/drive/My Drive/Tyre_Condition_Splits/test",
    seed=123,
    image_size=(128, 128),
    batch_size=64
)

# Load validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/drive/My Drive/Tyre_Condition_Splits/val",
    seed=123,
    image_size=(128, 128),
    batch_size=64
)

print("Datasets loaded successfully.")

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Adjust output layer as per your task
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save the model
model.save('/content/drive/My Drive/Tyre_Condition_Splits/tyre_condition_model.h5')
print("Model saved successfully.")


# In[ ]:


# Get and print class names
class_names = train_ds.class_names
print("Class names:", class_names)


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as img

# Create subplots for displaying images
fig, ax = plt.subplots(ncols=3, figsize=(20, 5))
fig.suptitle('Tyre Category')

# Load images from the correct paths in Google Drive
class_1_img = img.imread('/content/drive/My Drive/Tyre_Condition_Splits/train/class_1/NewNormal100.jpg')  # Update path
class_2_img = img.imread('/content/drive/My Drive/Tyre_Condition_Splits/train/class_2/Cracked-103.jpg')  # Update path
class_3_img = img.imread('/content/drive/My Drive/Tyre_Condition_Splits/train/class_3/Cracked-10.jpg')  # Update path

# Set titles for each subplot
ax[0].set_title('class_1')
ax[1].set_title('class_2')
ax[2].set_title('class_3')

# Display the images
ax[0].imshow(class_1_img)
ax[1].imshow(class_2_img)
ax[2].imshow(class_3_img)

# Show the plot
plt.show()


# In[ ]:


import tensorflow as tf
from tensorflow import keras

# Define the model
model = keras.models.Sequential()

# Rescale the input
model.add(keras.layers.Rescaling(1./255, input_shape=(128, 128, 3)))  # Change to 3 for RGB images

# First convolutional block
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.BatchNormalization())

# Second convolutional block
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.BatchNormalization())

# Dropout layer
model.add(keras.layers.Dropout(0.20))

# Third convolutional block
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.BatchNormalization())

# Dropout layer
model.add(keras.layers.Dropout(0.20))

# Fourth convolutional block
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.BatchNormalization())

# Dropout layer
model.add(keras.layers.Dropout(0.25))

# Fifth convolutional block
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.BatchNormalization())

# Dropout layer
model.add(keras.layers.Dropout(0.25))

# Flatten the output
model.add(keras.layers.Flatten())

# Fully connected layers
model.add(keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dense(3, activation="softmax"))  # Adjust number of classes

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' if labels are integers
              metrics=['accuracy'])

# Display the model summary
model.summary()


# In[ ]:


# Compile the model
model.compile(
    loss="sparse_categorical_crossentropy",  # Use this if your labels are integers
    optimizer=keras.optimizers.RMSprop(learning_rate=0.001),  # Specify learning rate
    metrics=["accuracy"]
)


# In[ ]:


# Train the model
hist = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    verbose=1  # Set to 1 to see training progress, or set to 0 to suppress output
)


# In[ ]:


model.summary()


# In[ ]:


get_ac = hist.history['accuracy']
get_los = hist.history['loss']
val_acc = hist.history['val_accuracy']
val_loss = hist.history['val_loss']


# In[ ]:


epochs = range(len(get_ac))
plt.plot(epochs, get_ac, 'g', label='Accuracy of Training data')
plt.plot(epochs, get_los, 'r', label='Loss of Training data')
plt.title('Training data accuracy and loss')
plt.legend(loc=0)
plt.figure()




# In[ ]:


plt.plot(epochs, get_ac, 'g', label='Accuracy of Training Data')
plt.plot(epochs, val_acc, 'r', label='Accuracy of Validation Data')
plt.title('Training and Validation Accuracy')
plt.legend(loc=0)
plt.figure()



# In[ ]:


plt.plot(epochs, get_los, 'g', label='Loss of Training Data')
plt.plot(epochs, val_loss, 'r', label='Loss of Validation Data')
plt.title('Training and Validation Loss')
plt.legend(loc=0)
plt.figure()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Step 1: Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Step 2: Make predictions on the test dataset
predictions = model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)  # Get the predicted class indices

# Step 3: Visualize predictions
plt.figure(figsize=(12, 12))
for images, labels in test_ds.take(1):  # Take one batch of images and labels
    for i in range(16):  # Display 16 images
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        # Get the true label and predicted label
        true_label = labels[i].numpy()
        predicted_label = predicted_classes[i]

        # Check if prediction is correct
        if true_label == predicted_label:
            plt.title(f"Actual: {class_names[true_label]}\nPredicted: {class_names[predicted_label]}", fontdict={'color':'green'})
        else:
            plt.title(f"Actual: {class_names[true_label]}\nPredicted: {class_names[predicted_label]}", fontdict={'color':'red'})

        plt.axis('off')  # Hide axes
plt.tight_layout()
plt.show()


# In[ ]:


import numpy as np
import tensorflow as tf

# Step 1: Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Step 2: Make predictions on the test dataset
predictions = model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)  # Get the predicted class indices

# Step 3: Calculate MAPE
true_classes = np.concatenate([y.numpy() for x, y in test_ds])  # Flatten true labels

# Filter out cases where true_classes is zero to avoid division by zero
valid_indices = true_classes != 0
mape = np.mean(np.abs((true_classes[valid_indices] - predicted_classes[valid_indices]) / true_classes[valid_indices])) * 100  # Calculate MAPE

# Step 4: Print MAPE
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Make predictions
predictions = model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)

# Step 2: Get true classes
true_classes = np.concatenate([y.numpy() for x, y in test_ds])  # Flatten true labels

# Step 3: Generate a classification report
report = classification_report(true_classes, predicted_classes, target_names=class_names)
print("Classification Report:\n", report)

# Step 4: Generate a confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:\n", conf_matrix)

# Optional: Plotting the confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:


# Final Output Summary
print("=== Model Performance Summary ===")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)




# In[ ]:


import numpy as np

# Predicting on the test dataset
predictions = model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)  # Get class indices

# Function to get tyre lifetime and maintenance suggestions based on predicted class
def get_tyres_lifetime_and_maintenance(predicted_class):
    lifetime_mapping = {
        0: {
            "lifetime": "5 years",  # class_1 (New Tyres)
            "maintenance": "Inspect every 6 months for wear and tear. Ensure proper inflation levels and alignment."
        },
        1: {
            "lifetime": "3 years",  # class_2 (Cracked Tyres)
            "maintenance": "Inspect monthly for cracks and damage. Consider rotating tyres to even out wear. Replace if cracks deepen."
        },
        2: {
            "lifetime": "1 year",  # class_3 (Severely Cracked Tyres)
            "maintenance": "Replace immediately to avoid safety risks. Avoid driving until the replacement is done."
        }
    }
    return lifetime_mapping.get(predicted_class, {"lifetime": "Unknown Class", "maintenance": "No suggestions available"})

# Get estimated lifetimes and maintenance suggestions
results = [get_tyres_lifetime_and_maintenance(cls) for cls in predicted_classes]

# Displaying the results
for i in range(len(predicted_classes)):
    print(f"Predicted Class: {predicted_classes[i]}")
    print(f"Estimated Lifetime: {results[i]['lifetime']}")
    print(f"Maintenance Suggestion: {results[i]['maintenance']}")
    print("-" * 50)

