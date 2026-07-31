import os
import sys

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.applications.resnet50 import preprocess_input

# Define constants


def extract_features(dataset, model):
    feature_extractor = tf.keras.models.Model(
        inputs=model.input,
        outputs=model.layers[-2].output
    )
    all_features = []
    all_labels = []

    for images, labels in dataset:
        # Preprocess images if required (ResNet50 uses preprocess_input)
        # print(images.shape)
        # print(images[0])
        preprocessed_images = preprocess_input(images)
        # print(preprocessed_images[0])
        # plt.imshow(preprocessed_images[0])
        # plt.show()
        features = feature_extractor.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


# Specify class names if known
print(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



# Split dataset into training and testing sets
train_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized3/Train"
val_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized3/Val"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 2
AUTOTUNE = tf.data.experimental.AUTOTUNE
class_names = ["CS_aug", "Healthy_aug"]

train_ds = image_dataset_from_directory(
    train_dir,
    seed=123,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

val_ds = image_dataset_from_directory(
    val_dir,
    seed=123,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

# Define the directories for the new folders
# output_base_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Organized"
# train_output_dir = os.path.join(output_base_dir, "train")
# val_output_dir = os.path.join(output_base_dir, "val")
#
# # Ensure output directories exist
# os.makedirs(train_output_dir, exist_ok=True)
# os.makedirs(val_output_dir, exist_ok=True)
#
# # Create subdirectories for each class in training and validation folders
# for class_name in class_names:
#     os.makedirs(os.path.join(train_output_dir, class_name), exist_ok=True)
#     os.makedirs(os.path.join(val_output_dir, class_name), exist_ok=True)
#
#
# # Function to copy images to their respective folders
# def organize_images(dataset, output_dir):
#     for images, labels in dataset:
#         for i in range(images.shape[0]):
#             # Get the image and its corresponding class name
#             img = images[i].numpy().astype("uint8")
#             label = labels[i].numpy()
#             class_name = class_names[label]
#
#             # Save the image to the appropriate class folder
#             class_folder = os.path.join(output_dir, class_name)
#             img_name = f"{len(os.listdir(class_folder)) + 1}.png"
#             img_path = os.path.join(class_folder, img_name)
#
#             # Save the image
#             tf.keras.preprocessing.image.save_img(img_path, img)
#
#
# # Organize training and validation datasets
# organize_images(train_ds, train_output_dir)
# organize_images(val_ds, val_output_dir)
#
# print("Images organized into train and validation folders successfully.")
#
# sys.exit(0)

# Configure dataset for performance
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
#
# Get Pretrained ResNet50 model
input_shape = (224, 224, 3)
base_model = tf.keras.models.load_model('./keypoints/keypoint_regressor3.keras')
base_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[-4].output)
base_model.summary()



x = base_model.output
x = tf.keras.layers.Dense(256, activation='relu', name='dense_256')(x)
x = tf.keras.layers.Dropout(0.5)(x)  # Regularization to reduce overfitting
x = tf.keras.layers.Dense(128, activation='relu', name='dense_128')(x)
x = tf.keras.layers.Dropout(0.3)(x)  # Regularization
x = tf.keras.layers.Dense(64, activation='relu', name='dense_64')(x)
classification_output = tf.keras.layers.Dense(2, activation='softmax', name="classification_output")(x)


model = tf.keras.models.Model(inputs=base_model.input, outputs=classification_output, name="classification_model")
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, mode='max',
                                              restore_best_weights=True)
# Train the model for 20 epochs

history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[early_stop])

model.save('cs8/cs8.h5')

model = tf.keras.models.load_model('cs8/cs8.h5')


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train_features, train_labels = extract_features(train_ds, model)
val_features, val_labels = extract_features(val_ds, model)

# Train an SVM on the extracted features
svm_classifier = SVC(kernel='linear', C=1)
print(train_features.shape)
svm_classifier.fit(train_features, train_labels)

# Make predictions on the validation data
val_predictions = svm_classifier.predict(val_features)
train_predictions = svm_classifier.predict(train_features)
# Evaluate the SVM classifier
accuracy = accuracy_score(val_labels, val_predictions)
print(f'Validation Accuracy: {accuracy:.4f}')

accuracy = accuracy_score(train_labels, train_predictions)
print(f'Training Accuracy: {accuracy:.4f}')
import joblib
#
# # # Save the model
joblib.dump(svm_classifier, 'documenting/cs8/svm_model8.pkl')

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#
# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()




