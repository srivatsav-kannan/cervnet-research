import sys

import numpy as np
from transformers import AutoImageProcessor, TFViTForImageClassification, TFAutoModelForImageClassification, DefaultDataCollator, create_optimizer
import tensorflow as tf
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

dataset = load_dataset("imagefolder", data_dir="/Users/srivatsavkannan/Datasets/CS", split="train")


print(type(dataset))
dataset = dataset.train_test_split(test_size=0.2)
print(dataset["train"][0])

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

size = (image_processor.size["height"], image_processor.size["width"])

train_data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomCrop(size[0], size[1]),
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(factor=0.02),
        tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="train_data_augmentation",
)

val_data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.CenterCrop(size[0], size[1]),
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ],
    name="val_data_augmentation",
)
def convert_to_tf_tensor(image: Image):
    np_image = np.array(image)
    tf_image = tf.convert_to_tensor(np_image)
    # `expand_dims()` is used to add a batch dimension since
    # the TF augmentation layers operates on batched inputs.
    return tf.expand_dims(tf_image, 0)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    images = [
        train_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    images = [
        val_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch

dataset["train"].set_transform(preprocess_train)
dataset["test"].set_transform(preprocess_val)

data_collator = DefaultDataCollator(return_tensors="tf")

batch_size = 16
num_epochs = 5
num_train_steps = len(dataset["train"]) * num_epochs
learning_rate = 3e-5
weight_decay_rate = 0.01

optimizer, lr_schedule = create_optimizer(
    init_lr=learning_rate,
    num_train_steps=num_train_steps,
    weight_decay_rate=weight_decay_rate,
    num_warmup_steps=0,
)

tf_train_dataset = dataset["train"].to_tf_dataset(
    columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
)

# converting our test dataset to tf.data.Dataset
tf_eval_dataset = dataset["test"].to_tf_dataset(
    columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
)



model = TFAutoModelForImageClassification.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=1)

model.save('model', save_mode='tf')

image = Image.open('image.JPG')
inputs = image_processor(image, return_tensors="tf")
outputs = model(**inputs, output_attentions=True)

logits = outputs.logits
answer = int(tf.math.argmax(logits, axis=-1)[0])

attention_weights = outputs.attentions  # This will be a list of tensors, one for each layer

print(attention_weights)
# Choose a layer for visualization
layer_index = 0  # Choose any layer you want to visualize

# Choose the head for visualization
head_index = 0  # Choose any head you want to visualize

# Plot the attention weights
plt.figure(figsize=(12, 8))
plt.imshow(attention_weights[layer_index][0][head_index], cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Attention Map for Layer {} Head {}'.format(layer_index, head_index))
plt.show()

print(answer)

