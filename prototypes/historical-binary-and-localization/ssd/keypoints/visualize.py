import json
import matplotlib.pyplot as plt

with open("history_ssd5.json", "r") as f:
    loaded_history = json.load(f)

# Access data
train_loss = loaded_history['loss']
val_loss = loaded_history['val_loss']
train_acc = loaded_history['mAP']
val_acc = loaded_history['val_mAP']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# #
# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b', label='Training mAP')
plt.plot(epochs, val_acc, 'r', label='Validation mAP')
plt.title('Training and Validation mAP')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()