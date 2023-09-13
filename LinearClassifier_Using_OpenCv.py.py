

# import os
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split as tts
# from sklearn.neighbors import KNeighborsClassifier as k
# from sklearn.metrics import accuracy_score
# from google.colab import drive
# from sklearn.svm import LinearSVC as LC

# im = []
# index = []

# for i, fo in enumerate((os.listdir(''))):
#     fp = os.path.join('', fo)
#     for fname in os.listdir(fp):
#        ip = os.path.join(fp, fname)
#        image = cv2.imread(ip)
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#        image = cv2.resize(image, (100, 100))
#        im.append(image)
#        index.append(i)
# im=np.array(im)
# index=np.array(index)
# im = im.astype('float32') / 255.0
# xt, xte, yt, yte = tts(im, index, test_size=0.38, random_state=35)
# lc = LC()
# lc.fit(xt.reshape(len(xt), -1),yt)
# yp = lc.predict(xte.reshape(len(xte), -1))
# ac = accuracy_score(yte, yp)
# print(ac)

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

# Function to load images and labels
def load_data(data_dir):
    images = []
    labels = []
    for label, folder in enumerate(sorted(os.listdir(data_dir))):
        folder_path = os.path.join(data_dir, folder)
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
            image = cv2.resize(image, (200, 200))  # Resize the image to a fixed size
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

# Path to the Flower Recognition dataset in Google Drive
data_dir = '/content/drive/MyDrive/flowers'
# Load data from Google Drive
images, labels = load_data(data_dir)

# Normalize pixel values to [0, 1]
images = images.astype('float32') / 255.0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.35, random_state=35)

# Function to compute multi-class SVM loss and gradient
def multi_class_svm_loss(W, X, y, reg_strength):
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    dW = np.zeros_like(W)

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # Margin for SVM loss
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    loss /= num_train
    dW /= num_train

    # Add regularization to the loss and gradient
    loss += 0.5 * reg_strength * np.sum(W * W)
    dW += reg_strength * W

    return loss, dW

# Initialize the weights matrix with small random values
num_classes = len(np.unique(labels))
W = 0.01 * np.random.randn(X_train.shape[1], num_classes)

# Hyperparameters
reg_strength = 1e-3
learning_rate = 1e-3
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    # Compute the loss and gradient using the multi_class_svm_loss function
    loss, dW = multi_class_svm_loss(W, X_train.reshape(len(X_train), -1), y_train, reg_strength)

    # Update the weights using gradient descent
    W -= learning_rate * dW

    # Print the loss for every few epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Make predictions on the test set
scores = X_test.reshape(len(X_test), -1).dot(W)
y_pred = np.argmax(scores, axis=1)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
