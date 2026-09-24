# Glossary

What this page covers:
This glossary defines technical terms, acronyms, and machine learning concepts used throughout the Farmer Eye documentation.
Each entry provides a simple, one-to-two sentence explanation.

---

## A

### Activation Function
A mathematical operation applied to the output of a neural network layer to help the model learn complex, non-linear relationships. In this project, Rectified Linear Unit (ReLU) and Softmax are the primary activation functions.

### Adam Optimizer
An algorithm used during neural network training that automatically adjusts the learning rate for each parameter based on historical gradients.

---

## B

### Base64
A binary-to-text encoding scheme that translates raw bytes (such as JPEG image data) into printable ASCII characters. This allows images to be transmitted inside text-based JSON network messages.

### Batch
A small group of images processed together by a neural network during a single training or inference step.

### Batch Normalization
A technique that standardizes the inputs of each layer inside a neural network during training. This stabilizes the learning process and speeds up model convergence.

---

## C

### Categorical Cross-Entropy
A loss function used in multi-class classification that measures the difference between the model's predicted probability distribution and the true one-hot class label.

### CNN (Convolutional Neural Network)
A deep learning architecture designed for image analysis that slides mathematical filters across pixel grids to recognize visual features like edges, textures, and lesion spots.

### Confidence Threshold
A cutoff percentage required before an automated system accepts a model prediction as valid. Predictions below this threshold are rejected or marked as uncertain.

### Confusion Matrix
A tabular grid comparing ground-truth class labels against model predictions. Rows represent actual categories while columns represent predicted categories.

### CSI (Camera Serial Interface)
A dedicated hardware ribbon cable connection on the Raspberry Pi used for direct, high-bandwidth communication between the camera sensor and the processor.

---

## D

### Dropout
A regularization method that randomly disables a percentage of neuron connections during training. This prevents the model from relying too heavily on specific features, which reduces overfitting.

---

## E

### Early Stopping
A training mechanism that monitors model performance on validation data and halts training when validation loss stops improving, restoring the best-performing weights.

### Epoch
One complete pass of the entire training dataset through the neural network during the optimization process.

---

## F

### F1-Score
The harmonic mean of precision and recall. It provides a balanced measurement of classification accuracy, especially when some classes have fewer samples than others.

### Float16 Quantization
A model compression method that converts 32-bit floating-point weight numbers into 16-bit floats. This reduces model file size by approximately half with minimal loss of accuracy.

### Frame
A single still image extracted from a continuous video stream.

---

## G

### Gradient
A vector of partial derivatives pointing in the direction of the steepest increase of a mathematical function. In deep learning, gradients guide how weights are updated to lower prediction error.

---

## H

### Holdout Test Set
A portion of the dataset set aside at the beginning of research and never seen by the model during training. It provides an unbiased final evaluation of accuracy.

### HSV Mask
A color-filtering method based on Hue, Saturation, and Value color channels. In this project, it isolates green plant foliage from background objects.

---

## I

### Inference
The process of feeding a new, unseen image into a trained neural network to generate a disease prediction.

---

## L

### L298N
A dual H-Bridge integrated circuit module used in robotics to control the speed and direction of direct current (DC) motors.

---

## M

### Macro Average
An evaluation metric calculated by computing the metric (such as precision or recall) independently for each class and then averaging those scores with equal weight.

---

## O

### Overfitting
A machine learning problem where a model memorizes the training data too closely, including image noise, and fails to generalize accurately to new, unseen images.

---

## P

### PiCamera2
The official Python library used on Raspberry Pi OS to configure and capture image frames from Camera Modules using the modern `libcamera` system architecture.

### Precision
The percentage of positive predictions for a specific class that were correct. High precision means the model rarely makes false-alarm predictions for that class.

---

## Q

### Quantization
The process of reducing the precision of numerical weights in a neural network (for example, from 32-bit floats to 8-bit integers) to decrease model size and speed up edge execution.

---

## R

### Recall
The percentage of actual positive instances of a class that the model detected correctly. High recall means the model rarely misses instances of that disease.

---

## S

### Saliency Map
A visual heatmap showing which pixels in an input image contributed most strongly to the neural network's classification decision.

### Softmax
A mathematical formula that takes an array of unconstrained numbers and transforms them into probabilities that sum to 1.0 (or 100%).

### Stratified Split
A dataset partitioning method that preserves the exact percentage of each class across the training, validation, and testing sets.

---

## T

### TensorFlow Lite (TFLite)
A lightweight software framework designed by Google to run trained machine learning models efficiently on resource-constrained mobile and embedded edge hardware.

---

## W

### Weighted Average
An evaluation metric calculated by weighting each class score by the number of true samples present in that class before computing the overall mean.

### WebSocket
A network protocol providing full-duplex, persistent communication channels over a single TCP connection between a client and a server.

---

## Next Steps

- Review the system architecture in [System Architecture](architecture.md).
- Follow a frame from camera to alert in [How It Works](how-it-works.md).
- Explore model benchmarks in [Model and Training](model.md).
