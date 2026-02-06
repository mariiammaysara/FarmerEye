# Load the necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from tensorflow.keras import backend as K

# Define the root directory containing the image dataset organized by class folders.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(BASE_DIR, 'data', 'plantvillage dataset', 'color')

# List the class folders (representing different plant diseases/conditions).
class_folders = os.listdir(data_dir)
print("Dataset Classes:")
for cls in class_folders:
    print(cls)

# Collect all image file paths and their corresponding labels (class folder names).
image_paths = []
labels = []
for class_folder in class_folders:
    class_path = os.path.join(data_dir, class_folder)
    image_files = os.listdir(class_path)
    for image_file in image_files:
        image_path = os.path.join(class_path, image_file)
        image_paths.append(image_path)
        labels.append(class_folder)

# Function to validate image files, ensuring they can be opened and are not corrupted.
def validate_images(image_paths):
    """
    Validate images and return a list of valid image paths.
    """
    valid_paths = []
    corrupted_paths = []
    
    for img_path in image_paths:
        try:
            # Try to open the image with PIL
            with Image.open(img_path) as img:
                img.verify()  # Verify it's actually an image
                # Try to load it with cv2 as well
                cv_img = cv2.imread(img_path)
                if cv_img is not None:
                    valid_paths.append(img_path)
                else:
                    corrupted_paths.append(img_path)
        except Exception as e:
            print(f"Corrupted or invalid image found: {img_path}")
            print(f"Error: {str(e)}")
            corrupted_paths.append(img_path)
    
    print(f"Total images checked: {len(image_paths)}")
    print(f"Valid images: {len(valid_paths)}")
    print(f"Corrupted images: {len(corrupted_paths)}")
    
    if corrupted_paths:
        print("\nCorrupted image paths:")
        for path in corrupted_paths:
            print(path)
    
    return valid_paths

# Validate images before creating the DataFrame
print("Validating images...")
image_paths = validate_images(image_paths)

# Create DataFrame with only valid images
df = pd.DataFrame({'image_path': image_paths, 'label': [os.path.basename(os.path.dirname(path)) for path in image_paths]})
df.head()
df.shape
print("The classes:\n", np.unique(df['label']))

# Analyze and visualize the distribution of images per class.
class_counts = df['label'].value_counts()
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=class_counts.values, y=class_counts.index, orient='h')
plt.title('Class Distribution')
plt.xlabel('Number of Images')
plt.ylabel('Plant Types')
plt.tight_layout() 
for i, v in enumerate(class_counts.values):
    ax.text(v + 5, i, str(v), color='black', va='center')
plt.show()

# Display sample images from each class
num_classes = len(df['label'].unique())
num_images_per_row = 3
num_rows = (num_classes + num_images_per_row - 1) // num_images_per_row
plt.figure(figsize=(15, 5 * num_rows))  
for i, plant_class in enumerate(df['label'].unique()):
    plt.subplot(num_rows, num_images_per_row, i + 1)
    image_path = os.path.join(data_dir, df[df['label'] == plant_class]['image_path'].iloc[0])
    if os.path.exists(image_path):
        sample_image = cv2.imread(image_path)
        if sample_image is not None:
            plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
            plt.title(plant_class)
            plt.axis('off')
        else:
            print(f"Error: Unable to load image from path: {image_path}")
    else:
        print(f"Error: Image path does not exist: {image_path}")
plt.tight_layout()
plt.show()

# Convert string labels to numerical indices for model training.
class_labels_dict = {class_label: idx for idx, class_label in enumerate(np.unique(df['label']))}
df['label'] = df['label'].map(class_labels_dict)

# Split the data into training, validation, and test sets. Stratify to maintain class proportions.
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])
train_df.shape, val_df.shape, test_df.shape

# Convert numerical labels back to strings for ImageDataGenerator compatibility.
train_df['label'] = train_df['label'].astype(str)
val_df['label'] = val_df['label'].astype(str)
test_df['label'] = test_df['label'].astype(str)
print(train_df['label'].unique())

# Define ImageDataGenerators for training (with augmentation), validation, and testing (only rescaling).
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators that flow data from the DataFrames.
train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='image_path', y_col='label', target_size=(224, 224),
    batch_size=32, class_mode='categorical', shuffle=True, seed=42
)
val_generator = val_datagen.flow_from_dataframe(
    val_df, x_col='image_path', y_col='label', target_size=(224, 224),
    batch_size=32, class_mode='categorical', shuffle=False
)
test_generator = test_datagen.flow_from_dataframe(
    test_df, x_col='image_path', y_col='label', target_size=(224, 224),
    batch_size=32, class_mode='categorical', shuffle=False
)
print(f'Training samples: {train_generator.samples}, Validation samples: {val_generator.samples}, Test samples: {test_generator.samples}')

# Helper function to display a batch of images from a generator.
def show_images(image_gen):
    class_dict = image_gen.class_indices
    classes = list(class_dict.keys())
    images, labels = next(image_gen)
    plt.figure(figsize=(20, 20))
    num_images = min(len(labels), 25)
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        image = images[i]
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name, color="green", fontsize=12)
        plt.axis('off')
    plt.show()

# Show example augmented images from the training generator.
show_images(train_generator)

# Define model parameters.
input_shape = (224, 224, 3)
n_classes = len(train_generator.class_indices)

# Build the Convolutional Neural Network (CNN) model architecture.
model = keras.Sequential([
    # Block 1
    keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(3, 3), padding='valid', use_bias=False, input_shape=input_shape),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(pool_size=(3, 3)),

    # Block 2
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    # Block 3
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    # Block 4
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    # Block 5
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    # Classifier Head
    keras.layers.Flatten(),
    keras.layers.Dense(256, use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(n_classes, activation='softmax')
])

# Compile the model with loss function, optimizer, and metrics.
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.optimizers.Adam(),
    metrics=['accuracy']
)

# model summary.
model.summary()

# Define callbacks for early stopping and learning rate reduction during training.
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Train the model using the prepared data generators and callbacks.
history = model.fit(
    train_generator, 
    batch_size=32,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the trained model on the unseen test dataset.
scores = model.evaluate(test_generator)
print(f"Initial Test Loss: {scores[0]}, Initial Test Accuracy: {scores[1]}")

# Make predictions on the test set to analyze performance.
test_predictions = model.predict(test_generator)
test_predicted_labels = np.argmax(test_predictions, axis=1)
test_true_labels = test_generator.classes

# Identify and visualize some misclassified images.
error_df = pd.DataFrame({'True Label': test_true_labels, 'Predicted Label': test_predicted_labels})
misclassified_indices = error_df[error_df['True Label'] != error_df['Predicted Label']].index
class_names_list = list(test_generator.class_indices.keys())
plt.figure(figsize=(15, 15))
for i, idx in enumerate(misclassified_indices[:9]): # Show first 9 errors
    img_path = test_df.iloc[idx]['image_path']
    img = keras_image.load_img(img_path, target_size=(224, 224))
    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    true_label_name = class_names_list[test_true_labels[idx]]
    pred_label_name = class_names_list[test_predicted_labels[idx]]
    plt.title(f'True: {true_label_name}\nPred: {pred_label_name}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# confusion matrix.
conf_matrix = confusion_matrix(test_true_labels, test_predicted_labels)
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_list, yticklabels=class_names_list)
plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Confusion Matrix')
plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
plt.show()

# classification report (precision, recall, F1-score).
print(classification_report(test_true_labels, test_predicted_labels, target_names=class_names_list))

# Plot learning rate changes during training
plt.figure(figsize=(10, 5))

# Get learning rates from history if available
if 'lr' in history.history:
    plt.plot(history.history['lr'], label='Learning Rate')
else:
    # If learning rate history is not available, just plot the initial learning rate
    try:
        initial_lr = float(model.optimizer.learning_rate.numpy())
        plt.axhline(y=initial_lr, color='r', linestyle='-', label='Initial Learning Rate')
    except:
        print("Could not retrieve learning rate information")

plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')  # Use log scale for better visualization
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot training metrics over time with learning rate
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy During Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot training progress summary
plt.figure(figsize=(15, 10))
metrics = ['loss', 'accuracy']
colors = ['b', 'r']
for i, metric in enumerate(metrics):
    plt.subplot(2, 1, i+1)
    
    # Plot training metric
    plt.plot(history.history[metric], 
             color=colors[i], label=f'Training {metric}', 
             marker='o', markersize=4)
    
    # Plot validation metric if available
    if f'val_{metric}' in history.history:
        plt.plot(history.history[f'val_{metric}'], 
                color=colors[i], label=f'Validation {metric}', 
                linestyle='--', marker='s', markersize=4)
    
    plt.title(f'Model {metric.capitalize()} Over Time')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Plot per-class accuracy
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
plt.figure(figsize=(15, 8))
sns.barplot(x=class_accuracies, y=class_names_list)
plt.title('Per-Class Accuracy')
plt.xlabel('Accuracy')
plt.tight_layout()
plt.show()

# Plot prediction confidence distribution
prediction_confidences = np.max(test_predictions, axis=1)
plt.figure(figsize=(10, 6))
sns.histplot(data=prediction_confidences, bins=50)
plt.title('Distribution of Prediction Confidences')
plt.xlabel('Confidence Score')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Plot top-k accuracy for different k values
k_values = [1, 2, 3, 4, 5]
top_k_accuracies = []
for k in k_values:
    top_k_correct = 0
    for i in range(len(test_predictions)):
        top_k_classes = np.argsort(test_predictions[i])[-k:]
        if test_true_labels[i] in top_k_classes:
            top_k_correct += 1
    top_k_accuracies.append(top_k_correct / len(test_predictions))

plt.figure(figsize=(8, 6))
plt.plot(k_values, top_k_accuracies, marker='o')
plt.title('Top-K Accuracy')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot misclassification matrix
misclass_matrix = conf_matrix.copy()
np.fill_diagonal(misclass_matrix, 0)  # Remove correct classifications
plt.figure(figsize=(15, 12))
sns.heatmap(misclass_matrix, annot=True, fmt='d', cmap='Reds', 
            xticklabels=class_names_list, yticklabels=class_names_list)
plt.title('Misclassification Matrix\n(excluding correct classifications)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Plot class distribution in training, validation and test sets
plt.figure(figsize=(15, 6))
train_dist = pd.Series(train_generator.classes).value_counts().sort_index()
val_dist = pd.Series(val_generator.classes).value_counts().sort_index()
test_dist = pd.Series(test_generator.classes).value_counts().sort_index()

plt.subplot(1, 3, 1)
train_dist.plot(kind='bar')
plt.title('Training Set Distribution')
plt.xlabel('Class Index')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
val_dist.plot(kind='bar')
plt.title('Validation Set Distribution')
plt.xlabel('Class Index')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
test_dist.plot(kind='bar')
plt.title('Test Set Distribution')
plt.xlabel('Class Index')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot precision-recall curve for each class
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# Binarize the labels for multi-class precision-recall curve
y_test_bin = label_binarize(test_true_labels, classes=range(n_classes))

plt.figure(figsize=(15, 10))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(
        y_test_bin[:, i],
        test_predictions[:, i]
    )
    avg_precision = average_precision_score(
        y_test_bin[:, i],
        test_predictions[:, i]
    )
    
    # Only plot if there are positive samples for this class
    if np.sum(y_test_bin[:, i]) > 0:
        plt.plot(recall, precision, lw=2, 
                label=f'{class_names_list[i]} (AP = {avg_precision:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for Each Class')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot correlation matrix between class predictions
prediction_correlations = np.corrcoef(test_predictions.T)
plt.figure(figsize=(15, 12))
sns.heatmap(prediction_correlations, cmap='coolwarm', center=0,
            xticklabels=class_names_list, yticklabels=class_names_list)
plt.title('Correlation Matrix of Class Predictions')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Plot model's uncertainty analysis
prediction_entropies = -np.sum(test_predictions * np.log2(test_predictions + 1e-10), axis=1)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=prediction_entropies, bins=50)
plt.title('Distribution of Prediction Entropy\n(Higher = More Uncertain)')
plt.xlabel('Entropy')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
correct_predictions = test_predicted_labels == test_true_labels
sns.boxplot(x=correct_predictions, y=prediction_entropies)
plt.title('Prediction Entropy vs Correctness')
plt.xlabel('Prediction Correct')
plt.ylabel('Entropy')
plt.tight_layout()
plt.show()

# Plot confusion matrix with percentages
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
plt.figure(figsize=(15, 12))
sns.heatmap(conf_matrix_percent, annot=True, fmt='.1f', cmap='YlOrRd',
            xticklabels=class_names_list, yticklabels=class_names_list)
plt.title('Confusion Matrix (Percentages)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Plot top misclassification pairs
misclass_pairs = []
for i in range(len(class_names_list)):
    for j in range(len(class_names_list)):
        if i != j:
            misclass_pairs.append({
                'True': class_names_list[i],
                'Predicted': class_names_list[j],
                'Count': conf_matrix[i, j]
            })

misclass_df = pd.DataFrame(misclass_pairs)
top_misclass = misclass_df.nlargest(10, 'Count')
plt.figure(figsize=(12, 6))
sns.barplot(data=top_misclass, x='Count', y='True', hue='Predicted')
plt.title('Top 10 Misclassification Pairs')
plt.xlabel('Number of Occurrences')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Analyze model complexity and parameters
total_params = model.count_params()
trainable_params = np.sum([np.prod(w.shape) for w in model.trainable_weights])
non_trainable_params = np.sum([np.prod(w.shape) for w in model.non_trainable_weights])

print("\nModel Complexity Analysis:")
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Non-trainable Parameters: {non_trainable_params:,}")

# Plot layer-wise parameter distribution with more details
layer_info = []
for layer in model.layers:
    params = layer.count_params()
    if params > 0:  # Only include layers with parameters
        layer_info.append({
            'name': layer.name,
            'type': layer.__class__.__name__,
            'params': params,
            'output_shape': str(layer.output_shape)
        })

# Create DataFrame for better visualization
layer_df = pd.DataFrame(layer_info)

# Plot parameter distribution
plt.figure(figsize=(15, 8))
ax = sns.barplot(data=layer_df, x='params', y='name')
plt.title('Layer-wise Parameter Distribution')
plt.xlabel('Number of Parameters (log scale)')
plt.ylabel('Layer Name')

# Add parameter count annotations
for i, v in enumerate(layer_df['params']):
    ax.text(v, i, f' {v:,}', va='center')

plt.xscale('log')  # Use log scale for better visualization
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.show()

# Plot layer types distribution
plt.figure(figsize=(10, 6))
layer_types = layer_df['type'].value_counts()
sns.barplot(x=layer_types.values, y=layer_types.index)
plt.title('Distribution of Layer Types')
plt.xlabel('Count')
plt.ylabel('Layer Type')
plt.tight_layout()
plt.show()

# Print detailed layer information
print("\nDetailed Layer Information:")
print(layer_df.to_string(index=False))

# Analyze prediction time
import time
batch_size = 32
n_samples = len(test_generator.labels)
prediction_times = []

for i in range(0, n_samples, batch_size):
    batch_images = next(test_generator)[0][:batch_size]
    start_time = time.time()
    _ = model.predict(batch_images)
    end_time = time.time()
    prediction_times.append(end_time - start_time)

avg_prediction_time = np.mean(prediction_times)
std_prediction_time = np.std(prediction_times)

print(f"\nPrediction Time Analysis:")
print(f"Average prediction time per batch: {avg_prediction_time:.4f} seconds")
print(f"Standard deviation: {std_prediction_time:.4f} seconds")
print(f"Average prediction time per image: {(avg_prediction_time/batch_size)*1000:.2f} ms")

# Plot prediction time distribution
plt.figure(figsize=(10, 6))
sns.histplot(prediction_times, bins=30)
plt.title('Distribution of Prediction Times per Batch')
plt.xlabel('Time (seconds)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Analyze model robustness to input variations
def apply_noise(image, noise_factor=0.1):
    noisy_image = image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=image.shape)
    return np.clip(noisy_image, 0., 1.)

# Test model with different noise levels
noise_levels = [0.0, 0.1, 0.2, 0.3]
noise_accuracies = []

for noise in noise_levels:
    noisy_predictions = []
    original_batch = next(test_generator)[0][:10]
    
    noisy_batch = np.array([apply_noise(img, noise) for img in original_batch])
    predictions = model.predict(noisy_batch)
    accuracy = np.mean(np.argmax(predictions, axis=1) == test_true_labels[:10])
    noise_accuracies.append(accuracy)

plt.figure(figsize=(10, 6))
plt.plot(noise_levels, noise_accuracies, marker='o')
plt.title('Model Robustness to Noise')
plt.xlabel('Noise Level')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize feature maps for a sample image
def visualize_feature_maps(model, image, layer_name):
    layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    feature_maps = layer_model.predict(np.expand_dims(image, axis=0))
    
    n_features = min(16, feature_maps.shape[-1])  # Show up to 16 features
    size = int(np.ceil(np.sqrt(n_features)))
    
    plt.figure(figsize=(12, 12))
    for i in range(n_features):
        plt.subplot(size, size, i + 1)
        plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
        plt.axis('off')
    plt.suptitle(f'Feature Maps from {layer_name}')
    plt.tight_layout()
    plt.show()

# Visualize feature maps for a sample image
sample_image = next(test_generator)[0][0]
conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name.lower()]
if conv_layers:
    visualize_feature_maps(model, sample_image, conv_layers[0])

# Add saliency map visualization
def compute_saliency_map(model, image, class_idx):
    image_tensor = tf.convert_to_tensor(np.expand_dims(image, axis=0))
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        predictions = model(image_tensor)
        loss = predictions[:, class_idx]
    gradients = tape.gradient(loss, image_tensor)
    saliency_map = tf.reduce_max(tf.abs(gradients), axis=-1)
    return saliency_map[0].numpy()

# Visualize saliency maps for correctly classified images
plt.figure(figsize=(15, 5))
correct_indices = np.where(test_predicted_labels == test_true_labels)[0][:3]
for i, idx in enumerate(correct_indices):
    img = next(test_generator)[0][idx]
    pred_class = test_predicted_labels[idx]
    
    saliency = compute_saliency_map(model, img, pred_class)
    
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    plt.title(f'Original\n{class_names_list[pred_class]}')
    plt.axis('off')
    
    plt.subplot(2, 3, i+4)
    plt.imshow(saliency, cmap='hot')
    plt.title('Saliency Map')
    plt.axis('off')
plt.tight_layout()
plt.show()

# --- Optional Fine-Tuning Phase ---
print("\n--- Starting Fine-Tuning ---")
for layer in model.layers: 
    layer.trainable = True

learning_rate_finetune = 0.00001
model.compile(optimizer=Adam(learning_rate=learning_rate_finetune),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
model.summary()

epochs_finetune = 10
history_finetune = model.fit(
    train_generator,
    epochs=epochs_finetune, 
    initial_epoch=history.epoch[-1], 
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr]
)
print("--- Fine-Tuning Finished ---")

# Evaluate the model again after fine-tuning.
test_loss_ft, test_accuracy_ft = model.evaluate(test_generator)
print(f"Fine-Tuned Test Loss: {test_loss_ft}, Fine-Tuned Test Accuracy: {test_accuracy_ft}")

# Plot accuracy and loss curves for the fine-tuning phase.
plt.figure(figsize=(12, 5))

# Get the available metrics from history
available_metrics = history_finetune.history.keys()

# Plot training curves if metrics are available
if 'accuracy' in available_metrics or 'acc' in available_metrics:
    accuracy_key = 'accuracy' if 'accuracy' in available_metrics else 'acc'
    val_accuracy_key = 'val_' + accuracy_key
    
    plt.subplot(1, 2, 1)
    plt.plot(history_finetune.history[accuracy_key], label='Training Accuracy (FT)')
    if val_accuracy_key in available_metrics:
        plt.plot(history_finetune.history[val_accuracy_key], label='Validation Accuracy (FT)')
    plt.title('Fine-Tuning Accuracy')
    plt.xlabel('Epochs (FT)')
    plt.ylabel('Accuracy')
    plt.legend()

if 'loss' in available_metrics:
    plt.subplot(1, 2, 2)
    plt.plot(history_finetune.history['loss'], label='Training Loss (FT)')
    if 'val_loss' in available_metrics:
        plt.plot(history_finetune.history['val_loss'], label='Validation Loss (FT)')
    plt.title('Fine-Tuning Loss')
    plt.xlabel('Epochs (FT)')
    plt.ylabel('Loss')
    plt.legend()

plt.tight_layout()
plt.show()

# --- Post Fine-Tuning Evaluation ---
test_predictions_ft = model.predict(test_generator)
test_predicted_labels_ft = np.argmax(test_predictions_ft, axis=1)
conf_matrix_ft = confusion_matrix(test_true_labels, test_predicted_labels_ft)

# Display the normalized confusion matrix.
try:
    normalized_conf_matrix = conf_matrix_ft.astype('float') / conf_matrix_ft.sum(axis=1)[:, np.newaxis]
    normalized_conf_matrix = np.nan_to_num(normalized_conf_matrix)
    plt.figure(figsize=(14, 12))
    sns.heatmap(normalized_conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names_list, yticklabels=class_names_list)
    plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Normalized Confusion Matrix (After Fine-tuning)')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error creating normalized confusion matrix: {str(e)}")

# Display ROC curve for each class
try:
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, roc_auc = dict(), dict(), dict()
    plt.figure(figsize=(12, 10))
    for i in range(n_classes):
        binary_true = (test_true_labels == i).astype(int)
        binary_pred_probs = test_predictions_ft[:, i]
        fpr[i], tpr[i], _ = roc_curve(binary_true, binary_pred_probs)
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_names_list[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve per Class (After Fine-tuning)'); plt.legend(loc="lower right")
    plt.tight_layout(); plt.show()
except Exception as e:
    print(f"Error creating ROC curves: {str(e)}")

# --- Prediction Function and Example ---
def predict(model, img_array, class_names_list):
    img_batch = tf.expand_dims(img_array, 0) # Create batch
    predictions = model.predict(img_batch)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names_list[predicted_index]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# Example: Predict labels for a few images from the test batch.
images_batch, labels_batch = next(test_generator)
for i in range(min(len(images_batch), 5)):
    predicted_class, confidence = predict(model, images_batch[i], class_names_list)
    actual_class = class_names_list[np.argmax(labels_batch[i])]
    print(f"Img {i+1} - Actual: {actual_class}, Predicted: {predicted_class}, Confidence: {confidence}%")

# --- Save and Load Model ---
# Save the final trained model.
model_save_path = 'plant_disease_model_final.h5'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
model.save(r'C:/Users/m2921/Downloads/code ai & Data/fine_tuned_model.h5')

# Load the saved model (for later use or deployment).
loaded_model = load_model(model_save_path)
print(f"Model loaded from {model_save_path}")

# Define the class_labels_dict with class labels and corresponding indices
predicted_class_index = {
    0: "Aphids_cotton", 1: "Army worm_cotton", 2: "Bacterial blight_cotton", 3: "cotton_curl_virus",
    4: "cotton_fussarium_wilt", 5: "Healthy_cotton", 6: "Pepper_bell__bacterial_spot",
    7: "Pepper_bell__healthy", 8: "Potato___Early_blight", 9: "Potato___healthy",
    10: "Potato___Late_blight", 11: "Powdery mildew_cotton",
    12: "Strawberry___healthy", 13: "Strawberry___Leaf_scorch",
    14: "Target spot_cotton", 15: "Tomato___Bacterial_spot", 16: "Tomato___Early_blight",
    17: "Tomato___healthy", 18: "Tomato___Late_blight", 19: "Tomato___Leaf_Mold",
    20: "Tomato___Septoria_leaf_spot", 21: "Tomato___Spider_mites Two-spotted_spider_mite",
    22: "Tomato___Target_Spot", 23: "Tomato___Tomato_mosaic_virus", 24: "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
}

# --- Prediction on a New Image and Treatment Lookup ---
# Define path to a new image for prediction.
new_image_path = "G:/ksiu/Level 4/Graduation Project (AIE493) Dr. Saeed/graduation project/code ai & Data/plantvillage dataset/color/Healthy_cotton/16.jpg"

# Load and preprocess the new image.
new_image = keras_image.load_img(new_image_path, target_size=(224, 224))
new_image_array = keras_image.img_to_array(new_image)
new_image_array = np.expand_dims(new_image_array, axis=0)
new_image_array = new_image_array / 255.0

# Predict the class for the new image using the loaded model.
predicted_probabilities = loaded_model.predict(new_image_array)[0]
predicted_class_index = np.argmax(predicted_probabilities)

# Map the predicted index back to the class name.
# Use the generator's mapping if available, otherwise use a manual mapping.
if 'test_generator' in locals():
    index_to_class = {v: k for k, v in test_generator.class_indices.items()}
    predicted_class_label = index_to_class.get(predicted_class_index, 'Unknown Class')
else:
    class_labels_dict_manual = {idx: name for idx, name in enumerate(class_names_list)}
    predicted_class_label = class_labels_dict_manual.get(predicted_class_index, 'Unknown Class')

confidence_score = round(np.max(predicted_probabilities) * 100, 2)

# Display the new image and its prediction.
plt.imshow(new_image)
plt.title(f"Predicted: {predicted_class_label}\nConfidence: {confidence_score}%")
plt.axis('off'); plt.show()

# --- Look up Treatment Information ---
# Read treatment data from an Excel file.
treatment_file_path = "G:/ksiu/Level 4/Graduation Project (AIE493) Dr. Saeed/graduation project/code ai & Data/plant disease.xlsx"
try:
    treatment_df = pd.read_excel(treatment_file_path)
    disease_column_name = treatment_df.columns[0]
    treatment_info = treatment_df[treatment_df[disease_column_name].str.strip().str.lower() == predicted_class_label.strip().lower()]

    if not treatment_info.empty:
        print("\n--- Treatment Information ---")
        print(f"Disease/Condition: {predicted_class_label}")
        print(f"\nTreatment (English):\n{treatment_info.iloc[0, 1]}") 
        print(f"\nTreatment (Arabic):\n{treatment_info.iloc[0, 2]}")   
        print(f"\nResources:\n{treatment_info.iloc[0, 3]}")
        print("-" * 30)
    else:
        print(f"\nNo treatment information found for '{predicted_class_label}'.")
except FileNotFoundError:
    print(f"Error: Treatment file not found at '{treatment_file_path}'")
except Exception as e:
    print(f"Error reading or processing treatment file: {str(e)}")