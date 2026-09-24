# Machine Learning & Computer Vision Guide

This document provides a comprehensive overview of the computer vision pipeline in **Farmer Eye**, including dataset structure, convolutional neural network (CNN) architecture, evaluation metrics, and TensorFlow Lite edge optimization.

---

## 1. Problem Formulation

The agricultural computer vision objective is formulated as a **25-class single-label image classification problem**:
Given a close-up image of a crop leaf $I \in \mathbb{R}^{H \times W \times 3}$, predict the probability distribution $P \in [0, 1]^{25}$ representing plant health status across 5 target crops: **Cotton, Tomato, Potato, Pepper, and Strawberry**.

---

## 2. Dataset & Stratified Splits

The research dataset aggregates **39,776 validated images** combining the open-source **PlantVillage** benchmark with additional real-world field imagery (specifically for cotton diseases).

### Partitioning Strategy
A two-stage stratified random split was applied with `random_state=42` to preserve class proportions across all splits:

| Split Partition | Sample Count | Percentage | Purpose |
|---|:---:|:---:|---|
| **Training Set** | 25,456 | 64.0% | Model weight parameter optimization |
| **Validation Set** | 6,365 | 16.0% | Hyperparameter tuning & early stopping monitoring |
| **Test Holdout Set** | 7,955 | 20.0% | Final unbiased performance assessment |
| **Total Validated** | **39,776** | **100.0%** | (1 corrupted image discarded during verification) |

Detailed dataset sources, license notes, and Kaggle download instructions are documented in the [data/README.md](../data/README.md) data card.

---

## 3. Supported Disease Classes (25 Classes)

The 25 target classes and their indexing order in [src/class_names.py](../src/class_names.py) are:

```
 0: Aphids_cotton                              13: Tomato_Bacterial_spot
 1: Army worm_cotton                           14: Tomato_Early_blight
 2: Bacterial blight_cotton                    15: Tomato_Late_blight
 3: Healthy_cotton                             16: Tomato_Leaf_Mold
 4: Pepper_bell_bacterial_spot                 17: Tomato_Septoria_leaf_spot
 5: Pepper_bell_healthy                        18: Tomato_Spider_mites Two-spotted_spider_mite
 6: Potato__Early_blight                       19: Tomato_Target_Spot
 7: Potato_Late_blight                         20: Tomato_Tomato_Yellow_Leaf_Curl_Virus
 8: Potato_healthy                             21: Tomato_Tomato_mosaic_virus
 9: Powdery mildew_cotton                      22: Tomato___healthy
10: Strawberry_Leaf_scorch                     23: cotton_curl_virus
11: Strawberry_healthy                         24: cotton_fussarium_wilt
12: Target spot_cotton
```

---

## 4. CNN Model Architecture

The classifier is a **custom 5-block Convolutional Neural Network (CNN)** trained from scratch:

```
 Input (224 x 224 x 3 RGB Image)
            │
            ▼
 ┌─────────────────────────────────────────┐
 │ Block 1: Conv2D(32, 3x3) + BN + MaxPool │  ──> Output: (112 x 112 x 32)
 └────────────────────┬────────────────────┘
                      ▼
 ┌─────────────────────────────────────────┐
 │ Block 2: Conv2D(64, 3x3) + BN + MaxPool │  ──> Output: (56 x 56 x 64)
 └────────────────────┬────────────────────┘
                      ▼
 ┌─────────────────────────────────────────┐
 │ Block 3: Conv2D(128, 3x3) + BN + MaxPool│  ──> Output: (28 x 28 x 128)
 └────────────────────┬────────────────────┘
                      ▼
 ┌─────────────────────────────────────────┐
 │ Block 4: Conv2D(256, 3x3) + BN + MaxPool│  ──> Output: (14 x 14 x 256)
 └────────────────────┬────────────────────┘
                      ▼
 ┌─────────────────────────────────────────┐
 │ Block 5: Conv2D(512, 3x3) + BN + MaxPool│  ──> Output: (7 x 7 x 512)
 └────────────────────┬────────────────────┘
                      ▼
 ┌─────────────────────────────────────────┐
 │ Flatten + Dense(512) + Dropout(0.5)     │
 └────────────────────┬────────────────────┘
                      ▼
 ┌─────────────────────────────────────────┐
 │ Dense(25, Softmax Activation)           │  ──> Output: Probability Vector [25]
 └─────────────────────────────────────────┘
```

### Key Architectural Choices:
- **Batch Normalization (BN)**: Applied after each convolutional layer to stabilize gradient descent and accelerate convergence.
- **Progressive Filter Growth**: Filters increase from 32 to 512, capturing low-level edges in early blocks and complex pathological textures in deeper blocks.
- **Dropout (0.5)**: Placed before the final classification head to mitigate overfitting on homogenous background leaves.

---

## 5. Training Strategy & Hyperparameters

The model was trained in TensorFlow/Keras with the following configuration:
- **Optimizer**: Adam ($\beta_1=0.9, \beta_2=0.999$, initial learning rate $\eta = 10^{-3}$)
- **Loss Function**: Categorical Cross-Entropy
- **Batch Size**: 32 samples
- **Callbacks**:
  - `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)`

---

## 6. Experimental Evaluation & Results

Evaluated on the independent **7,955-image holdout test set**:

- **Overall Test Accuracy**: **97.95%**
- **Test Loss**: **0.0787**
- **Macro Average**: Precision: `0.97` | Recall: `0.98` | F1-Score: `0.98`
- **Weighted Average**: Precision: `0.98` | Recall: `0.98` | F1-Score: `0.98`

> [!NOTE]
> Metrics were evaluated offline on a workstation using a random hold-out split. Real-field performance under dynamic outdoor lighting, soil clutter, and multi-leaf occlusion was not evaluated and may exhibit a domain gap.

### Visual Assets & Curves

#### Training & Validation History (50 Epochs)
![Training Curves](assets/results/training_validation_curves.png)

#### Holdout Test Set Confusion Matrix
![Confusion Matrix](assets/results/confusion_matrix.png)

---

## 7. Edge Optimization with TensorFlow Lite (TFLite)

To deploy the deep learning model onto the Raspberry Pi with low latency and lower memory overhead, the repository provides an automated conversion script: [src/convert_tflite.py](../src/convert_tflite.py).

### Quantization Modes

| Mode | Flag | Target Spec | Typical Model Size | Target Hardware |
|---|---|---|:---:|---|
| **Float16 Quantization** | `-q float16` *(Default)* | `tf.float16` weights | **~2.45 MB** (~50% reduction) | Raspberry Pi 4 GPU / Modern CPU |
| **Dynamic Range** | `-q dynamic` | 8-bit integer weights | **~1.30 MB** (~74% reduction) | Low-power ARM CPUs |
| **Unquantized Float32** | `-q none` | Standard float32 | **~4.86 MB** | Desktop testing |

### How to Convert:
```bash
# Convert using default float16 quantization (saves to models_tflite/)
python src/convert_tflite.py

# Convert using 8-bit dynamic quantization
python src/convert_tflite.py -q dynamic
```

---

## 8. Running Offline Model Evaluation

The [src/evaluate.py](../src/evaluate.py) script allows evaluating both Keras (`.h5`) and TFLite (`.tflite`) models against any test directory structured in class subfolders:

```bash
# Evaluate Keras model
python src/evaluate.py --model models/plant_disease_model_final.h5 --data path/to/test_data --output docs/assets/results/keras

# Evaluate TFLite model to compare accuracy
python src/evaluate.py --model models_tflite/plant_disease_model_float16.tflite --data path/to/test_data --output docs/assets/results/tflite
```

### Outputs Generated:
1. `metrics.json`: Structured JSON containing overall accuracy, macro/weighted averages, and per-class precision/recall/F1.
2. `confusion_matrix.png`: High-resolution 300 DPI annotated heatmap.
3. `classification_report.txt`: Complete textual precision/recall report.
