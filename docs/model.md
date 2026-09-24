# Model and Training

What this page covers:
This page details the deep learning model used for crop disease classification.
It describes the dataset, image preprocessing, CNN layer design, training configuration, evaluation benchmarks, and the known fine-tuning behavior.

---

## Dataset Overview

The model was developed using a research dataset of **39,776 validated images** across 25 crop conditions (24 disease categories and 1 healthy category) covering 5 crops: Cotton, Tomato, Potato, Pepper, and Strawberry.

The dataset was partitioned using a two-stage stratified random split with `random_state=42`:
- **Training Set**: 25,456 images (64.0%)
- **Validation Set**: 6,365 images (16.0%)
- **Test Set (Holdout)**: 7,955 images (20.0%)

For full dataset provenance, citations, and download instructions, see the [Dataset Card (data/README.md)](../data/README.md).

---

## Image Preprocessing

All images are transformed to match the format used during model training:

1. **Spatial Resizing**: Resized to 224 pixels by 224 pixels using bilinear interpolation.
2. **Channel Format**: 3 color channels in standard RGB order.
3. **Pixel Scaling**: Integer pixel values (0 to 255) are cast to 32-bit floats and divided by 255.0 to yield values between 0.0 and 1.0.
4. **Data Augmentation (Training Only)**: Random rotation (20 degrees), width/height shifts (10%), shearing (20%), zoom (20%), and horizontal flipping.

---

## CNN Model Architecture

The classifier is a custom 5-block Convolutional Neural Network (CNN) built with Keras and trained from scratch:

| Block / Stage | Layer Type | Output Shape | Parameters | Purpose |
|---|---|---|:---:|---|
| **Input** | InputLayer | `(224, 224, 3)` | 0 | Raw preprocessed image tensor. |
| **Block 1** | Conv2D (64 filters, 7x7, stride 3, valid) + BN + ReLU + MaxPool (3x3) | `(24, 24, 64)` | 9,664 | Large kernel captures coarse spatial edges; stride 3 reduces dimensions. |
| **Block 2** | Conv2D (128 filters, 3x3, same) + BN + ReLU + MaxPool (2x2) | `(12, 12, 128)` | 74,496 | Extracts compound corner and texture patterns. |
| **Block 3** | Conv2D (128 filters, 3x3, same) + BN + ReLU + MaxPool (2x2) | `(6, 6, 128)` | 148,224 | Deep feature extraction across leaf surfaces. |
| **Block 4** | Conv2D (256 filters, 3x3, same) + BN + ReLU + MaxPool (2x2) | `(3, 3, 256)` | 296,192 | Extracts localized lesion and spot shapes. |
| **Block 5** | Conv2D (256 filters, 3x3, same) + BN + ReLU + MaxPool (2x2) | `(1, 1, 256)` | 591,104 | High-level disease feature representations. |
| **Head** | Flatten + Dense(256) + BN + ReLU + Dense(256) + Dropout(0.5) | `(256)` | 131,840 | Fully connected classification layers; 50% dropout prevents overfitting. |
| **Output** | Dense(25, Softmax) | `(25)` | 6,425 | Produces probability vector summing to 1.0 across all 25 classes. |

### Parameter Summary
- **Total Parameters**: 1,257,433 (~4.86 MB in 32-bit floats)
- **Trainable Parameters**: 1,255,257
- **Non-Trainable Parameters**: 2,176 (from Batch Normalization layers)

---

## Training Setup

The training script (`src/app.py`) configures the optimization process with the following settings:
- **Optimizer**: Adam with default learning rate $\eta = 0.001$.
- **Loss Function**: Categorical Cross-Entropy.
- **Batch Size**: 32 images.
- **Maximum Epochs**: 50 epochs.
- **Early Stopping**: Monitors validation loss (`val_loss`) with a patience of 10 epochs, restoring best weights upon completion.
- **Learning Rate Decay**: `ReduceLROnPlateau` reduces learning rate by factor 0.2 when validation loss stalls for 5 epochs (minimum rate $10^{-5}$).

---

## Evaluation Benchmark and Results

The trained model checkpoint (`models/plant_disease_model_final.h5`) was evaluated on the independent 7,955-sample holdout test set:

- **Overall Test Accuracy**: **97.95%** (`0.979510`)
- **Test Loss**: **0.0787**
- **Macro Average**: Precision: `0.97` | Recall: `0.98` | F1-Score: `0.98`
- **Weighted Average**: Precision: `0.98` | Recall: `0.98` | F1-Score: `0.98`

*Note: These metrics were measured offline on a desktop workstation. Performance under varying outdoor illumination, motion blur, and overlapping leaves in physical fields has not been quantified.*

### Full 25-Class Holdout Test Performance

The table below shows the exact classification results across all 25 classes on the holdout test set:

| Class Index | Crop | Disease / Condition Name | Precision | Recall | F1-Score | Support |
|:---:|:---:|---|:---:|:---:|:---:|:---:|
| 0 | Cotton | `Aphids_cotton` | 0.98 | 0.98 | 0.98 | 449 |
| 1 | Cotton | `Army worm_cotton` | 0.98 | 0.99 | 0.98 | 448 |
| 2 | Cotton | `Bacterial blight_cotton` | 0.95 | 0.95 | 0.95 | 529 |
| 3 | Cotton | `Healthy_cotton` | 0.98 | 0.99 | 0.99 | 533 |
| 4 | Pepper | `Pepper_bell__bacterial_spot` | 0.99 | 0.92 | 0.96 | 213 |
| 5 | Pepper | `Pepper_bell__healthy` | 0.98 | 0.96 | 0.97 | 308 |
| 6 | Potato | `Potato___Early_blight` | 0.97 | 0.94 | 0.95 | 219 |
| 7 | Potato | `Potato___Late_blight` | 0.95 | 0.92 | 0.94 | 219 |
| 8 | Potato | `Potato___healthy` | 0.91 | 1.00 | 0.95 | 30 |
| 9 | Cotton | `Powdery mildew_cotton` | 0.99 | 0.99 | 0.99 | 448 |
| 10 | Strawberry | `Strawberry___Leaf_scorch` | 1.00 | 1.00 | 1.00 | 222 |
| 11 | Strawberry | `Strawberry___healthy` | 0.99 | 1.00 | 0.99 | 91 |
| 12 | Cotton | `Target spot_cotton` | 0.94 | 0.97 | 0.95 | 447 |
| 13 | Tomato | `Tomato___Bacterial_spot` | 0.99 | 0.99 | 0.99 | 426 |
| 14 | Tomato | `Tomato___Early_blight` | 0.98 | 0.94 | 0.96 | 200 |
| 15 | Tomato | `Tomato___Late_blight` | 0.99 | 0.98 | 0.98 | 382 |
| 16 | Tomato | `Tomato___Leaf_Mold` | 1.00 | 1.00 | 1.00 | 190 |
| 17 | Tomato | `Tomato___Septoria_leaf_spot` | 0.99 | 1.00 | 0.99 | 355 |
| 18 | Tomato | `Tomato___Spider_mites Two-spotted_spider_mite` | 0.97 | 0.99 | 0.98 | 335 |
| 19 | Tomato | `Tomato___Target_Spot` | 0.98 | 0.97 | 0.98 | 281 |
| 20 | Tomato | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 1.00 | 1.00 | 1.00 | 1072 |
| 21 | Tomato | `Tomato___Tomato_mosaic_virus` | 1.00 | 1.00 | 1.00 | 75 |
| 22 | Tomato | `Tomato___healthy` | 1.00 | 1.00 | 1.00 | 318 |
| 23 | Cotton | `cotton_curl_virus` | 0.90 | 0.99 | 0.94 | 82 |
| 24 | Cotton | `cotton_fussarium_wilt` | 0.94 | 1.00 | 0.97 | 83 |
| — | — | **Overall Accuracy** | — | — | **0.98** | **7,955** |
| — | — | **Macro Average** | **0.97** | **0.98** | **0.98** | **7,955** |
| — | — | **Weighted Average** | **0.98** | **0.98** | **0.98** | **7,955** |

### Visual Artifacts

The figures below show training progression and test set classification distributions:

| Training and Validation Curves | Confusion Matrix Heatmap |
|:---:|:---:|
| ![Training Curves](assets/results/training_validation_curves.png) | ![Confusion Matrix](assets/results/confusion_matrix.png) |

---

## The Known Fine-Tuning Issue

The training script (`src/app.py`) and research notebook include an optional fine-tuning section:

```python
# From src/app.py lines 818-825
epochs_finetune = 10
history_finetune = model.fit(
    train_generator,
    epochs=epochs_finetune,
    initial_epoch=history.epoch[-1],
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr]
)
```

### Why Fine-Tuning Did Not Train:
In Keras, `initial_epoch` defines the index at which training begins, while `epochs` specifies the total target epoch count.
Because initial training completed 50 epochs (indices 0 through 49), `history.epoch[-1]` was 49.
When `model.fit()` was invoked with `epochs=10` and `initial_epoch=49`, Keras observed that the current epoch (49) was already greater than the target total epochs (10).
As a result, Keras terminated the call immediately without executing a single training step.
The model saved as `models/plant_disease_model_final.h5` is therefore the result of initial training, not subsequent fine-tuning.

---

## How to Retrain the Model

To execute the training pipeline in this repository:

1. Download and extract the PlantVillage dataset to `data/plantvillage dataset/color/`.
2. Run the training script:
   ```bash
   python src/app.py
   ```
   *Note: A dedicated `src/train_model.py` command-line script is not included in this repository. All research training logic resides in `src/app.py` and `notebooks/research_and_training.ipynb`.*

---

## Edge Optimization with TensorFlow Lite

For efficient execution on Raspberry Pi CPUs, the repository provides `src/convert_tflite.py` to quantize the model:

| Quantization Flag | Precision | Resulting Size | Target Platform |
|---|---|:---:|---|
| `--quantization float16` | 16-bit float weights | ~2.45 MB | Raspberry Pi 4 GPU / Modern CPU |
| `--quantization dynamic` | 8-bit integer weights | ~1.30 MB | Low-power ARM CPUs |
| `--quantization none` | Standard 32-bit float | ~4.86 MB | Workstation testing |

Conversion command:
```bash
python src/convert_tflite.py --model models/plant_disease_model_final.h5 --quantization float16
```

---

## Next Steps

- Explore how predictions are matched with treatments in [Treatment Database](treatment-database.md).
- Review network message formats in [WebSocket API](websocket-api.md).
- Understand operational trade-offs in [Limitations and Future Work](limitations.md).
