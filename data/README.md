# 🌿 FarmerEye Dataset Card

This document describes the dataset utilized for training, validating, and testing the plant disease detection model in **Farmer Eye**.

---

## 1. Dataset Overview

* **Total Images Validated**: **39,776** images (1 corrupted image discarded during initial verification in the research notebook).
* **Target Classes**: **25** distinct conditions (diseased and healthy states) across 5 crop types: Cotton, Tomato, Potato, Pepper, and Strawberry.
* **Image Channels & Size**: RGB images resized and normalized to $224 \times 224 \times 3$ for model training.

---

## 2. Data Sources

1. **PlantVillage Dataset**
   * **Kaggle**: [PlantVillage Dataset (emmarex/plantdisease)](https://www.kaggle.com/datasets/emmarex/plantdisease)
   * **Original GitHub Repository**: [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
   * **Reference Paper**: *“Using Deep Learning for Image-Based Plant Disease Detection”* (Hughes & Salathé, 2015), [arXiv:1511.08060](https://arxiv.org/abs/1511.08060).
   * **Coverage**: Tomato, Potato, Pepper Bell, and Strawberry classes.

2. **Additional Real-World / Crop Images (Cotton & Field Subsets)**
   * **Kaggle Reference**: [Cotton Leaf Disease Dataset](https://www.kaggle.com/datasets/dhanushnarayananr/cotton-leaf-disease-dataset)
   * **Coverage**: Cotton disease and pest classes (`Aphids_cotton`, `Army worm_cotton`, `Bacterial blight_cotton`, `Healthy_cotton`, `Powdery mildew_cotton`, `Target spot_cotton`, `cotton_curl_virus`, `cotton_fussarium_wilt`).

> [!NOTE]
> **License Information**: Licenses vary across datasets. Please verify the respective license on each source repository or Kaggle dataset page before redistribution or commercial use. External image source licenses: **TBD**.

---

## 3. Dataset Splits

The dataset was partitioned using a **two-stage stratified random split** via scikit-learn (`train_test_split(..., random_state=42, stratify=df['label'])`):

1. **Stage 1 (Test Holdout)**: 20% of the total dataset was set aside as an unseen test holdout set.
2. **Stage 2 (Train / Validation)**: The remaining 80% was partitioned into 80% training and 20% validation.

| Split | Samples | Percentage of Total | Stratified Random State |
|---|:---:|:---:|:---:|
| **Training** | 25,456 | ~64.0% | `42` |
| **Validation** | 6,365 | ~16.0% | `42` |
| **Testing** | 7,955 | ~20.0% | `42` |
| **Total Validated** | **39,776** | **100.0%** | — |

---

## 4. Class List & Per-Class Sample Distribution

The 25 target classes correspond to [src/class_names.py](../src/class_names.py). Total per-class counts across all splits were visualized via Seaborn bar plots in the research notebook rather than printed as numerical tables (overall counts marked **TBD**). The exact test set support (20% stratified sample) printed in the notebook classification report (Cell 26) is documented below:

| Class Index | Class Name | Crop | Test Set Support (Cell 26) | Total Class Count |
|:---:|---|---|:---:|:---:|
| 0 | `Aphids_cotton` | Cotton | 449 | TBD |
| 1 | `Army worm_cotton` | Cotton | 448 | TBD |
| 2 | `Bacterial blight_cotton` | Cotton | 529 | TBD |
| 3 | `Healthy_cotton` | Cotton | 533 | TBD |
| 4 | `Pepper_bell_bacterial_spot` | Pepper | 213 | TBD |
| 5 | `Pepper_bell_healthy` | Pepper | 308 | TBD |
| 6 | `Potato__Early_blight` | Potato | 219 | TBD |
| 7 | `Potato_Late_blight` | Potato | 219 | TBD |
| 8 | `Potato_healthy` | Potato | 30 | TBD |
| 9 | `Powdery mildew_cotton` | Cotton | 448 | TBD |
| 10 | `Strawberry_Leaf_scorch` | Strawberry | 222 | TBD |
| 11 | `Strawberry_healthy` | Strawberry | 91 | TBD |
| 12 | `Target spot_cotton` | Cotton | 447 | TBD |
| 13 | `Tomato_Bacterial_spot` | Tomato | 426 | TBD |
| 14 | `Tomato_Early_blight` | Tomato | 200 | TBD |
| 15 | `Tomato_Late_blight` | Tomato | 382 | TBD |
| 16 | `Tomato_Leaf_Mold` | Tomato | 190 | TBD |
| 17 | `Tomato_Septoria_leaf_spot` | Tomato | 355 | TBD |
| 18 | `Tomato_Spider_mites Two-spotted_spider_mite` | Tomato | 335 | TBD |
| 19 | `Tomato_Target_Spot` | Tomato | 281 | TBD |
| 20 | `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | Tomato | 1072 | TBD |
| 21 | `Tomato_Tomato_mosaic_virus` | Tomato | 75 | TBD |
| 22 | `Tomato___healthy` | Tomato | 318 | TBD |
| 23 | `cotton_curl_virus` | Cotton | 82 | TBD |
| 24 | `cotton_fussarium_wilt` | Cotton | 83 | TBD |
| — | **Total** | — | **7,955** | **39,776** |

---

## 5. Expected Folder Layout

The training and validation scripts expect raw images structured into one folder per class:

```text
data/
├── README.md                      # This data card
├── plant_disease_data.xlsx        # Bilingual treatment reference database
└── dataset/
    ├── Aphids_cotton/
    │   ├── img_001.jpg
    │   └── ...
    ├── Army worm_cotton/
    │   └── ...
    ├── Pepper_bell__bacterial_spot/
    │   └── ...
    └── Tomato___healthy/
        └── ...
```

---

## 6. How to Obtain the Data

To download the dataset using the Kaggle API:

1. **Install Kaggle CLI**:
   ```bash
   pip install kaggle
   ```

2. **Configure API Token**:
   * Create an API token from your Kaggle account profile (`kaggle.json`).
   * Place `kaggle.json` into:
     * **Windows**: `%USERPROFILE%\.kaggle\kaggle.json`
     * **Linux / macOS**: `~/.kaggle/kaggle.json` (ensure `chmod 600 ~/.kaggle/kaggle.json`)

3. **Download Dataset**:
   ```bash
   kaggle datasets download -d emmarex/plantdisease
   ```

4. **Extract Files**:
   ```bash
   unzip <archive.zip> -d data/dataset/
   ```
