# Treatment Database

What this page covers:
This page describes the agricultural treatment reference database used by Farmer Eye.
It explains the Excel data schema, the name normalization algorithm, and the step-by-step process for adding new disease classes.

---

## Database File and Schema

The treatment database is stored as an Excel spreadsheet at `data/plant_disease_data.xlsx`.
The file contains advisory records organized across four columns:

| Column Index | Column Name | Data Type | Purpose |
|:---:|---|---|---|
| `0` | Disease / Condition Name | Text | Category identifier matching classes in `src/class_names.py`. |
| `1` | Treatment (English) | Text | Practical steps in English for curing or mitigating the condition. |
| `2` | Treatment (Arabic) | Text | Practical steps in Arabic (العلاج باللغة العربية) for local growers. |
| `3` | Resources / Chemical Notes | Text | Recommended fungicides, pesticides, or agronomic references. |

---

## Why Name Normalization Is Necessary

In machine learning pipelines, category names often contain formatting inconsistencies:
- File paths and dataset folders often use multiple underscores (for example, `Pepper_bell__bacterial_spot` or `Tomato___Early_blight`).
- Human-entered spreadsheet entries often use standard spaces (for example, `Pepper bell bacterial spot`).
- Letter casing may differ between research scripts and spreadsheets.

To ensure reliable lookups, the system uses `normalize_disease_name()` in [`../src/class_names.py`](../src/class_names.py):

```python
def normalize_disease_name(name: str) -> str:
    # 1. Convert all characters to lowercase
    # 2. Replace all underscores with spaces
    # 3. Collapse multiple consecutive spaces into a single space
    # 4. Strip leading and trailing whitespace
    return " ".join(name.replace("_", " ").lower().split())
```

When querying the database, both the model's predicted class name and the spreadsheet's disease column are normalized through this function.
This ensures a 100% match rate regardless of underscore count or letter case.

---

## Step-by-Step: How to Add a New Disease Class

When expanding the model to recognize a new crop condition, follow these steps:

### Step 1: Update Class Names in Code
Open [`../src/class_names.py`](../src/class_names.py) and append the new category name to the **end** of the `CLASS_NAMES` list:

```python
CLASS_NAMES = [
    # ... existing 25 classes (indices 0 to 24) ...
    "New_Crop__New_Disease"  # Index 25
]
```

> [!IMPORTANT]
> **Order Rule**: Never insert a new class in the middle of `CLASS_NAMES` or re-sort the list alphabetically.
> The neural network outputs a vector of numbers where each position corresponds to a specific integer index.
> Changing existing positions will corrupt the predictions of any previously trained model weights.

### Step 2: Add a Row to the Treatment Spreadsheet
Open `data/plant_disease_data.xlsx` in Excel or LibreOffice and append a new row:
1. **Column A**: Enter the new disease name (for example, `New Crop New Disease`).
2. **Column B**: Enter the English treatment instructions.
3. **Column C**: Enter the Arabic treatment instructions.
4. **Column D**: Enter agronomic resource notes or chemical recommendations.
Save the file.

### Step 3: Update and Run Verification Tests
Open [`../tests/test_treatment_lookup.py`](../tests/test_treatment_lookup.py) and run the test suite to ensure the database entry resolves:

```bash
pytest tests/test_treatment_lookup.py -v
```

---

## Advisory Treatment Disclaimer

> [!CAUTION]
> All treatment recommendations provided by Farmer Eye are informational and advisory.
> Chemical dosages, spray intervals, and pesticide applications must be evaluated and confirmed by a certified agronomist or local agricultural authority before application to crops.

---

## Next Steps

- Review model training and class indexing in [Model and Training](model.md).
- See how treatment alerts are delivered over the network in [WebSocket API](websocket-api.md).
- Explore edge hardware connections in [Hardware Setup](hardware.md).
