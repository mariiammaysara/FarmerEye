# Limitations and Future Work

What this page covers:
This page provides an objective assessment of the technical boundaries of Farmer Eye.
It details dataset characteristics, network security constraints, software inconsistencies, and areas planned for future development.

---

## Technical Limitations

### 1. Dataset Domain Gap
The convolutional neural network was trained primarily on the PlantVillage dataset.
Most images in this benchmark show single, detached crop leaves photographed against plain, uniform laboratory backgrounds.
Real agricultural fields present complex backgrounds with weeds, dry soil, shadows, changing sunlight angles, and overlapping leaves.
Accuracy on physical farms may be lower than laboratory benchmark scores due to this domain gap.

### 2. Image Partitioning Method
As documented in the [Dataset Card (data/README.md)](../data/README.md), the dataset was split using a two-stage stratified random sample of individual images.
Because individual frames from the same plant or photoshoot may exist across both training and test partitions, performance metrics reflect holdout test accuracy rather than performance on unseen agricultural farms.

### 3. Unencrypted Network Connections
The current WebSocket servers (`src/combined_detection_stream.py` and `src/real_time_detection.py`) use unencrypted `ws://` connections.
All video frames, telemetry, and treatment payloads travel as plain text across the local network without Transport Layer Security (TLS/WSS) or authentication tokens.
Anyone on the local Wi-Fi subnet can view the video feed or send mock commands.

### 4. Excel-Based Knowledge Base
Treatments are stored in an Excel workbook (`data/plant_disease_data.xlsx`) read using Pandas into memory at server startup.
While convenient for editing, this approach cannot handle concurrent writes, does not support automated database migrations, and requires restarting the server process to pick up content edits.

### 5. Inconsistent Confidence Thresholds
The repository includes two different edge server implementations with different decision criteria:
- `src/real_time_detection.py` uses a confidence threshold of `0.70` (70%).
- `src/combined_detection_stream.py` uses a confidence threshold of `0.98` (98%).
Operators should be aware that the two scripts produce different false-positive and false-negative detection behaviors.

### 6. Advisory Nature of Treatments
Agricultural treatment recommendations are algorithmic references based on compiled advisory sheets.
They are not substitutes for professional agronomic diagnosis.
Chemical application rates, seasonal restrictions, and pest resistance vary widely by geographic region and crop variety.

### 7. External Components Not Included
- **Mobile Application**: The Flutter mobile dashboard client is developed and hosted in a separate repository.
- **Motor Locomotion**: Chassis movement controllers and physical motor drivers are not included in this repository.

---

## Future Work

The development roadmap includes the following planned improvements:

1. **In-Field Dataset Collection**: Collect and annotate diverse in-situ images under outdoor lighting conditions across varying growth stages.
2. **Encrypted WebSockets**: Implement TLS encryption (`wss://`) with token-based client authentication.
3. **Embedded Relational Database**: Migrate treatment storage from Excel to an embedded SQLite or PostgreSQL database with versioned schema migrations.
4. **Unified Configuration**: Consolidate server logic and confidence thresholds into a centralized configuration file.
5. **Edge Optimization**: Complete automated pipeline benchmarks for INT8 quantized TensorFlow Lite execution on edge accelerators.

---

## Next Steps

- Review model metrics and benchmark results in [Model and Training](model.md).
- Learn about development standards and testing in [Development and Testing](development.md).
- Check the complete list of terms in [Glossary](glossary.md).
