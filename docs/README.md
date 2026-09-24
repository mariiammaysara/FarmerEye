# Farmer Eye Documentation

What this page covers:
This page indexes the complete documentation for the Farmer Eye project.
It describes the contents of each guide and provides a recommended reading order for beginners.

---

## Documentation Index

The following guides document the architecture, machine learning model, network interfaces, and operational steps of the Farmer Eye platform:

| Document | Purpose |
|---|---|
| [How It Works](how-it-works.md) | Follows one camera image through capture, processing, deep learning inference, and treatment transmission. |
| [Getting Started](getting-started.md) | Step-by-step setup instructions for development laptops and edge Raspberry Pi hardware. |
| [System Architecture](architecture.md) | Component layouts, data flow, server options, and threading models. |
| [Model and Training](model.md) | CNN architecture, data preprocessing, training parameters, performance benchmarks, and fine-tuning analysis. |
| [WebSocket API](websocket-api.md) | Message formats, connection handshakes, and event definitions for client-server communication. |
| [Treatment Database](treatment-database.md) | Excel data structure, name normalization rules, and instructions for adding disease categories. |
| [Hardware Setup](hardware.md) | Camera specifications, connection steps, and verification procedures on physical hardware. |
| [Development and Testing](development.md) | Unit testing with hardware mocks, continuous integration workflows, and coding conventions. |
| [Limitations and Future Work](limitations.md) | Known technical constraints, domain gaps, and planned system extensions. |
| [Project Context and Credits](project-context.md) | Project academic background, team members, supervisors, and funding programs. |
| [Glossary](glossary.md) | Simple definitions for all technical terms used across the documentation set. |

---

## Suggested Reading Order

For engineers and contributors new to the repository, read the documentation in this order:

1. **[How It Works](how-it-works.md)**: Understand the core pipeline from camera frame to mobile alert.
2. **[Getting Started](getting-started.md)**: Set up your local environment and run the test suite.
3. **[System Architecture](architecture.md)**: Learn how the modules, servers, and network ports interact.
4. **[Model and Training](model.md)**: Review the neural network design, training procedures, and benchmark metrics.
5. **[Treatment Database](treatment-database.md)**: Inspect how diagnostic labels map to English and Arabic treatment advice.
6. **[WebSocket API](websocket-api.md)**: Review the JSON message schemas used by the streaming server.
7. **[Hardware Setup](hardware.md)**: Review the physical camera interface and hardware boundaries.
8. **[Development and Testing](development.md)**: Understand testing practices, hardware mocking, and CI checks.
9. **[Limitations and Future Work](limitations.md)**: Review known real-world boundaries and future roadmap items.
10. **[Glossary](glossary.md)**: Look up technical definitions and acronyms.

---

## Next Steps

- Start learning the pipeline step-by-step in [How It Works](how-it-works.md).
- Set up your machine with [Getting Started](getting-started.md).
