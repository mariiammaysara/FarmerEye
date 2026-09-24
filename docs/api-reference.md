# WebSocket API & Protocol Specification

This document defines the real-time WebSocket protocol used to transmit continuous video streams, diagnostic detection alerts, and keepalive signals between the Raspberry Pi edge device and the mobile client application.

---

## 1. Protocol Architecture & Connection

- **Protocol**: WebSocket over TCP (`ws://`)
- **Default Port**: `8765`
- **Default Host**: `0.0.0.0` (binds to all available local network interfaces)
- **URL Format**: `ws://<raspberry_pi_ip_address>:8765`
- **Serialization**: JSON text messages with Base64-encoded binary image payloads

### Environmental Configuration
The edge server parameters can be customized via system environment variables:

| Environment Variable | Default Value | Description |
|---|:---:|---|
| `WEBSOCKET_HOST` | `0.0.0.0` | IP interface to bind server |
| `WEBSOCKET_PORT` | `8765` | TCP listening port |
| `DETECTION_THRESHOLD`| `0.98` | Minimum model confidence score required to trigger alert |
| `DETECTION_COOLDOWN` | `2.0` | Seconds between consecutive inference scans |
| `PING_INTERVAL` | `20.0` | Heartbeat interval (seconds) |
| `PING_TIMEOUT` | `35.0` | Disconnect timeout for unresponsive clients (seconds) |

---

## 2. Server-to-Client Messages

### A. Connection Handshake (`welcome`)
Sent by the edge server immediately upon a new client connection:

```json
{
  "type": "welcome",
  "message": "Connected to Disease Detection Server",
  "timestamp": "2026-09-24T18:30:00.123456"
}
```

---

### B. Video Stream Frame (`camera_frame`)
Broadcast periodically at ~20 FPS. The `image` property contains the JPEG image encoded as a standard Base64 string:

```json
{
  "type": "camera_frame",
  "timestamp": "2026-09-24T18:30:01.050123",
  "image": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA..."
}
```

#### Client Rendering Guidance:
- The base64 string can be passed directly to `Image.memory(base64Decode(imageStr))` in Flutter.

---

### C. Disease Detection Alert (`detection`)
Dispatched when the convolutional neural network detects a plant disease with confidence exceeding `DETECTION_THRESHOLD`. Contains full clinical and pharmaceutical guidance:

```json
{
  "type": "detection",
  "status": "detection",
  "timestamp": "2026-09-24T18:30:02.500890",
  "disease_name": "Tomato___Early_blight",
  "confidence": 0.9854,
  "treatment_en": "Apply copper-based fungicides at first symptom. Prune lower affected leaves to increase aeration and minimize soil splash.",
  "treatment_ar": "استخدام مبيدات فطرية نحاسية عند ظهور أولى الأعراض. إزالة الأوراق السفلية المصابة لتحسين التهوية والحد من رذاذ التربة.",
  "resources": "https://www.apsnet.org/edcenter/disandpath/fungalasco/pdlessons/Pages/EarlyBlight.aspx",
  "image": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA..."
}
```

| Field | Data Type | Description |
|---|:---:|---|
| `type` | String | Event identifier (`detection`) |
| `status` | String | Status flag (`detection`) |
| `timestamp` | String | ISO 8601 UTC timestamp |
| `disease_name` | String | Canonical plant condition name |
| `confidence` | Float | Softmax output probability `[0.0, 1.0]` |
| `treatment_en` | String | English pharmaceutical and agricultural advice |
| `treatment_ar` | String | Arabic agricultural treatment guidance |
| `resources` | String | Academic / extension reference URL |
| `image` | String | Base64-encoded annotated JPEG crop |

---

### D. Scan Status (`no_detection`)
Emitted periodically when leaves in view are healthy or confidence is below threshold:

```json
{
  "timestamp": "2026-09-24T18:30:04.000000",
  "status": "no_detection",
  "message": "No plants or diseases detected"
}
```

---

### E. Heartbeat Response (`pong`)
Returned in response to a client `ping`:

```json
{
  "type": "pong",
  "timestamp": "2026-09-24T18:30:20.000000"
}
```

---

## 3. Client-to-Server Messages

### Keepalive Ping (`ping`)
Clients should send a ping every 15–20 seconds to keep the session alive:

```json
{
  "type": "ping"
}
```

---

## 4. Mobile Client Integration (Flutter / Dart Example)

Below is a complete, minimal Dart implementation demonstrating how a Flutter mobile app connects, handles the live video stream, and surfaces detection dialogs:

```dart
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:web_socket_channel/io.dart';

class FarmerEyeFeedView extends StatefulWidget {
  final String serverIp;
  const FarmerEyeFeedView({Key? key, required this.serverIp}) : super(key: key);

  @override
  _FarmerEyeFeedViewState createState() => _FarmerEyeFeedViewState();
}

class _FarmerEyeFeedViewState extends State<FarmerEyeFeedView> {
  late IOWebSocketChannel _channel;
  Uint8List? _latestFrameBytes;

  @override
  void initState() {
    super.initState();
    // Connect to edge Raspberry Pi WebSocket
    _channel = IOWebSocketChannel.connect('ws://${widget.serverIp}:8765');
    _channel.stream.listen(_handleIncomingMessage);
  }

  void _handleIncomingMessage(dynamic rawMessage) {
    final Map<String, dynamic> data = jsonDecode(rawMessage);
    final String type = data['type'] ?? data['status'] ?? '';

    if (type == 'camera_frame' && data['image'] != null) {
      setState(() {
        _latestFrameBytes = base64Decode(data['image']);
      });
    } else if (type == 'detection') {
      _showDiseaseAlert(data);
    }
  }

  void _showDiseaseAlert(Map<String, dynamic> alert) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('⚠️ Alert: ${alert['disease_name']}'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Confidence: ${(alert['confidence'] * 100).toStringAsFixed(1)}%'),
            const Divider(),
            const Text('Treatment (English):', style: TextStyle(fontWeight: FontWeight.bold)),
            Text(alert['treatment_en'] ?? 'N/A'),
            const SizedBox(height: 8),
            const Text('العلاج (العربية):', style: TextStyle(fontWeight: FontWeight.bold)),
            Text(alert['treatment_ar'] ?? 'غير متوفر', textDirection: TextDirection.rtl),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('OK')),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _channel.sink.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Farmer Eye - Live Telemetry')),
      body: Center(
        child: _latestFrameBytes != null
            ? Image.memory(_latestFrameBytes!, gaplessPlayback: true)
            : const CircularProgressIndicator(),
      ),
    );
  }
}
```
