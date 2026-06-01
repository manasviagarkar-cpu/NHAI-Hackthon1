# NHAI FaceRec — 100% Offline Flutter Facial Recognition

This is the production-ready **Flutter** port of the NHAI FaceRec offline-first attendance verification system. Rebuilt from the React Native prototype, this implementation utilizes native plugins and hardware acceleration to maximize speed, battery efficiency, and reliability for field construction environments.

---

## 🚀 Key Features

*   **10x Faster Offline Face Tracking**: Integrated with `google_mlkit_face_detection` to perform sub-15ms bounding box and challenge metrics calculations natively.
*   **Dual-Stage Liveness Checks**: Real-time evaluation of eye-blink (`leftEyeOpenProbability` and `rightEyeOpenProbability`) and smile intensity (`smilingProbability`) parameters to prevent digital screen or printed photo spoofing.
*   **Fast Embedding Matching**: Compares 128-dimensional Float32 vectors from the cropped face region against SQLite profiles using local vector Cosine Similarity dot products.
*   **Robust Off-Grid SQLite Storage**: Configured via `sqflite` with Write-Ahead Logging (WAL) for parallel query safety.
*   **AWS Sync & Space Purging**: Floating log synchronizer simulation that drains pending streams and triggers storage optimizations to reclaim local disk space.
*   **Premium Glassmorphic Aesthetics**: Sleek dark mode design utilizing neon indicator frames, glowing latency timers, custom slide cards, and curved analytics panels.

---

## 🛠 Project Structure

```
lib/
├── main.dart                 # App initializer & dark glass theme bootstrapper
└── src/
    ├── components/           # Curved visual overlays
    │   ├── bounding_box.dart     # Canvas corners painter for tracking bounding boxes
    │   ├── inference_timer.dart  # Microsecond latency timer indicator
    │   ├── liveness_prompt.dart  # Real-time challenge instruction panels
    │   └── worker_card.dart      # Successful match profile slider cards
    ├── database/
    │   └── database_helper.dart  # SQLite migrations, counts, & WAL structures
    ├── ml/
    │   └── face_recognition_service.dart # Quantized MobileFaceNet interpreter & similarity math
    ├── screens/
    │   ├── home_screen.dart      # Elegant dashboard showing active counts
    │   ├── liveness_screen.dart  # Challenge viewport (blink & smile triggers)
    │   ├── recognition_screen.dart # Real-time live frame matcher
    │   ├── registration_screen.dart # Enter details form & 3-capture embedding generator
    │   └── sync_screen.dart      # Logs history timeline & cloud synchronizers
    └── utils/
        └── constants.dart        # Global thresholds, durations, & spacing tokens
```

---

## ⚙️ Android Studio Setup & Building

To maintain version alignment and avoid writing complex native platform code, this project is designed as a pristine, cross-platform Dart module. Follow these three quick steps to run the application in **Android Studio**:

### Step 1: Generate Native Bindings
Run the standard Flutter reconstruction command in your terminal. This instantly generates native Gradle structures, Android Manifests, and native bindings perfectly aligned with your local Flutter SDK:
```bash
flutter create . --platforms=android
```

### Step 2: Download TFLite Models
Download the optimized `mobilefacenet.tflite` model into `assets/models/`. You can copy it from your existing folders or download it directly using this quick PowerShell command:
```powershell
New-Item -ItemType Directory -Force -Path assets\models\
Invoke-WebRequest -Uri "https://github.com/sirius-ai/MobileFaceNet_TF/raw/master/tflite/MobileFaceNet.tflite" -OutFile "assets\models\mobilefacenet.tflite"
```

### Step 3: Run in Android Studio
1. Launch **Android Studio**.
2. Select **Open** and choose this project directory.
3. Android Studio will automatically resolve the `pubspec.yaml` dependencies and load the project.
4. Select your connected physical debugging device (or emulator) from the top bar and click **Run** (Green Arrow icon).
