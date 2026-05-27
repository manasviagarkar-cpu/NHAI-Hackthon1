#!/usr/bin/env bash
# =============================================================================
# NHAI FaceRec — Model Download Script
# Downloads all required TFLite models from official open-source repositories.
#
# Usage:
#   chmod +x scripts/download_models.sh
#   ./scripts/download_models.sh
#
# Models downloaded:
#   1. MobileFaceNet (INT8 quantized) — Face embedding extraction
#   2. BlazeFace Short Range — Fast face detection
#   3. MediaPipe Face Mesh (Face Landmark) — 468-point landmark detection
# =============================================================================

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/assets/models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  NHAI FaceRec — Model Downloader${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

# Create models directory
mkdir -p "$MODELS_DIR"
echo -e "${GREEN}[✓]${NC} Models directory: $MODELS_DIR"
echo ""

# ---------------------------------------------------------------------------
# 1. MobileFaceNet — Face Embedding Model (~4MB INT8)
# Source: sirius-ai/MobileFaceNet_TF (converted to TFLite)
# Alternative: insightface/MobileFaceNet
# ---------------------------------------------------------------------------
MOBILEFACENET_URL="https://github.com/sirius-ai/MobileFaceNet_TF/raw/master/tflite/MobileFaceNet.tflite"
MOBILEFACENET_FILE="$MODELS_DIR/mobilefacenet.tflite"

if [ -f "$MOBILEFACENET_FILE" ]; then
    echo -e "${YELLOW}[→]${NC} MobileFaceNet already exists, skipping..."
else
    echo -e "${BLUE}[↓]${NC} Downloading MobileFaceNet (INT8, ~4MB)..."
    if curl -fSL --retry 3 --retry-delay 5 -o "$MOBILEFACENET_FILE" "$MOBILEFACENET_URL" 2>/dev/null; then
        FILE_SIZE=$(wc -c < "$MOBILEFACENET_FILE" | tr -d ' ')
        echo -e "${GREEN}[✓]${NC} MobileFaceNet downloaded (${FILE_SIZE} bytes)"
    else
        echo -e "${RED}[✗]${NC} Failed to download MobileFaceNet from primary source."
        echo -e "${YELLOW}    Trying alternative source...${NC}"
        ALT_URL="https://raw.githubusercontent.com/sirius-ai/MobileFaceNet_TF/master/tflite/MobileFaceNet.tflite"
        if curl -fSL --retry 3 --retry-delay 5 -o "$MOBILEFACENET_FILE" "$ALT_URL" 2>/dev/null; then
            FILE_SIZE=$(wc -c < "$MOBILEFACENET_FILE" | tr -d ' ')
            echo -e "${GREEN}[✓]${NC} MobileFaceNet downloaded from alt source (${FILE_SIZE} bytes)"
        else
            echo -e "${RED}[✗]${NC} Both sources failed. Download manually from:"
            echo -e "    ${MOBILEFACENET_URL}"
            echo -e "    Place as: $MOBILEFACENET_FILE"
        fi
    fi
fi

echo ""

# ---------------------------------------------------------------------------
# 2. BlazeFace Short Range — Face Detection (~0.4MB)
# Source: Google MediaPipe official models
# ---------------------------------------------------------------------------
BLAZEFACE_URL="https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
BLAZEFACE_FILE="$MODELS_DIR/blazeface.tflite"

if [ -f "$BLAZEFACE_FILE" ]; then
    echo -e "${YELLOW}[→]${NC} BlazeFace already exists, skipping..."
else
    echo -e "${BLUE}[↓]${NC} Downloading BlazeFace Short Range (~0.4MB)..."
    if curl -fSL --retry 3 --retry-delay 5 -o "$BLAZEFACE_FILE" "$BLAZEFACE_URL" 2>/dev/null; then
        FILE_SIZE=$(wc -c < "$BLAZEFACE_FILE" | tr -d ' ')
        echo -e "${GREEN}[✓]${NC} BlazeFace downloaded (${FILE_SIZE} bytes)"
    else
        echo -e "${RED}[✗]${NC} Failed to download BlazeFace. Download manually from:"
        echo -e "    ${BLAZEFACE_URL}"
        echo -e "    Place as: $BLAZEFACE_FILE"
    fi
fi

echo ""

# ---------------------------------------------------------------------------
# 3. MediaPipe Face Mesh (Face Landmark) — 468 Landmarks (~8MB)
# Source: Google MediaPipe official models
# ---------------------------------------------------------------------------
FACEMESH_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
FACEMESH_FILE="$MODELS_DIR/face_landmark.tflite"

if [ -f "$FACEMESH_FILE" ]; then
    echo -e "${YELLOW}[→]${NC} Face Mesh already exists, skipping..."
else
    echo -e "${BLUE}[↓]${NC} Downloading MediaPipe Face Mesh (~8MB)..."
    if curl -fSL --retry 3 --retry-delay 5 -o "$FACEMESH_FILE" "$FACEMESH_URL" 2>/dev/null; then
        FILE_SIZE=$(wc -c < "$FACEMESH_FILE" | tr -d ' ')
        echo -e "${GREEN}[✓]${NC} Face Mesh downloaded (${FILE_SIZE} bytes)"
    else
        echo -e "${YELLOW}    Trying alternative direct TFLite source...${NC}"
        ALT_FACEMESH_URL="https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2_with_blendshapes.task"
        if curl -fSL --retry 3 --retry-delay 5 -o "$FACEMESH_FILE" "$ALT_FACEMESH_URL" 2>/dev/null; then
            FILE_SIZE=$(wc -c < "$FACEMESH_FILE" | tr -d ' ')
            echo -e "${GREEN}[✓]${NC} Face Mesh downloaded from alt source (${FILE_SIZE} bytes)"
        else
            echo -e "${RED}[✗]${NC} Both sources failed. Download manually from:"
            echo -e "    ${FACEMESH_URL}"
            echo -e "    Place as: $FACEMESH_FILE"
        fi
    fi
fi

echo ""

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  Download Summary${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

TOTAL_SIZE=0
ALL_PRESENT=true

for MODEL_FILE in "$MOBILEFACENET_FILE" "$BLAZEFACE_FILE" "$FACEMESH_FILE"; do
    MODEL_NAME=$(basename "$MODEL_FILE")
    if [ -f "$MODEL_FILE" ]; then
        SIZE=$(wc -c < "$MODEL_FILE" | tr -d ' ')
        SIZE_MB=$(echo "scale=2; $SIZE / 1048576" | bc 2>/dev/null || echo "?")
        TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
        echo -e "  ${GREEN}✓${NC} ${MODEL_NAME} — ${SIZE_MB} MB"
    else
        echo -e "  ${RED}✗${NC} ${MODEL_NAME} — MISSING"
        ALL_PRESENT=false
    fi
done

echo ""

if [ "$ALL_PRESENT" = true ]; then
    TOTAL_MB=$(echo "scale=2; $TOTAL_SIZE / 1048576" | bc 2>/dev/null || echo "?")
    echo -e "${GREEN}All models downloaded successfully!${NC}"
    echo -e "Total bundle size: ${TOTAL_MB} MB"
    
    if [ "$TOTAL_SIZE" -lt 20971520 ]; then
        echo -e "${GREEN}✓ Under 20MB target${NC}"
    else
        echo -e "${YELLOW}⚠ Exceeds 20MB target — consider further quantization${NC}"
    fi
else
    echo -e "${RED}Some models are missing. See instructions above.${NC}"
    echo -e "You can also download them manually and place in:"
    echo -e "  $MODELS_DIR/"
    exit 1
fi

echo ""
echo -e "${GREEN}Setup complete! You can now build the app.${NC}"
