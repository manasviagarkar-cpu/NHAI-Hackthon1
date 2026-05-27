/**
 * @module FaceDetector
 * Wrapper around BlazeFace TFLite model for real-time face detection.
 * Uses react-native-fast-tflite for zero-bridge-overhead inference
 * directly on the camera frame processor thread.
 *
 * BlazeFace Short Range outputs:
 *   - Bounding box (xCenter, yCenter, width, height) in normalized [0,1] coords
 *   - 6 keypoints (right eye, left eye, nose, mouth, right ear, left ear)
 *   - Confidence score
 */

import { FACE_DETECTION_CONFIDENCE, BLAZEFACE_INPUT_SIZE } from '../utils/Constants';

/**
 * Represents a detected face bounding box in normalized coordinates [0, 1].
 */
export interface FaceDetection {
  /** X coordinate of the bounding box center (0.0 – 1.0) */
  xCenter: number;
  /** Y coordinate of the bounding box center (0.0 – 1.0) */
  yCenter: number;
  /** Width of the bounding box (0.0 – 1.0) */
  width: number;
  /** Height of the bounding box (0.0 – 1.0) */
  height: number;
  /** Detection confidence score (0.0 – 1.0) */
  confidence: number;
  /** Pixel coordinates for UI overlay (set after denormalization) */
  pixelBox: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
}

/**
 * Sigmoid activation function for converting raw logits to probabilities.
 *
 * @param {number} x - Raw logit value
 * @returns {number} Probability in range (0.0, 1.0)
 */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Parses the raw TFLite output tensors from BlazeFace into face detections.
 * BlazeFace Short Range outputs two tensors:
 *   - regressors: [1, 896, 16] — bounding box + keypoint coordinates
 *   - classificators: [1, 896, 1] — confidence scores
 *
 * This function applies non-maximum suppression (NMS) and filters by confidence.
 *
 * @param {Float32Array} regressors - Raw bounding box output tensor
 * @param {Float32Array} classificators - Raw confidence score tensor
 * @param {number} frameWidth - Width of the camera frame in pixels
 * @param {number} frameHeight - Height of the camera frame in pixels
 * @param {number} [confidenceThreshold=FACE_DETECTION_CONFIDENCE] - Minimum confidence
 * @returns {FaceDetection[]} Array of detected faces sorted by confidence (highest first)
 */
export function parseBlazeFaceOutput(
  regressors: Float32Array,
  classificators: Float32Array,
  frameWidth: number,
  frameHeight: number,
  confidenceThreshold: number = FACE_DETECTION_CONFIDENCE
): FaceDetection[] {
  const detections: FaceDetection[] = [];
  const numAnchors = 896;
  const numRegressorValues = 16; // 4 bbox + 6 keypoints × 2

  // Generate anchors for BlazeFace Short Range (128×128 input)
  const anchors = generateAnchors();

  for (let i = 0; i < numAnchors; i++) {
    const score = sigmoid(classificators[i]);

    if (score < confidenceThreshold) {
      continue;
    }

    const offset = i * numRegressorValues;
    const anchor = anchors[i];

    // Decode bounding box relative to anchor
    const xCenter = regressors[offset + 0] / BLAZEFACE_INPUT_SIZE + anchor.x;
    const yCenter = regressors[offset + 1] / BLAZEFACE_INPUT_SIZE + anchor.y;
    const w = regressors[offset + 2] / BLAZEFACE_INPUT_SIZE;
    const h = regressors[offset + 3] / BLAZEFACE_INPUT_SIZE;

    // Convert normalized coords to pixel coords for UI overlay
    const pixelX = (xCenter - w / 2) * frameWidth;
    const pixelY = (yCenter - h / 2) * frameHeight;
    const pixelW = w * frameWidth;
    const pixelH = h * frameHeight;

    detections.push({
      xCenter,
      yCenter,
      width: w,
      height: h,
      confidence: score,
      pixelBox: {
        x: Math.max(0, pixelX),
        y: Math.max(0, pixelY),
        w: Math.min(pixelW, frameWidth - pixelX),
        h: Math.min(pixelH, frameHeight - pixelY),
      },
    });
  }

  // Sort by confidence descending
  detections.sort((a, b) => b.confidence - a.confidence);

  // Apply simple NMS — keep only the highest confidence detection
  // (we only need one face for single-person recognition)
  if (detections.length > 1) {
    return [detections[0]];
  }

  return detections;
}

/**
 * Anchor point for BlazeFace decoder.
 */
interface Anchor {
  x: number;
  y: number;
}

/**
 * Generates anchor points for BlazeFace Short Range model.
 * The model uses a fixed set of 896 anchors arranged in an SSD-style grid.
 *
 * Grid structure (128×128 input):
 *   - Layers at strides [8, 16] with [2, 6] anchors per location
 *
 * @returns {Anchor[]} Array of 896 anchor points in normalized [0,1] coordinates
 */
function generateAnchors(): Anchor[] {
  const anchors: Anchor[] = [];
  const strides = [8, 16];
  const anchorsPerStride = [2, 6];

  for (let strideIdx = 0; strideIdx < strides.length; strideIdx++) {
    const stride = strides[strideIdx];
    const gridSize = Math.floor(BLAZEFACE_INPUT_SIZE / stride);
    const numAnchorsAtStride = anchorsPerStride[strideIdx];

    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        const anchorX = (x + 0.5) / gridSize;
        const anchorY = (y + 0.5) / gridSize;

        for (let n = 0; n < numAnchorsAtStride; n++) {
          anchors.push({ x: anchorX, y: anchorY });
        }
      }
    }
  }

  return anchors;
}

/**
 * Extracts the face crop region from a detection, with padding for
 * better embedding extraction. Returns pixel coordinates expanded
 * by a margin factor.
 *
 * @param {FaceDetection} detection - The face detection to crop from
 * @param {number} frameWidth - Camera frame width in pixels
 * @param {number} frameHeight - Camera frame height in pixels
 * @param {number} [paddingFactor=0.3] - Extra margin around the face (30% default)
 * @returns {{ x: number; y: number; width: number; height: number }} Padded crop region
 */
export function getFaceCropRegion(
  detection: FaceDetection,
  frameWidth: number,
  frameHeight: number,
  paddingFactor: number = 0.3
): { x: number; y: number; width: number; height: number } {
  const { x, y, w, h } = detection.pixelBox;

  const padX = w * paddingFactor;
  const padY = h * paddingFactor;

  const cropX = Math.max(0, Math.round(x - padX));
  const cropY = Math.max(0, Math.round(y - padY));
  const cropW = Math.min(Math.round(w + 2 * padX), frameWidth - cropX);
  const cropH = Math.min(Math.round(h + 2 * padY), frameHeight - cropY);

  return { x: cropX, y: cropY, width: cropW, height: cropH };
}

/** Cached anchor array for repeated calls */
let cachedAnchors: Anchor[] | null = null;

/**
 * Returns cached anchors to avoid regeneration on every frame.
 * Anchors are deterministic and never change during runtime.
 *
 * @returns {Anchor[]} Cached array of 896 anchor points
 */
export function getCachedAnchors(): Anchor[] {
  if (!cachedAnchors) {
    cachedAnchors = generateAnchors();
  }
  return cachedAnchors;
}
