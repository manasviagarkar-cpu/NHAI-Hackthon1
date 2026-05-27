/**
 * @module ImagePreprocessor
 * Utility functions for camera frame preprocessing before ML inference.
 * Handles the conversion from VisionCamera frame formats to the tensors
 * expected by our TFLite models.
 *
 * VisionCamera v4 frame processors provide frames in platform-specific formats.
 * This module abstracts those differences and provides a unified preprocessing API.
 */

import {
  MOBILEFACENET_INPUT_SIZE,
  BLAZEFACE_INPUT_SIZE,
  FACEMESH_INPUT_SIZE,
  PIXEL_NORMALIZE_MEAN,
  PIXEL_NORMALIZE_STD,
} from './Constants';

/**
 * Describes the dimensions and format of a camera frame.
 */
export interface FrameInfo {
  /** Frame width in pixels */
  width: number;
  /** Frame height in pixels */
  height: number;
  /** Bytes per row (may include padding) */
  bytesPerRow: number;
  /** Pixel format string */
  pixelFormat: 'rgb' | 'rgba' | 'bgra' | 'yuv';
}

/**
 * Resizes an RGB image using nearest-neighbor interpolation.
 * Fastest method — suitable for real-time frame processing where
 * bilinear quality improvement is negligible at 112×112 target.
 *
 * @param {Uint8Array} srcPixels - Source RGB pixel data
 * @param {number} srcWidth - Source image width
 * @param {number} srcHeight - Source image height
 * @param {number} dstWidth - Target width
 * @param {number} dstHeight - Target height
 * @returns {Uint8Array} Resized RGB pixel data
 */
export function resizeRgbNearest(
  srcPixels: Uint8Array,
  srcWidth: number,
  srcHeight: number,
  dstWidth: number,
  dstHeight: number
): Uint8Array {
  const dst = new Uint8Array(dstWidth * dstHeight * 3);
  const scaleX = srcWidth / dstWidth;
  const scaleY = srcHeight / dstHeight;

  for (let y = 0; y < dstHeight; y++) {
    const srcY = Math.min(Math.floor(y * scaleY), srcHeight - 1);
    for (let x = 0; x < dstWidth; x++) {
      const srcX = Math.min(Math.floor(x * scaleX), srcWidth - 1);

      const srcIdx = (srcY * srcWidth + srcX) * 3;
      const dstIdx = (y * dstWidth + x) * 3;

      dst[dstIdx + 0] = srcPixels[srcIdx + 0];
      dst[dstIdx + 1] = srcPixels[srcIdx + 1];
      dst[dstIdx + 2] = srcPixels[srcIdx + 2];
    }
  }

  return dst;
}

/**
 * Converts BGRA pixel buffer to RGB. Common on iOS where the camera
 * outputs in BGRA format.
 *
 * @param {Uint8Array} bgraPixels - BGRA pixel data (width × height × 4)
 * @param {number} pixelCount - Total number of pixels
 * @returns {Uint8Array} RGB pixel data (width × height × 3)
 */
export function bgraToRgb(bgraPixels: Uint8Array, pixelCount: number): Uint8Array {
  const rgb = new Uint8Array(pixelCount * 3);

  for (let i = 0; i < pixelCount; i++) {
    const bgraIdx = i * 4;
    const rgbIdx = i * 3;

    rgb[rgbIdx + 0] = bgraPixels[bgraIdx + 2]; // R ← third byte in BGRA
    rgb[rgbIdx + 1] = bgraPixels[bgraIdx + 1]; // G ← second byte
    rgb[rgbIdx + 2] = bgraPixels[bgraIdx + 0]; // B ← first byte
  }

  return rgb;
}

/**
 * Creates a float32 tensor normalized for BlazeFace input.
 * BlazeFace expects 128×128 RGB normalized to [0.0, 1.0].
 *
 * @param {Uint8Array} rgbPixels - Full frame RGB data
 * @param {number} width - Frame width
 * @param {number} height - Frame height
 * @returns {Float32Array} Tensor of shape [1, 128, 128, 3] normalized to [0, 1]
 */
export function prepareBlazeFaceInput(
  rgbPixels: Uint8Array,
  width: number,
  height: number
): Float32Array {
  const inputSize = BLAZEFACE_INPUT_SIZE;
  const tensor = new Float32Array(inputSize * inputSize * 3);
  const scaleX = width / inputSize;
  const scaleY = height / inputSize;

  for (let y = 0; y < inputSize; y++) {
    const srcY = Math.min(Math.floor(y * scaleY), height - 1);
    for (let x = 0; x < inputSize; x++) {
      const srcX = Math.min(Math.floor(x * scaleX), width - 1);

      const srcIdx = (srcY * width + srcX) * 3;
      const dstIdx = (y * inputSize + x) * 3;

      // Normalize to [0.0, 1.0]
      tensor[dstIdx + 0] = rgbPixels[srcIdx + 0] / 255.0;
      tensor[dstIdx + 1] = rgbPixels[srcIdx + 1] / 255.0;
      tensor[dstIdx + 2] = rgbPixels[srcIdx + 2] / 255.0;
    }
  }

  return tensor;
}

/**
 * Creates a float32 tensor normalized for MobileFaceNet input.
 * MobileFaceNet expects 112×112 RGB normalized to [-1.0, 1.0].
 *
 * @param {Uint8Array} rgbPixels - Cropped face RGB data
 * @param {number} width - Crop width
 * @param {number} height - Crop height
 * @returns {Float32Array} Tensor of shape [1, 112, 112, 3] normalized to [-1, 1]
 */
export function prepareMobileFaceNetInput(
  rgbPixels: Uint8Array,
  width: number,
  height: number
): Float32Array {
  const inputSize = MOBILEFACENET_INPUT_SIZE;
  const tensor = new Float32Array(inputSize * inputSize * 3);
  const scaleX = width / inputSize;
  const scaleY = height / inputSize;

  for (let y = 0; y < inputSize; y++) {
    const srcY = Math.min(Math.floor(y * scaleY), height - 1);
    for (let x = 0; x < inputSize; x++) {
      const srcX = Math.min(Math.floor(x * scaleX), width - 1);

      const srcIdx = (srcY * width + srcX) * 3;
      const dstIdx = (y * inputSize + x) * 3;

      // Normalize to [-1.0, 1.0]: (pixel - 127.5) / 127.5
      tensor[dstIdx + 0] = (rgbPixels[srcIdx + 0] - PIXEL_NORMALIZE_MEAN) / PIXEL_NORMALIZE_STD;
      tensor[dstIdx + 1] = (rgbPixels[srcIdx + 1] - PIXEL_NORMALIZE_MEAN) / PIXEL_NORMALIZE_STD;
      tensor[dstIdx + 2] = (rgbPixels[srcIdx + 2] - PIXEL_NORMALIZE_MEAN) / PIXEL_NORMALIZE_STD;
    }
  }

  return tensor;
}

/**
 * Creates a float32 tensor normalized for MediaPipe Face Mesh input.
 * Face Mesh expects 192×192 RGB normalized to [0.0, 1.0].
 *
 * @param {Uint8Array} rgbPixels - Cropped face RGB data
 * @param {number} width - Crop width
 * @param {number} height - Crop height
 * @returns {Float32Array} Tensor of shape [1, 192, 192, 3] normalized to [0, 1]
 */
export function prepareFaceMeshInput(
  rgbPixels: Uint8Array,
  width: number,
  height: number
): Float32Array {
  const inputSize = FACEMESH_INPUT_SIZE;
  const tensor = new Float32Array(inputSize * inputSize * 3);
  const scaleX = width / inputSize;
  const scaleY = height / inputSize;

  for (let y = 0; y < inputSize; y++) {
    const srcY = Math.min(Math.floor(y * scaleY), height - 1);
    for (let x = 0; x < inputSize; x++) {
      const srcX = Math.min(Math.floor(x * scaleX), width - 1);

      const srcIdx = (srcY * width + srcX) * 3;
      const dstIdx = (y * inputSize + x) * 3;

      // Normalize to [0.0, 1.0]
      tensor[dstIdx + 0] = rgbPixels[srcIdx + 0] / 255.0;
      tensor[dstIdx + 1] = rgbPixels[srcIdx + 1] / 255.0;
      tensor[dstIdx + 2] = rgbPixels[srcIdx + 2] / 255.0;
    }
  }

  return tensor;
}

/**
 * Validates that pixel data dimensions match expected buffer size.
 * Used as a safety check before processing to prevent buffer overrun.
 *
 * @param {Uint8Array} pixels - Raw pixel data
 * @param {number} width - Expected width
 * @param {number} height - Expected height
 * @param {number} channels - Number of channels (3 for RGB, 4 for RGBA/BGRA)
 * @returns {boolean} True if buffer size matches expected dimensions
 */
export function validatePixelBuffer(
  pixels: Uint8Array,
  width: number,
  height: number,
  channels: number
): boolean {
  const expectedSize = width * height * channels;
  return pixels.length >= expectedSize;
}
