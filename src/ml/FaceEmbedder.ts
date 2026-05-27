/**
 * @module FaceEmbedder
 * Wrapper for the MobileFaceNet TFLite model that extracts 128-dimensional
 * face embeddings from cropped face images.
 *
 * Pipeline:
 *   1. Receive cropped face region (from BlazeFace detection)
 *   2. Resize to 112×112 pixels
 *   3. Normalize pixel values to [-1.0, 1.0]
 *   4. Run MobileFaceNet inference
 *   5. Return L2-normalized 128-dim embedding
 *
 * The embedding can then be compared via cosine similarity for recognition.
 */

import {
  EMBEDDING_DIM,
  MOBILEFACENET_INPUT_SIZE,
  PIXEL_NORMALIZE_MEAN,
  PIXEL_NORMALIZE_STD,
} from '../utils/Constants';

/**
 * Result of a face embedding extraction.
 */
export interface EmbeddingResult {
  /** The 128-dimensional face embedding vector */
  embedding: Float32Array;
  /** Time taken for preprocessing + inference in milliseconds */
  inferenceTimeMs: number;
  /** Whether the extraction was successful */
  success: boolean;
  /** Error message if extraction failed */
  error?: string;
}

/**
 * Preprocesses raw pixel data for MobileFaceNet input.
 * Converts RGB uint8 pixel values to float32 normalized to [-1.0, 1.0].
 *
 * MobileFaceNet expects: (pixel_value - 127.5) / 127.5
 * This maps [0, 255] → [-1.0, 1.0]
 *
 * @param {Uint8Array} rgbPixels - Raw RGB pixel data (width × height × 3)
 * @param {number} width - Input image width in pixels
 * @param {number} height - Input image height in pixels
 * @returns {Float32Array} Normalized float32 tensor of shape [1, 112, 112, 3]
 */
export function preprocessForMobileFaceNet(
  rgbPixels: Uint8Array,
  width: number,
  height: number
): Float32Array {
  const inputSize = MOBILEFACENET_INPUT_SIZE;
  const tensorData = new Float32Array(inputSize * inputSize * 3);

  // Bilinear interpolation for resizing to 112×112
  const scaleX = width / inputSize;
  const scaleY = height / inputSize;

  for (let y = 0; y < inputSize; y++) {
    for (let x = 0; x < inputSize; x++) {
      // Source coordinates in the original image
      const srcX = x * scaleX;
      const srcY = y * scaleY;

      // Nearest-neighbor for speed (bilinear is more accurate but slower)
      const srcXi = Math.min(Math.floor(srcX), width - 1);
      const srcYi = Math.min(Math.floor(srcY), height - 1);

      const srcIdx = (srcYi * width + srcXi) * 3;
      const dstIdx = (y * inputSize + x) * 3;

      // Normalize each channel: (value - 127.5) / 127.5 → [-1.0, 1.0]
      tensorData[dstIdx + 0] = (rgbPixels[srcIdx + 0] - PIXEL_NORMALIZE_MEAN) / PIXEL_NORMALIZE_STD;
      tensorData[dstIdx + 1] = (rgbPixels[srcIdx + 1] - PIXEL_NORMALIZE_MEAN) / PIXEL_NORMALIZE_STD;
      tensorData[dstIdx + 2] = (rgbPixels[srcIdx + 2] - PIXEL_NORMALIZE_MEAN) / PIXEL_NORMALIZE_STD;
    }
  }

  return tensorData;
}

/**
 * Post-processes the raw TFLite output into an L2-normalized embedding.
 * MobileFaceNet outputs a raw 128-dim vector which needs normalization
 * for stable cosine similarity comparisons.
 *
 * @param {Float32Array} rawOutput - Raw model output (128 floats)
 * @returns {Float32Array} L2-normalized 128-dimensional embedding
 */
export function postprocessEmbedding(rawOutput: Float32Array): Float32Array {
  if (rawOutput.length < EMBEDDING_DIM) {
    throw new Error(
      `Model output too short: expected at least ${EMBEDDING_DIM} values, got ${rawOutput.length}`
    );
  }

  // Extract the first 128 values (model may output more)
  const embedding = new Float32Array(EMBEDDING_DIM);
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    embedding[i] = rawOutput[i];
  }

  // L2 normalization: v / ||v||
  let norm = 0;
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    norm += embedding[i] * embedding[i];
  }
  norm = Math.sqrt(norm);

  if (norm > 1e-10) {
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      embedding[i] /= norm;
    }
  }

  return embedding;
}

/**
 * Converts RGBA pixel buffer (from camera frame) to RGB by stripping
 * the alpha channel. VisionCamera frame processors output RGBA.
 *
 * @param {Uint8Array} rgbaPixels - RGBA pixel data (width × height × 4)
 * @param {number} pixelCount - Total number of pixels (width × height)
 * @returns {Uint8Array} RGB pixel data (width × height × 3)
 */
export function rgbaToRgb(rgbaPixels: Uint8Array, pixelCount: number): Uint8Array {
  const rgb = new Uint8Array(pixelCount * 3);
  for (let i = 0; i < pixelCount; i++) {
    rgb[i * 3 + 0] = rgbaPixels[i * 4 + 0]; // R
    rgb[i * 3 + 1] = rgbaPixels[i * 4 + 1]; // G
    rgb[i * 3 + 2] = rgbaPixels[i * 4 + 2]; // B
  }
  return rgb;
}

/**
 * Crops a region from an RGB pixel buffer.
 * Used to extract the face region detected by BlazeFace before
 * passing to MobileFaceNet.
 *
 * @param {Uint8Array} rgbPixels - Full frame RGB data
 * @param {number} frameWidth - Full frame width
 * @param {number} cropX - X coordinate of crop origin
 * @param {number} cropY - Y coordinate of crop origin
 * @param {number} cropWidth - Width of crop region
 * @param {number} cropHeight - Height of crop region
 * @returns {Uint8Array} Cropped RGB pixel data
 */
export function cropRgbRegion(
  rgbPixels: Uint8Array,
  frameWidth: number,
  cropX: number,
  cropY: number,
  cropWidth: number,
  cropHeight: number
): Uint8Array {
  const cropped = new Uint8Array(cropWidth * cropHeight * 3);

  for (let y = 0; y < cropHeight; y++) {
    const srcRow = (cropY + y) * frameWidth * 3;
    const dstRow = y * cropWidth * 3;

    for (let x = 0; x < cropWidth; x++) {
      const srcIdx = srcRow + (cropX + x) * 3;
      const dstIdx = dstRow + x * 3;

      cropped[dstIdx + 0] = rgbPixels[srcIdx + 0];
      cropped[dstIdx + 1] = rgbPixels[srcIdx + 1];
      cropped[dstIdx + 2] = rgbPixels[srcIdx + 2];
    }
  }

  return cropped;
}

/**
 * Full pipeline: takes a cropped face image and returns its embedding.
 * This is the main entry point used by the recognition and registration screens.
 *
 * @param {Uint8Array} faceRgbPixels - Cropped face region in RGB format
 * @param {number} faceWidth - Width of the cropped face image
 * @param {number} faceHeight - Height of the cropped face image
 * @param {(input: Float32Array) => Float32Array} runInference - TFLite model inference function
 * @returns {EmbeddingResult} The extraction result with embedding and timing
 */
export function extractFaceEmbedding(
  faceRgbPixels: Uint8Array,
  faceWidth: number,
  faceHeight: number,
  runInference: (input: Float32Array) => Float32Array
): EmbeddingResult {
  const startTime = performance.now();

  try {
    // Step 1: Preprocess — resize to 112×112 and normalize to [-1, 1]
    const tensorInput = preprocessForMobileFaceNet(
      faceRgbPixels,
      faceWidth,
      faceHeight
    );

    // Step 2: Run MobileFaceNet inference
    const rawOutput = runInference(tensorInput);

    // Step 3: Post-process — extract 128-dim and L2-normalize
    const embedding = postprocessEmbedding(rawOutput);

    const inferenceTimeMs = performance.now() - startTime;

    return {
      embedding,
      inferenceTimeMs,
      success: true,
    };
  } catch (error) {
    const inferenceTimeMs = performance.now() - startTime;
    return {
      embedding: new Float32Array(EMBEDDING_DIM),
      inferenceTimeMs,
      success: false,
      error: error instanceof Error ? error.message : 'Unknown embedding error',
    };
  }
}
