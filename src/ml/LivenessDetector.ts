/**
 * @module LivenessDetector
 * Anti-spoofing module using MediaPipe Face Mesh 468-landmark output.
 * Detects blinks (Eye Aspect Ratio) and smiles (mouth curve ratio)
 * to verify the subject is a live person, not a printed photo or screen.
 *
 * Liveness Challenge Flow:
 *   1. "Please blink" → EAR drops below threshold for N consecutive frames
 *   2. "Please smile" → Mouth curve ratio exceeds threshold
 *   3. Both passed → Subject is live → proceed to recognition
 *
 * Landmark Reference:
 *   MediaPipe Face Mesh outputs 468 3D landmarks.
 *   See: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
 */

import {
  EAR_BLINK_THRESHOLD,
  EAR_CONSECUTIVE_FRAMES,
  SMILE_THRESHOLD,
  LEFT_EYE_LANDMARKS,
  RIGHT_EYE_LANDMARKS,
  UPPER_LIP_TOP,
  LOWER_LIP_BOTTOM,
  MOUTH_LEFT,
  MOUTH_RIGHT,
  FACEMESH_LANDMARK_COUNT,
} from '../utils/Constants';

/**
 * Represents a 2D/3D facial landmark point.
 */
export interface Landmark {
  /** X coordinate (normalized 0.0 – 1.0 or pixel coords) */
  x: number;
  /** Y coordinate (normalized 0.0 – 1.0 or pixel coords) */
  y: number;
  /** Z coordinate (depth, normalized — may be zero for 2D models) */
  z: number;
}

/**
 * Liveness challenge types that the detector supports.
 */
export type LivenessChallenge = 'blink' | 'smile';

/**
 * Current state of the liveness detection process.
 */
export interface LivenessState {
  /** The current challenge being presented */
  currentChallenge: LivenessChallenge;
  /** Whether the blink challenge has been completed */
  blinkDetected: boolean;
  /** Whether the smile challenge has been completed */
  smileDetected: boolean;
  /** Current Eye Aspect Ratio value (for debug display) */
  currentEAR: number;
  /** Current smile ratio value (for debug display) */
  currentSmileRatio: number;
  /** Number of consecutive low-EAR frames (for blink detection) */
  consecutiveLowEARFrames: number;
  /** Whether all liveness challenges have been passed */
  isLive: boolean;
}

/**
 * Parses the raw Face Mesh TFLite output tensor into an array of 468 landmarks.
 * The model outputs a flat array of [x, y, z] triplets.
 *
 * @param {Float32Array} rawOutput - Raw model output (468 × 3 = 1404 values)
 * @returns {Landmark[]} Array of 468 parsed landmark points
 * @throws {Error} If output size doesn't match expected landmark count
 */
export function parseFaceMeshOutput(rawOutput: Float32Array): Landmark[] {
  const expectedSize = FACEMESH_LANDMARK_COUNT * 3;

  if (rawOutput.length < expectedSize) {
    throw new Error(
      `Face Mesh output too short: expected ${expectedSize} values, got ${rawOutput.length}`
    );
  }

  const landmarks: Landmark[] = [];

  for (let i = 0; i < FACEMESH_LANDMARK_COUNT; i++) {
    landmarks.push({
      x: rawOutput[i * 3 + 0],
      y: rawOutput[i * 3 + 1],
      z: rawOutput[i * 3 + 2],
    });
  }

  return landmarks;
}

/**
 * Calculates the Euclidean distance between two 2D landmark points.
 *
 * @param {Landmark} p1 - First point
 * @param {Landmark} p2 - Second point
 * @returns {number} Euclidean distance
 */
function euclideanDistance(p1: Landmark, p2: Landmark): number {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Computes the Eye Aspect Ratio (EAR) for a single eye.
 *
 * EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
 *
 * Where the 6 landmarks are ordered as:
 *   p1 = outer corner, p2 = upper-outer, p3 = upper-inner
 *   p4 = inner corner, p5 = lower-inner, p6 = lower-outer
 *
 * When the eye is open, EAR ≈ 0.3–0.4
 * When the eye is closed, EAR drops below 0.25
 *
 * Reference: Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks" (2016)
 *
 * @param {Landmark[]} landmarks - Full array of 468 face mesh landmarks
 * @param {readonly number[]} eyeIndices - 6 landmark indices for this eye [outer, top1, top2, inner, bottom1, bottom2]
 * @returns {number} Eye Aspect Ratio value
 */
export function calculateEAR(
  landmarks: Landmark[],
  eyeIndices: readonly number[]
): number {
  const p1 = landmarks[eyeIndices[0]]; // outer corner
  const p2 = landmarks[eyeIndices[1]]; // upper outer
  const p3 = landmarks[eyeIndices[2]]; // upper inner
  const p4 = landmarks[eyeIndices[3]]; // inner corner
  const p5 = landmarks[eyeIndices[4]]; // lower inner
  const p6 = landmarks[eyeIndices[5]]; // lower outer

  // Vertical distances (eye height)
  const vertical1 = euclideanDistance(p2, p6);
  const vertical2 = euclideanDistance(p3, p5);

  // Horizontal distance (eye width)
  const horizontal = euclideanDistance(p1, p4);

  // Prevent division by zero
  if (horizontal < 1e-6) {
    return 0;
  }

  return (vertical1 + vertical2) / (2.0 * horizontal);
}

/**
 * Computes the average EAR across both eyes.
 * Using both eyes reduces noise from asymmetric expressions.
 *
 * @param {Landmark[]} landmarks - Full array of 468 face mesh landmarks
 * @returns {number} Average Eye Aspect Ratio
 */
export function calculateAverageEAR(landmarks: Landmark[]): number {
  const leftEAR = calculateEAR(landmarks, LEFT_EYE_LANDMARKS);
  const rightEAR = calculateEAR(landmarks, RIGHT_EYE_LANDMARKS);
  return (leftEAR + rightEAR) / 2.0;
}

/**
 * Computes the smile ratio from mouth landmarks.
 *
 * Smile Ratio = mouth_height / mouth_width
 *
 * When the mouth is neutral, ratio ≈ 0.1–0.2
 * When smiling, mouth widens and ratio changes → we use a threshold of 0.3
 *
 * Actually for smile detection, we look at the vertical stretch of the mouth
 * relative to its width. A smile widens the mouth and slightly opens it.
 *
 * @param {Landmark[]} landmarks - Full array of 468 face mesh landmarks
 * @returns {number} Smile ratio value
 */
export function calculateSmileRatio(landmarks: Landmark[]): number {
  const upperLip = landmarks[UPPER_LIP_TOP];
  const lowerLip = landmarks[LOWER_LIP_BOTTOM];
  const mouthLeft = landmarks[MOUTH_LEFT];
  const mouthRight = landmarks[MOUTH_RIGHT];

  const mouthHeight = euclideanDistance(upperLip, lowerLip);
  const mouthWidth = euclideanDistance(mouthLeft, mouthRight);

  // Prevent division by zero
  if (mouthWidth < 1e-6) {
    return 0;
  }

  return mouthHeight / mouthWidth;
}

/**
 * Creates a fresh liveness state for starting a new liveness check session.
 *
 * @returns {LivenessState} Initial liveness state with blink challenge active
 */
export function createInitialLivenessState(): LivenessState {
  return {
    currentChallenge: 'blink',
    blinkDetected: false,
    smileDetected: false,
    currentEAR: 0,
    currentSmileRatio: 0,
    consecutiveLowEARFrames: 0,
    isLive: false,
  };
}

/**
 * Processes a single frame's landmarks against the current liveness state.
 * This is the main function called on each camera frame during liveness check.
 *
 * State machine flow:
 *   1. If blink not yet detected → check EAR for blink
 *   2. If blink detected but smile not → check smile ratio
 *   3. If both detected → mark as live
 *
 * @param {Landmark[]} landmarks - 468 face mesh landmarks from current frame
 * @param {LivenessState} currentState - Current liveness detection state
 * @returns {LivenessState} Updated liveness state
 */
export function processLivenessFrame(
  landmarks: Landmark[],
  currentState: LivenessState
): LivenessState {
  const newState = { ...currentState };

  // Calculate current metrics
  const ear = calculateAverageEAR(landmarks);
  const smileRatio = calculateSmileRatio(landmarks);

  newState.currentEAR = ear;
  newState.currentSmileRatio = smileRatio;

  // ── Challenge 1: Blink Detection ──
  if (!newState.blinkDetected) {
    newState.currentChallenge = 'blink';

    if (ear < EAR_BLINK_THRESHOLD) {
      newState.consecutiveLowEARFrames += 1;
    } else {
      // EAR went back up — if we had enough consecutive low frames, it was a blink
      if (newState.consecutiveLowEARFrames >= EAR_CONSECUTIVE_FRAMES) {
        newState.blinkDetected = true;
        newState.currentChallenge = 'smile';
      }
      newState.consecutiveLowEARFrames = 0;
    }

    return newState;
  }

  // ── Challenge 2: Smile Detection ──
  if (!newState.smileDetected) {
    newState.currentChallenge = 'smile';

    if (smileRatio > SMILE_THRESHOLD) {
      newState.smileDetected = true;
    }
  }

  // ── Both challenges passed ──
  if (newState.blinkDetected && newState.smileDetected) {
    newState.isLive = true;
  }

  return newState;
}

/**
 * Preprocesses raw pixel data for the Face Mesh model input.
 * MediaPipe Face Mesh expects 192×192 RGB input normalized to [0.0, 1.0].
 *
 * @param {Uint8Array} rgbPixels - Raw RGB pixel data
 * @param {number} width - Input image width
 * @param {number} height - Input image height
 * @param {number} targetSize - Target resize dimension (192 for Face Mesh)
 * @returns {Float32Array} Normalized float32 tensor
 */
export function preprocessForFaceMesh(
  rgbPixels: Uint8Array,
  width: number,
  height: number,
  targetSize: number
): Float32Array {
  const tensorData = new Float32Array(targetSize * targetSize * 3);

  const scaleX = width / targetSize;
  const scaleY = height / targetSize;

  for (let y = 0; y < targetSize; y++) {
    for (let x = 0; x < targetSize; x++) {
      const srcXi = Math.min(Math.floor(x * scaleX), width - 1);
      const srcYi = Math.min(Math.floor(y * scaleY), height - 1);

      const srcIdx = (srcYi * width + srcXi) * 3;
      const dstIdx = (y * targetSize + x) * 3;

      // Normalize to [0.0, 1.0] (Face Mesh uses this range, unlike MobileFaceNet's [-1, 1])
      tensorData[dstIdx + 0] = rgbPixels[srcIdx + 0] / 255.0;
      tensorData[dstIdx + 1] = rgbPixels[srcIdx + 1] / 255.0;
      tensorData[dstIdx + 2] = rgbPixels[srcIdx + 2] / 255.0;
    }
  }

  return tensorData;
}
