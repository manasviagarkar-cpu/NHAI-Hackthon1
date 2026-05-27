/**
 * @module Constants
 * Central configuration for all thresholds, dimensions, and app-wide settings.
 * ALL tunable values live here — never hardcode thresholds inline.
 */

export const DEMO_MODE = false;

// =============================================================================
// Face Recognition
// =============================================================================

/** Dimensionality of the face embedding vector produced by MobileFaceNet */
export const EMBEDDING_DIM = 128;

/** Cosine similarity threshold for positive face match (0.0 – 1.0) */
export const COSINE_THRESHOLD = 0.75;

/** Number of frames to capture and average during registration */
export const REGISTRATION_FRAME_COUNT = 3;

/** Input size expected by MobileFaceNet (112×112 RGB) */
export const MOBILEFACENET_INPUT_SIZE = 112;

/** Pixel value normalization range for MobileFaceNet: [-1.0, 1.0] */
export const PIXEL_NORMALIZE_MEAN = 127.5;
export const PIXEL_NORMALIZE_STD = 127.5;

// =============================================================================
// Face Detection (BlazeFace)
// =============================================================================

/** Minimum confidence score for BlazeFace face detection */
export const FACE_DETECTION_CONFIDENCE = 0.7;

/** BlazeFace input size (128×128 RGB) */
export const BLAZEFACE_INPUT_SIZE = 128;

// =============================================================================
// Liveness Detection
// =============================================================================

/** Eye Aspect Ratio threshold below which a blink is detected */
export const EAR_BLINK_THRESHOLD = 0.25;

/** Number of consecutive frames with low EAR to confirm a blink */
export const EAR_CONSECUTIVE_FRAMES = 2;

/** Smile ratio threshold above which a smile is detected */
export const SMILE_THRESHOLD = 0.3;

/** Maximum time (ms) allowed to complete each liveness challenge */
export const LIVENESS_CHALLENGE_TIMEOUT_MS = 10000;

/** MediaPipe Face Mesh input size (192×192 RGB) */
export const FACEMESH_INPUT_SIZE = 192;

/** Total number of landmarks from MediaPipe Face Mesh */
export const FACEMESH_LANDMARK_COUNT = 468;

// =============================================================================
// MediaPipe Face Mesh Landmark Indices
// Landmark indices for Eye Aspect Ratio (EAR) calculation.
// Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
// =============================================================================

/** Left eye landmark indices for EAR: [outer, top1, top2, inner, bottom1, bottom2] */
export const LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144] as const;

/** Right eye landmark indices for EAR: [outer, top1, top2, inner, bottom1, bottom2] */
export const RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380] as const;

/** Upper lip top landmark index */
export const UPPER_LIP_TOP = 13;

/** Lower lip bottom landmark index */
export const LOWER_LIP_BOTTOM = 14;

/** Mouth left corner landmark index */
export const MOUTH_LEFT = 61;

/** Mouth right corner landmark index */
export const MOUTH_RIGHT = 291;

// =============================================================================
// Performance
// =============================================================================

/** Throttle interval (ms) for recognition inference — process every N ms, not every frame */
export const THROTTLE_MS = 200;

/** Maximum acceptable inference time (ms) for a single recognition pass */
export const MAX_INFERENCE_TIME_MS = 1000;

/** Inference timer color thresholds */
export const INFERENCE_TIMER_GREEN_MS = 500;
export const INFERENCE_TIMER_YELLOW_MS = 1000;

// =============================================================================
// Camera
// =============================================================================

/** Camera facing direction */
export const CAMERA_FACING: 'front' | 'back' = 'front';

/** Camera frame target FPS */
export const CAMERA_FPS = 30;

/** Photo quality for registration frames (0.0 – 1.0) */
export const PHOTO_QUALITY = 0.85;

// =============================================================================
// Database
// =============================================================================

/** SQLite database filename */
export const DB_NAME = 'nhai_facerec.db';

/** AsyncStorage keys */
export const STORAGE_KEYS = {
  LAST_SYNC_TIMESTAMP: '@nhai/last_sync_timestamp',
  APP_INITIALIZED: '@nhai/app_initialized',
  TOTAL_RECOGNITIONS: '@nhai/total_recognitions',
} as const;

// =============================================================================
// Model File Paths (relative to assets/models/)
// =============================================================================

/** TFLite model asset filenames */
export const MODEL_FILES = {
  MOBILEFACENET: 'mobilefacenet.tflite',
  BLAZEFACE: 'blazeface.tflite',
  FACE_LANDMARK: 'face_landmark.tflite',
} as const;

// =============================================================================
// UI
// =============================================================================

/** App color palette */
export const COLORS = {
  PRIMARY: '#00C896',
  PRIMARY_DARK: '#00A67A',
  ACCENT: '#FFB800',
  BACKGROUND: '#0A0A0A',
  SURFACE: '#1A1A2E',
  SURFACE_LIGHT: '#242444',
  TEXT_PRIMARY: '#FFFFFF',
  TEXT_SECONDARY: '#A0A0B8',
  TEXT_MUTED: '#6B6B80',
  SUCCESS: '#00E676',
  WARNING: '#FFB800',
  ERROR: '#FF5252',
  BORDER: '#2A2A3E',
  OVERLAY: 'rgba(0, 0, 0, 0.7)',
  CARD_GRADIENT_START: '#1E1E3A',
  CARD_GRADIENT_END: '#16162B',
} as const;

/** Border radius values */
export const RADIUS = {
  SM: 8,
  MD: 12,
  LG: 16,
  XL: 24,
  FULL: 999,
} as const;

/** Spacing scale */
export const SPACING = {
  XS: 4,
  SM: 8,
  MD: 16,
  LG: 24,
  XL: 32,
  XXL: 48,
} as const;
