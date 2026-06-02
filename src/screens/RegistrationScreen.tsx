/**
 * @module RegistrationScreen
 * Captures face embeddings for a new worker registration.
 *
 * Flow:
 *   1. User enters worker name and ID
 *   2. Camera opens in portrait mode
 *   3. BlazeFace detects face → draws bounding box
 *   4. User taps "Capture" 3 times
 *   5. MobileFaceNet extracts embedding from each capture
 *   6. Embeddings are averaged and stored in SQLite
 *   7. Success confirmation shown
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  Alert,
  Animated,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { useRunOnJS } from 'react-native-worklets-core';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { BoundingBox } from '../components/BoundingBox';
import { insertWorker, workerExists } from '../database/WorkerRepository';
import { averageEmbeddings } from '../ml/CosineSimilarity';
import { extractFaceEmbedding } from '../ml/FaceEmbedder';
import { extractRgbPixelsFromFrame } from '../utils/ImagePreprocessor';
import {
  COLORS,
  RADIUS,
  SPACING,
  REGISTRATION_FRAME_COUNT,
  CAMERA_FACING,
  FACE_DETECTION_CONFIDENCE,
} from '../utils/Constants';

type RegistrationScreenProps = {
  navigation: NativeStackNavigationProp<any>;
};

/** Registration workflow stages */
type RegistrationStage = 'input' | 'camera' | 'capturing' | 'processing' | 'success';

/**
 * Registration screen for enrolling new workers with face embeddings.
 *
 * @param {RegistrationScreenProps} props - Navigation prop
 * @returns {React.ReactElement} The registration screen UI
 */
export default function RegistrationScreen({
  navigation,
}: RegistrationScreenProps): React.ReactElement {
  // Form state
  const [workerName, setWorkerName] = useState('');
  const [workerId, setWorkerId] = useState('');
  const [stage, setStage] = useState<RegistrationStage>('input');

  // Camera state
  const device = useCameraDevice(CAMERA_FACING);
  const { hasPermission, requestPermission } = useCameraPermission();

  // Face detection state
  const [faceDetected, setFaceDetected] = useState(false);
  const [faceBox, setFaceBox] = useState({ x: 0, y: 0, w: 0, h: 0 });

  // Capture state
  const [capturedCount, setCapturedCount] = useState(0);
  const [embeddings, setEmbeddings] = useState<Float32Array[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  // Ref to hold the latest camera frame so handleCapture (JS thread) can access it
  const frameProcessorFrame = useRef<any>(null);

  // Animations
  const successAnim = useRef(new Animated.Value(0)).current;

  // Load TFLite models
  const faceDetectionModel = useTensorflowModel(
    require('../../assets/models/blazeface.tflite')
  );
  const embeddingModel = useTensorflowModel(
    require('../../assets/models/mobilefacenet.tflite')
  );

  const modelsReady = faceDetectionModel.state === 'loaded' && embeddingModel.state === 'loaded';

  // Request camera permission on mount
  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  /**
   * Callback to update face detection state from frame processor.
   */
  const onFaceDetected = useRunOnJS((detected: boolean, x: number, y: number, w: number, h: number) => {
    setFaceDetected(detected);
    if (detected) {
      setFaceBox({ x, y, w, h });
    }
  }, []);

  /**
   * Callback to receive embedding from frame processor capture.
   */
  const onEmbeddingCaptured = useRunOnJS((embeddingArray: number[]) => {
    const embedding = new Float32Array(embeddingArray);
    setEmbeddings(prev => [...prev, embedding]);
    setCapturedCount(prev => prev + 1);
  }, []);

  /**
   * Frame processor — runs BlazeFace on each frame for live face detection.
   * Does NOT extract embeddings on every frame (only on explicit capture).
   */
  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    if (faceDetectionModel.state !== 'loaded') return;

    // Store the latest frame so handleCapture (running on JS thread) can access pixel data
    frameProcessorFrame.current = frame;

    try {
      const model = faceDetectionModel.model;
      const outputs = model.runSync([frame as any]);

      if (outputs && outputs.length >= 2) {
        // BlazeFace outputs: [regressors, classificators]
        const classificators = outputs[1] as Float32Array;

        // Check if any detection exceeds confidence threshold
        let bestScore = 0;
        let bestIdx = -1;
        const numAnchors = Math.min(classificators.length, 896);

        for (let i = 0; i < numAnchors; i++) {
          const score = 1 / (1 + Math.exp(-classificators[i])); // sigmoid
          if (score > bestScore) {
            bestScore = score;
            bestIdx = i;
          }
        }

        if (bestScore > FACE_DETECTION_CONFIDENCE && bestIdx >= 0) {
          const regressors = outputs[0] as Float32Array;
          const offset = bestIdx * 16;

          // Decode bbox (simplified — normalized coords)
          const cx = regressors[offset + 0];
          const cy = regressors[offset + 1];
          const w = regressors[offset + 2];
          const h = regressors[offset + 3];

          // Convert to pixel coords (approximate)
          const frameW = frame.width;
          const frameH = frame.height;
          const pixelX = Math.max(0, (cx - w / 2)) * frameW * 0.8;
          const pixelY = Math.max(0, (cy - h / 2)) * frameH * 0.8;
          const pixelW = w * frameW * 0.8;
          const pixelH = h * frameH * 0.8;

          onFaceDetected(true, pixelX, pixelY, pixelW, pixelH);
        } else {
          onFaceDetected(false, 0, 0, 0, 0);
        }
      }
    } catch {
      onFaceDetected(false, 0, 0, 0, 0);
    }
  }, [faceDetectionModel, onFaceDetected]);

  /**
   * Handles the capture button press — extracts embedding from current frame.
   */
  const handleCapture = useCallback(async () => {
    if (!faceDetected || !modelsReady || isProcessing) return;
    if (capturedCount >= REGISTRATION_FRAME_COUNT) return;

    setIsProcessing(true);

    try {
      // Ensure we have a recent camera frame stored by the frame processor
      const latestFrame = frameProcessorFrame.current;
      if (!latestFrame) {
        throw new Error('No camera frame available — please wait a moment and try again.');
      }

      // Use the face bounding box tracked by the live frame processor (already in state)
      const { x: bx, y: by, w: bw, h: bh } = faceBox;

      if (bw <= 0 || bh <= 0) {
        throw new Error('Face bounding box is invalid — please center your face and try again.');
      }

      // Step 1: Extract RGB pixel data for the detected face crop from the latest frame
      // extractRgbPixelsFromFrame uses frame.toArrayBuffer() (VisionCamera v4 API)
      // and slices the crop region from the RGB backing buffer
      const croppedRgbPixels = extractRgbPixelsFromFrame(latestFrame, bx, by, bw, bh);

      if (croppedRgbPixels.length === 0) {
        throw new Error(
          'Could not extract face pixels from frame. Please ensure camera permission is granted and try again.'
        );
      }

      // Step 2: Run MobileFaceNet on the cropped, preprocessed face image
      // extractFaceEmbedding handles: resize to 112×112 → normalize to [-1,1] → inference → L2 normalize
      const { embedding: realEmbedding, success, error: embError } = extractFaceEmbedding(
        croppedRgbPixels,
        Math.round(bw),
        Math.round(bh),
        (inputTensor) => embeddingModel.model.runSync([inputTensor]) as Float32Array
      );

      if (!success || !realEmbedding) {
        throw new Error(embError || 'Embedding extraction failed — please try again.');
      }

      // Step 3: Store embedding and update capture count
      setEmbeddings(prev => [...prev, realEmbedding]);
      setCapturedCount(prev => prev + 1);

      if (capturedCount + 1 >= REGISTRATION_FRAME_COUNT) {
        // All captures done — average embeddings and save to database
        setTimeout(() => saveRegistration([...embeddings, realEmbedding]), 500);
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Failed to capture face. Please try again.';
      Alert.alert('Capture Error', msg);
    } finally {
      setIsProcessing(false);
    }
  }, [faceDetected, modelsReady, isProcessing, capturedCount, faceBox, embeddingModel, embeddings, saveRegistration]);

  /**
   * Averages captured embeddings and saves the worker to the database.
   */
  const saveRegistration = useCallback(
    (allEmbeddings: Float32Array[]) => {
      setStage('processing');

      try {
        const averagedEmbedding = averageEmbeddings(allEmbeddings);
        insertWorker(workerId, workerName, averagedEmbedding);

        setStage('success');

        // Play success animation
        Animated.spring(successAnim, {
          toValue: 1,
          friction: 6,
          tension: 50,
          useNativeDriver: true,
        }).start();
      } catch (error) {
        const msg =
          error instanceof Error ? error.message : 'Unknown error during registration';
        Alert.alert('Registration Failed', msg);
        setStage('camera');
      }
    },
    [workerId, workerName, successAnim]
  );

  /**
   * Validates form inputs before proceeding to camera.
   */
  const handleProceedToCamera = useCallback(() => {
    const trimmedName = workerName.trim();
    const trimmedId = workerId.trim();

    if (!trimmedName) {
      Alert.alert('Missing Name', 'Please enter the worker\'s name.');
      return;
    }
    if (!trimmedId) {
      Alert.alert('Missing ID', 'Please enter the worker\'s ID.');
      return;
    }
    if (workerExists(trimmedId)) {
      Alert.alert(
        'ID Already Registered',
        `A worker with ID "${trimmedId}" already exists. Use a different ID or delete the existing registration.`
      );
      return;
    }

    setStage('camera');
  }, [workerName, workerId]);

  // ── Input Stage ──
  if (stage === 'input') {
    return (
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <View style={styles.inputContainer}>
          <Text style={styles.screenTitle}>Register Worker</Text>
          <Text style={styles.screenSubtitle}>
            Enter worker details, then capture 3 face images for enrollment.
          </Text>

          <View style={styles.formGroup}>
            <Text style={styles.label}>Worker Name</Text>
            <TextInput
              style={styles.textInput}
              placeholder="e.g., Rajesh Kumar"
              placeholderTextColor={COLORS.TEXT_MUTED}
              value={workerName}
              onChangeText={setWorkerName}
              autoCapitalize="words"
              returnKeyType="next"
            />
          </View>

          <View style={styles.formGroup}>
            <Text style={styles.label}>Worker ID</Text>
            <TextInput
              style={styles.textInput}
              placeholder="e.g., NHAI-2025-001"
              placeholderTextColor={COLORS.TEXT_MUTED}
              value={workerId}
              onChangeText={setWorkerId}
              autoCapitalize="characters"
              returnKeyType="done"
            />
          </View>

          <TouchableOpacity
            style={styles.proceedButton}
            activeOpacity={0.85}
            onPress={handleProceedToCamera}
          >
            <Text style={styles.proceedButtonText}>Open Camera →</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.backButton}
            onPress={() => navigation.goBack()}
          >
            <Text style={styles.backButtonText}>← Back</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    );
  }

  // ── Success Stage ──
  if (stage === 'success') {
    return (
      <View style={[styles.container, styles.successContainer]}>
        <Animated.View
          style={[
            styles.successContent,
            {
              opacity: successAnim,
              transform: [
                {
                  scale: successAnim.interpolate({
                    inputRange: [0, 1],
                    outputRange: [0.5, 1],
                  }),
                },
              ],
            },
          ]}
        >
          <Text style={styles.successIcon}>✓</Text>
          <Text style={styles.successTitle}>Registration Complete</Text>
          <Text style={styles.successSubtitle}>
            {workerName} (ID: {workerId}) has been registered successfully.
          </Text>
          <Text style={styles.successDetail}>
            {REGISTRATION_FRAME_COUNT} face captures averaged into a single embedding.
          </Text>

          <TouchableOpacity
            style={styles.proceedButton}
            activeOpacity={0.85}
            onPress={() => navigation.goBack()}
          >
            <Text style={styles.proceedButtonText}>Done</Text>
          </TouchableOpacity>
        </Animated.View>
      </View>
    );
  }

  // ── Processing Stage ──
  if (stage === 'processing') {
    return (
      <View style={[styles.container, styles.successContainer]}>
        <ActivityIndicator size="large" color={COLORS.PRIMARY} />
        <Text style={[styles.successTitle, { marginTop: SPACING.LG }]}>
          Processing Embeddings...
        </Text>
        <Text style={styles.successSubtitle}>
          Averaging {REGISTRATION_FRAME_COUNT} captures for optimal accuracy.
        </Text>
      </View>
    );
  }

  // ── Camera Stage ──
  if (!device) {
    return (
      <View style={[styles.container, styles.successContainer]}>
        <Text style={styles.successTitle}>No Camera Found</Text>
        <Text style={styles.successSubtitle}>
          This device does not have a {CAMERA_FACING} camera available.
        </Text>
      </View>
    );
  }

  if (!hasPermission) {
    return (
      <View style={[styles.container, styles.successContainer]}>
        <Text style={styles.successTitle}>Camera Permission Required</Text>
        <Text style={styles.successSubtitle}>
          Please grant camera access to register workers.
        </Text>
        <TouchableOpacity style={styles.proceedButton} onPress={requestPermission}>
          <Text style={styles.proceedButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Camera Preview */}
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={stage === 'camera'}
        frameProcessor={frameProcessor}
        pixelFormat="rgb"
      />

      {/* Bounding Box Overlay */}
      <BoundingBox
        x={faceBox.x}
        y={faceBox.y}
        width={faceBox.w}
        height={faceBox.h}
        color={faceDetected ? COLORS.SUCCESS : COLORS.ERROR}
        visible={faceDetected}
        label={faceDetected ? 'Face Detected' : undefined}
      />

      {/* Top status bar */}
      <View style={styles.cameraTopBar}>
        <TouchableOpacity onPress={() => { setStage('input'); setCapturedCount(0); setEmbeddings([]); }}>
          <Text style={styles.cameraBackText}>← Back</Text>
        </TouchableOpacity>
        <View style={styles.captureCounter}>
          <Text style={styles.captureCounterText}>
            {capturedCount} / {REGISTRATION_FRAME_COUNT}
          </Text>
        </View>
      </View>

      {/* Worker info overlay */}
      <View style={styles.workerInfoOverlay}>
        <Text style={styles.workerInfoName}>{workerName}</Text>
        <Text style={styles.workerInfoId}>ID: {workerId}</Text>
      </View>

      {/* Capture progress dots */}
      <View style={styles.captureDotsContainer}>
        {Array.from({ length: REGISTRATION_FRAME_COUNT }, (_, i) => (
          <View
            key={i}
            style={[
              styles.captureDot,
              i < capturedCount ? styles.captureDotFilled : styles.captureDotEmpty,
            ]}
          />
        ))}
      </View>

      {/* Model loading indicator */}
      {!modelsReady && (
        <View style={styles.modelLoadingOverlay}>
          <ActivityIndicator size="large" color={COLORS.PRIMARY} />
          <Text style={styles.modelLoadingText}>Loading ML models...</Text>
        </View>
      )}

      {/* Capture Button */}
      <View style={styles.captureButtonContainer}>
        <TouchableOpacity
          style={[
            styles.captureButton,
            (!faceDetected || !modelsReady || isProcessing) && styles.captureButtonDisabled,
          ]}
          activeOpacity={0.7}
          onPress={handleCapture}
          disabled={!faceDetected || !modelsReady || isProcessing || capturedCount >= REGISTRATION_FRAME_COUNT}
        >
          <View style={styles.captureButtonInner}>
            {isProcessing ? (
              <ActivityIndicator size="small" color={COLORS.BACKGROUND} />
            ) : (
              <Text style={styles.captureButtonText}>
                {capturedCount >= REGISTRATION_FRAME_COUNT ? '✓' : '📷'}
              </Text>
            )}
          </View>
        </TouchableOpacity>
        <Text style={styles.captureHintText}>
          {!faceDetected
            ? 'Position your face in the frame'
            : isProcessing
              ? 'Processing...'
              : capturedCount >= REGISTRATION_FRAME_COUNT
                ? 'All captures complete!'
                : `Tap to capture (${REGISTRATION_FRAME_COUNT - capturedCount} remaining)`}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.BACKGROUND,
  },
  inputContainer: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: SPACING.XL,
  },
  screenTitle: {
    fontSize: 28,
    fontWeight: '900',
    color: COLORS.TEXT_PRIMARY,
    marginBottom: SPACING.SM,
  },
  screenSubtitle: {
    fontSize: 14,
    color: COLORS.TEXT_SECONDARY,
    marginBottom: SPACING.XL,
    lineHeight: 20,
  },
  formGroup: {
    marginBottom: SPACING.LG,
  },
  label: {
    fontSize: 13,
    fontWeight: '700',
    color: COLORS.TEXT_SECONDARY,
    marginBottom: SPACING.SM,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  textInput: {
    backgroundColor: COLORS.SURFACE,
    borderRadius: RADIUS.MD,
    paddingHorizontal: SPACING.MD,
    paddingVertical: SPACING.MD,
    fontSize: 16,
    color: COLORS.TEXT_PRIMARY,
    borderWidth: 1,
    borderColor: COLORS.BORDER,
  },
  proceedButton: {
    backgroundColor: COLORS.PRIMARY,
    borderRadius: RADIUS.MD,
    paddingVertical: SPACING.MD,
    alignItems: 'center',
    marginTop: SPACING.LG,
  },
  proceedButtonText: {
    color: COLORS.BACKGROUND,
    fontSize: 16,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  backButton: {
    alignItems: 'center',
    marginTop: SPACING.MD,
    paddingVertical: SPACING.SM,
  },
  backButtonText: {
    color: COLORS.TEXT_MUTED,
    fontSize: 14,
  },
  // Camera overlay styles
  cameraTopBar: {
    position: 'absolute',
    top: SPACING.XXL,
    left: SPACING.MD,
    right: SPACING.MD,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    zIndex: 20,
  },
  cameraBackText: {
    color: COLORS.TEXT_PRIMARY,
    fontSize: 16,
    fontWeight: '600',
    backgroundColor: 'rgba(0,0,0,0.5)',
    paddingHorizontal: SPACING.MD,
    paddingVertical: SPACING.SM,
    borderRadius: RADIUS.SM,
  },
  captureCounter: {
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingHorizontal: SPACING.MD,
    paddingVertical: SPACING.SM,
    borderRadius: RADIUS.SM,
  },
  captureCounterText: {
    color: COLORS.TEXT_PRIMARY,
    fontSize: 16,
    fontWeight: '800',
    fontVariant: ['tabular-nums'],
  },
  workerInfoOverlay: {
    position: 'absolute',
    top: SPACING.XXL + 50,
    left: SPACING.MD,
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: SPACING.MD,
    paddingVertical: SPACING.SM,
    borderRadius: RADIUS.SM,
    zIndex: 20,
  },
  workerInfoName: {
    color: COLORS.TEXT_PRIMARY,
    fontSize: 15,
    fontWeight: '700',
  },
  workerInfoId: {
    color: COLORS.TEXT_SECONDARY,
    fontSize: 12,
  },
  captureDotsContainer: {
    position: 'absolute',
    bottom: 140,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'center',
    gap: SPACING.SM,
    zIndex: 20,
  },
  captureDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    borderWidth: 2,
  },
  captureDotFilled: {
    backgroundColor: COLORS.SUCCESS,
    borderColor: COLORS.SUCCESS,
  },
  captureDotEmpty: {
    backgroundColor: 'transparent',
    borderColor: COLORS.TEXT_SECONDARY,
  },
  captureButtonContainer: {
    position: 'absolute',
    bottom: SPACING.XXL,
    left: 0,
    right: 0,
    alignItems: 'center',
    zIndex: 20,
  },
  captureButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: COLORS.PRIMARY,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 4,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  captureButtonDisabled: {
    backgroundColor: COLORS.TEXT_MUTED,
    opacity: 0.6,
  },
  captureButtonInner: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: COLORS.PRIMARY,
    alignItems: 'center',
    justifyContent: 'center',
  },
  captureButtonText: {
    fontSize: 24,
  },
  captureHintText: {
    color: COLORS.TEXT_PRIMARY,
    fontSize: 13,
    fontWeight: '500',
    marginTop: SPACING.SM,
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: SPACING.MD,
    paddingVertical: SPACING.XS,
    borderRadius: RADIUS.SM,
  },
  modelLoadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.8)',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 50,
  },
  modelLoadingText: {
    color: COLORS.TEXT_PRIMARY,
    fontSize: 16,
    marginTop: SPACING.MD,
  },
  // Success styles
  successContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: SPACING.XL,
  },
  successContent: {
    alignItems: 'center',
  },
  successIcon: {
    fontSize: 64,
    color: COLORS.SUCCESS,
    marginBottom: SPACING.LG,
    backgroundColor: 'rgba(0, 230, 118, 0.1)',
    width: 100,
    height: 100,
    lineHeight: 100,
    textAlign: 'center',
    borderRadius: 50,
    overflow: 'hidden',
  },
  successTitle: {
    fontSize: 24,
    fontWeight: '900',
    color: COLORS.TEXT_PRIMARY,
    textAlign: 'center',
  },
  successSubtitle: {
    fontSize: 15,
    color: COLORS.TEXT_SECONDARY,
    textAlign: 'center',
    marginTop: SPACING.SM,
    lineHeight: 22,
  },
  successDetail: {
    fontSize: 13,
    color: COLORS.TEXT_MUTED,
    textAlign: 'center',
    marginTop: SPACING.SM,
  },
});
