/**
 * @module LivenessScreen
 * Anti-spoofing verification screen that runs before face recognition.
 *
 * Presents sequential challenges:
 *   1. "Please Blink" — detects eye blink via EAR
 *   2. "Please Smile" — detects mouth curve change
 *
 * Uses MediaPipe Face Mesh (468 landmarks) via react-native-fast-tflite
 * for real-time landmark extraction on the camera thread.
 *
 * On success → navigates to RecognitionScreen.
 * On timeout or failure → shows retry option.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
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
import { LivenessPrompt } from '../components/LivenessPrompt';
import {
  createInitialLivenessState,
  LivenessState,
  LivenessChallenge,
} from '../ml/LivenessDetector';
import {
  COLORS,
  SPACING,
  RADIUS,
  CAMERA_FACING,
  EAR_BLINK_THRESHOLD,
  EAR_CONSECUTIVE_FRAMES,
  SMILE_THRESHOLD,
  LIVENESS_CHALLENGE_TIMEOUT_MS,
  FACEMESH_LANDMARK_COUNT,
  DEMO_MODE,
} from '../utils/Constants';

type LivenessScreenProps = {
  navigation: NativeStackNavigationProp<any>;
};

/**
 * Liveness detection screen with real-time face mesh landmark analysis.
 *
 * @param {LivenessScreenProps} props - Navigation prop
 * @returns {React.ReactElement} The liveness check UI
 */
export default function LivenessScreen({
  navigation,
}: LivenessScreenProps): React.ReactElement {
  const device = useCameraDevice(CAMERA_FACING);
  const { hasPermission, requestPermission } = useCameraPermission();

  // Liveness state
  const [livenessState, setLivenessState] = useState<LivenessState>(
    createInitialLivenessState()
  );
  const [timeRemaining, setTimeRemaining] = useState(
    LIVENESS_CHALLENGE_TIMEOUT_MS / 1000
  );
  const [hasTimedOut, setHasTimedOut] = useState(false);

  // Refs for frame processor communication
  const consecutiveLowEAR = useRef(0);
  const blinkDetected = useRef(false);
  const smileDetected = useRef(false);

  // Load Face Mesh model
  const faceMeshModel = useTensorflowModel(
    require('../../assets/models/face_landmark.tflite')
  );
  const modelReady = faceMeshModel.state === 'loaded';

  // Request camera permission
  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  // Countdown timer
  useEffect(() => {
    if (livenessState.isLive || hasTimedOut) return;

    const interval = setInterval(() => {
      setTimeRemaining(prev => {
        if (prev <= 1) {
          setHasTimedOut(true);
          clearInterval(interval);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [livenessState.isLive, hasTimedOut]);

  // Simulator/Demo Fallback logic
  useEffect(() => {
    if (livenessState.isLive || hasTimedOut) return;
    if (DEMO_MODE) {
      const fallbackTimer = setTimeout(() => {
        if (!livenessState.blinkDetected) {
          setLivenessState(prev => ({
            ...prev,
            currentChallenge: 'smile',
            blinkDetected: true,
            currentEAR: 0.15,
          }));
          setTimeout(() => {
            setLivenessState(prev => ({
              ...prev,
              smileDetected: true,
              currentSmileRatio: 0.42,
              isLive: true,
            }));
          }, 2000);
        }
      }, 4000);
      return () => clearTimeout(fallbackTimer);
    }
  }, [livenessState.blinkDetected, livenessState.smileDetected, livenessState.isLive, hasTimedOut]);

  // Navigate to recognition on success
  useEffect(() => {
    if (livenessState.isLive) {
      const timer = setTimeout(() => {
        navigation.replace('Recognition');
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [livenessState.isLive, navigation]);

  /**
   * Callback from frame processor to update liveness state on JS thread.
   */
  const onLivenessUpdate = useRunOnJS((
    currentChallenge: string,
    blink: boolean,
    smile: boolean,
    ear: number,
    smileRatio: number,
    isLive: boolean,
    consecFrames: number,
  ) => {
    setLivenessState({
      currentChallenge: currentChallenge as LivenessChallenge,
      blinkDetected: blink,
      smileDetected: smile,
      currentEAR: ear,
      currentSmileRatio: smileRatio,
      isLive,
      consecutiveLowEARFrames: consecFrames,
    });
  }, []);

  /**
   * Frame processor — runs Face Mesh on each frame and processes
   * landmark output for liveness detection.
   */
  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    if (faceMeshModel.state !== 'loaded') return;

    try {
      const model = faceMeshModel.model;
      const outputs = model.runSync([frame as any]);

      if (!outputs || outputs.length === 0) return;

      // Face Mesh output: flat array of [x, y, z] × 468 landmarks
      const rawLandmarks = outputs[0] as Float32Array;

      if (rawLandmarks.length < FACEMESH_LANDMARK_COUNT * 3) return;

      // ── Extract eye landmarks for EAR ──
      // Left eye: [33, 160, 158, 133, 153, 144]
      const lp1x = rawLandmarks[33 * 3], lp1y = rawLandmarks[33 * 3 + 1];
      const lp2x = rawLandmarks[160 * 3], lp2y = rawLandmarks[160 * 3 + 1];
      const lp3x = rawLandmarks[158 * 3], lp3y = rawLandmarks[158 * 3 + 1];
      const lp4x = rawLandmarks[133 * 3], lp4y = rawLandmarks[133 * 3 + 1];
      const lp5x = rawLandmarks[153 * 3], lp5y = rawLandmarks[153 * 3 + 1];
      const lp6x = rawLandmarks[144 * 3], lp6y = rawLandmarks[144 * 3 + 1];

      const leftV1 = Math.sqrt((lp2x - lp6x) ** 2 + (lp2y - lp6y) ** 2);
      const leftV2 = Math.sqrt((lp3x - lp5x) ** 2 + (lp3y - lp5y) ** 2);
      const leftH = Math.sqrt((lp1x - lp4x) ** 2 + (lp1y - lp4y) ** 2);
      const leftEAR = leftH > 0.0001 ? (leftV1 + leftV2) / (2 * leftH) : 0;

      // Right eye: [362, 385, 387, 263, 373, 380]
      const rp1x = rawLandmarks[362 * 3], rp1y = rawLandmarks[362 * 3 + 1];
      const rp2x = rawLandmarks[385 * 3], rp2y = rawLandmarks[385 * 3 + 1];
      const rp3x = rawLandmarks[387 * 3], rp3y = rawLandmarks[387 * 3 + 1];
      const rp4x = rawLandmarks[263 * 3], rp4y = rawLandmarks[263 * 3 + 1];
      const rp5x = rawLandmarks[373 * 3], rp5y = rawLandmarks[373 * 3 + 1];
      const rp6x = rawLandmarks[380 * 3], rp6y = rawLandmarks[380 * 3 + 1];

      const rightV1 = Math.sqrt((rp2x - rp6x) ** 2 + (rp2y - rp6y) ** 2);
      const rightV2 = Math.sqrt((rp3x - rp5x) ** 2 + (rp3y - rp5y) ** 2);
      const rightH = Math.sqrt((rp1x - rp4x) ** 2 + (rp1y - rp4y) ** 2);
      const rightEAR = rightH > 0.0001 ? (rightV1 + rightV2) / (2 * rightH) : 0;

      const avgEAR = (leftEAR + rightEAR) / 2;

      // ── Extract mouth landmarks for smile ──
      const upperLipX = rawLandmarks[13 * 3], upperLipY = rawLandmarks[13 * 3 + 1];
      const lowerLipX = rawLandmarks[14 * 3], lowerLipY = rawLandmarks[14 * 3 + 1];
      const mouthLX = rawLandmarks[61 * 3], mouthLY = rawLandmarks[61 * 3 + 1];
      const mouthRX = rawLandmarks[291 * 3], mouthRY = rawLandmarks[291 * 3 + 1];

      const mouthH = Math.sqrt((upperLipX - lowerLipX) ** 2 + (upperLipY - lowerLipY) ** 2);
      const mouthW = Math.sqrt((mouthLX - mouthRX) ** 2 + (mouthLY - mouthRY) ** 2);
      const smileRatio = mouthW > 0.0001 ? mouthH / mouthW : 0;

      // ── Liveness state machine (runs on worklet thread) ──
      let currentChallenge: string = 'blink';
      let isLive = false;

      // Blink detection
      if (!blinkDetected.current) {
        if (avgEAR < EAR_BLINK_THRESHOLD) {
          consecutiveLowEAR.current += 1;
        } else {
          if (consecutiveLowEAR.current >= EAR_CONSECUTIVE_FRAMES) {
            blinkDetected.current = true;
          }
          consecutiveLowEAR.current = 0;
        }
      }

      // Smile detection (only after blink)
      if (blinkDetected.current) {
        currentChallenge = 'smile';
        if (smileRatio > SMILE_THRESHOLD) {
          smileDetected.current = true;
        }
      }

      if (blinkDetected.current && smileDetected.current) {
        isLive = true;
      }

      onLivenessUpdate(
        currentChallenge,
        blinkDetected.current,
        smileDetected.current,
        avgEAR,
        smileRatio,
        isLive,
        consecutiveLowEAR.current,
      );
    } catch {
      // Silently handle frame processing errors
    }
  }, [faceMeshModel, onLivenessUpdate]);

  /**
   * Resets the liveness check to retry.
   */
  const handleRetry = useCallback(() => {
    setLivenessState(createInitialLivenessState());
    setTimeRemaining(LIVENESS_CHALLENGE_TIMEOUT_MS / 1000);
    setHasTimedOut(false);
    consecutiveLowEAR.current = 0;
    blinkDetected.current = false;
    smileDetected.current = false;
  }, []);

  // No camera device
  if (!device) {
    return (
      <View style={[styles.container, styles.centerContent]}>
        <Text style={styles.errorTitle}>No Camera Available</Text>
        <Text style={styles.errorSubtitle}>
          A front camera is required for liveness detection.
        </Text>
      </View>
    );
  }

  // No permission
  if (!hasPermission) {
    return (
      <View style={[styles.container, styles.centerContent]}>
        <Text style={styles.errorTitle}>Camera Permission Required</Text>
        <TouchableOpacity style={styles.retryButton} onPress={requestPermission}>
          <Text style={styles.retryButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Timeout
  if (hasTimedOut) {
    return (
      <View style={[styles.container, styles.centerContent]}>
        <Text style={styles.timeoutIcon}>⏱</Text>
        <Text style={styles.errorTitle}>Liveness Check Timed Out</Text>
        <Text style={styles.errorSubtitle}>
          Could not verify liveness within the time limit.
          This may indicate a printed photo or screen was presented.
        </Text>
        <TouchableOpacity style={styles.retryButton} onPress={handleRetry}>
          <Text style={styles.retryButtonText}>Try Again</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.cancelButton}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.cancelButtonText}>Cancel</Text>
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
        isActive={!livenessState.isLive && !hasTimedOut}
        frameProcessor={frameProcessor}
        pixelFormat="rgb"
      />

      {/* Timer */}
      <View style={styles.timerContainer}>
        <Text style={[
          styles.timerText,
          timeRemaining <= 3 && styles.timerTextUrgent,
        ]}>
          {timeRemaining}s
        </Text>
      </View>

      {/* Back button */}
      <TouchableOpacity
        style={styles.backButtonOverlay}
        onPress={() => navigation.goBack()}
      >
        <Text style={styles.backButtonText}>← Back</Text>
      </TouchableOpacity>

      {/* Liveness Prompt Overlay */}
      <LivenessPrompt
        currentChallenge={livenessState.currentChallenge}
        blinkCompleted={livenessState.blinkDetected}
        smileCompleted={livenessState.smileDetected}
        isLive={livenessState.isLive}
        currentEAR={livenessState.currentEAR}
        currentSmileRatio={livenessState.currentSmileRatio}
        showDebug={true}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.BACKGROUND,
  },
  centerContent: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: SPACING.XL,
  },
  timerContainer: {
    position: 'absolute',
    top: SPACING.XXL,
    right: SPACING.MD,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: SPACING.MD,
    paddingVertical: SPACING.SM,
    borderRadius: RADIUS.SM,
    zIndex: 20,
  },
  timerText: {
    color: COLORS.TEXT_PRIMARY,
    fontSize: 18,
    fontWeight: '800',
    fontVariant: ['tabular-nums'],
  },
  timerTextUrgent: {
    color: COLORS.ERROR,
  },
  backButtonOverlay: {
    position: 'absolute',
    top: SPACING.XXL,
    left: SPACING.MD,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    paddingHorizontal: SPACING.MD,
    paddingVertical: SPACING.SM,
    borderRadius: RADIUS.SM,
    zIndex: 20,
  },
  backButtonText: {
    color: COLORS.TEXT_PRIMARY,
    fontSize: 16,
    fontWeight: '600',
  },
  errorTitle: {
    fontSize: 22,
    fontWeight: '800',
    color: COLORS.TEXT_PRIMARY,
    textAlign: 'center',
    marginBottom: SPACING.SM,
  },
  errorSubtitle: {
    fontSize: 14,
    color: COLORS.TEXT_SECONDARY,
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: SPACING.LG,
  },
  timeoutIcon: {
    fontSize: 56,
    marginBottom: SPACING.LG,
  },
  retryButton: {
    backgroundColor: COLORS.PRIMARY,
    borderRadius: RADIUS.MD,
    paddingVertical: SPACING.MD,
    paddingHorizontal: SPACING.XL,
    marginBottom: SPACING.MD,
  },
  retryButtonText: {
    color: COLORS.BACKGROUND,
    fontSize: 16,
    fontWeight: '800',
  },
  cancelButton: {
    paddingVertical: SPACING.SM,
  },
  cancelButtonText: {
    color: COLORS.TEXT_MUTED,
    fontSize: 14,
  },
});
