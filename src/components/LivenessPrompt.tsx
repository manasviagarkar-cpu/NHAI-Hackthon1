/**
 * @module LivenessPrompt
 * UI component that displays liveness challenge instructions with
 * animated transitions and progress indicators.
 *
 * Shows sequential prompts:
 *   Step 1: "Please Blink" with eye icon
 *   Step 2: "Please Smile" with smile icon
 *   Done: "Liveness Verified ✓"
 */

import React, { useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import { COLORS, RADIUS, SPACING } from '../utils/Constants';
import { LivenessChallenge } from '../ml/LivenessDetector';

/**
 * Props for the LivenessPrompt component.
 */
interface LivenessPromptProps {
  /** The current liveness challenge being presented */
  currentChallenge: LivenessChallenge;
  /** Whether the blink challenge has been completed */
  blinkCompleted: boolean;
  /** Whether the smile challenge has been completed */
  smileCompleted: boolean;
  /** Whether all challenges are complete (subject is live) */
  isLive: boolean;
  /** Current EAR value (for debug display) */
  currentEAR?: number;
  /** Current smile ratio (for debug display) */
  currentSmileRatio?: number;
  /** Whether to show debug metrics */
  showDebug?: boolean;
}

/**
 * Returns the display configuration for the current challenge state.
 *
 * @param {LivenessPromptProps} props - Current liveness state
 * @returns {{ icon: string; text: string; subtext: string; color: string }}
 */
function getChallengeDisplay(props: LivenessPromptProps): {
  icon: string;
  text: string;
  subtext: string;
  color: string;
} {
  if (props.isLive) {
    return {
      icon: '✓',
      text: 'Liveness Verified',
      subtext: 'Proceeding to recognition...',
      color: COLORS.SUCCESS,
    };
  }

  if (props.currentChallenge === 'blink') {
    return {
      icon: '👁',
      text: 'Please Blink',
      subtext: 'Close and open your eyes naturally',
      color: COLORS.ACCENT,
    };
  }

  // smile challenge
  return {
    icon: '😊',
    text: 'Please Smile',
    subtext: 'Give a natural smile',
    color: COLORS.PRIMARY,
  };
}

/**
 * Renders the liveness challenge prompt with animated transitions.
 * Provides clear visual feedback on which challenge is active
 * and which have been completed.
 *
 * @param {LivenessPromptProps} props - Liveness prompt configuration
 * @returns {React.ReactElement} The liveness prompt overlay
 */
export function LivenessPrompt(props: LivenessPromptProps): React.ReactElement {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.8)).current;
  const progressAnim = useRef(new Animated.Value(0)).current;

  const { currentChallenge, blinkCompleted, smileCompleted, isLive } = props;
  const display = getChallengeDisplay(props);

  useEffect(() => {
    // Animate in
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        friction: 8,
        tension: 80,
        useNativeDriver: true,
      }),
    ]).start();

    return () => {
      fadeAnim.setValue(0);
      scaleAnim.setValue(0.8);
    };
  }, [currentChallenge, isLive, fadeAnim, scaleAnim]);

  // Animate progress bar
  useEffect(() => {
    let targetProgress = 0;
    if (blinkCompleted) targetProgress = 0.5;
    if (smileCompleted) targetProgress = 1.0;

    Animated.timing(progressAnim, {
      toValue: targetProgress,
      duration: 400,
      useNativeDriver: false, // layout animation needs false
    }).start();
  }, [blinkCompleted, smileCompleted, progressAnim]);

  return (
    <View style={styles.container}>
      {/* Progress bar */}
      <View style={styles.progressBarContainer}>
        <Animated.View
          style={[
            styles.progressBarFill,
            {
              width: progressAnim.interpolate({
                inputRange: [0, 1],
                outputRange: ['0%', '100%'],
              }),
              backgroundColor: display.color,
            },
          ]}
        />
      </View>

      {/* Step indicators */}
      <View style={styles.stepsContainer}>
        <View style={styles.stepRow}>
          <View
            style={[
              styles.stepDot,
              blinkCompleted ? styles.stepCompleted : (currentChallenge === 'blink' ? styles.stepActive : styles.stepInactive),
            ]}
          >
            <Text style={styles.stepDotText}>
              {blinkCompleted ? '✓' : '1'}
            </Text>
          </View>
          <Text style={[styles.stepLabel, blinkCompleted && styles.stepLabelCompleted]}>
            Blink
          </Text>
        </View>

        <View style={styles.stepConnector} />

        <View style={styles.stepRow}>
          <View
            style={[
              styles.stepDot,
              smileCompleted ? styles.stepCompleted : (currentChallenge === 'smile' ? styles.stepActive : styles.stepInactive),
            ]}
          >
            <Text style={styles.stepDotText}>
              {smileCompleted ? '✓' : '2'}
            </Text>
          </View>
          <Text style={[styles.stepLabel, smileCompleted && styles.stepLabelCompleted]}>
            Smile
          </Text>
        </View>
      </View>

      {/* Main prompt */}
      <Animated.View
        style={[
          styles.promptContainer,
          {
            opacity: fadeAnim,
            transform: [{ scale: scaleAnim }],
            borderColor: display.color,
          },
        ]}
      >
        <Text style={styles.iconText}>{display.icon}</Text>
        <Text style={[styles.promptText, { color: display.color }]}>
          {display.text}
        </Text>
        <Text style={styles.subtextText}>{display.subtext}</Text>
      </Animated.View>

      {/* Debug metrics (optional) */}
      {props.showDebug && (
        <View style={styles.debugContainer}>
          <Text style={styles.debugText}>
            EAR: {(props.currentEAR ?? 0).toFixed(3)}
          </Text>
          <Text style={styles.debugText}>
            Smile: {(props.currentSmileRatio ?? 0).toFixed(3)}
          </Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    paddingHorizontal: SPACING.LG,
    paddingBottom: SPACING.XXL,
    paddingTop: SPACING.LG,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    borderTopLeftRadius: RADIUS.XL,
    borderTopRightRadius: RADIUS.XL,
  },
  progressBarContainer: {
    height: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 2,
    marginBottom: SPACING.MD,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 2,
  },
  stepsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: SPACING.LG,
  },
  stepRow: {
    alignItems: 'center',
    gap: SPACING.XS,
  },
  stepDot: {
    width: 28,
    height: 28,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepDotText: {
    color: COLORS.TEXT_PRIMARY,
    fontSize: 13,
    fontWeight: '700',
  },
  stepActive: {
    backgroundColor: COLORS.ACCENT,
  },
  stepCompleted: {
    backgroundColor: COLORS.SUCCESS,
  },
  stepInactive: {
    backgroundColor: COLORS.SURFACE_LIGHT,
  },
  stepConnector: {
    width: 40,
    height: 2,
    backgroundColor: COLORS.BORDER,
    marginHorizontal: SPACING.SM,
  },
  stepLabel: {
    color: COLORS.TEXT_SECONDARY,
    fontSize: 11,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  stepLabelCompleted: {
    color: COLORS.SUCCESS,
  },
  promptContainer: {
    alignItems: 'center',
    paddingVertical: SPACING.MD,
    borderWidth: 1,
    borderRadius: RADIUS.LG,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
  },
  iconText: {
    fontSize: 36,
    marginBottom: SPACING.SM,
  },
  promptText: {
    fontSize: 22,
    fontWeight: '800',
    letterSpacing: 0.5,
    marginBottom: SPACING.XS,
  },
  subtextText: {
    color: COLORS.TEXT_SECONDARY,
    fontSize: 14,
    fontWeight: '400',
  },
  debugContainer: {
    marginTop: SPACING.MD,
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  debugText: {
    color: COLORS.TEXT_MUTED,
    fontSize: 12,
    fontFamily: 'monospace',
  },
});

export default LivenessPrompt;
