/**
 * @module InferenceTimer
 * Displays the current inference time in milliseconds with color-coded
 * feedback. Shows green when fast (<500ms), yellow when acceptable (<1000ms),
 * and red when too slow (>1000ms).
 *
 * Positioned as an overlay on the recognition screen camera preview.
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import {
  COLORS,
  RADIUS,
  SPACING,
  INFERENCE_TIMER_GREEN_MS,
  INFERENCE_TIMER_YELLOW_MS,
} from '../utils/Constants';

/**
 * Props for the InferenceTimer component.
 */
interface InferenceTimerProps {
  /** Current inference time in milliseconds */
  timeMs: number;
  /** Whether the timer is actively measuring (shows pulsing dot) */
  active?: boolean;
  /** Whether to show the component */
  visible?: boolean;
}

/**
 * Returns the appropriate color based on inference time performance.
 *
 * @param {number} timeMs - Inference time in milliseconds
 * @returns {string} Hex color string
 */
function getTimerColor(timeMs: number): string {
  if (timeMs <= 0) return COLORS.TEXT_MUTED;
  if (timeMs < INFERENCE_TIMER_GREEN_MS) return COLORS.SUCCESS;
  if (timeMs < INFERENCE_TIMER_YELLOW_MS) return COLORS.WARNING;
  return COLORS.ERROR;
}

/**
 * Returns a performance label based on inference time.
 *
 * @param {number} timeMs - Inference time in milliseconds
 * @returns {string} Human-readable performance label
 */
function getPerformanceLabel(timeMs: number): string {
  if (timeMs <= 0) return 'Waiting...';
  if (timeMs < 300) return 'Excellent';
  if (timeMs < INFERENCE_TIMER_GREEN_MS) return 'Fast';
  if (timeMs < INFERENCE_TIMER_YELLOW_MS) return 'OK';
  return 'Slow';
}

/**
 * Renders a real-time inference timer overlay showing processing speed.
 * Designed to be positioned in the top-right corner of the camera preview.
 *
 * @param {InferenceTimerProps} props - Timer configuration
 * @returns {React.ReactElement | null} The timer overlay or null if not visible
 */
export function InferenceTimer({
  timeMs,
  active = false,
  visible = true,
}: InferenceTimerProps): React.ReactElement | null {
  if (!visible) return null;

  const color = getTimerColor(timeMs);
  const label = getPerformanceLabel(timeMs);
  const displayMs = timeMs > 0 ? Math.round(timeMs) : 0;

  return (
    <View style={styles.container}>
      <View style={styles.innerContainer}>
        {/* Status indicator dot */}
        <View style={[styles.statusDot, { backgroundColor: active ? color : COLORS.TEXT_MUTED }]} />

        {/* Time display */}
        <Text style={[styles.timeText, { color }]}>
          {displayMs}
          <Text style={styles.unitText}>ms</Text>
        </Text>

        {/* Performance label */}
        <Text style={[styles.labelText, { color }]}>
          {label}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: SPACING.MD,
    right: SPACING.MD,
    zIndex: 20,
  },
  innerContainer: {
    backgroundColor: 'rgba(0, 0, 0, 0.75)',
    borderRadius: RADIUS.MD,
    paddingHorizontal: SPACING.MD,
    paddingVertical: SPACING.SM,
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.SM,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  timeText: {
    fontSize: 18,
    fontWeight: '800',
    fontVariant: ['tabular-nums'],
  },
  unitText: {
    fontSize: 12,
    fontWeight: '500',
    opacity: 0.7,
  },
  labelText: {
    fontSize: 11,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    opacity: 0.8,
  },
});

export default InferenceTimer;
