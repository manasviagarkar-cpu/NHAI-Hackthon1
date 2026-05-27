/**
 * @module WorkerCard
 * Displays a matched worker's information after successful face recognition.
 * Shows name, worker ID, confidence score, and timestamp in a card
 * with slide-in animation.
 */

import React, { useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import { COLORS, RADIUS, SPACING, COSINE_THRESHOLD } from '../utils/Constants';

/**
 * Props for the WorkerCard component.
 */
interface WorkerCardProps {
  /** Worker display name */
  name: string;
  /** Worker external ID (employee badge) */
  workerId: string;
  /** Cosine similarity confidence score (0.0 – 1.0) */
  confidence: number;
  /** Whether the match meets the threshold */
  isMatch: boolean;
  /** Whether to show the card (triggers animation) */
  visible: boolean;
  /** Optional timestamp string */
  timestamp?: string;
}

/**
 * Returns a confidence level descriptor and color.
 *
 * @param {number} confidence - Cosine similarity score
 * @returns {{ label: string; color: string }} Confidence display info
 */
function getConfidenceDisplay(confidence: number): { label: string; color: string } {
  if (confidence >= 0.9) return { label: 'Very High', color: COLORS.SUCCESS };
  if (confidence >= 0.8) return { label: 'High', color: COLORS.PRIMARY };
  if (confidence >= COSINE_THRESHOLD) return { label: 'Moderate', color: COLORS.WARNING };
  return { label: 'Low', color: COLORS.ERROR };
}

/**
 * Renders a card showing the recognition result for a matched worker.
 * Slides in from the bottom with a spring animation.
 *
 * @param {WorkerCardProps} props - Worker card configuration
 * @returns {React.ReactElement | null} The worker card or null if not visible
 */
export function WorkerCard({
  name,
  workerId,
  confidence,
  isMatch,
  visible,
  timestamp,
}: WorkerCardProps): React.ReactElement | null {
  const slideAnim = useRef(new Animated.Value(100)).current;
  const opacityAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (visible) {
      Animated.parallel([
        Animated.spring(slideAnim, {
          toValue: 0,
          friction: 10,
          tension: 60,
          useNativeDriver: true,
        }),
        Animated.timing(opacityAnim, {
          toValue: 1,
          duration: 250,
          useNativeDriver: true,
        }),
      ]).start();
    } else {
      Animated.parallel([
        Animated.timing(slideAnim, {
          toValue: 100,
          duration: 200,
          useNativeDriver: true,
        }),
        Animated.timing(opacityAnim, {
          toValue: 0,
          duration: 200,
          useNativeDriver: true,
        }),
      ]).start();
    }
  }, [visible, slideAnim, opacityAnim]);

  if (!visible && !isMatch) return null;

  const confidenceDisplay = getConfidenceDisplay(confidence);
  const confidencePercent = Math.round(confidence * 100);
  const borderColor = isMatch ? COLORS.SUCCESS : COLORS.ERROR;

  return (
    <Animated.View
      style={[
        styles.container,
        {
          transform: [{ translateY: slideAnim }],
          opacity: opacityAnim,
          borderLeftColor: borderColor,
        },
      ]}
    >
      {/* Status indicator */}
      <View style={[styles.statusBadge, { backgroundColor: borderColor }]}>
        <Text style={styles.statusText}>
          {isMatch ? '✓ MATCHED' : '✗ NO MATCH'}
        </Text>
      </View>

      {/* Worker info */}
      <View style={styles.infoContainer}>
        {/* Avatar placeholder */}
        <View style={[styles.avatar, { borderColor }]}>
          <Text style={styles.avatarText}>
            {name.charAt(0).toUpperCase()}
          </Text>
        </View>

        <View style={styles.detailsContainer}>
          <Text style={styles.nameText} numberOfLines={1}>
            {name}
          </Text>
          <Text style={styles.idText}>ID: {workerId}</Text>
          {timestamp && (
            <Text style={styles.timestampText}>{timestamp}</Text>
          )}
        </View>

        {/* Confidence gauge */}
        <View style={styles.confidenceContainer}>
          <Text style={[styles.confidenceValue, { color: confidenceDisplay.color }]}>
            {confidencePercent}%
          </Text>
          <Text style={[styles.confidenceLabel, { color: confidenceDisplay.color }]}>
            {confidenceDisplay.label}
          </Text>

          {/* Mini progress bar */}
          <View style={styles.confidenceBarBg}>
            <View
              style={[
                styles.confidenceBarFill,
                {
                  width: `${confidencePercent}%`,
                  backgroundColor: confidenceDisplay.color,
                },
              ]}
            />
          </View>
        </View>
      </View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    bottom: SPACING.XXL,
    left: SPACING.MD,
    right: SPACING.MD,
    backgroundColor: COLORS.SURFACE,
    borderRadius: RADIUS.LG,
    borderLeftWidth: 4,
    paddingVertical: SPACING.MD,
    paddingHorizontal: SPACING.MD,
    zIndex: 30,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 16,
    elevation: 12,
  },
  statusBadge: {
    position: 'absolute',
    top: -12,
    right: SPACING.MD,
    paddingHorizontal: SPACING.SM,
    paddingVertical: SPACING.XS,
    borderRadius: RADIUS.SM,
  },
  statusText: {
    color: COLORS.BACKGROUND,
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1,
  },
  infoContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.MD,
  },
  avatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: COLORS.SURFACE_LIGHT,
    borderWidth: 2,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    color: COLORS.TEXT_PRIMARY,
    fontSize: 20,
    fontWeight: '700',
  },
  detailsContainer: {
    flex: 1,
    gap: 2,
  },
  nameText: {
    color: COLORS.TEXT_PRIMARY,
    fontSize: 18,
    fontWeight: '700',
  },
  idText: {
    color: COLORS.TEXT_SECONDARY,
    fontSize: 13,
    fontWeight: '500',
  },
  timestampText: {
    color: COLORS.TEXT_MUTED,
    fontSize: 11,
  },
  confidenceContainer: {
    alignItems: 'flex-end',
    gap: 2,
  },
  confidenceValue: {
    fontSize: 22,
    fontWeight: '800',
    fontVariant: ['tabular-nums'],
  },
  confidenceLabel: {
    fontSize: 10,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  confidenceBarBg: {
    width: 60,
    height: 3,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 2,
    overflow: 'hidden',
    marginTop: 2,
  },
  confidenceBarFill: {
    height: '100%',
    borderRadius: 2,
  },
});

export default WorkerCard;
