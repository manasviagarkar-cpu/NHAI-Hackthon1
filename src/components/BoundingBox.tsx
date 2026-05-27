/**
 * @module BoundingBox
 * Animated SVG overlay that draws a bounding box around a detected face.
 * Renders on top of the camera preview with a pulsing animation effect.
 *
 * Colors:
 *   - Green when face is detected and ready
 *   - Yellow when processing
 *   - Red when no match or error
 */

import React, { useEffect, useRef } from 'react';
import { View, StyleSheet, Animated } from 'react-native';
import Svg, { Rect, Line } from 'react-native-svg';
import { COLORS, RADIUS } from '../utils/Constants';

/**
 * Props for the BoundingBox component.
 */
interface BoundingBoxProps {
  /** X coordinate of the bounding box (pixels) */
  x: number;
  /** Y coordinate of the bounding box (pixels) */
  y: number;
  /** Width of the bounding box (pixels) */
  width: number;
  /** Height of the bounding box (pixels) */
  height: number;
  /** Color of the bounding box border */
  color?: string;
  /** Whether to show the pulsing animation */
  animated?: boolean;
  /** Whether the box is visible */
  visible?: boolean;
  /** Label to display above the box */
  label?: string;
}

/**
 * Renders an animated bounding box overlay on the camera preview.
 * Uses corner brackets instead of a full rectangle for a modern look.
 *
 * @param {BoundingBoxProps} props - Bounding box configuration
 * @returns {React.ReactElement | null} The bounding box overlay or null if not visible
 */
export function BoundingBox({
  x,
  y,
  width,
  height,
  color = COLORS.SUCCESS,
  animated = true,
  visible = true,
  label,
}: BoundingBoxProps): React.ReactElement | null {
  const pulseAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    if (!animated || !visible) {
      pulseAnim.setValue(1);
      return;
    }

    const animation = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 0.5,
          duration: 800,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 800,
          useNativeDriver: true,
        }),
      ])
    );

    animation.start();

    return () => {
      animation.stop();
    };
  }, [animated, visible, pulseAnim]);

  if (!visible || width <= 0 || height <= 0) {
    return null;
  }

  const cornerLength = Math.min(width, height) * 0.2;
  const strokeWidth = 3;

  return (
    <Animated.View
      style={[
        styles.container,
        {
          left: x,
          top: y,
          width,
          height,
          opacity: pulseAnim,
        },
      ]}
    >
      {/* Label above box */}
      {label ? (
        <View style={[styles.labelContainer, { backgroundColor: color }]}>
          <Animated.Text style={styles.labelText}>{label}</Animated.Text>
        </View>
      ) : null}

      {/* Corner brackets SVG */}
      <Svg width={width} height={height} style={StyleSheet.absoluteFill}>
        {/* Top-left corner */}
        <Line
          x1={0} y1={0}
          x2={cornerLength} y2={0}
          stroke={color} strokeWidth={strokeWidth} strokeLinecap="round"
        />
        <Line
          x1={0} y1={0}
          x2={0} y2={cornerLength}
          stroke={color} strokeWidth={strokeWidth} strokeLinecap="round"
        />

        {/* Top-right corner */}
        <Line
          x1={width - cornerLength} y1={0}
          x2={width} y2={0}
          stroke={color} strokeWidth={strokeWidth} strokeLinecap="round"
        />
        <Line
          x1={width} y1={0}
          x2={width} y2={cornerLength}
          stroke={color} strokeWidth={strokeWidth} strokeLinecap="round"
        />

        {/* Bottom-left corner */}
        <Line
          x1={0} y1={height - cornerLength}
          x2={0} y2={height}
          stroke={color} strokeWidth={strokeWidth} strokeLinecap="round"
        />
        <Line
          x1={0} y1={height}
          x2={cornerLength} y2={height}
          stroke={color} strokeWidth={strokeWidth} strokeLinecap="round"
        />

        {/* Bottom-right corner */}
        <Line
          x1={width} y1={height - cornerLength}
          x2={width} y2={height}
          stroke={color} strokeWidth={strokeWidth} strokeLinecap="round"
        />
        <Line
          x1={width - cornerLength} y1={height}
          x2={width} y2={height}
          stroke={color} strokeWidth={strokeWidth} strokeLinecap="round"
        />
      </Svg>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    zIndex: 10,
  },
  labelContainer: {
    position: 'absolute',
    top: -28,
    left: 0,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: RADIUS.SM,
    alignSelf: 'flex-start',
  },
  labelText: {
    color: COLORS.BACKGROUND,
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
});

export default BoundingBox;
