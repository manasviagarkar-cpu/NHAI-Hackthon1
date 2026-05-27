/**
 * @module HomeScreen
 * Main navigation hub for the NHAI FaceRec app.
 * Displays dashboard statistics and provides navigation to all features:
 *   - Register a new worker
 *   - Start face recognition (with liveness check)
 *   - Manage sync & purge
 *
 * Dark theme with glassmorphism cards and animated metrics.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
  StatusBar,
  ScrollView,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { getDatabaseStats } from '../database/DatabaseManager';
import { getPendingSyncCount } from '../database/AttendanceRepository';
import { COLORS, RADIUS, SPACING } from '../utils/Constants';

type HomeScreenProps = {
  navigation: NativeStackNavigationProp<any>;
};

/**
 * Dashboard statistics displayed on the home screen.
 */
interface DashboardStats {
  workerCount: number;
  pendingSync: number;
  totalLogs: number;
}

/**
 * Home screen component — the app's landing page and navigation hub.
 *
 * @param {HomeScreenProps} props - Navigation prop from React Navigation
 * @returns {React.ReactElement} The home screen UI
 */
export default function HomeScreen({ navigation }: HomeScreenProps): React.ReactElement {
  const [stats, setStats] = useState<DashboardStats>({
    workerCount: 0,
    pendingSync: 0,
    totalLogs: 0,
  });

  // Animations
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(30)).current;
  const cardAnims = useRef([
    new Animated.Value(0),
    new Animated.Value(0),
    new Animated.Value(0),
  ]).current;

  /**
   * Refreshes dashboard statistics from the database.
   * Called on initial mount and each time the screen gains focus.
   */
  const refreshStats = useCallback(() => {
    try {
      const dbStats = getDatabaseStats();
      setStats(dbStats);
    } catch {
      // Database might not be initialized yet on first render
      setStats({ workerCount: 0, pendingSync: 0, totalLogs: 0 });
    }
  }, []);

  // Refresh stats when screen gains focus (returning from other screens)
  useFocusEffect(
    useCallback(() => {
      refreshStats();
    }, [refreshStats])
  );

  // Entry animation
  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 600,
        useNativeDriver: true,
      }),
      Animated.spring(slideAnim, {
        toValue: 0,
        friction: 8,
        tension: 60,
        useNativeDriver: true,
      }),
    ]).start();

    // Stagger card animations
    const staggerDelay = 150;
    cardAnims.forEach((anim, index) => {
      Animated.timing(anim, {
        toValue: 1,
        duration: 400,
        delay: 300 + index * staggerDelay,
        useNativeDriver: true,
      }).start();
    });
  }, [fadeAnim, slideAnim, cardAnims]);

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.BACKGROUND} />
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <Animated.View
          style={[
            styles.header,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            },
          ]}
        >
          <Text style={styles.logoText}>⬡</Text>
          <Text style={styles.titleText}>NHAI FaceRec</Text>
          <Text style={styles.subtitleText}>
            Offline Facial Recognition System
          </Text>
          <View style={styles.offlineBadge}>
            <View style={styles.offlineDot} />
            <Text style={styles.offlineBadgeText}>Offline Ready</Text>
          </View>
        </Animated.View>

        {/* Stats Row */}
        <Animated.View
          style={[
            styles.statsRow,
            { opacity: fadeAnim },
          ]}
        >
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{stats.workerCount}</Text>
            <Text style={styles.statLabel}>Workers</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{stats.totalLogs}</Text>
            <Text style={styles.statLabel}>Records</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Text style={[styles.statValue, stats.pendingSync > 0 && styles.statValueWarning]}>
              {stats.pendingSync}
            </Text>
            <Text style={styles.statLabel}>Pending</Text>
          </View>
        </Animated.View>

        {/* Action Cards */}
        <View style={styles.cardsContainer}>
          {/* Register Card */}
          <Animated.View style={{ opacity: cardAnims[0], transform: [{ translateY: cardAnims[0].interpolate({ inputRange: [0, 1], outputRange: [40, 0] }) }] }}>
            <TouchableOpacity
              style={[styles.card, styles.cardRegister]}
              activeOpacity={0.85}
              onPress={() => navigation.navigate('Registration')}
            >
              <View style={styles.cardIconContainer}>
                <Text style={styles.cardIcon}>👤</Text>
              </View>
              <View style={styles.cardContent}>
                <Text style={styles.cardTitle}>Register Worker</Text>
                <Text style={styles.cardDescription}>
                  Capture face embeddings for a new worker. Takes 3 photos for optimal accuracy.
                </Text>
              </View>
              <Text style={styles.cardArrow}>→</Text>
            </TouchableOpacity>
          </Animated.View>

          {/* Recognize Card */}
          <Animated.View style={{ opacity: cardAnims[1], transform: [{ translateY: cardAnims[1].interpolate({ inputRange: [0, 1], outputRange: [40, 0] }) }] }}>
            <TouchableOpacity
              style={[styles.card, styles.cardRecognize]}
              activeOpacity={0.85}
              onPress={() => navigation.navigate('Liveness')}
              disabled={stats.workerCount === 0}
            >
              <View style={[styles.cardIconContainer, styles.cardIconRecognize]}>
                <Text style={styles.cardIcon}>🔍</Text>
              </View>
              <View style={styles.cardContent}>
                <Text style={styles.cardTitle}>Recognize Face</Text>
                <Text style={styles.cardDescription}>
                  {stats.workerCount === 0
                    ? 'Register at least one worker first to begin recognition.'
                    : `Liveness check → face match against ${stats.workerCount} registered worker${stats.workerCount !== 1 ? 's' : ''}.`}
                </Text>
              </View>
              <Text style={styles.cardArrow}>→</Text>
            </TouchableOpacity>
          </Animated.View>

          {/* Sync Card */}
          <Animated.View style={{ opacity: cardAnims[2], transform: [{ translateY: cardAnims[2].interpolate({ inputRange: [0, 1], outputRange: [40, 0] }) }] }}>
            <TouchableOpacity
              style={[styles.card, styles.cardSync]}
              activeOpacity={0.85}
              onPress={() => navigation.navigate('Sync')}
            >
              <View style={[styles.cardIconContainer, styles.cardIconSync]}>
                <Text style={styles.cardIcon}>☁️</Text>
              </View>
              <View style={styles.cardContent}>
                <Text style={styles.cardTitle}>Sync & Purge</Text>
                <Text style={styles.cardDescription}>
                  {stats.pendingSync > 0
                    ? `${stats.pendingSync} record${stats.pendingSync !== 1 ? 's' : ''} pending sync to AWS.`
                    : 'All records synced. Manage data and view history.'}
                </Text>
              </View>
              {stats.pendingSync > 0 && (
                <View style={styles.badge}>
                  <Text style={styles.badgeText}>{stats.pendingSync}</Text>
                </View>
              )}
              <Text style={styles.cardArrow}>→</Text>
            </TouchableOpacity>
          </Animated.View>
        </View>

        {/* Footer */}
        <View style={styles.footer}>
          <Text style={styles.footerText}>NHAI Innovation Hackathon 7.0</Text>
          <Text style={styles.footerSubtext}>100% Offline • On-Device ML • Open Source</Text>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.BACKGROUND,
  },
  scrollContent: {
    paddingHorizontal: SPACING.LG,
    paddingTop: SPACING.XXL + SPACING.MD,
    paddingBottom: SPACING.XXL,
  },
  header: {
    alignItems: 'center',
    marginBottom: SPACING.XL,
  },
  logoText: {
    fontSize: 48,
    color: COLORS.PRIMARY,
    marginBottom: SPACING.SM,
  },
  titleText: {
    fontSize: 32,
    fontWeight: '900',
    color: COLORS.TEXT_PRIMARY,
    letterSpacing: 1,
  },
  subtitleText: {
    fontSize: 14,
    color: COLORS.TEXT_SECONDARY,
    marginTop: SPACING.XS,
    fontWeight: '400',
  },
  offlineBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: SPACING.MD,
    backgroundColor: 'rgba(0, 200, 150, 0.1)',
    paddingHorizontal: SPACING.MD,
    paddingVertical: SPACING.XS,
    borderRadius: RADIUS.FULL,
    borderWidth: 1,
    borderColor: 'rgba(0, 200, 150, 0.3)',
  },
  offlineDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: COLORS.PRIMARY,
    marginRight: SPACING.SM,
  },
  offlineBadgeText: {
    color: COLORS.PRIMARY,
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  statsRow: {
    flexDirection: 'row',
    backgroundColor: COLORS.SURFACE,
    borderRadius: RADIUS.LG,
    padding: SPACING.LG,
    marginBottom: SPACING.XL,
    borderWidth: 1,
    borderColor: COLORS.BORDER,
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 28,
    fontWeight: '800',
    color: COLORS.TEXT_PRIMARY,
    fontVariant: ['tabular-nums'],
  },
  statValueWarning: {
    color: COLORS.WARNING,
  },
  statLabel: {
    fontSize: 11,
    color: COLORS.TEXT_MUTED,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginTop: 2,
    fontWeight: '600',
  },
  statDivider: {
    width: 1,
    backgroundColor: COLORS.BORDER,
  },
  cardsContainer: {
    gap: SPACING.MD,
  },
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.SURFACE,
    borderRadius: RADIUS.LG,
    padding: SPACING.LG,
    borderWidth: 1,
    borderColor: COLORS.BORDER,
  },
  cardRegister: {
    borderLeftWidth: 3,
    borderLeftColor: COLORS.PRIMARY,
  },
  cardRecognize: {
    borderLeftWidth: 3,
    borderLeftColor: COLORS.ACCENT,
  },
  cardSync: {
    borderLeftWidth: 3,
    borderLeftColor: '#8B5CF6',
  },
  cardIconContainer: {
    width: 48,
    height: 48,
    borderRadius: RADIUS.MD,
    backgroundColor: 'rgba(0, 200, 150, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: SPACING.MD,
  },
  cardIconRecognize: {
    backgroundColor: 'rgba(255, 184, 0, 0.1)',
  },
  cardIconSync: {
    backgroundColor: 'rgba(139, 92, 246, 0.1)',
  },
  cardIcon: {
    fontSize: 22,
  },
  cardContent: {
    flex: 1,
  },
  cardTitle: {
    fontSize: 17,
    fontWeight: '700',
    color: COLORS.TEXT_PRIMARY,
    marginBottom: 4,
  },
  cardDescription: {
    fontSize: 13,
    color: COLORS.TEXT_SECONDARY,
    lineHeight: 18,
  },
  cardArrow: {
    fontSize: 20,
    color: COLORS.TEXT_MUTED,
    marginLeft: SPACING.SM,
  },
  badge: {
    backgroundColor: COLORS.WARNING,
    borderRadius: RADIUS.FULL,
    minWidth: 22,
    height: 22,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 6,
    marginRight: SPACING.SM,
  },
  badgeText: {
    color: COLORS.BACKGROUND,
    fontSize: 12,
    fontWeight: '800',
  },
  footer: {
    alignItems: 'center',
    marginTop: SPACING.XXL,
    paddingTop: SPACING.LG,
    borderTopWidth: 1,
    borderTopColor: COLORS.BORDER,
  },
  footerText: {
    color: COLORS.TEXT_MUTED,
    fontSize: 13,
    fontWeight: '600',
  },
  footerSubtext: {
    color: COLORS.TEXT_MUTED,
    fontSize: 11,
    marginTop: 4,
    opacity: 0.6,
  },
});
