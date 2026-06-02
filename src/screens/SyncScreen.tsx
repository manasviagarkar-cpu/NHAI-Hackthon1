/**
 * @module SyncScreen
 * Manages sync and purge of attendance records.
 *
 * Features:
 *   - Shows count of pending (unsynced) records
 *   - "Sync to AWS" button — mock sync that marks records as synced
 *   - "Purge Synced Data" button — deletes synced records to free storage
 *   - Shows last sync timestamp
 *   - Displays recent attendance history
 */

import React, { useCallback, useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
  FlatList,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import {
  getPendingSyncCount,
  getPendingLogs,
  markAllAsSynced,
  purgeSyncedLogs,
  getAllLogs,
  getSyncedCount,
  AttendanceLog,
} from '../database/AttendanceRepository';
import { COLORS, RADIUS, SPACING, STORAGE_KEYS } from '../utils/Constants';

type SyncScreenProps = {
  navigation: NativeStackNavigationProp<any>;
};

/**
 * Sync & Purge screen for managing offline attendance data.
 *
 * @param {SyncScreenProps} props - Navigation prop
 * @returns {React.ReactElement} The sync screen UI
 */
export default function SyncScreen({
  navigation,
}: SyncScreenProps): React.ReactElement {
  const [pendingCount, setPendingCount] = useState(0);
  const [syncedCount, setSyncedCount] = useState(0);
  const [lastSyncTimestamp, setLastSyncTimestamp] = useState<string | null>(null);
  const [recentLogs, setRecentLogs] = useState<AttendanceLog[]>([]);
  const [isSyncing, setIsSyncing] = useState(false);
  const [isPurging, setIsPurging] = useState(false);

  /**
   * Refreshes all data from the database and AsyncStorage.
   */
  const refreshData = useCallback(async () => {
    try {
      const pending = getPendingSyncCount();
      const synced = getSyncedCount();
      const logs = getAllLogs(50);
      const lastSync = await AsyncStorage.getItem(STORAGE_KEYS.LAST_SYNC_TIMESTAMP);

      setPendingCount(pending);
      setSyncedCount(synced);
      setRecentLogs(logs);
      setLastSyncTimestamp(lastSync);
    } catch (error) {
      console.error('Failed to refresh sync data:', error);
    }
  }, []);

  // Refresh on screen focus
  useFocusEffect(
    useCallback(() => {
      refreshData();
    }, [refreshData])
  );

  // AWS endpoint — replace with actual API Gateway URL before production deployment
  // For hackathon demo: sync is attempted, fails gracefully with offline message
  const AWS_ENDPOINT =
    'https://nhai-facerec-sync.execute-api.ap-south-1.amazonaws.com/prod/sync';

  /**
   * Handles the "Sync to AWS" button press.
   * Checks network connectivity, retrieves pending logs from SQLite,
   * POSTs them to the AWS endpoint, then marks them as synced locally.
   * Falls back gracefully when the device is offline.
   */
  const handleSync = useCallback(async () => {
    if (isSyncing || pendingCount === 0) return;
    setIsSyncing(true);

    try {
      // Check network connectivity first
      const isConnected = await fetch('https://www.google.com', { method: 'HEAD' })
        .then(() => true)
        .catch(() => false);

      if (!isConnected) {
        Alert.alert('No Connection', 'Please connect to internet to sync.');
        setIsSyncing(false);
        return;
      }

      // Get all pending logs from SQLite
      const pendingLogs = getPendingLogs();

      // POST to AWS endpoint
      const response = await fetch(AWS_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': 'NHAI_FACEREC_2026',
        },
        body: JSON.stringify({
          device_id: 'NHAI_DEVICE_001',
          logs: pendingLogs,
          synced_at: new Date().toISOString(),
        }),
      });

      if (response.ok) {
        // Mark as synced in local DB
        const ids = pendingLogs.map((l: AttendanceLog) => l.id);
        markAllAsSynced();
        await AsyncStorage.setItem(STORAGE_KEYS.LAST_SYNC_TIMESTAMP, new Date().toISOString());
        await refreshData();
        setPendingCount(0);
        Alert.alert('Sync Complete', `${ids.length} records synced to AWS.`);
      } else {
        throw new Error(`Server error: ${response.status}`);
      }
    } catch (error: any) {
      // Graceful fallback — if AWS is unreachable, show clear message
      Alert.alert(
        'Sync Failed',
        'Could not reach AWS server. Data is safely stored locally and will sync when connection is restored.',
      );
    } finally {
      setIsSyncing(false);
    }
  }, [isSyncing, pendingCount, refreshData]);

  /**
   * Handles the "Purge Synced Data" button press.
   * Permanently deletes all synced records from the local database.
   */
  const handlePurge = useCallback(async () => {
    if (syncedCount === 0) {
      Alert.alert('Nothing to Purge', 'There are no synced records to delete.');
      return;
    }

    Alert.alert(
      'Confirm Purge',
      `This will permanently delete ${syncedCount} synced record${syncedCount !== 1 ? 's' : ''} from local storage. This cannot be undone.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Purge',
          style: 'destructive',
          onPress: async () => {
            setIsPurging(true);

            try {
              // Simulate processing delay
              await new Promise(resolve => setTimeout(resolve, 800));

              const purgedCount = purgeSyncedLogs();
              await refreshData();

              Alert.alert(
                'Purge Complete',
                `Deleted ${purgedCount} synced record${purgedCount !== 1 ? 's' : ''} from local storage.`
              );
            } catch (error) {
              Alert.alert('Purge Failed', 'An error occurred. Please try again.');
            } finally {
              setIsPurging(false);
            }
          },
        },
      ]
    );
  }, [syncedCount, refreshData]);

  /**
   * Formats an ISO timestamp string for display.
   *
   * @param {string} isoString - ISO 8601 timestamp
   * @returns {string} Formatted date/time string
   */
  const formatTimestamp = (isoString: string): string => {
    try {
      const date = new Date(isoString);
      return date.toLocaleString();
    } catch {
      return isoString;
    }
  };

  /**
   * Renders a single attendance log item in the history list.
   */
  const renderLogItem = ({ item }: { item: AttendanceLog }) => (
    <View style={styles.logItem}>
      <View style={styles.logItemLeft}>
        <View style={[styles.logAvatar, { borderColor: item.synced ? COLORS.SUCCESS : COLORS.WARNING }]}>
          <Text style={styles.logAvatarText}>
            {item.workerName.charAt(0).toUpperCase()}
          </Text>
        </View>
        <View style={styles.logDetails}>
          <Text style={styles.logName}>{item.workerName}</Text>
          <Text style={styles.logId}>ID: {item.workerId}</Text>
          <Text style={styles.logTime}>{formatTimestamp(item.timestamp)}</Text>
        </View>
      </View>
      <View style={styles.logItemRight}>
        <Text style={[styles.logConfidence, { color: item.confidence >= 0.9 ? COLORS.SUCCESS : COLORS.WARNING }]}>
          {Math.round(item.confidence * 100)}%
        </Text>
        <View style={[styles.syncBadge, { backgroundColor: item.synced ? 'rgba(0, 230, 118, 0.15)' : 'rgba(255, 184, 0, 0.15)' }]}>
          <Text style={[styles.syncBadgeText, { color: item.synced ? COLORS.SUCCESS : COLORS.WARNING }]}>
            {item.synced ? 'Synced' : 'Pending'}
          </Text>
        </View>
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <Text style={styles.backText}>← Back</Text>
          </TouchableOpacity>
          <Text style={styles.title}>Sync & Purge</Text>
          <Text style={styles.subtitle}>
            Manage attendance data synchronization
          </Text>
        </View>

        {/* Status Cards */}
        <View style={styles.statusCardsRow}>
          <View style={[styles.statusCard, styles.statusCardPending]}>
            <Text style={[styles.statusCardValue, pendingCount > 0 && { color: COLORS.WARNING }]}>
              {pendingCount}
            </Text>
            <Text style={styles.statusCardLabel}>Pending Sync</Text>
          </View>

          <View style={[styles.statusCard, styles.statusCardSynced]}>
            <Text style={[styles.statusCardValue, { color: COLORS.SUCCESS }]}>
              {syncedCount}
            </Text>
            <Text style={styles.statusCardLabel}>Synced (Local)</Text>
          </View>
        </View>

        {/* Last Sync Info */}
        <View style={styles.lastSyncContainer}>
          <Text style={styles.lastSyncLabel}>Last Sync</Text>
          <Text style={styles.lastSyncValue}>
            {lastSyncTimestamp ? formatTimestamp(lastSyncTimestamp) : 'Never synced'}
          </Text>
        </View>

        {/* Action Buttons */}
        <View style={styles.actionsContainer}>
          {/* Sync Button */}
          <TouchableOpacity
            style={[
              styles.actionButton,
              styles.syncButton,
              (isSyncing || pendingCount === 0) && styles.actionButtonDisabled,
            ]}
            activeOpacity={0.85}
            onPress={handleSync}
            disabled={isSyncing || pendingCount === 0}
          >
            {isSyncing ? (
              <ActivityIndicator size="small" color={COLORS.BACKGROUND} />
            ) : (
              <Text style={styles.actionButtonIcon}>☁️</Text>
            )}
            <Text style={styles.actionButtonText}>
              {isSyncing ? 'Syncing...' : 'Sync to AWS'}
            </Text>
            {pendingCount > 0 && !isSyncing && (
              <View style={styles.actionBadge}>
                <Text style={styles.actionBadgeText}>{pendingCount}</Text>
              </View>
            )}
          </TouchableOpacity>

          {/* Purge Button */}
          <TouchableOpacity
            style={[
              styles.actionButton,
              styles.purgeButton,
              (isPurging || syncedCount === 0) && styles.actionButtonDisabled,
            ]}
            activeOpacity={0.85}
            onPress={handlePurge}
            disabled={isPurging || syncedCount === 0}
          >
            {isPurging ? (
              <ActivityIndicator size="small" color={COLORS.TEXT_PRIMARY} />
            ) : (
              <Text style={styles.actionButtonIcon}>🗑️</Text>
            )}
            <Text style={[styles.actionButtonText, styles.purgeButtonText]}>
              {isPurging ? 'Purging...' : 'Purge Synced Data'}
            </Text>
            {syncedCount > 0 && !isPurging && (
              <View style={[styles.actionBadge, styles.purgeBadge]}>
                <Text style={styles.actionBadgeText}>{syncedCount}</Text>
              </View>
            )}
          </TouchableOpacity>
        </View>

        {/* Recent History */}
        <View style={styles.historyContainer}>
          <Text style={styles.historyTitle}>
            Recent Activity
          </Text>
          {recentLogs.length === 0 ? (
            <View style={styles.emptyHistory}>
              <Text style={styles.emptyHistoryIcon}>📋</Text>
              <Text style={styles.emptyHistoryText}>
                No attendance records yet.
              </Text>
              <Text style={styles.emptyHistorySubtext}>
                Records will appear here after successful face recognition.
              </Text>
            </View>
          ) : (
            <View style={styles.logsList}>
              {recentLogs.slice(0, 20).map((log) => (
                <View key={log.id}>
                  {renderLogItem({ item: log })}
                </View>
              ))}
            </View>
          )}
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
    marginBottom: SPACING.XL,
  },
  backText: {
    color: COLORS.TEXT_MUTED,
    fontSize: 14,
    marginBottom: SPACING.MD,
  },
  title: {
    fontSize: 28,
    fontWeight: '900',
    color: COLORS.TEXT_PRIMARY,
  },
  subtitle: {
    fontSize: 14,
    color: COLORS.TEXT_SECONDARY,
    marginTop: SPACING.XS,
  },
  statusCardsRow: {
    flexDirection: 'row',
    gap: SPACING.MD,
    marginBottom: SPACING.LG,
  },
  statusCard: {
    flex: 1,
    backgroundColor: COLORS.SURFACE,
    borderRadius: RADIUS.LG,
    padding: SPACING.LG,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: COLORS.BORDER,
  },
  statusCardPending: {
    borderTopWidth: 3,
    borderTopColor: COLORS.WARNING,
  },
  statusCardSynced: {
    borderTopWidth: 3,
    borderTopColor: COLORS.SUCCESS,
  },
  statusCardValue: {
    fontSize: 36,
    fontWeight: '900',
    color: COLORS.TEXT_PRIMARY,
    fontVariant: ['tabular-nums'],
  },
  statusCardLabel: {
    fontSize: 12,
    color: COLORS.TEXT_MUTED,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    fontWeight: '600',
    marginTop: SPACING.XS,
  },
  lastSyncContainer: {
    backgroundColor: COLORS.SURFACE,
    borderRadius: RADIUS.MD,
    padding: SPACING.MD,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: SPACING.LG,
    borderWidth: 1,
    borderColor: COLORS.BORDER,
  },
  lastSyncLabel: {
    fontSize: 13,
    color: COLORS.TEXT_MUTED,
    fontWeight: '600',
  },
  lastSyncValue: {
    fontSize: 13,
    color: COLORS.TEXT_SECONDARY,
    fontWeight: '500',
  },
  actionsContainer: {
    gap: SPACING.MD,
    marginBottom: SPACING.XL,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: SPACING.MD + 2,
    borderRadius: RADIUS.MD,
    gap: SPACING.SM,
  },
  syncButton: {
    backgroundColor: COLORS.PRIMARY,
  },
  purgeButton: {
    backgroundColor: 'transparent',
    borderWidth: 1.5,
    borderColor: COLORS.ERROR,
  },
  actionButtonDisabled: {
    opacity: 0.4,
  },
  actionButtonIcon: {
    fontSize: 18,
  },
  actionButtonText: {
    color: COLORS.BACKGROUND,
    fontSize: 16,
    fontWeight: '800',
  },
  purgeButtonText: {
    color: COLORS.ERROR,
  },
  actionBadge: {
    backgroundColor: 'rgba(0, 0, 0, 0.2)',
    borderRadius: RADIUS.FULL,
    minWidth: 24,
    height: 24,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 6,
  },
  purgeBadge: {
    backgroundColor: 'rgba(255, 82, 82, 0.2)',
  },
  actionBadgeText: {
    color: COLORS.TEXT_PRIMARY,
    fontSize: 12,
    fontWeight: '800',
  },
  historyContainer: {
    marginTop: SPACING.SM,
  },
  historyTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: COLORS.TEXT_PRIMARY,
    marginBottom: SPACING.MD,
  },
  logsList: {
    gap: SPACING.SM,
  },
  logItem: {
    backgroundColor: COLORS.SURFACE,
    borderRadius: RADIUS.MD,
    padding: SPACING.MD,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: COLORS.BORDER,
  },
  logItemLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    gap: SPACING.MD,
  },
  logAvatar: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: COLORS.SURFACE_LIGHT,
    borderWidth: 2,
    alignItems: 'center',
    justifyContent: 'center',
  },
  logAvatarText: {
    color: COLORS.TEXT_PRIMARY,
    fontSize: 14,
    fontWeight: '700',
  },
  logDetails: {
    flex: 1,
  },
  logName: {
    color: COLORS.TEXT_PRIMARY,
    fontSize: 14,
    fontWeight: '600',
  },
  logId: {
    color: COLORS.TEXT_MUTED,
    fontSize: 11,
  },
  logTime: {
    color: COLORS.TEXT_MUTED,
    fontSize: 10,
    marginTop: 2,
  },
  logItemRight: {
    alignItems: 'flex-end',
    gap: 4,
  },
  logConfidence: {
    fontSize: 14,
    fontWeight: '800',
    fontVariant: ['tabular-nums'],
  },
  syncBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: RADIUS.SM,
  },
  syncBadgeText: {
    fontSize: 10,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  emptyHistory: {
    alignItems: 'center',
    paddingVertical: SPACING.XXL,
  },
  emptyHistoryIcon: {
    fontSize: 40,
    marginBottom: SPACING.MD,
  },
  emptyHistoryText: {
    color: COLORS.TEXT_SECONDARY,
    fontSize: 15,
    fontWeight: '600',
  },
  emptyHistorySubtext: {
    color: COLORS.TEXT_MUTED,
    fontSize: 13,
    textAlign: 'center',
    marginTop: SPACING.XS,
    lineHeight: 18,
  },
});
