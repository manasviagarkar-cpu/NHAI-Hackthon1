/**
 * @module App
 * Entry point for the NHAI FaceRec React Native application.
 * Initializes the SQLite database, sets up the Gesture Handler root view,
 * provides Safe Area Context, and boots the navigation container.
 */

import React, { useEffect, useState } from 'react';
import { StyleSheet, View, Text, ActivityIndicator, StatusBar } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import AppNavigator from './src/navigation/AppNavigator';
import { initDatabase } from './src/database/DatabaseManager';
import { COLORS, SPACING } from './src/utils/Constants';

/**
 * Root component of the application.
 * Handles database bootstrapping and displays a loading screen during initialization.
 *
 * @returns {React.ReactElement} The configured application component
 */
export default function App(): React.ReactElement {
  const [dbInitialized, setDbInitialized] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function setupApp() {
      try {
        await initDatabase();
        setDbInitialized(true);
      } catch (err) {
        console.error('Failed to initialize database:', err);
        setError('Failed to initialize local database. Please restart the app.');
      }
    }
    setupApp();
  }, []);

  if (error) {
    return (
      <View style={styles.errorContainer}>
        <StatusBar barStyle="light-content" backgroundColor={COLORS.BACKGROUND} />
        <Text style={styles.errorIcon}>⚠️</Text>
        <Text style={styles.errorText}>{error}</Text>
      </View>
    );
  }

  if (!dbInitialized) {
    return (
      <View style={styles.loadingContainer}>
        <StatusBar barStyle="light-content" backgroundColor={COLORS.BACKGROUND} />
        <ActivityIndicator size="large" color={COLORS.PRIMARY} />
        <Text style={styles.loadingText}>Initializing Local Storage...</Text>
      </View>
    );
  }

  return (
    <GestureHandlerRootView style={styles.container}>
      <SafeAreaProvider>
        <StatusBar barStyle="light-content" backgroundColor={COLORS.BACKGROUND} />
        <AppNavigator />
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.BACKGROUND,
  },
  loadingContainer: {
    flex: 1,
    backgroundColor: COLORS.BACKGROUND,
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    color: COLORS.TEXT_SECONDARY,
    fontSize: 16,
    marginTop: SPACING.MD,
    fontWeight: '600',
  },
  errorContainer: {
    flex: 1,
    backgroundColor: COLORS.BACKGROUND,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: SPACING.XL,
  },
  errorIcon: {
    fontSize: 48,
    marginBottom: SPACING.MD,
  },
  errorText: {
    color: COLORS.ERROR,
    fontSize: 16,
    textAlign: 'center',
    lineHeight: 24,
    fontWeight: '600',
  },
});
