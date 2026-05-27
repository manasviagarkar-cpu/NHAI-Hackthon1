/**
 * @module AppNavigator
 * Navigation configuration for NHAI FaceRec.
 * Sets up the native stack navigator with dark mode styles matching the premium theme.
 */

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import HomeScreen from '../screens/HomeScreen';
import RegistrationScreen from '../screens/RegistrationScreen';
import LivenessScreen from '../screens/LivenessScreen';
import RecognitionScreen from '../screens/RecognitionScreen';
import SyncScreen from '../screens/SyncScreen';
import { COLORS } from '../utils/Constants';

/**
 * Type definitions for the Root Stack Navigator.
 */
export type RootStackParamList = {
  Home: undefined;
  Registration: undefined;
  Liveness: undefined;
  Recognition: undefined;
  Sync: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

/**
 * Main application navigation component.
 * Configures screen transitions, headers, and global dark theme matching the NHAI colors.
 *
 * @returns {React.ReactElement} The configured navigation container
 */
export default function AppNavigator(): React.ReactElement {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Home"
        screenOptions={{
          headerShown: false,
          animation: 'slide_from_right',
          contentStyle: { backgroundColor: COLORS.BACKGROUND },
        }}
      >
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="Registration" component={RegistrationScreen} />
        <Stack.Screen name="Liveness" component={LivenessScreen} />
        <Stack.Screen name="Recognition" component={RecognitionScreen} />
        <Stack.Screen name="Sync" component={SyncScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
