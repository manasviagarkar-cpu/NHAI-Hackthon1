const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Register .tflite model files as assets so they are bundled with the app
config.resolver.assetExts.push('tflite');

module.exports = config;
