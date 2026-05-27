/**
 * @module DatabaseManager
 * Initializes and manages the SQLite database for offline storage.
 * Uses expo-sqlite with WAL journal mode for optimal read/write performance.
 *
 * Tables:
 *   - workers: Stores registered worker profiles with face embeddings
 *   - attendance_logs: Records attendance events for later sync
 */

import * as SQLite from 'expo-sqlite';
import { DB_NAME } from '../utils/Constants';

/** Singleton database instance */
let db: SQLite.SQLiteDatabase | null = null;

/**
 * Opens (or creates) the SQLite database and returns the singleton instance.
 * The database is created in the app's default document directory.
 *
 * @returns {SQLite.SQLiteDatabase} The opened database instance
 */
export function getDatabase(): SQLite.SQLiteDatabase {
  if (!db) {
    db = SQLite.openDatabaseSync(DB_NAME);
  }
  return db;
}

/**
 * Initializes the database schema. Creates all required tables if they
 * don't already exist, and enables WAL journal mode for performance.
 *
 * Must be called once at app startup before any database operations.
 *
 * @returns {Promise<void>}
 */
export async function initDatabase(): Promise<void> {
  const database = getDatabase();

  // Enable WAL mode for better concurrent read/write performance
  database.execSync('PRAGMA journal_mode = WAL;');

  // Enable foreign keys
  database.execSync('PRAGMA foreign_keys = ON;');

  // Create workers table
  // - id: Auto-incrementing primary key
  // - worker_id: External worker identifier (e.g., employee ID)
  // - name: Worker display name
  // - embedding: Face embedding stored as BLOB (Float32Array → ArrayBuffer)
  // - created_at: ISO 8601 timestamp of registration
  database.execSync(`
    CREATE TABLE IF NOT EXISTS workers (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      worker_id TEXT NOT NULL UNIQUE,
      name TEXT NOT NULL,
      embedding BLOB NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    );
  `);

  // Create index on worker_id for fast lookups
  database.execSync(`
    CREATE INDEX IF NOT EXISTS idx_workers_worker_id ON workers(worker_id);
  `);

  // Create attendance_logs table
  // - id: Auto-incrementing primary key
  // - worker_id: References the worker who was recognized
  // - worker_name: Denormalized name for quick display without JOINs
  // - confidence: Cosine similarity score of the match
  // - timestamp: ISO 8601 timestamp of the recognition event
  // - synced: Boolean flag (0 = pending, 1 = synced to AWS)
  database.execSync(`
    CREATE TABLE IF NOT EXISTS attendance_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      worker_id TEXT NOT NULL,
      worker_name TEXT NOT NULL,
      confidence REAL NOT NULL DEFAULT 0.0,
      timestamp TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
      synced INTEGER NOT NULL DEFAULT 0
    );
  `);

  // Create index on synced for efficient pending count queries
  database.execSync(`
    CREATE INDEX IF NOT EXISTS idx_attendance_synced ON attendance_logs(synced);
  `);

  // Create index on worker_id for attendance history lookups
  database.execSync(`
    CREATE INDEX IF NOT EXISTS idx_attendance_worker ON attendance_logs(worker_id);
  `);
}

/**
 * Drops all tables and recreates the schema. USE WITH EXTREME CAUTION.
 * This is intended for development/testing only.
 *
 * @returns {Promise<void>}
 */
export async function resetDatabase(): Promise<void> {
  const database = getDatabase();

  database.execSync('DROP TABLE IF EXISTS attendance_logs;');
  database.execSync('DROP TABLE IF EXISTS workers;');

  await initDatabase();
}

/**
 * Returns basic database statistics for the home screen dashboard.
 *
 * @returns {{ workerCount: number; pendingSync: number; totalLogs: number }}
 */
export function getDatabaseStats(): {
  workerCount: number;
  pendingSync: number;
  totalLogs: number;
} {
  const database = getDatabase();

  const workerRow = database.getFirstSync<{ count: number }>(
    'SELECT COUNT(*) as count FROM workers;'
  );
  const pendingRow = database.getFirstSync<{ count: number }>(
    'SELECT COUNT(*) as count FROM attendance_logs WHERE synced = 0;'
  );
  const totalRow = database.getFirstSync<{ count: number }>(
    'SELECT COUNT(*) as count FROM attendance_logs;'
  );

  return {
    workerCount: workerRow?.count ?? 0,
    pendingSync: pendingRow?.count ?? 0,
    totalLogs: totalRow?.count ?? 0,
  };
}

/**
 * Closes the database connection. Call on app shutdown if needed.
 */
export function closeDatabase(): void {
  if (db) {
    db.closeSync();
    db = null;
  }
}
