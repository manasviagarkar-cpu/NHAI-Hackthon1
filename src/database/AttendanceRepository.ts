/**
 * @module AttendanceRepository
 * CRUD operations for the attendance_logs table.
 * Records are created with synced=false after successful recognition,
 * and later marked as synced after AWS upload (or mock sync).
 */

import { getDatabase } from './DatabaseManager';

/**
 * Represents a single attendance log entry.
 */
export interface AttendanceLog {
  /** Auto-incremented database row ID */
  id: number;
  /** External worker identifier who was recognized */
  workerId: string;
  /** Worker display name (denormalized for quick access) */
  workerName: string;
  /** Cosine similarity confidence score of the match */
  confidence: number;
  /** ISO 8601 timestamp of the recognition event */
  timestamp: string;
  /** Whether this record has been synced to the server */
  synced: boolean;
}

/** Raw row shape from SQLite */
interface AttendanceRow {
  id: number;
  worker_id: string;
  worker_name: string;
  confidence: number;
  timestamp: string;
  synced: number;
}

/**
 * Converts a raw database row into a typed AttendanceLog object.
 *
 * @param {AttendanceRow} row - Raw row from SQLite query
 * @returns {AttendanceLog} Typed attendance log object
 */
function rowToAttendanceLog(row: AttendanceRow): AttendanceLog {
  return {
    id: row.id,
    workerId: row.worker_id,
    workerName: row.worker_name,
    confidence: row.confidence,
    timestamp: row.timestamp,
    synced: row.synced === 1,
  };
}

/**
 * Inserts a new attendance log entry after a successful face recognition.
 * The record is created with synced=false (pending sync to AWS).
 *
 * @param {string} workerId - The recognized worker's external ID
 * @param {string} workerName - The recognized worker's display name
 * @param {number} confidence - Cosine similarity score (0.0 – 1.0)
 * @returns {number} The auto-generated row ID of the new log entry
 */
export function logAttendance(
  workerId: string,
  workerName: string,
  confidence: number
): number {
  const db = getDatabase();

  const result = db.runSync(
    `INSERT INTO attendance_logs (worker_id, worker_name, confidence) 
     VALUES (?, ?, ?);`,
    [workerId, workerName, confidence]
  );

  return result.lastInsertRowId;
}

/**
 * Returns the count of attendance records that have not yet been synced.
 * Used by the Sync screen badge and dashboard display.
 *
 * @returns {number} Count of records where synced = false
 */
export function getPendingSyncCount(): number {
  const db = getDatabase();
  const row = db.getFirstSync<{ count: number }>(
    'SELECT COUNT(*) as count FROM attendance_logs WHERE synced = 0;'
  );
  return row?.count ?? 0;
}

/**
 * Returns all attendance logs that have not yet been synced.
 * Used by the sync process to batch upload to AWS.
 *
 * @returns {AttendanceLog[]} Array of unsynced attendance records
 */
export function getPendingLogs(): AttendanceLog[] {
  const db = getDatabase();
  const rows = db.getAllSync<AttendanceRow>(
    'SELECT * FROM attendance_logs WHERE synced = 0 ORDER BY timestamp DESC;'
  );
  return rows.map(rowToAttendanceLog);
}

/**
 * Marks the specified attendance log IDs as synced (synced = true).
 * Called after successful upload to AWS.
 *
 * @param {number[]} ids - Array of attendance log IDs to mark as synced
 * @returns {number} Number of records updated
 */
export function markAsSynced(ids: number[]): number {
  if (ids.length === 0) return 0;

  const db = getDatabase();
  const placeholders = ids.map(() => '?').join(', ');

  const result = db.runSync(
    `UPDATE attendance_logs SET synced = 1 WHERE id IN (${placeholders});`,
    ids
  );

  return result.changes;
}

/**
 * Marks ALL pending attendance logs as synced.
 * Used by the mock "Sync to AWS" button for hackathon demo.
 *
 * @returns {number} Number of records updated
 */
export function markAllAsSynced(): number {
  const db = getDatabase();
  const result = db.runSync(
    'UPDATE attendance_logs SET synced = 1 WHERE synced = 0;'
  );
  return result.changes;
}

/**
 * Permanently deletes all attendance logs that have been synced.
 * Used by the "Purge synced data" feature to free device storage.
 *
 * @returns {number} Number of records deleted
 */
export function purgeSyncedLogs(): number {
  const db = getDatabase();
  const result = db.runSync(
    'DELETE FROM attendance_logs WHERE synced = 1;'
  );
  return result.changes;
}

/**
 * Returns all attendance logs, ordered by most recent first.
 * Optionally limited to a maximum number of records.
 *
 * @param {number} [limit] - Maximum number of records to return (optional)
 * @returns {AttendanceLog[]} Array of attendance records
 */
export function getAllLogs(limit?: number): AttendanceLog[] {
  const db = getDatabase();
  const query = limit
    ? 'SELECT * FROM attendance_logs ORDER BY timestamp DESC LIMIT ?;'
    : 'SELECT * FROM attendance_logs ORDER BY timestamp DESC;';
  const params = limit ? [limit] : [];

  const rows = db.getAllSync<AttendanceRow>(query, params);
  return rows.map(rowToAttendanceLog);
}

/**
 * Returns attendance logs for a specific worker, ordered by most recent first.
 *
 * @param {string} workerId - The worker's external ID
 * @returns {AttendanceLog[]} Array of attendance records for this worker
 */
export function getLogsByWorker(workerId: string): AttendanceLog[] {
  const db = getDatabase();
  const rows = db.getAllSync<AttendanceRow>(
    'SELECT * FROM attendance_logs WHERE worker_id = ? ORDER BY timestamp DESC;',
    [workerId]
  );
  return rows.map(rowToAttendanceLog);
}

/**
 * Returns the total count of all attendance logs (synced + unsynced).
 *
 * @returns {number} Total count of attendance records
 */
export function getTotalLogCount(): number {
  const db = getDatabase();
  const row = db.getFirstSync<{ count: number }>(
    'SELECT COUNT(*) as count FROM attendance_logs;'
  );
  return row?.count ?? 0;
}

/**
 * Returns the count of synced attendance logs (available for purge).
 *
 * @returns {number} Count of synced records
 */
export function getSyncedCount(): number {
  const db = getDatabase();
  const row = db.getFirstSync<{ count: number }>(
    'SELECT COUNT(*) as count FROM attendance_logs WHERE synced = 1;'
  );
  return row?.count ?? 0;
}
