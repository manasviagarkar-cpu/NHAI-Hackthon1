/**
 * @module WorkerRepository
 * CRUD operations for the workers table.
 * Handles face embedding serialization (Float32Array ↔ BLOB) and
 * provides lookup methods for the recognition pipeline.
 */

import { getDatabase } from './DatabaseManager';
import { EMBEDDING_DIM } from '../utils/Constants';

/**
 * Represents a registered worker with their face embedding.
 */
export interface Worker {
  /** Auto-incremented database row ID */
  id: number;
  /** External worker identifier (e.g., employee badge ID) */
  workerId: string;
  /** Worker display name */
  name: string;
  /** 128-dimensional face embedding vector */
  embedding: Float32Array;
  /** ISO 8601 registration timestamp */
  createdAt: string;
}

/** Raw row shape from SQLite (embedding is ArrayBuffer from BLOB) */
interface WorkerRow {
  id: number;
  worker_id: string;
  name: string;
  embedding: ArrayBuffer;
  created_at: string;
}

/**
 * Converts a Float32Array to an ArrayBuffer for SQLite BLOB storage.
 * The embedding is stored as raw IEEE 754 float32 bytes.
 *
 * @param {Float32Array} embedding - The face embedding vector
 * @returns {ArrayBuffer} Binary representation for BLOB storage
 */
function embeddingToBlob(embedding: Float32Array): ArrayBuffer {
  return embedding.buffer.slice(
    embedding.byteOffset,
    embedding.byteOffset + embedding.byteLength
  );
}

/**
 * Converts an ArrayBuffer from SQLite BLOB back to a Float32Array.
 *
 * @param {ArrayBuffer} blob - The raw bytes from the BLOB column
 * @returns {Float32Array} The reconstructed embedding vector
 */
function blobToEmbedding(blob: ArrayBuffer): Float32Array {
  return new Float32Array(blob);
}

/**
 * Converts a raw database row into a typed Worker object.
 *
 * @param {WorkerRow} row - Raw row from SQLite query
 * @returns {Worker} Typed worker object with Float32Array embedding
 */
function rowToWorker(row: WorkerRow): Worker {
  return {
    id: row.id,
    workerId: row.worker_id,
    name: row.name,
    embedding: blobToEmbedding(row.embedding),
    createdAt: row.created_at,
  };
}

/**
 * Inserts a new worker with their averaged face embedding into the database.
 * If a worker with the same worker_id already exists, the operation will
 * throw a UNIQUE constraint error.
 *
 * @param {string} workerId - Unique external identifier for the worker
 * @param {string} name - Display name of the worker
 * @param {Float32Array} embedding - 128-dimensional averaged face embedding
 * @returns {number} The auto-generated row ID of the inserted worker
 * @throws {Error} If worker_id already exists or embedding dimension is wrong
 */
export function insertWorker(
  workerId: string,
  name: string,
  embedding: Float32Array
): number {
  if (embedding.length !== EMBEDDING_DIM) {
    throw new Error(
      `Embedding dimension mismatch: expected ${EMBEDDING_DIM}, got ${embedding.length}`
    );
  }

  const db = getDatabase();
  const blob = embeddingToBlob(embedding);

  const result = db.runSync(
    'INSERT INTO workers (worker_id, name, embedding) VALUES (?, ?, ?);',
    [workerId, name, blob as any]
  );

  return result.lastInsertRowId;
}

/**
 * Updates an existing worker's face embedding. Useful for re-registration
 * to improve recognition accuracy over time.
 *
 * @param {string} workerId - The external worker ID to update
 * @param {Float32Array} embedding - New 128-dimensional face embedding
 * @returns {boolean} True if a row was updated, false if worker not found
 */
export function updateWorkerEmbedding(
  workerId: string,
  embedding: Float32Array
): boolean {
  if (embedding.length !== EMBEDDING_DIM) {
    throw new Error(
      `Embedding dimension mismatch: expected ${EMBEDDING_DIM}, got ${embedding.length}`
    );
  }

  const db = getDatabase();
  const blob = embeddingToBlob(embedding);

  const result = db.runSync(
    'UPDATE workers SET embedding = ? WHERE worker_id = ?;',
    [blob as any, workerId]
  );

  return result.changes > 0;
}

/**
 * Retrieves all registered workers with their embeddings.
 * Used by the recognition pipeline to compare live embeddings
 * against the full database.
 *
 * @returns {Worker[]} Array of all registered workers
 */
export function getAllWorkers(): Worker[] {
  const db = getDatabase();
  const rows = db.getAllSync<WorkerRow>('SELECT * FROM workers ORDER BY name;');
  return rows.map(rowToWorker);
}

/**
 * Retrieves a single worker by their external worker ID.
 *
 * @param {string} workerId - The external worker ID to look up
 * @returns {Worker | null} The worker if found, null otherwise
 */
export function getWorkerById(workerId: string): Worker | null {
  const db = getDatabase();
  const row = db.getFirstSync<WorkerRow>(
    'SELECT * FROM workers WHERE worker_id = ?;',
    [workerId]
  );
  return row ? rowToWorker(row) : null;
}

/**
 * Deletes a worker and their embedding from the database.
 * Does NOT delete associated attendance logs (those are historical records).
 *
 * @param {string} workerId - The external worker ID to delete
 * @returns {boolean} True if a worker was deleted, false if not found
 */
export function deleteWorker(workerId: string): boolean {
  const db = getDatabase();
  const result = db.runSync(
    'DELETE FROM workers WHERE worker_id = ?;',
    [workerId]
  );
  return result.changes > 0;
}

/**
 * Returns the total number of registered workers.
 *
 * @returns {number} Count of workers in the database
 */
export function getWorkerCount(): number {
  const db = getDatabase();
  const row = db.getFirstSync<{ count: number }>(
    'SELECT COUNT(*) as count FROM workers;'
  );
  return row?.count ?? 0;
}

/**
 * Checks whether a worker with the given ID already exists.
 *
 * @param {string} workerId - The external worker ID to check
 * @returns {boolean} True if the worker exists
 */
export function workerExists(workerId: string): boolean {
  const db = getDatabase();
  const row = db.getFirstSync<{ count: number }>(
    'SELECT COUNT(*) as count FROM workers WHERE worker_id = ?;',
    [workerId]
  );
  return (row?.count ?? 0) > 0;
}
