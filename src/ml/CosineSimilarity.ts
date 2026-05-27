/**
 * @module CosineSimilarity
 * Computes cosine similarity between face embedding vectors and finds
 * the best match against the database of registered workers.
 *
 * Cosine similarity measures the angle between two vectors in the
 * embedding space: 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite.
 */

import { Worker } from '../database/WorkerRepository';
import { COSINE_THRESHOLD, EMBEDDING_DIM } from '../utils/Constants';

/**
 * Result of a face match attempt against the worker database.
 */
export interface MatchResult {
  /** Whether a match was found above the confidence threshold */
  matched: boolean;
  /** The matched worker (null if no match) */
  worker: Worker | null;
  /** Cosine similarity score of the best match (0.0 – 1.0) */
  score: number;
  /** Time taken for the similarity search in milliseconds */
  searchTimeMs: number;
}

/**
 * Computes the cosine similarity between two embedding vectors.
 *
 * Formula: cos(θ) = (A · B) / (||A|| × ||B||)
 *
 * @param {Float32Array} a - First embedding vector (length must equal EMBEDDING_DIM)
 * @param {Float32Array} b - Second embedding vector (length must equal EMBEDDING_DIM)
 * @returns {number} Cosine similarity in range [-1.0, 1.0]
 * @throws {Error} If vectors have different lengths or don't match EMBEDDING_DIM
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(
      `Vector length mismatch: ${a.length} vs ${b.length}`
    );
  }
  if (a.length !== EMBEDDING_DIM) {
    throw new Error(
      `Expected ${EMBEDDING_DIM}-dim vectors, got ${a.length}-dim`
    );
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  // Single-pass computation of dot product and norms
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    const ai = a[i];
    const bi = b[i];
    dotProduct += ai * bi;
    normA += ai * ai;
    normB += bi * bi;
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);

  // Prevent division by zero (zero vector edge case)
  if (denominator === 0) {
    return 0;
  }

  return dotProduct / denominator;
}

/**
 * Searches through all registered workers to find the best matching face.
 * Returns the worker with the highest cosine similarity score above the threshold.
 *
 * Performance: O(n × d) where n = number of workers, d = embedding dimension.
 * For 100 workers with 128-dim embeddings, this takes < 1ms on modern devices.
 *
 * @param {Float32Array} liveEmbedding - The face embedding from the live camera frame
 * @param {Worker[]} workers - Array of all registered workers with their embeddings
 * @param {number} [threshold=COSINE_THRESHOLD] - Minimum similarity score for a match
 * @returns {MatchResult} The best match result with timing information
 */
export function findBestMatch(
  liveEmbedding: Float32Array,
  workers: Worker[],
  threshold: number = COSINE_THRESHOLD
): MatchResult {
  const startTime = performance.now();

  if (workers.length === 0) {
    return {
      matched: false,
      worker: null,
      score: 0,
      searchTimeMs: performance.now() - startTime,
    };
  }

  let bestScore = -1;
  let bestWorker: Worker | null = null;

  for (let i = 0; i < workers.length; i++) {
    const score = cosineSimilarity(liveEmbedding, workers[i].embedding);
    if (score > bestScore) {
      bestScore = score;
      bestWorker = workers[i];
    }
  }

  const searchTimeMs = performance.now() - startTime;

  return {
    matched: bestScore >= threshold,
    worker: bestScore >= threshold ? bestWorker : null,
    score: bestScore,
    searchTimeMs,
  };
}

/**
 * Averages multiple face embeddings into a single representative embedding.
 * Used during registration to combine multiple captures for robustness.
 *
 * The averaged embedding is L2-normalized to maintain unit length,
 * which is important for consistent cosine similarity comparisons.
 *
 * @param {Float32Array[]} embeddings - Array of embeddings to average (must all be EMBEDDING_DIM length)
 * @returns {Float32Array} The averaged and L2-normalized embedding
 * @throws {Error} If no embeddings provided or dimension mismatch
 */
export function averageEmbeddings(embeddings: Float32Array[]): Float32Array {
  if (embeddings.length === 0) {
    throw new Error('Cannot average zero embeddings');
  }

  const count = embeddings.length;
  const averaged = new Float32Array(EMBEDDING_DIM);

  // Sum all embeddings element-wise
  for (let i = 0; i < count; i++) {
    if (embeddings[i].length !== EMBEDDING_DIM) {
      throw new Error(
        `Embedding ${i} dimension mismatch: expected ${EMBEDDING_DIM}, got ${embeddings[i].length}`
      );
    }
    for (let j = 0; j < EMBEDDING_DIM; j++) {
      averaged[j] += embeddings[i][j];
    }
  }

  // Divide by count to get mean
  for (let j = 0; j < EMBEDDING_DIM; j++) {
    averaged[j] /= count;
  }

  // L2-normalize the result for consistent cosine similarity
  let norm = 0;
  for (let j = 0; j < EMBEDDING_DIM; j++) {
    norm += averaged[j] * averaged[j];
  }
  norm = Math.sqrt(norm);

  if (norm > 0) {
    for (let j = 0; j < EMBEDDING_DIM; j++) {
      averaged[j] /= norm;
    }
  }

  return averaged;
}
