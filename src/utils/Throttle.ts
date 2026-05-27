/**
 * @module Throttle
 * Frame processing throttle utility.
 * Ensures ML inference runs at most once every THROTTLE_MS (200ms),
 * preventing CPU overload on mid-range devices while maintaining
 * responsive recognition.
 */

import { THROTTLE_MS } from './Constants';

/**
 * Creates a throttled version of a function that can only be called
 * once per interval. Subsequent calls within the interval are silently dropped.
 *
 * Unlike debounce (which delays execution), throttle guarantees the function
 * runs at regular intervals during continuous invocation — ideal for
 * camera frame processing where we want consistent 5 FPS inference
 * from a 30 FPS camera feed.
 *
 * @template T - The function signature
 * @param {T} fn - The function to throttle
 * @param {number} [intervalMs=THROTTLE_MS] - Minimum interval between calls in milliseconds
 * @returns {T} Throttled version of the function
 *
 * @example
 * const throttledInference = createThrottle(processFrame, 200);
 * // In frame processor callback:
 * camera.onFrame((frame) => throttledInference(frame));
 */
export function createThrottle<T extends (...args: any[]) => any>(
  fn: T,
  intervalMs: number = THROTTLE_MS
): T {
  let lastCallTime = 0;
  let isRunning = false;

  const throttled = ((...args: any[]) => {
    const now = performance.now();

    // Skip if we're still within the throttle interval
    if (now - lastCallTime < intervalMs) {
      return undefined;
    }

    // Skip if previous invocation is still running (async safety)
    if (isRunning) {
      return undefined;
    }

    lastCallTime = now;

    // Handle both sync and async functions
    const result = fn(...args);

    if (result instanceof Promise) {
      isRunning = true;
      result
        .then(() => {
          isRunning = false;
        })
        .catch(() => {
          isRunning = false;
        });
    }

    return result;
  }) as T;

  return throttled;
}

/**
 * Simple timestamp-based throttle check without wrapping a function.
 * Useful in frame processors where you want inline throttle control.
 *
 * @example
 * const throttle = new ThrottleTimer(200);
 * // In frame processor:
 * if (throttle.shouldProcess()) {
 *   runInference(frame);
 * }
 */
export class ThrottleTimer {
  private lastProcessTime: number = 0;
  private readonly intervalMs: number;

  /**
   * Creates a new ThrottleTimer.
   *
   * @param {number} [intervalMs=THROTTLE_MS] - Minimum interval between allowed operations
   */
  constructor(intervalMs: number = THROTTLE_MS) {
    this.intervalMs = intervalMs;
  }

  /**
   * Checks if enough time has passed since the last processed frame.
   * If yes, updates the internal timestamp and returns true.
   * If no, returns false without side effects.
   *
   * @returns {boolean} True if the caller should process, false to skip
   */
  shouldProcess(): boolean {
    const now = performance.now();

    if (now - this.lastProcessTime >= this.intervalMs) {
      this.lastProcessTime = now;
      return true;
    }

    return false;
  }

  /**
   * Resets the throttle timer, allowing the next call to process immediately.
   */
  reset(): void {
    this.lastProcessTime = 0;
  }

  /**
   * Returns milliseconds until the next allowed process call.
   *
   * @returns {number} Milliseconds remaining (0 if ready to process)
   */
  timeUntilNext(): number {
    const elapsed = performance.now() - this.lastProcessTime;
    return Math.max(0, this.intervalMs - elapsed);
  }
}
