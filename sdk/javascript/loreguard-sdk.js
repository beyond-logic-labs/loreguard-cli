/**
 * Loreguard SDK for JavaScript / Node.js / Electron
 *
 * This SDK helps JavaScript-based games discover and connect to loreguard-client.
 *
 * Usage (Node.js/Electron):
 *     const { getBaseUrl, chat } = require('./loreguard-sdk');
 *
 *     // Get the URL for loreguard-client
 *     const url = getBaseUrl();  // e.g., "http://127.0.0.1:52341"
 *
 *     // Chat with an NPC (streaming)
 *     for await (const event of chat("merchant-npc", "Hello!")) {
 *       if (event.t) {
 *         process.stdout.write(event.t);
 *       } else if (event.speech) {
 *         console.log("\nVerified:", event.verified);
 *       }
 *     }
 *
 * Usage (Browser - requires bundler):
 *     import { getBaseUrl, chatFetch } from './loreguard-sdk.mjs';
 */

const fs = require('fs');
const path = require('path');
const os = require('os');

/**
 * Get the loreguard data directory path.
 * @returns {string} Path to the loreguard data directory
 */
function getDataDir() {
  switch (process.platform) {
    case 'win32':
      return path.join(process.env.APPDATA || path.join(os.homedir(), 'AppData', 'Roaming'), 'loreguard');
    case 'darwin':
      return path.join(os.homedir(), 'Library', 'Application Support', 'loreguard');
    default:
      // Linux and others
      const xdg = process.env.XDG_DATA_HOME || path.join(os.homedir(), '.local', 'share');
      return path.join(xdg, 'loreguard');
  }
}

/**
 * Get runtime info from loreguard-client.
 * @returns {Object|null} Runtime info if running, null otherwise
 */
function getRuntimeInfo() {
  const runtimePath = path.join(getDataDir(), 'runtime.json');

  if (!fs.existsSync(runtimePath)) {
    return null;
  }

  try {
    const content = fs.readFileSync(runtimePath, 'utf8');
    return JSON.parse(content);
  } catch (e) {
    return null;
  }
}

/**
 * Get the port loreguard-client is running on.
 * @returns {number} Port number
 * @throws {Error} If loreguard-client is not running
 */
function getLocalPort() {
  const info = getRuntimeInfo();
  if (!info || !info.port) {
    throw new Error('loreguard-client not running. Start it with: loreguard');
  }
  return info.port;
}

/**
 * Get the base URL for loreguard-client API.
 * @returns {string} URL like "http://127.0.0.1:52341"
 */
function getBaseUrl() {
  return `http://127.0.0.1:${getLocalPort()}`;
}

function getAuthHeaders() {
  const info = getRuntimeInfo();
  if (!info || !info.api_token) {
    throw new Error('loreguard-client runtime credential is missing; restart loreguard-client');
  }
  return { Authorization: `Bearer ${info.api_token}` };
}

/**
 * Check if loreguard-client is running.
 * @returns {boolean}
 */
function isRunning() {
  try {
    getLocalPort();
    return true;
  } catch {
    return false;
  }
}

/**
 * Chat with an NPC (async generator for streaming).
 * @param {string} characterId - NPC ID
 * @param {string} message - Player's message
 * @param {Object} options - Optional settings
 * @param {string} options.playerHandle - Player's display name
 * @param {string} options.currentContext - Game context
 * @yields {Object} Token events {t: "..."} or completion {speech: "...", verified: true}
 */
async function* chat(characterId, message, options = {}) {
  const url = `${getBaseUrl()}/api/chat`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({
      character_id: characterId,
      message: message,
      player_handle: options.playerHandle || '',
      player_id: options.playerId || '',
      current_context: options.currentContext || '',
      scenario_id: options.scenarioId || '',
      history: options.history || [],
      chunk_mode: options.chunkMode || '',
      manage_history: options.manageHistory || false,
      max_speech_tokens: options.maxSpeechTokens || 0,
      verbose: options.verbose || false,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // Keep incomplete line

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            yield data;

            // Stop after completion or error
            if (data.speech || data.error) {
              return;
            }
          } catch (e) {
            // Ignore parse errors for incomplete data
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Chat with an NPC (simple non-streaming).
 * @param {string} characterId - NPC ID
 * @param {string} message - Player's message
 * @param {Object} options - Optional settings
 * @returns {Promise<Object>} Complete response
 */
async function chatSimple(characterId, message, options = {}) {
  const url = `${getBaseUrl()}/api/chat`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({
      character_id: characterId,
      message: message,
      player_handle: options.playerHandle || '',
      player_id: options.playerId || '',
      current_context: options.currentContext || '',
      scenario_id: options.scenarioId || '',
      history: options.history || [],
      chunk_mode: options.chunkMode || '',
      manage_history: options.manageHistory || false,
      max_speech_tokens: options.maxSpeechTokens || 0,
      verbose: options.verbose || false,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Health check for loreguard-client.
 * @returns {Promise<Object>} Health status
 */
async function healthCheck() {
  const url = `${getBaseUrl()}/health`;
  const response = await fetch(url, { headers: getAuthHeaders() });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  return response.json();
}

// CommonJS exports
module.exports = {
  getDataDir,
  getRuntimeInfo,
  getLocalPort,
  getBaseUrl,
  getAuthHeaders,
  isRunning,
  chat,
  chatSimple,
  healthCheck,
};
