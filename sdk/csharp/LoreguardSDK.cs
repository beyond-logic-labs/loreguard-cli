/*
 * Loreguard SDK for Unity / C#
 *
 * This SDK helps Unity games discover and connect to loreguard-client.
 *
 * Usage:
 *     // Get the URL for loreguard-client
 *     string url = LoreguardSDK.GetBaseUrl();  // e.g., "http://127.0.0.1:52341"
 *
 *     // Chat with an NPC
 *     StartCoroutine(LoreguardSDK.Chat("merchant-npc", "Hello!", OnToken, OnComplete));
 *
 * Installation:
 *     1. Copy this file to your Unity project's Assets/Scripts folder
 *     2. Add Newtonsoft.Json package (or use Unity's JsonUtility)
 */

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace Loreguard
{
    /// <summary>
    /// SDK for connecting Unity games to loreguard-client.
    /// </summary>
    public static class LoreguardSDK
    {
        private static int? _cachedPort = null;
        private static float _cacheTime = 0f;
        private const float CACHE_DURATION = 5f; // Re-check port every 5 seconds

        /// <summary>
        /// Get the loreguard data directory path.
        /// </summary>
        public static string GetDataDir()
        {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "loreguard"
            );
#elif UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.Personal),
                "Library/Application Support/loreguard"
            );
#else
            // Linux
            var xdg = Environment.GetEnvironmentVariable("XDG_DATA_HOME");
            var baseDir = string.IsNullOrEmpty(xdg)
                ? Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.Personal),
                    ".local/share"
                )
                : xdg;
            return Path.Combine(baseDir, "loreguard");
#endif
        }

        /// <summary>
        /// Get runtime info from loreguard-client.
        /// </summary>
        /// <returns>RuntimeInfo if running, null otherwise</returns>
        public static RuntimeInfo GetRuntimeInfo()
        {
            var runtimePath = Path.Combine(GetDataDir(), "runtime.json");

            if (!File.Exists(runtimePath))
                return null;

            try
            {
                var json = File.ReadAllText(runtimePath);
                return JsonUtility.FromJson<RuntimeInfo>(json);
            }
            catch (Exception)
            {
                return null;
            }
        }

        /// <summary>
        /// Get the port loreguard-client is running on.
        /// </summary>
        /// <returns>Port number</returns>
        /// <exception cref="Exception">If loreguard-client is not running</exception>
        public static int GetLocalPort()
        {
            // Use cached port if recent
            if (_cachedPort.HasValue && Time.realtimeSinceStartup - _cacheTime < CACHE_DURATION)
            {
                return _cachedPort.Value;
            }

            var info = GetRuntimeInfo();
            if (info == null || info.port == 0)
            {
                throw new Exception(
                    "loreguard-client not running. Start it with: loreguard"
                );
            }

            _cachedPort = info.port;
            _cacheTime = Time.realtimeSinceStartup;
            return info.port;
        }

        /// <summary>
        /// Get the base URL for loreguard-client API.
        /// </summary>
        /// <returns>URL like "http://127.0.0.1:52341"</returns>
        public static string GetBaseUrl()
        {
            return $"http://127.0.0.1:{GetLocalPort()}";
        }

        /// <summary>
        /// Check if loreguard-client is running.
        /// </summary>
        public static bool IsRunning()
        {
            try
            {
                GetLocalPort();
                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Chat with an NPC (coroutine for Unity).
        /// </summary>
        /// <param name="characterId">NPC ID</param>
        /// <param name="message">Player's message</param>
        /// <param name="onToken">Called for each token (streaming)</param>
        /// <param name="onComplete">Called when done with full response</param>
        /// <param name="onError">Called on error</param>
        /// <param name="playerHandle">Player's display name (optional)</param>
        public static IEnumerator Chat(
            string characterId,
            string message,
            Action<string> onToken = null,
            Action<ChatResponse> onComplete = null,
            Action<string> onError = null,
            string playerHandle = ""
        )
        {
            string url;
            try
            {
                url = $"{GetBaseUrl()}/api/chat";
            }
            catch (Exception e)
            {
                onError?.Invoke(e.Message);
                yield break;
            }

            var request = new ChatRequest
            {
                character_id = characterId,
                message = message,
                player_handle = playerHandle
            };

            var body = JsonUtility.ToJson(request);
            var bodyBytes = Encoding.UTF8.GetBytes(body);

            using var webRequest = new UnityWebRequest(url, "POST");
            webRequest.uploadHandler = new UploadHandlerRaw(bodyBytes);
            webRequest.downloadHandler = new DownloadHandlerBuffer();
            webRequest.SetRequestHeader("Content-Type", "application/json");
            webRequest.SetRequestHeader("Accept", "text/event-stream");

            webRequest.SendWebRequest();

            var buffer = new StringBuilder();
            var lastLength = 0;

            // Poll for streaming data
            while (!webRequest.isDone)
            {
                if (webRequest.downloadHandler.data != null &&
                    webRequest.downloadHandler.data.Length > lastLength)
                {
                    var newData = Encoding.UTF8.GetString(
                        webRequest.downloadHandler.data,
                        lastLength,
                        webRequest.downloadHandler.data.Length - lastLength
                    );
                    lastLength = webRequest.downloadHandler.data.Length;

                    // Parse SSE events
                    buffer.Append(newData);
                    // Handle both Unix (\n) and Windows (\r\n) line endings
                    var lines = buffer.ToString().Replace("\r\n", "\n").Split('\n');
                    buffer.Clear();

                    foreach (var rawLine in lines)
                    {
                        var line = rawLine.Trim();  // Remove any remaining \r
                        if (line.StartsWith("data: "))
                        {
                            var json = line.Substring(6).Trim();
                            try
                            {
                                // Check for token
                                if (json.Contains("\"t\""))
                                {
                                    var token = JsonUtility.FromJson<TokenEvent>(json);
                                    onToken?.Invoke(token.t);
                                }
                                // Check for completion
                                else if (json.Contains("\"speech\""))
                                {
                                    var response = JsonUtility.FromJson<ChatResponse>(json);
                                    onComplete?.Invoke(response);
                                }
                                // Check for error
                                else if (json.Contains("\"error\""))
                                {
                                    var error = JsonUtility.FromJson<ErrorEvent>(json);
                                    onError?.Invoke(error.error);
                                }
                            }
                            catch (Exception)
                            {
                                // Keep partial line for next iteration
                                buffer.Append(line);
                            }
                        }
                        else if (!string.IsNullOrWhiteSpace(line))
                        {
                            buffer.Append(line);
                        }
                    }
                }

                yield return null;
            }

            if (webRequest.result != UnityWebRequest.Result.Success)
            {
                onError?.Invoke(webRequest.error);
            }
        }

        /// <summary>
        /// Simple non-streaming chat (coroutine).
        /// </summary>
        public static IEnumerator ChatSimple(
            string characterId,
            string message,
            Action<ChatResponse> onComplete,
            Action<string> onError = null,
            string playerHandle = ""
        )
        {
            string url;
            try
            {
                url = $"{GetBaseUrl()}/api/chat";
            }
            catch (Exception e)
            {
                onError?.Invoke(e.Message);
                yield break;
            }

            var request = new ChatRequest
            {
                character_id = characterId,
                message = message,
                player_handle = playerHandle
            };

            var body = JsonUtility.ToJson(request);

            using var webRequest = UnityWebRequest.Post(url, body, "application/json");
            yield return webRequest.SendWebRequest();

            if (webRequest.result == UnityWebRequest.Result.Success)
            {
                var response = JsonUtility.FromJson<ChatResponse>(
                    webRequest.downloadHandler.text
                );
                onComplete?.Invoke(response);
            }
            else
            {
                onError?.Invoke(webRequest.error);
            }
        }

        // Data classes
        [Serializable]
        public class RuntimeInfo
        {
            public int port;
            public int pid;
            public string started_at;
            public string version;
            public bool backend_connected;
        }

        [Serializable]
        public class ChatRequest
        {
            public string character_id;
            public string message;
            public string player_handle;
            public string current_context;
        }

        [Serializable]
        public class ChatResponse
        {
            public string speech;
            public string thoughts;
            public bool verified;
            public int retries;
            public long latency_ms;
        }

        [Serializable]
        private class TokenEvent
        {
            public string t;
            public int i;
        }

        [Serializable]
        private class ErrorEvent
        {
            public string error;
        }
    }
}
