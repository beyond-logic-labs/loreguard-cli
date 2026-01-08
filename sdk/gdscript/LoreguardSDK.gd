## Loreguard SDK for Godot 4
##
## This SDK helps Godot games discover and connect to loreguard-client.
##
## Usage:
##     # Get the URL for loreguard-client
##     var url = LoreguardSDK.get_base_url()  # e.g., "http://127.0.0.1:52341"
##
##     # Chat with an NPC
##     var sdk = LoreguardSDK.new()
##     add_child(sdk)
##     sdk.chat_completed.connect(_on_chat_completed)
##     sdk.chat("merchant-npc", "Hello!")
##
## Installation:
##     1. Copy this file to your Godot project
##     2. Add as autoload or instantiate as needed

class_name LoreguardSDK
extends Node

## Emitted for each token during streaming
signal token_received(token: String)

## Emitted when chat completes
signal chat_completed(response: Dictionary)

## Emitted on error
signal chat_error(message: String)

var _http_request: HTTPRequest = null


static func get_data_dir() -> String:
	"""Get the loreguard data directory path."""
	match OS.get_name():
		"Windows":
			var appdata = OS.get_environment("APPDATA")
			if appdata.is_empty():
				appdata = OS.get_environment("USERPROFILE") + "/AppData/Roaming"
			return appdata + "/loreguard"
		"macOS":
			return OS.get_environment("HOME") + "/Library/Application Support/loreguard"
		_:
			# Linux and others
			var xdg = OS.get_environment("XDG_DATA_HOME")
			if xdg.is_empty():
				xdg = OS.get_environment("HOME") + "/.local/share"
			return xdg + "/loreguard"


static func get_runtime_info() -> Variant:
	"""Load runtime info from loreguard-client.

	Returns Dictionary if running, null otherwise.
	"""
	var runtime_path = get_data_dir() + "/runtime.json"

	if not FileAccess.file_exists(runtime_path):
		return null

	var file = FileAccess.open(runtime_path, FileAccess.READ)
	if file == null:
		return null

	var content = file.get_as_text()
	file.close()

	var json = JSON.new()
	var error = json.parse(content)
	if error != OK:
		return null

	return json.data


static func get_local_port() -> int:
	"""Get the port loreguard-client is running on.

	Returns -1 if not running.
	"""
	var info = get_runtime_info()
	if info == null or not info.has("port"):
		push_error("loreguard-client not running. Start it with: loreguard")
		return -1
	return info["port"]


static func get_base_url() -> String:
	"""Get the base URL for loreguard-client API."""
	var port = get_local_port()
	if port == -1:
		return ""
	return "http://127.0.0.1:%d" % port


static func is_running() -> bool:
	"""Check if loreguard-client is running."""
	return get_local_port() != -1


func _ready():
	_http_request = HTTPRequest.new()
	add_child(_http_request)
	_http_request.request_completed.connect(_on_request_completed)


func chat(
	character_id: String,
	message: String,
	player_handle: String = "",
	current_context: String = ""
) -> void:
	"""Chat with an NPC.

	Emits token_received for each token (if streaming is supported),
	then chat_completed with the full response, or chat_error on failure.
	"""
	var url = get_base_url()
	if url.is_empty():
		chat_error.emit("loreguard-client not running")
		return

	var body = JSON.stringify({
		"character_id": character_id,
		"message": message,
		"player_handle": player_handle,
		"current_context": current_context
	})

	var headers = [
		"Content-Type: application/json",
		"Accept: application/json"  # Non-streaming for simplicity
	]

	var error = _http_request.request(
		url + "/api/chat",
		headers,
		HTTPClient.METHOD_POST,
		body
	)

	if error != OK:
		chat_error.emit("Failed to send request: %s" % error)


func _on_request_completed(
	result: int,
	response_code: int,
	headers: PackedStringArray,
	body: PackedByteArray
) -> void:
	if result != HTTPRequest.RESULT_SUCCESS:
		chat_error.emit("Request failed with result: %d" % result)
		return

	if response_code != 200:
		chat_error.emit("Server returned error: %d" % response_code)
		return

	var json = JSON.new()
	var error = json.parse(body.get_string_from_utf8())
	if error != OK:
		chat_error.emit("Failed to parse response")
		return

	chat_completed.emit(json.data)


## Streaming chat using HTTPClient (more complex but supports SSE)
func chat_streaming(
	character_id: String,
	message: String,
	player_handle: String = "",
	current_context: String = ""
) -> void:
	"""Chat with streaming tokens.

	Uses HTTPClient for chunked reading to support SSE.
	Emits token_received for each token, then chat_completed.
	"""
	var port = get_local_port()
	if port == -1:
		chat_error.emit("loreguard-client not running")
		return

	# Run in a thread to avoid blocking
	var thread = Thread.new()
	thread.start(_streaming_request.bind(port, character_id, message, player_handle, current_context))


func _streaming_request(
	port: int,
	character_id: String,
	message: String,
	player_handle: String,
	current_context: String
) -> void:
	var http = HTTPClient.new()
	var err = http.connect_to_host("127.0.0.1", port)
	if err != OK:
		call_deferred("emit_signal", "chat_error", "Failed to connect")
		return

	# Wait for connection
	while http.get_status() == HTTPClient.STATUS_CONNECTING or \
		  http.get_status() == HTTPClient.STATUS_RESOLVING:
		http.poll()
		OS.delay_msec(100)

	if http.get_status() != HTTPClient.STATUS_CONNECTED:
		call_deferred("emit_signal", "chat_error", "Connection failed")
		return

	var body = JSON.stringify({
		"character_id": character_id,
		"message": message,
		"player_handle": player_handle,
		"current_context": current_context
	})

	var headers = [
		"Content-Type: application/json",
		"Accept: text/event-stream"
	]

	err = http.request(HTTPClient.METHOD_POST, "/api/chat", headers, body)
	if err != OK:
		call_deferred("emit_signal", "chat_error", "Request failed")
		return

	# Wait for response
	while http.get_status() == HTTPClient.STATUS_REQUESTING:
		http.poll()
		OS.delay_msec(100)

	if not http.has_response():
		call_deferred("emit_signal", "chat_error", "No response")
		return

	# Read response body in chunks
	var buffer = ""
	while http.get_status() == HTTPClient.STATUS_BODY:
		http.poll()
		var chunk = http.read_response_body_chunk()
		if chunk.size() > 0:
			buffer += chunk.get_string_from_utf8()

			# Parse SSE events
			var lines = buffer.split("\n")
			buffer = lines[-1]  # Keep incomplete line

			for i in range(lines.size() - 1):
				var line = lines[i]
				if line.begins_with("data: "):
					var json_str = line.substr(6)
					var json = JSON.new()
					if json.parse(json_str) == OK:
						var data = json.data
						if data.has("t"):
							call_deferred("emit_signal", "token_received", data["t"])
						elif data.has("speech"):
							call_deferred("emit_signal", "chat_completed", data)
						elif data.has("error"):
							call_deferred("emit_signal", "chat_error", data["error"])

		OS.delay_msec(10)
