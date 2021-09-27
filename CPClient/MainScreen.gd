extends CanvasLayer

func _ready():
	$NewPlayerHTTPRequest.connect("request_completed", self, "_on_request_completed")
	$JoinGameHTTPRequest.connect("request_completed", self, "_on_request_completed")
	$MakeReadyHTTPRequest.connect("request_completed", self, "_on_request_completed")
	$GameStateHTTPRequest.connect("request_completed", self, "_on_request_completed")

func _on_TextureButton_pressed():
	print("Pressed join game")
	var msg = JSON.print({"playerid":GameManager.playerId})
	print(msg)
	var result=$JoinGameHTTPRequest.request(GameManager.url+"/joinnextgame/", GameManager.headers, false, HTTPClient.METHOD_PUT, msg)
	print(result)

func _on_NewPlayerHTTPRequest_request_completed(result, response_code, headers, body):
	var json = JSON.parse(body.get_string_from_utf8())
	print(json.result)
	GameManager.playerId=int(json.result)

func _on_NameInputField_text_entered(new_text):
	var msg = JSON.print({"name":new_text})
	print(msg)
	print($NewPlayerHTTPRequest.request(GameManager.url+"/newplayer/", GameManager.headers, false, HTTPClient.METHOD_POST, msg))

func _on_JoinGameHTTPRequest_request_completed(result, response_code, headers, body):
	var json = JSON.parse(body.get_string_from_utf8())
	print("JoinGameRequest return ", json.result)
	GameManager.myGameId=int(json.result)

func _on_MakeReadyButton_pressed():
	print("Pressed make ready")
	var msg = JSON.print({"playerid":GameManager.playerId})
	print(msg)
	print($MakeReadyHTTPRequest.request(GameManager.url+"/makeready/", GameManager.headers, false, HTTPClient.METHOD_PUT, msg))

func _on_MakeReadyHTTPRequest_request_completed(result, response_code, headers, body):
	var json = JSON.parse(body.get_string_from_utf8())
	print("MakeReadyRequest return", json.result)

func _on_TickTimer_timeout():
	if GameManager.myGameId>=0:
		var msg = JSON.print({"gameid":GameManager.myGameId})
		print("Calling getgamestate with ",msg)
		print("GameState result ",$GameStateHTTPRequest.request(GameManager.url+"/getgamestate/", GameManager.headers, false, HTTPClient.METHOD_GET, msg))

func _on_GameStateHTTPRequest_request_completed(result, response_code, headers, body):
	var json = JSON.parse(body.get_string_from_utf8())
	print("GameStateHTTPRequest completed", json.result)
	if json.result["active"]:
		print("Activating playing table.")
		get_tree().change_scene("res://PlayingTable.tscn")
	else:
		$PlayeList.clear()
		for p in json.result['player_names']:
			var name = json.result['player_names'][p]
			if p in json.result['player_ready'] and json.result['player_ready'][p]:
				name = ' * '+name
			else:
				name = '   '+name
			$PlayeList.add_item(name)
				
