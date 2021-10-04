extends CanvasLayer

func _ready():
	print("MainScene activated")
	GameManager.myGameId=-1
	$JoinGameButton.disabled=true
	if GameManager.playerId>=0:
		$NameInputField.set_text(GameManager.playerName)
		$NameInputField.editable=true
		$JoinGameButton.disabled=false
	$MakeReadyButton.disabled=true

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
	GameManager.playerName=new_text
	print(msg)
	print($NewPlayerHTTPRequest.request(GameManager.url+"/newplayer/", GameManager.headers, false, HTTPClient.METHOD_POST, msg))
	$NameInputField.set_editable(false)
	$JoinGameButton.disabled=false

func _on_JoinGameHTTPRequest_request_completed(result, response_code, headers, body):
	var json = JSON.parse(body.get_string_from_utf8())
	print("JoinGameRequest return ", json.result)
	GameManager.myGameId=int(json.result)
	$MakeReadyButton.disabled=false

func _on_MakeReadyButton_pressed():
	print("Pressed make ready")
	var msg = JSON.print({"playerid":GameManager.playerId})
	print(msg)
	print($MakeReadyHTTPRequest.request(GameManager.url+"/makeready/", GameManager.headers, false, HTTPClient.METHOD_PUT, msg))

func _on_MakeReadyHTTPRequest_request_completed(result, response_code, headers, body):
	var json = JSON.parse(body.get_string_from_utf8())
	print("MakeReadyRequest return", json.result)

func _on_TickTimer_timeout():
	$PlayerListHTTPRequest.request(GameManager.url+"/playerstats/", GameManager.headers, false, HTTPClient.METHOD_POST, "")
	if GameManager.myGameId>=0:
		var msg = JSON.print({"gameid":GameManager.myGameId})
		print(msg)
		print("Calling getgamestate with ",msg)
		print("GameState result ",$GameStateHTTPRequest.request(GameManager.url+"/getgamestate/", GameManager.headers, false, HTTPClient.METHOD_POST, msg))

func _on_GameStateHTTPRequest_request_completed(result, response_code, headers, body):
	var json = JSON.parse(body.get_string_from_utf8())
	print("GameStateHTTPRequest completed", json.result)
	if "active" in json.result and json.result["active"]:
		print("Activating playing table.")
		get_tree().change_scene("res://PlayingTable.tscn")

func _on_PlayerListHTTPRequest_request_completed(result, response_code, headers, body):
	var json = JSON.parse(body.get_string_from_utf8())
	$PlayerList.clear()
	for pId in json.result.keys():
		var p=json.result[pId]
		var name = p[0]
		var inPending = p[1]
		var isReady = p[2]
		var activeGames = p[3]
		if isReady==1:
			name = ' * '+name
		elif inPending==1:
			name = ' + '+name
		else:
			name = '   '+name
		name = name + " " + str(activeGames)
		$PlayerList.add_item(name)
			
