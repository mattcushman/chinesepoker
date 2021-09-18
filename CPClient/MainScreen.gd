extends CanvasLayer

var url = "http://localhost:105/"
var headers = ["Content-Type: application/json"]
var playerId=-1
var ready=false
var myGameId=-1

func _ready():
	$NewPlayerHTTPRequest.connect("request_completed", self, "_on_request_completed")
	$JoinGameHTTPRequest.connect("request_completed", self, "_on_request_completed")
	$MakeReadyHTTPRequest.connect("request_completed", self, "_on_request_completed")

func _on_TextureButton_pressed():
	print("Pressed join game")
	var msg = JSON.print({"playerid":playerId})
	print(msg)
	var result=$JoinGameHTTPRequest.request(url+"/joinnextgame/", headers, false, HTTPClient.METHOD_PUT, msg)
	print(result)

func _on_NewPlayerHTTPRequest_request_completed(result, response_code, headers, body):
	var json = JSON.parse(body.get_string_from_utf8())
	print(json.result)
	playerId=int(json.result)

func _on_NameInputField_text_entered(new_text):
	var msg = JSON.print({"name":new_text})
	print(msg)
	print($NewPlayerHTTPRequest.request(url+"/newplayer/", headers, false, HTTPClient.METHOD_POST, msg))

func _on_JoinGameHTTPRequest_request_completed(result, response_code, headers, body):
	var json = JSON.parse(body.get_string_from_utf8())
	print("JoinGameRequest return ", json.result)
	myGameId=int(json.result)

func _on_MakeReadyButton_pressed():
	print("Pressed make ready")
	ready=true
	var msg = JSON.print({"playerid":playerId})
	print(msg)
	print($MakeReadyHTTPRequest.request(url+"/makeready/", headers, false, HTTPClient.METHOD_PUT, msg))


func _on_MakeReadyHTTPRequest_request_completed(result, response_code, headers, body):
	var json = JSON.parse(body.get_string_from_utf8())
	print("MakeReadyRequest return", json.result)

func _on_TickTimer_timeout():
	if myGameId>=0:
		var msg = JSON.print({"gameid":myGameId})
		print(msg)
		print($MakeReadyHTTPRequest.request(url+"/getgamestate/", headers, false, HTTPClient.METHOD_GET, msg))
	
