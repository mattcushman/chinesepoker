extends CanvasLayer

var first=true
var activeCards=[]
var cardsToPlay=[]
var toMove=-1
var playerPositions=[]


func _ready():
	print("In ready of PlayingTable")

	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
#func _process(delta):
#	pass


func _on_Timer_timeout():
	var msg = JSON.print({"gameid":GameManager.myGameId})
	print("Calling getgamestate with ",msg)
	print("GameState result ",$GameStateHTTPRequest.request(GameManager.url+"/getgamestate/", GameManager.headers, false, HTTPClient.METHOD_POST, msg))


func _on_GameStateHTTPRequest_request_completed(result, response_code, headers, body):
	var json = JSON.parse(body.get_string_from_utf8())
	print("GameStateHTTPRequest completed", json.result)
	if first:
		first=false
		var players=[]
		for p in json.result["players"]:
			players.append(int(p))
		var myPosition=players.find(GameManager.playerId)
		print("my position: ",myPosition, " players: ",players)
		if myPosition>0:
			playerPositions = (players.slice(myPosition, players.size()-1,1,true) + players.slice(0, myPosition-1,1,true))
		else:
			playerPositions = players		
		print("playerPositions= ",playerPositions)
		for cardFloat in json.result['hands'][str(GameManager.playerId)]:
			var c=int(cardFloat)
			var card=Card.new(c%4,c/4)
			$PlayerCards.add_child(card)
			card.connect("button_up", self, "card_clicked", [card])
			activeCards.append(card)
	toMove=int(json.result['to_move'])
	var playerHands=[$PlayerCards1, $PlayerCards2, $PlayerCards3, $PlayerCards4]
	var playerLabels=[$Label1, $Label2, $Label3, $Label4]
	for playerPos in range(playerPositions.size()):
		var playerId=playerPositions[playerPos]
		for n in playerHands[playerPos].get_children():
			playerHands[playerPos].remove_child(n)
		name = json.result['player_names'][str(playerId)]
		var cardsRemaining=str(json.result['hands'][str(playerPositions[playerPos])].size())
		var labelNameBBCode = "[color=white]"+name+"[/color] ("+cardsRemaining+")"
		if toMove==playerId:
			labelNameBBCode = "[color=red]"+name+"[/color] ("+cardsRemaining+")"
		playerLabels[playerPos].bbcode_text=labelNameBBCode
		for cardRank in json.result['last_move'][str(playerPositions[playerPos])]:
			print("adding card"+str(cardRank))
			playerHands[playerPos].add_child(Card.new(int(cardRank)%4, int(cardRank)/4))

	if toMove==GameManager.playerId:
		$YourMoveNotifier.show()
		$Button.disabled=false
	else:
		$YourMoveNotifier.hide()
		$Button.disabled=true
	if json.result['winner']>=0:
		var gameOverMsg="Fatality.  You have been pwned by "+json.result['player_names'][str(json.result['winner'])]
		if json.result['winner']==GameManager.playerId:
			gameOverMsg="You have pwned the table"
		print("Game over: "+gameOverMsg)
		$GameOverMessage.set_text(gameOverMsg)
		$GameOverMessage.popup()

		
func card_clicked(card):
	card.toggle_gray()

func _on_Button_pressed():
	print("Playing hand")
	var ranksToPlay=[]
	for card in activeCards:
		if card.get_state()==1:
			cardsToPlay.append(card)
			ranksToPlay.append(card.rank())
	var msg=JSON.print({"gameid":GameManager.myGameId, "move":ranksToPlay, 'playerid':GameManager.playerId, 'token':GameManager.token})
	print(msg)
	$PlayHandHTTPRequest.request(GameManager.url+"/implementmove/", GameManager.headers, false, HTTPClient.METHOD_PUT, msg)


func _on_PlayHandHTTPRequest_request_completed(result, response_code, headers, body):
	var json = JSON.parse(body.get_string_from_utf8())
	print("PlayHandHTTPRequest completed ", json.result)
	if json.result:
		print("Removing cards from list")
		for c in cardsToPlay:
			$PlayerCards.remove_child(c)
			activeCards.erase(c)
	cardsToPlay=[]

	
	
	


func _on_GameOverMessage_confirmed():
	get_tree().change_scene("res://MainScreen.tscn")	
