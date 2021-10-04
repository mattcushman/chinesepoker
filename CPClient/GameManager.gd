extends Node

var url = "http://172.104.211.116:5051"
#var url = "http://localhost:105/"

var headers = ["Content-Type: application/json"]
 
var suiteNameOld = ["Diamonds", "Clubs", "Hearts", "Spades"]
var suiteName = ["D", "C", "H", "S"]
var valueName = ["4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "3"]

var playerId=-1
var myGameId=-1
var playerName=""

  
func _ready():
	if OS.has_environment("CP_SERVER"):
		url = OS.get_environment("CP_SERVER")

