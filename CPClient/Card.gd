extends TextureButton

class_name Card

var suite
var value 
var face 
var back
var state
var grayedFace
var move_x=0
var move_y=-50


func _ready():
	pass

func _init(var s, var v):
	value=v
	suite=s
	face=load("res://assets/cards2colour/"+GameManager.suiteName[suite]+GameManager.valueName[value]+"L.png")
	state=0
	set_normal_texture(face)

func toggle_gray():
	if state==0:
		state=1
		self.rect_position += Vector2(move_x, move_y)
	elif state==1:
		state=0
		self.rect_position -= Vector2(move_x, move_y)		
	else:
		return "Error"

func _on_Timer_timeout():
	pass # Replace with function body.
	
func get_state():
	return state

func rank():
	return 4*value+suite
