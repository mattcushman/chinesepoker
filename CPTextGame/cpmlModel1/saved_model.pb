??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02unknown8??
?
hand_layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namehand_layer1/kernel
}
&hand_layer1/kernel/Read/ReadVariableOpReadVariableOphand_layer1/kernel*"
_output_shapes
:*
dtype0
x
hand_layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namehand_layer1/bias
q
$hand_layer1/bias/Read/ReadVariableOpReadVariableOphand_layer1/bias*
_output_shapes
:*
dtype0
~
conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_3/kernel
w
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*"
_output_shapes
:*
dtype0
r
conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_3/bias
k
!conv1d_3/bias/Read/ReadVariableOpReadVariableOpconv1d_3/bias*
_output_shapes
:*
dtype0
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:*
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
:*
dtype0
?
hand_layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*#
shared_namehand_layer2/kernel
y
&hand_layer2/kernel/Read/ReadVariableOpReadVariableOphand_layer2/kernel*
_output_shapes

:@*
dtype0
x
hand_layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_namehand_layer2/bias
q
$hand_layer2/bias/Read/ReadVariableOpReadVariableOphand_layer2/bias*
_output_shapes
:@*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@ *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
z
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
??*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:?*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	?@*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:@*
dtype0
z
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_7/kernel
s
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel* 
_output_shapes
:
??*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:?*
dtype0
z
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_8/kernel
s
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel* 
_output_shapes
:
??*
dtype0
q
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_8/bias
j
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes	
:?*
dtype0
y
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	?*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?>
value?>B?> B?>
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer-13
layer_with_weights-7
layer-14
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
 
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
h

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
h

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
R
1	variables
2trainable_variables
3regularization_losses
4	keras_api
R
5	variables
6trainable_variables
7regularization_losses
8	keras_api
h

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
h

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
h

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
R
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
R
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
h

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
R
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
h

]kernel
^bias
_	variables
`trainable_variables
aregularization_losses
b	keras_api
h

ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
 
?
0
1
2
 3
%4
&5
+6
,7
98
:9
?10
@11
E12
F13
S14
T15
]16
^17
c18
d19
?
0
1
2
 3
%4
&5
+6
,7
98
:9
?10
@11
E12
F13
S14
T15
]16
^17
c18
d19
 
?
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
 
^\
VARIABLE_VALUEhand_layer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEhand_layer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv1d_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
?
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
!	variables
"trainable_variables
#regularization_losses
[Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1

%0
&1
 
?
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
'	variables
(trainable_variables
)regularization_losses
^\
VARIABLE_VALUEhand_layer2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEhand_layer2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

+0
,1
 
?
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
1	variables
2trainable_variables
3regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1

90
:1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
;	variables
<trainable_variables
=regularization_losses
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

?0
@1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

E0
F1

E0
F1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1

S0
T1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

]0
^1

]0
^1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
_	variables
`trainable_variables
aregularization_losses
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

c0
d1

c0
d1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
e	variables
ftrainable_variables
gregularization_losses
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17

?0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables

serving_default_handPlaceholder*+
_output_shapes
:?????????4*
dtype0* 
shape:?????????4
?
serving_default_historyPlaceholder*/
_output_shapes
:?????????@4*
dtype0*$
shape:?????????@4

serving_default_movePlaceholder*+
_output_shapes
:?????????4*
dtype0* 
shape:?????????4
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_handserving_default_historyserving_default_movehand_layer1/kernelhand_layer1/biashand_layer2/kernelhand_layer2/biasconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasconv2d_1/kernelconv2d_1/biasdense_6/kerneldense_6/biasdense_5/kerneldense_5/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *1
f,R*
(__inference_signature_wrapper_1879666903
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&hand_layer1/kernel/Read/ReadVariableOp$hand_layer1/bias/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp!conv1d_3/bias/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp&hand_layer2/kernel/Read/ReadVariableOp$hand_layer2/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_save_1879667708
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehand_layer1/kernelhand_layer1/biasconv1d_3/kernelconv1d_3/biasconv1d_2/kernelconv1d_2/biashand_layer2/kernelhand_layer2/biasconv2d_1/kernelconv2d_1/biasdense_6/kerneldense_6/biasdense_5/kerneldense_5/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biastotalcount*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference__traced_restore_1879667784ݙ
?

?
G__inference_dense_7_layer_call_and_return_conditional_losses_1879666291

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
G__inference_dense_5_layer_call_and_return_conditional_losses_1879667505

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
G__inference_dense_9_layer_call_and_return_conditional_losses_1879666331

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
G__inference_dense_8_layer_call_and_return_conditional_losses_1879667598

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
K__inference_hand_layer2_layer_call_and_return_conditional_losses_1879666145

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:}
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:?????????@?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????@@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?F
?	
G__inference_model_1_layer_call_and_return_conditional_losses_1879666793
move
hand
history,
hand_layer1_1879666737:$
hand_layer1_1879666739:(
hand_layer2_1879666742:@$
hand_layer2_1879666744:@)
conv1d_2_1879666747:!
conv1d_2_1879666749:)
conv1d_3_1879666752:!
conv1d_3_1879666754:-
conv2d_1_1879666757:@ !
conv2d_1_1879666759: &
dense_6_1879666764:
??!
dense_6_1879666766:	?%
dense_5_1879666769:	?@ 
dense_5_1879666771:@&
dense_7_1879666776:
??!
dense_7_1879666778:	?&
dense_8_1879666782:
??!
dense_8_1879666784:	?%
dense_9_1879666787:	? 
dense_9_1879666789:
identity?? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?#hand_layer1/StatefulPartitionedCall?#hand_layer2/StatefulPartitionedCall?
#hand_layer1/StatefulPartitionedCallStatefulPartitionedCallhistoryhand_layer1_1879666737hand_layer1_1879666739*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_hand_layer1_layer_call_and_return_conditional_losses_1879666108?
#hand_layer2/StatefulPartitionedCallStatefulPartitionedCall,hand_layer1/StatefulPartitionedCall:output:0hand_layer2_1879666742hand_layer2_1879666744*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_hand_layer2_layer_call_and_return_conditional_losses_1879666145?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallhandconv1d_2_1879666747conv1d_2_1879666749*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv1d_2_layer_call_and_return_conditional_losses_1879666167?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCallmoveconv1d_3_1879666752conv1d_3_1879666754*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv1d_3_layer_call_and_return_conditional_losses_1879666189?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall,hand_layer2/StatefulPartitionedCall:output:0conv2d_1_1879666757conv2d_1_1879666759*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_1_layer_call_and_return_conditional_losses_1879666206?
flatten_4/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_flatten_4_layer_call_and_return_conditional_losses_1879666218?
flatten_5/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_flatten_5_layer_call_and_return_conditional_losses_1879666226?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_6_1879666764dense_6_1879666766*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_6_layer_call_and_return_conditional_losses_1879666239?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_5_1879666769dense_5_1879666771*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_5_layer_call_and_return_conditional_losses_1879666256?
flatten_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_flatten_3_layer_call_and_return_conditional_losses_1879666268?
concatenate_1/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0(dense_5/StatefulPartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_concatenate_1_layer_call_and_return_conditional_losses_1879666278?
dense_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_7_1879666776dense_7_1879666778*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_7_layer_call_and_return_conditional_losses_1879666291?
dropout_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_dropout_1_layer_call_and_return_conditional_losses_1879666302?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_8_1879666782dense_8_1879666784*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_8_layer_call_and_return_conditional_losses_1879666315?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_1879666787dense_9_1879666789*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_9_layer_call_and_return_conditional_losses_1879666331w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall$^hand_layer1/StatefulPartitionedCall$^hand_layer2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????4:?????????4:?????????@4: : : : : : : : : : : : : : : : : : : : 2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2J
#hand_layer1/StatefulPartitionedCall#hand_layer1/StatefulPartitionedCall2J
#hand_layer2/StatefulPartitionedCall#hand_layer2/StatefulPartitionedCall:Q M
+
_output_shapes
:?????????4

_user_specified_namemove:QM
+
_output_shapes
:?????????4

_user_specified_namehand:XT
/
_output_shapes
:?????????@4
!
_user_specified_name	history
?
e
I__inference_flatten_4_layer_call_and_return_conditional_losses_1879667445

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
h
I__inference_dropout_1_layer_call_and_return_conditional_losses_1879666421

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_conv1d_2_layer_call_fn_1879667367

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv1d_2_layer_call_and_return_conditional_losses_1879666167s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????4: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????4
 
_user_specified_nameinputs
?
?
M__inference_concatenate_1_layer_call_and_return_conditional_losses_1879667531
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:??????????:?????????@:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?

?
G__inference_dense_6_layer_call_and_return_conditional_losses_1879666239

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_7_layer_call_fn_1879667540

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_7_layer_call_and_return_conditional_losses_1879666291p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
I__inference_dropout_1_layer_call_and_return_conditional_losses_1879666302

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
M__inference_concatenate_1_layer_call_and_return_conditional_losses_1879666278

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:??????????:?????????@:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_model_1_layer_call_fn_1879666950
inputs_0
inputs_1
inputs_2
unknown:
	unknown_0:
	unknown_1:@
	unknown_2:@
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:@ 
	unknown_8: 
	unknown_9:
??

unknown_10:	?

unknown_11:	?@

unknown_12:@

unknown_13:
??

unknown_14:	?

unknown_15:
??

unknown_16:	?

unknown_17:	?

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_model_1_layer_call_and_return_conditional_losses_1879666338o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????4:?????????4:?????????@4: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????4
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????4
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:?????????@4
"
_user_specified_name
inputs/2
?2
?
#__inference__traced_save_1879667708
file_prefix1
-savev2_hand_layer1_kernel_read_readvariableop/
+savev2_hand_layer1_bias_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop1
-savev2_hand_layer2_kernel_read_readvariableop/
+savev2_hand_layer2_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_hand_layer1_kernel_read_readvariableop+savev2_hand_layer1_bias_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop-savev2_hand_layer2_kernel_read_readvariableop+savev2_hand_layer2_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::::@:@:@ : :
??:?:	?@:@:
??:?:
??:?:	?:: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
:@:,	(
&
_output_shapes
:@ : 


_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_dense_5_layer_call_fn_1879667494

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_5_layer_call_and_return_conditional_losses_1879666256o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_flatten_3_layer_call_and_return_conditional_losses_1879667516

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
e
I__inference_flatten_4_layer_call_and_return_conditional_losses_1879666218

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?G
?

G__inference_model_1_layer_call_and_return_conditional_losses_1879666854
move
hand
history,
hand_layer1_1879666798:$
hand_layer1_1879666800:(
hand_layer2_1879666803:@$
hand_layer2_1879666805:@)
conv1d_2_1879666808:!
conv1d_2_1879666810:)
conv1d_3_1879666813:!
conv1d_3_1879666815:-
conv2d_1_1879666818:@ !
conv2d_1_1879666820: &
dense_6_1879666825:
??!
dense_6_1879666827:	?%
dense_5_1879666830:	?@ 
dense_5_1879666832:@&
dense_7_1879666837:
??!
dense_7_1879666839:	?&
dense_8_1879666843:
??!
dense_8_1879666845:	?%
dense_9_1879666848:	? 
dense_9_1879666850:
identity?? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?#hand_layer1/StatefulPartitionedCall?#hand_layer2/StatefulPartitionedCall?
#hand_layer1/StatefulPartitionedCallStatefulPartitionedCallhistoryhand_layer1_1879666798hand_layer1_1879666800*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_hand_layer1_layer_call_and_return_conditional_losses_1879666108?
#hand_layer2/StatefulPartitionedCallStatefulPartitionedCall,hand_layer1/StatefulPartitionedCall:output:0hand_layer2_1879666803hand_layer2_1879666805*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_hand_layer2_layer_call_and_return_conditional_losses_1879666145?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallhandconv1d_2_1879666808conv1d_2_1879666810*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv1d_2_layer_call_and_return_conditional_losses_1879666167?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCallmoveconv1d_3_1879666813conv1d_3_1879666815*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv1d_3_layer_call_and_return_conditional_losses_1879666189?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall,hand_layer2/StatefulPartitionedCall:output:0conv2d_1_1879666818conv2d_1_1879666820*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_1_layer_call_and_return_conditional_losses_1879666206?
flatten_4/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_flatten_4_layer_call_and_return_conditional_losses_1879666218?
flatten_5/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_flatten_5_layer_call_and_return_conditional_losses_1879666226?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_6_1879666825dense_6_1879666827*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_6_layer_call_and_return_conditional_losses_1879666239?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_5_1879666830dense_5_1879666832*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_5_layer_call_and_return_conditional_losses_1879666256?
flatten_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_flatten_3_layer_call_and_return_conditional_losses_1879666268?
concatenate_1/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0(dense_5/StatefulPartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_concatenate_1_layer_call_and_return_conditional_losses_1879666278?
dense_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_7_1879666837dense_7_1879666839*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_7_layer_call_and_return_conditional_losses_1879666291?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_dropout_1_layer_call_and_return_conditional_losses_1879666421?
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_8_1879666843dense_8_1879666845*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_8_layer_call_and_return_conditional_losses_1879666315?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_1879666848dense_9_1879666850*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_9_layer_call_and_return_conditional_losses_1879666331w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall$^hand_layer1/StatefulPartitionedCall$^hand_layer2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????4:?????????4:?????????@4: : : : : : : : : : : : : : : : : : : : 2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2J
#hand_layer1/StatefulPartitionedCall#hand_layer1/StatefulPartitionedCall2J
#hand_layer2/StatefulPartitionedCall#hand_layer2/StatefulPartitionedCall:Q M
+
_output_shapes
:?????????4

_user_specified_namemove:QM
+
_output_shapes
:?????????4

_user_specified_namehand:XT
/
_output_shapes
:?????????@4
!
_user_specified_name	history
?	
?
G__inference_dense_9_layer_call_and_return_conditional_losses_1879667617

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?*
?
K__inference_hand_layer1_layer_call_and_return_conditional_losses_1879666108

inputsA
+conv1d_expanddims_1_readvariableop_resource:@
2squeeze_batch_dims_biasadd_readvariableop_resource:
identity??"Conv1D/ExpandDims_1/ReadVariableOp?)squeeze_batch_dims/BiasAdd/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:?????????@4?
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:V
Conv1D/ShapeShapeConv1D/ExpandDims:output:0*
T0*
_output_shapes
:d
Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Conv1D/strided_sliceStridedSliceConv1D/Shape:output:0#Conv1D/strided_slice/stack:output:0%Conv1D/strided_slice/stack_1:output:0%Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   4      ?
Conv1D/ReshapeReshapeConv1D/ExpandDims:output:0Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????4?
Conv1D/Conv2DConv2DConv1D/Reshape:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
k
Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         ]
Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/concatConcatV2Conv1D/strided_slice:output:0Conv1D/concat/values_1:output:0Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Conv1D/Reshape_1ReshapeConv1D/Conv2D:output:0Conv1D/concat:output:0*
T0*3
_output_shapes!
:?????????@?
Conv1D/SqueezeSqueezeConv1D/Reshape_1:output:0*
T0*/
_output_shapes
:?????????@*
squeeze_dims

?????????_
squeeze_batch_dims/ShapeShapeConv1D/Squeeze:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      ?
squeeze_batch_dims/ReshapeReshapeConv1D/Squeeze:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:??????????
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????s
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????@m
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp#^Conv1D/ExpandDims_1/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@4: : 2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:?????????@4
 
_user_specified_nameinputs
?
l
2__inference_concatenate_1_layer_call_fn_1879667523
inputs_0
inputs_1
inputs_2
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_concatenate_1_layer_call_and_return_conditional_losses_1879666278a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:??????????:?????????@:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?
?
0__inference_hand_layer2_layer_call_fn_1879667392

inputs
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_hand_layer2_layer_call_and_return_conditional_losses_1879666145w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
J
.__inference_flatten_3_layer_call_fn_1879667510

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_flatten_3_layer_call_and_return_conditional_losses_1879666268a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
e
I__inference_flatten_5_layer_call_and_return_conditional_losses_1879666226

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?F
?	
G__inference_model_1_layer_call_and_return_conditional_losses_1879666338

inputs
inputs_1
inputs_2,
hand_layer1_1879666109:$
hand_layer1_1879666111:(
hand_layer2_1879666146:@$
hand_layer2_1879666148:@)
conv1d_2_1879666168:!
conv1d_2_1879666170:)
conv1d_3_1879666190:!
conv1d_3_1879666192:-
conv2d_1_1879666207:@ !
conv2d_1_1879666209: &
dense_6_1879666240:
??!
dense_6_1879666242:	?%
dense_5_1879666257:	?@ 
dense_5_1879666259:@&
dense_7_1879666292:
??!
dense_7_1879666294:	?&
dense_8_1879666316:
??!
dense_8_1879666318:	?%
dense_9_1879666332:	? 
dense_9_1879666334:
identity?? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?#hand_layer1/StatefulPartitionedCall?#hand_layer2/StatefulPartitionedCall?
#hand_layer1/StatefulPartitionedCallStatefulPartitionedCallinputs_2hand_layer1_1879666109hand_layer1_1879666111*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_hand_layer1_layer_call_and_return_conditional_losses_1879666108?
#hand_layer2/StatefulPartitionedCallStatefulPartitionedCall,hand_layer1/StatefulPartitionedCall:output:0hand_layer2_1879666146hand_layer2_1879666148*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_hand_layer2_layer_call_and_return_conditional_losses_1879666145?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv1d_2_1879666168conv1d_2_1879666170*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv1d_2_layer_call_and_return_conditional_losses_1879666167?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_3_1879666190conv1d_3_1879666192*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv1d_3_layer_call_and_return_conditional_losses_1879666189?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall,hand_layer2/StatefulPartitionedCall:output:0conv2d_1_1879666207conv2d_1_1879666209*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_1_layer_call_and_return_conditional_losses_1879666206?
flatten_4/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_flatten_4_layer_call_and_return_conditional_losses_1879666218?
flatten_5/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_flatten_5_layer_call_and_return_conditional_losses_1879666226?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_6_1879666240dense_6_1879666242*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_6_layer_call_and_return_conditional_losses_1879666239?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_5_1879666257dense_5_1879666259*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_5_layer_call_and_return_conditional_losses_1879666256?
flatten_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_flatten_3_layer_call_and_return_conditional_losses_1879666268?
concatenate_1/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0(dense_5/StatefulPartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_concatenate_1_layer_call_and_return_conditional_losses_1879666278?
dense_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_7_1879666292dense_7_1879666294*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_7_layer_call_and_return_conditional_losses_1879666291?
dropout_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_dropout_1_layer_call_and_return_conditional_losses_1879666302?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_8_1879666316dense_8_1879666318*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_8_layer_call_and_return_conditional_losses_1879666315?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_1879666332dense_9_1879666334*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_9_layer_call_and_return_conditional_losses_1879666331w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall$^hand_layer1/StatefulPartitionedCall$^hand_layer2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????4:?????????4:?????????@4: : : : : : : : : : : : : : : : : : : : 2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2J
#hand_layer1/StatefulPartitionedCall#hand_layer1/StatefulPartitionedCall2J
#hand_layer2/StatefulPartitionedCall#hand_layer2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????4
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????4
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????@4
 
_user_specified_nameinputs
?
?
,__inference_model_1_layer_call_fn_1879666381
move
hand
history
unknown:
	unknown_0:
	unknown_1:@
	unknown_2:@
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:@ 
	unknown_8: 
	unknown_9:
??

unknown_10:	?

unknown_11:	?@

unknown_12:@

unknown_13:
??

unknown_14:	?

unknown_15:
??

unknown_16:	?

unknown_17:	?

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmovehandhistoryunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_model_1_layer_call_and_return_conditional_losses_1879666338o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????4:?????????4:?????????@4: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
+
_output_shapes
:?????????4

_user_specified_namemove:QM
+
_output_shapes
:?????????4

_user_specified_namehand:XT
/
_output_shapes
:?????????@4
!
_user_specified_name	history
??
?
G__inference_model_1_layer_call_and_return_conditional_losses_1879667138
inputs_0
inputs_1
inputs_2M
7hand_layer1_conv1d_expanddims_1_readvariableop_resource:L
>hand_layer1_squeeze_batch_dims_biasadd_readvariableop_resource:?
-hand_layer2_tensordot_readvariableop_resource:@9
+hand_layer2_biasadd_readvariableop_resource:@J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_2_biasadd_readvariableop_resource:J
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_3_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:@ 6
(conv2d_1_biasadd_readvariableop_resource: :
&dense_6_matmul_readvariableop_resource:
??6
'dense_6_biasadd_readvariableop_resource:	?9
&dense_5_matmul_readvariableop_resource:	?@5
'dense_5_biasadd_readvariableop_resource:@:
&dense_7_matmul_readvariableop_resource:
??6
'dense_7_biasadd_readvariableop_resource:	?:
&dense_8_matmul_readvariableop_resource:
??6
'dense_8_biasadd_readvariableop_resource:	?9
&dense_9_matmul_readvariableop_resource:	?5
'dense_9_biasadd_readvariableop_resource:
identity??conv1d_2/BiasAdd/ReadVariableOp?+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp?conv1d_3/BiasAdd/ReadVariableOp?+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?.hand_layer1/Conv1D/ExpandDims_1/ReadVariableOp?5hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOp?"hand_layer2/BiasAdd/ReadVariableOp?$hand_layer2/Tensordot/ReadVariableOpl
!hand_layer1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
hand_layer1/Conv1D/ExpandDims
ExpandDimsinputs_2*hand_layer1/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:?????????@4?
.hand_layer1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp7hand_layer1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0e
#hand_layer1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
hand_layer1/Conv1D/ExpandDims_1
ExpandDims6hand_layer1/Conv1D/ExpandDims_1/ReadVariableOp:value:0,hand_layer1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n
hand_layer1/Conv1D/ShapeShape&hand_layer1/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:p
&hand_layer1/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(hand_layer1/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(hand_layer1/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 hand_layer1/Conv1D/strided_sliceStridedSlice!hand_layer1/Conv1D/Shape:output:0/hand_layer1/Conv1D/strided_slice/stack:output:01hand_layer1/Conv1D/strided_slice/stack_1:output:01hand_layer1/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 hand_layer1/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   4      ?
hand_layer1/Conv1D/ReshapeReshape&hand_layer1/Conv1D/ExpandDims:output:0)hand_layer1/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????4?
hand_layer1/Conv1D/Conv2DConv2D#hand_layer1/Conv1D/Reshape:output:0(hand_layer1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
w
"hand_layer1/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         i
hand_layer1/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
hand_layer1/Conv1D/concatConcatV2)hand_layer1/Conv1D/strided_slice:output:0+hand_layer1/Conv1D/concat/values_1:output:0'hand_layer1/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
hand_layer1/Conv1D/Reshape_1Reshape"hand_layer1/Conv1D/Conv2D:output:0"hand_layer1/Conv1D/concat:output:0*
T0*3
_output_shapes!
:?????????@?
hand_layer1/Conv1D/SqueezeSqueeze%hand_layer1/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:?????????@*
squeeze_dims

?????????w
$hand_layer1/squeeze_batch_dims/ShapeShape#hand_layer1/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:|
2hand_layer1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
4hand_layer1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????~
4hand_layer1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,hand_layer1/squeeze_batch_dims/strided_sliceStridedSlice-hand_layer1/squeeze_batch_dims/Shape:output:0;hand_layer1/squeeze_batch_dims/strided_slice/stack:output:0=hand_layer1/squeeze_batch_dims/strided_slice/stack_1:output:0=hand_layer1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
,hand_layer1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      ?
&hand_layer1/squeeze_batch_dims/ReshapeReshape#hand_layer1/Conv1D/Squeeze:output:05hand_layer1/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:??????????
5hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp>hand_layer1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
&hand_layer1/squeeze_batch_dims/BiasAddBiasAdd/hand_layer1/squeeze_batch_dims/Reshape:output:0=hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????
.hand_layer1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      u
*hand_layer1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%hand_layer1/squeeze_batch_dims/concatConcatV25hand_layer1/squeeze_batch_dims/strided_slice:output:07hand_layer1/squeeze_batch_dims/concat/values_1:output:03hand_layer1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
(hand_layer1/squeeze_batch_dims/Reshape_1Reshape/hand_layer1/squeeze_batch_dims/BiasAdd:output:0.hand_layer1/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????@?
hand_layer1/ReluRelu1hand_layer1/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????@?
$hand_layer2/Tensordot/ReadVariableOpReadVariableOp-hand_layer2_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0d
hand_layer2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:o
hand_layer2/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          i
hand_layer2/Tensordot/ShapeShapehand_layer1/Relu:activations:0*
T0*
_output_shapes
:e
#hand_layer2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
hand_layer2/Tensordot/GatherV2GatherV2$hand_layer2/Tensordot/Shape:output:0#hand_layer2/Tensordot/free:output:0,hand_layer2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
%hand_layer2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
 hand_layer2/Tensordot/GatherV2_1GatherV2$hand_layer2/Tensordot/Shape:output:0#hand_layer2/Tensordot/axes:output:0.hand_layer2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
hand_layer2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
hand_layer2/Tensordot/ProdProd'hand_layer2/Tensordot/GatherV2:output:0$hand_layer2/Tensordot/Const:output:0*
T0*
_output_shapes
: g
hand_layer2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
hand_layer2/Tensordot/Prod_1Prod)hand_layer2/Tensordot/GatherV2_1:output:0&hand_layer2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: c
!hand_layer2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
hand_layer2/Tensordot/concatConcatV2#hand_layer2/Tensordot/free:output:0#hand_layer2/Tensordot/axes:output:0*hand_layer2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
hand_layer2/Tensordot/stackPack#hand_layer2/Tensordot/Prod:output:0%hand_layer2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
hand_layer2/Tensordot/transpose	Transposehand_layer1/Relu:activations:0%hand_layer2/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????@?
hand_layer2/Tensordot/ReshapeReshape#hand_layer2/Tensordot/transpose:y:0$hand_layer2/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
hand_layer2/Tensordot/MatMulMatMul&hand_layer2/Tensordot/Reshape:output:0,hand_layer2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@g
hand_layer2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@e
#hand_layer2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
hand_layer2/Tensordot/concat_1ConcatV2'hand_layer2/Tensordot/GatherV2:output:0&hand_layer2/Tensordot/Const_2:output:0,hand_layer2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
hand_layer2/TensordotReshape&hand_layer2/Tensordot/MatMul:product:0'hand_layer2/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????@@?
"hand_layer2/BiasAdd/ReadVariableOpReadVariableOp+hand_layer2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
hand_layer2/BiasAddBiasAddhand_layer2/Tensordot:output:0*hand_layer2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@p
hand_layer2/ReluReluhand_layer2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@i
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d_2/Conv1D/ExpandDims
ExpandDimsinputs_1'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????4?
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????f
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????i
conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d_3/Conv1D/ExpandDims
ExpandDimsinputs_0'conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????4?
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_3/Conv1D/ExpandDims_1
ExpandDims3conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
conv1d_3/Conv1DConv2D#conv1d_3/Conv1D/ExpandDims:output:0%conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv1d_3/Conv1D/SqueezeSqueezeconv1d_3/Conv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1d_3/BiasAddBiasAdd conv1d_3/Conv1D/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????f
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:??????????
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
conv2d_1/Conv2DConv2Dhand_layer2/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? `
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
flatten_4/ReshapeReshapeconv1d_2/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????`
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
flatten_5/ReshapeReshapeconv1d_3/Relu:activations:0flatten_5/Const:output:0*
T0*(
_output_shapes
:???????????
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_6/MatMulMatMulflatten_5/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense_5/MatMulMatMulflatten_4/Reshape:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@`
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_3/ReshapeReshapeconv2d_1/Relu:activations:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_1/concatConcatV2dense_6/Relu:activations:0dense_5/Relu:activations:0flatten_3/Reshape:output:0"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_7/MatMulMatMulconcatenate_1/concat:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????m
dropout_1/IdentityIdentitydense_7/Relu:activations:0*
T0*(
_output_shapes
:???????????
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_8/MatMulMatMuldropout_1/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp/^hand_layer1/Conv1D/ExpandDims_1/ReadVariableOp6^hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOp#^hand_layer2/BiasAdd/ReadVariableOp%^hand_layer2/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????4:?????????4:?????????@4: : : : : : : : : : : : : : : : : : : : 2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2`
.hand_layer1/Conv1D/ExpandDims_1/ReadVariableOp.hand_layer1/Conv1D/ExpandDims_1/ReadVariableOp2n
5hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOp5hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOp2H
"hand_layer2/BiasAdd/ReadVariableOp"hand_layer2/BiasAdd/ReadVariableOp2L
$hand_layer2/Tensordot/ReadVariableOp$hand_layer2/Tensordot/ReadVariableOp:U Q
+
_output_shapes
:?????????4
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????4
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:?????????@4
"
_user_specified_name
inputs/2
?
J
.__inference_dropout_1_layer_call_fn_1879667556

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_dropout_1_layer_call_and_return_conditional_losses_1879666302a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?W
?
&__inference__traced_restore_1879667784
file_prefix9
#assignvariableop_hand_layer1_kernel:1
#assignvariableop_1_hand_layer1_bias:8
"assignvariableop_2_conv1d_3_kernel:.
 assignvariableop_3_conv1d_3_bias:8
"assignvariableop_4_conv1d_2_kernel:.
 assignvariableop_5_conv1d_2_bias:7
%assignvariableop_6_hand_layer2_kernel:@1
#assignvariableop_7_hand_layer2_bias:@<
"assignvariableop_8_conv2d_1_kernel:@ .
 assignvariableop_9_conv2d_1_bias: 6
"assignvariableop_10_dense_6_kernel:
??/
 assignvariableop_11_dense_6_bias:	?5
"assignvariableop_12_dense_5_kernel:	?@.
 assignvariableop_13_dense_5_bias:@6
"assignvariableop_14_dense_7_kernel:
??/
 assignvariableop_15_dense_7_bias:	?6
"assignvariableop_16_dense_8_kernel:
??/
 assignvariableop_17_dense_8_bias:	?5
"assignvariableop_18_dense_9_kernel:	?.
 assignvariableop_19_dense_9_bias:#
assignvariableop_20_total: #
assignvariableop_21_count: 
identity_23??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp#assignvariableop_hand_layer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp#assignvariableop_1_hand_layer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv1d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv1d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_hand_layer2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_hand_layer2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_6_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_5_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_5_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_7_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_7_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_8_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_8_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_9_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_9_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
,__inference_dense_9_layer_call_fn_1879667607

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_9_layer_call_and_return_conditional_losses_1879666331o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
%__inference__wrapped_model_1879666059
move
hand
historyU
?model_1_hand_layer1_conv1d_expanddims_1_readvariableop_resource:T
Fmodel_1_hand_layer1_squeeze_batch_dims_biasadd_readvariableop_resource:G
5model_1_hand_layer2_tensordot_readvariableop_resource:@A
3model_1_hand_layer2_biasadd_readvariableop_resource:@R
<model_1_conv1d_2_conv1d_expanddims_1_readvariableop_resource:>
0model_1_conv1d_2_biasadd_readvariableop_resource:R
<model_1_conv1d_3_conv1d_expanddims_1_readvariableop_resource:>
0model_1_conv1d_3_biasadd_readvariableop_resource:I
/model_1_conv2d_1_conv2d_readvariableop_resource:@ >
0model_1_conv2d_1_biasadd_readvariableop_resource: B
.model_1_dense_6_matmul_readvariableop_resource:
??>
/model_1_dense_6_biasadd_readvariableop_resource:	?A
.model_1_dense_5_matmul_readvariableop_resource:	?@=
/model_1_dense_5_biasadd_readvariableop_resource:@B
.model_1_dense_7_matmul_readvariableop_resource:
??>
/model_1_dense_7_biasadd_readvariableop_resource:	?B
.model_1_dense_8_matmul_readvariableop_resource:
??>
/model_1_dense_8_biasadd_readvariableop_resource:	?A
.model_1_dense_9_matmul_readvariableop_resource:	?=
/model_1_dense_9_biasadd_readvariableop_resource:
identity??'model_1/conv1d_2/BiasAdd/ReadVariableOp?3model_1/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp?'model_1/conv1d_3/BiasAdd/ReadVariableOp?3model_1/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp?'model_1/conv2d_1/BiasAdd/ReadVariableOp?&model_1/conv2d_1/Conv2D/ReadVariableOp?&model_1/dense_5/BiasAdd/ReadVariableOp?%model_1/dense_5/MatMul/ReadVariableOp?&model_1/dense_6/BiasAdd/ReadVariableOp?%model_1/dense_6/MatMul/ReadVariableOp?&model_1/dense_7/BiasAdd/ReadVariableOp?%model_1/dense_7/MatMul/ReadVariableOp?&model_1/dense_8/BiasAdd/ReadVariableOp?%model_1/dense_8/MatMul/ReadVariableOp?&model_1/dense_9/BiasAdd/ReadVariableOp?%model_1/dense_9/MatMul/ReadVariableOp?6model_1/hand_layer1/Conv1D/ExpandDims_1/ReadVariableOp?=model_1/hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOp?*model_1/hand_layer2/BiasAdd/ReadVariableOp?,model_1/hand_layer2/Tensordot/ReadVariableOpt
)model_1/hand_layer1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%model_1/hand_layer1/Conv1D/ExpandDims
ExpandDimshistory2model_1/hand_layer1/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:?????????@4?
6model_1/hand_layer1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp?model_1_hand_layer1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0m
+model_1/hand_layer1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
'model_1/hand_layer1/Conv1D/ExpandDims_1
ExpandDims>model_1/hand_layer1/Conv1D/ExpandDims_1/ReadVariableOp:value:04model_1/hand_layer1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:~
 model_1/hand_layer1/Conv1D/ShapeShape.model_1/hand_layer1/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:x
.model_1/hand_layer1/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
0model_1/hand_layer1/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????z
0model_1/hand_layer1/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(model_1/hand_layer1/Conv1D/strided_sliceStridedSlice)model_1/hand_layer1/Conv1D/Shape:output:07model_1/hand_layer1/Conv1D/strided_slice/stack:output:09model_1/hand_layer1/Conv1D/strided_slice/stack_1:output:09model_1/hand_layer1/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
(model_1/hand_layer1/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   4      ?
"model_1/hand_layer1/Conv1D/ReshapeReshape.model_1/hand_layer1/Conv1D/ExpandDims:output:01model_1/hand_layer1/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????4?
!model_1/hand_layer1/Conv1D/Conv2DConv2D+model_1/hand_layer1/Conv1D/Reshape:output:00model_1/hand_layer1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides

*model_1/hand_layer1/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         q
&model_1/hand_layer1/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
!model_1/hand_layer1/Conv1D/concatConcatV21model_1/hand_layer1/Conv1D/strided_slice:output:03model_1/hand_layer1/Conv1D/concat/values_1:output:0/model_1/hand_layer1/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
$model_1/hand_layer1/Conv1D/Reshape_1Reshape*model_1/hand_layer1/Conv1D/Conv2D:output:0*model_1/hand_layer1/Conv1D/concat:output:0*
T0*3
_output_shapes!
:?????????@?
"model_1/hand_layer1/Conv1D/SqueezeSqueeze-model_1/hand_layer1/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:?????????@*
squeeze_dims

??????????
,model_1/hand_layer1/squeeze_batch_dims/ShapeShape+model_1/hand_layer1/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:?
:model_1/hand_layer1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<model_1/hand_layer1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
<model_1/hand_layer1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4model_1/hand_layer1/squeeze_batch_dims/strided_sliceStridedSlice5model_1/hand_layer1/squeeze_batch_dims/Shape:output:0Cmodel_1/hand_layer1/squeeze_batch_dims/strided_slice/stack:output:0Emodel_1/hand_layer1/squeeze_batch_dims/strided_slice/stack_1:output:0Emodel_1/hand_layer1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
4model_1/hand_layer1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      ?
.model_1/hand_layer1/squeeze_batch_dims/ReshapeReshape+model_1/hand_layer1/Conv1D/Squeeze:output:0=model_1/hand_layer1/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:??????????
=model_1/hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpFmodel_1_hand_layer1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
.model_1/hand_layer1/squeeze_batch_dims/BiasAddBiasAdd7model_1/hand_layer1/squeeze_batch_dims/Reshape:output:0Emodel_1/hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????
6model_1/hand_layer1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      }
2model_1/hand_layer1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
-model_1/hand_layer1/squeeze_batch_dims/concatConcatV2=model_1/hand_layer1/squeeze_batch_dims/strided_slice:output:0?model_1/hand_layer1/squeeze_batch_dims/concat/values_1:output:0;model_1/hand_layer1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
0model_1/hand_layer1/squeeze_batch_dims/Reshape_1Reshape7model_1/hand_layer1/squeeze_batch_dims/BiasAdd:output:06model_1/hand_layer1/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????@?
model_1/hand_layer1/ReluRelu9model_1/hand_layer1/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????@?
,model_1/hand_layer2/Tensordot/ReadVariableOpReadVariableOp5model_1_hand_layer2_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0l
"model_1/hand_layer2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:w
"model_1/hand_layer2/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          y
#model_1/hand_layer2/Tensordot/ShapeShape&model_1/hand_layer1/Relu:activations:0*
T0*
_output_shapes
:m
+model_1/hand_layer2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&model_1/hand_layer2/Tensordot/GatherV2GatherV2,model_1/hand_layer2/Tensordot/Shape:output:0+model_1/hand_layer2/Tensordot/free:output:04model_1/hand_layer2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-model_1/hand_layer2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(model_1/hand_layer2/Tensordot/GatherV2_1GatherV2,model_1/hand_layer2/Tensordot/Shape:output:0+model_1/hand_layer2/Tensordot/axes:output:06model_1/hand_layer2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#model_1/hand_layer2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
"model_1/hand_layer2/Tensordot/ProdProd/model_1/hand_layer2/Tensordot/GatherV2:output:0,model_1/hand_layer2/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%model_1/hand_layer2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
$model_1/hand_layer2/Tensordot/Prod_1Prod1model_1/hand_layer2/Tensordot/GatherV2_1:output:0.model_1/hand_layer2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)model_1/hand_layer2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
$model_1/hand_layer2/Tensordot/concatConcatV2+model_1/hand_layer2/Tensordot/free:output:0+model_1/hand_layer2/Tensordot/axes:output:02model_1/hand_layer2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
#model_1/hand_layer2/Tensordot/stackPack+model_1/hand_layer2/Tensordot/Prod:output:0-model_1/hand_layer2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
'model_1/hand_layer2/Tensordot/transpose	Transpose&model_1/hand_layer1/Relu:activations:0-model_1/hand_layer2/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????@?
%model_1/hand_layer2/Tensordot/ReshapeReshape+model_1/hand_layer2/Tensordot/transpose:y:0,model_1/hand_layer2/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
$model_1/hand_layer2/Tensordot/MatMulMatMul.model_1/hand_layer2/Tensordot/Reshape:output:04model_1/hand_layer2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@o
%model_1/hand_layer2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@m
+model_1/hand_layer2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&model_1/hand_layer2/Tensordot/concat_1ConcatV2/model_1/hand_layer2/Tensordot/GatherV2:output:0.model_1/hand_layer2/Tensordot/Const_2:output:04model_1/hand_layer2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
model_1/hand_layer2/TensordotReshape.model_1/hand_layer2/Tensordot/MatMul:product:0/model_1/hand_layer2/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????@@?
*model_1/hand_layer2/BiasAdd/ReadVariableOpReadVariableOp3model_1_hand_layer2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model_1/hand_layer2/BiasAddBiasAdd&model_1/hand_layer2/Tensordot:output:02model_1/hand_layer2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@?
model_1/hand_layer2/ReluRelu$model_1/hand_layer2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@q
&model_1/conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"model_1/conv1d_2/Conv1D/ExpandDims
ExpandDimshand/model_1/conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????4?
3model_1/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp<model_1_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0j
(model_1/conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
$model_1/conv1d_2/Conv1D/ExpandDims_1
ExpandDims;model_1/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:01model_1/conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
model_1/conv1d_2/Conv1DConv2D+model_1/conv1d_2/Conv1D/ExpandDims:output:0-model_1/conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
model_1/conv1d_2/Conv1D/SqueezeSqueeze model_1/conv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
'model_1/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/conv1d_2/BiasAddBiasAdd(model_1/conv1d_2/Conv1D/Squeeze:output:0/model_1/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????v
model_1/conv1d_2/ReluRelu!model_1/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????q
&model_1/conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"model_1/conv1d_3/Conv1D/ExpandDims
ExpandDimsmove/model_1/conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????4?
3model_1/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp<model_1_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0j
(model_1/conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
$model_1/conv1d_3/Conv1D/ExpandDims_1
ExpandDims;model_1/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:01model_1/conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
model_1/conv1d_3/Conv1DConv2D+model_1/conv1d_3/Conv1D/ExpandDims:output:0-model_1/conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
model_1/conv1d_3/Conv1D/SqueezeSqueeze model_1/conv1d_3/Conv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
'model_1/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/conv1d_3/BiasAddBiasAdd(model_1/conv1d_3/Conv1D/Squeeze:output:0/model_1/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????v
model_1/conv1d_3/ReluRelu!model_1/conv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:??????????
&model_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
model_1/conv2d_1/Conv2DConv2D&model_1/hand_layer2/Relu:activations:0.model_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
'model_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model_1/conv2d_1/BiasAddBiasAdd model_1/conv2d_1/Conv2D:output:0/model_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? z
model_1/conv2d_1/ReluRelu!model_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? h
model_1/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
model_1/flatten_4/ReshapeReshape#model_1/conv1d_2/Relu:activations:0 model_1/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????h
model_1/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
model_1/flatten_5/ReshapeReshape#model_1/conv1d_3/Relu:activations:0 model_1/flatten_5/Const:output:0*
T0*(
_output_shapes
:???????????
%model_1/dense_6/MatMul/ReadVariableOpReadVariableOp.model_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
model_1/dense_6/MatMulMatMul"model_1/flatten_5/Reshape:output:0-model_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&model_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_1/dense_6/BiasAddBiasAdd model_1/dense_6/MatMul:product:0.model_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
model_1/dense_6/ReluRelu model_1/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
model_1/dense_5/MatMulMatMul"model_1/flatten_4/Reshape:output:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@p
model_1/dense_5/ReluRelu model_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@h
model_1/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
model_1/flatten_3/ReshapeReshape#model_1/conv2d_1/Relu:activations:0 model_1/flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????c
!model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model_1/concatenate_1/concatConcatV2"model_1/dense_6/Relu:activations:0"model_1/dense_5/Relu:activations:0"model_1/flatten_3/Reshape:output:0*model_1/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
%model_1/dense_7/MatMul/ReadVariableOpReadVariableOp.model_1_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
model_1/dense_7/MatMulMatMul%model_1/concatenate_1/concat:output:0-model_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&model_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_1/dense_7/BiasAddBiasAdd model_1/dense_7/MatMul:product:0.model_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
model_1/dense_7/ReluRelu model_1/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????}
model_1/dropout_1/IdentityIdentity"model_1/dense_7/Relu:activations:0*
T0*(
_output_shapes
:???????????
%model_1/dense_8/MatMul/ReadVariableOpReadVariableOp.model_1_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
model_1/dense_8/MatMulMatMul#model_1/dropout_1/Identity:output:0-model_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&model_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_1/dense_8/BiasAddBiasAdd model_1/dense_8/MatMul:product:0.model_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
model_1/dense_8/ReluRelu model_1/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
%model_1/dense_9/MatMul/ReadVariableOpReadVariableOp.model_1_dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model_1/dense_9/MatMulMatMul"model_1/dense_8/Relu:activations:0-model_1/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&model_1/dense_9/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/dense_9/BiasAddBiasAdd model_1/dense_9/MatMul:product:0.model_1/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????o
IdentityIdentity model_1/dense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^model_1/conv1d_2/BiasAdd/ReadVariableOp4^model_1/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp(^model_1/conv1d_3/BiasAdd/ReadVariableOp4^model_1/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp(^model_1/conv2d_1/BiasAdd/ReadVariableOp'^model_1/conv2d_1/Conv2D/ReadVariableOp'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp'^model_1/dense_6/BiasAdd/ReadVariableOp&^model_1/dense_6/MatMul/ReadVariableOp'^model_1/dense_7/BiasAdd/ReadVariableOp&^model_1/dense_7/MatMul/ReadVariableOp'^model_1/dense_8/BiasAdd/ReadVariableOp&^model_1/dense_8/MatMul/ReadVariableOp'^model_1/dense_9/BiasAdd/ReadVariableOp&^model_1/dense_9/MatMul/ReadVariableOp7^model_1/hand_layer1/Conv1D/ExpandDims_1/ReadVariableOp>^model_1/hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOp+^model_1/hand_layer2/BiasAdd/ReadVariableOp-^model_1/hand_layer2/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????4:?????????4:?????????@4: : : : : : : : : : : : : : : : : : : : 2R
'model_1/conv1d_2/BiasAdd/ReadVariableOp'model_1/conv1d_2/BiasAdd/ReadVariableOp2j
3model_1/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp3model_1/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2R
'model_1/conv1d_3/BiasAdd/ReadVariableOp'model_1/conv1d_3/BiasAdd/ReadVariableOp2j
3model_1/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp3model_1/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2R
'model_1/conv2d_1/BiasAdd/ReadVariableOp'model_1/conv2d_1/BiasAdd/ReadVariableOp2P
&model_1/conv2d_1/Conv2D/ReadVariableOp&model_1/conv2d_1/Conv2D/ReadVariableOp2P
&model_1/dense_5/BiasAdd/ReadVariableOp&model_1/dense_5/BiasAdd/ReadVariableOp2N
%model_1/dense_5/MatMul/ReadVariableOp%model_1/dense_5/MatMul/ReadVariableOp2P
&model_1/dense_6/BiasAdd/ReadVariableOp&model_1/dense_6/BiasAdd/ReadVariableOp2N
%model_1/dense_6/MatMul/ReadVariableOp%model_1/dense_6/MatMul/ReadVariableOp2P
&model_1/dense_7/BiasAdd/ReadVariableOp&model_1/dense_7/BiasAdd/ReadVariableOp2N
%model_1/dense_7/MatMul/ReadVariableOp%model_1/dense_7/MatMul/ReadVariableOp2P
&model_1/dense_8/BiasAdd/ReadVariableOp&model_1/dense_8/BiasAdd/ReadVariableOp2N
%model_1/dense_8/MatMul/ReadVariableOp%model_1/dense_8/MatMul/ReadVariableOp2P
&model_1/dense_9/BiasAdd/ReadVariableOp&model_1/dense_9/BiasAdd/ReadVariableOp2N
%model_1/dense_9/MatMul/ReadVariableOp%model_1/dense_9/MatMul/ReadVariableOp2p
6model_1/hand_layer1/Conv1D/ExpandDims_1/ReadVariableOp6model_1/hand_layer1/Conv1D/ExpandDims_1/ReadVariableOp2~
=model_1/hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOp=model_1/hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOp2X
*model_1/hand_layer2/BiasAdd/ReadVariableOp*model_1/hand_layer2/BiasAdd/ReadVariableOp2\
,model_1/hand_layer2/Tensordot/ReadVariableOp,model_1/hand_layer2/Tensordot/ReadVariableOp:Q M
+
_output_shapes
:?????????4

_user_specified_namemove:QM
+
_output_shapes
:?????????4

_user_specified_namehand:XT
/
_output_shapes
:?????????@4
!
_user_specified_name	history
?
J
.__inference_flatten_4_layer_call_fn_1879667439

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_flatten_4_layer_call_and_return_conditional_losses_1879666218a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_hand_layer1_layer_call_fn_1879667295

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_hand_layer1_layer_call_and_return_conditional_losses_1879666108w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@4: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@4
 
_user_specified_nameinputs
?
e
I__inference_flatten_3_layer_call_and_return_conditional_losses_1879666268

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
-__inference_conv1d_3_layer_call_fn_1879667342

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv1d_3_layer_call_and_return_conditional_losses_1879666189s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????4: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????4
 
_user_specified_nameinputs
?
?
-__inference_conv2d_1_layer_call_fn_1879667454

inputs!
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_1_layer_call_and_return_conditional_losses_1879666206w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
H__inference_conv1d_3_layer_call_and_return_conditional_losses_1879667358

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????4?
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????4
 
_user_specified_nameinputs
?
?
,__inference_dense_6_layer_call_fn_1879667474

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_6_layer_call_and_return_conditional_losses_1879666239p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_signature_wrapper_1879666903
hand
history
move
unknown:
	unknown_0:
	unknown_1:@
	unknown_2:@
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:@ 
	unknown_8: 
	unknown_9:
??

unknown_10:	?

unknown_11:	?@

unknown_12:@

unknown_13:
??

unknown_14:	?

unknown_15:
??

unknown_16:	?

unknown_17:	?

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmovehandhistoryunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference__wrapped_model_1879666059o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????4:?????????@4:?????????4: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
+
_output_shapes
:?????????4

_user_specified_namehand:XT
/
_output_shapes
:?????????@4
!
_user_specified_name	history:QM
+
_output_shapes
:?????????4

_user_specified_namemove
?
?
K__inference_hand_layer2_layer_call_and_return_conditional_losses_1879667423

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:}
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:?????????@?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????@@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
e
I__inference_flatten_5_layer_call_and_return_conditional_losses_1879667434

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_dense_8_layer_call_fn_1879667587

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_8_layer_call_and_return_conditional_losses_1879666315p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?G
?

G__inference_model_1_layer_call_and_return_conditional_losses_1879666642

inputs
inputs_1
inputs_2,
hand_layer1_1879666586:$
hand_layer1_1879666588:(
hand_layer2_1879666591:@$
hand_layer2_1879666593:@)
conv1d_2_1879666596:!
conv1d_2_1879666598:)
conv1d_3_1879666601:!
conv1d_3_1879666603:-
conv2d_1_1879666606:@ !
conv2d_1_1879666608: &
dense_6_1879666613:
??!
dense_6_1879666615:	?%
dense_5_1879666618:	?@ 
dense_5_1879666620:@&
dense_7_1879666625:
??!
dense_7_1879666627:	?&
dense_8_1879666631:
??!
dense_8_1879666633:	?%
dense_9_1879666636:	? 
dense_9_1879666638:
identity?? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?#hand_layer1/StatefulPartitionedCall?#hand_layer2/StatefulPartitionedCall?
#hand_layer1/StatefulPartitionedCallStatefulPartitionedCallinputs_2hand_layer1_1879666586hand_layer1_1879666588*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_hand_layer1_layer_call_and_return_conditional_losses_1879666108?
#hand_layer2/StatefulPartitionedCallStatefulPartitionedCall,hand_layer1/StatefulPartitionedCall:output:0hand_layer2_1879666591hand_layer2_1879666593*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_hand_layer2_layer_call_and_return_conditional_losses_1879666145?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv1d_2_1879666596conv1d_2_1879666598*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv1d_2_layer_call_and_return_conditional_losses_1879666167?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_3_1879666601conv1d_3_1879666603*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv1d_3_layer_call_and_return_conditional_losses_1879666189?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall,hand_layer2/StatefulPartitionedCall:output:0conv2d_1_1879666606conv2d_1_1879666608*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_1_layer_call_and_return_conditional_losses_1879666206?
flatten_4/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_flatten_4_layer_call_and_return_conditional_losses_1879666218?
flatten_5/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_flatten_5_layer_call_and_return_conditional_losses_1879666226?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_6_1879666613dense_6_1879666615*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_6_layer_call_and_return_conditional_losses_1879666239?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_5_1879666618dense_5_1879666620*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_5_layer_call_and_return_conditional_losses_1879666256?
flatten_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_flatten_3_layer_call_and_return_conditional_losses_1879666268?
concatenate_1/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0(dense_5/StatefulPartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_concatenate_1_layer_call_and_return_conditional_losses_1879666278?
dense_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_7_1879666625dense_7_1879666627*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_7_layer_call_and_return_conditional_losses_1879666291?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_dropout_1_layer_call_and_return_conditional_losses_1879666421?
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_8_1879666631dense_8_1879666633*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_8_layer_call_and_return_conditional_losses_1879666315?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_1879666636dense_9_1879666638*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_9_layer_call_and_return_conditional_losses_1879666331w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall$^hand_layer1/StatefulPartitionedCall$^hand_layer2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????4:?????????4:?????????@4: : : : : : : : : : : : : : : : : : : : 2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2J
#hand_layer1/StatefulPartitionedCall#hand_layer1/StatefulPartitionedCall2J
#hand_layer2/StatefulPartitionedCall#hand_layer2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????4
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????4
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????@4
 
_user_specified_nameinputs
?
?
H__inference_conv2d_1_layer_call_and_return_conditional_losses_1879667465

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
,__inference_model_1_layer_call_fn_1879666997
inputs_0
inputs_1
inputs_2
unknown:
	unknown_0:
	unknown_1:@
	unknown_2:@
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:@ 
	unknown_8: 
	unknown_9:
??

unknown_10:	?

unknown_11:	?@

unknown_12:@

unknown_13:
??

unknown_14:	?

unknown_15:
??

unknown_16:	?

unknown_17:	?

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_model_1_layer_call_and_return_conditional_losses_1879666642o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????4:?????????4:?????????@4: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????4
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????4
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:?????????@4
"
_user_specified_name
inputs/2
?

?
G__inference_dense_8_layer_call_and_return_conditional_losses_1879666315

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
I__inference_dropout_1_layer_call_and_return_conditional_losses_1879667566

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_conv1d_3_layer_call_and_return_conditional_losses_1879666189

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????4?
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????4
 
_user_specified_nameinputs
?*
?
K__inference_hand_layer1_layer_call_and_return_conditional_losses_1879667333

inputsA
+conv1d_expanddims_1_readvariableop_resource:@
2squeeze_batch_dims_biasadd_readvariableop_resource:
identity??"Conv1D/ExpandDims_1/ReadVariableOp?)squeeze_batch_dims/BiasAdd/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:?????????@4?
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:V
Conv1D/ShapeShapeConv1D/ExpandDims:output:0*
T0*
_output_shapes
:d
Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Conv1D/strided_sliceStridedSliceConv1D/Shape:output:0#Conv1D/strided_slice/stack:output:0%Conv1D/strided_slice/stack_1:output:0%Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   4      ?
Conv1D/ReshapeReshapeConv1D/ExpandDims:output:0Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????4?
Conv1D/Conv2DConv2DConv1D/Reshape:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
k
Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         ]
Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/concatConcatV2Conv1D/strided_slice:output:0Conv1D/concat/values_1:output:0Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Conv1D/Reshape_1ReshapeConv1D/Conv2D:output:0Conv1D/concat:output:0*
T0*3
_output_shapes!
:?????????@?
Conv1D/SqueezeSqueezeConv1D/Reshape_1:output:0*
T0*/
_output_shapes
:?????????@*
squeeze_dims

?????????_
squeeze_batch_dims/ShapeShapeConv1D/Squeeze:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      ?
squeeze_batch_dims/ReshapeReshapeConv1D/Squeeze:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:??????????
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????s
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????@m
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp#^Conv1D/ExpandDims_1/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@4: : 2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:?????????@4
 
_user_specified_nameinputs
?
?
H__inference_conv1d_2_layer_call_and_return_conditional_losses_1879667383

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????4?
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????4
 
_user_specified_nameinputs
?

?
G__inference_dense_5_layer_call_and_return_conditional_losses_1879666256

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_model_1_layer_call_fn_1879666732
move
hand
history
unknown:
	unknown_0:
	unknown_1:@
	unknown_2:@
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:@ 
	unknown_8: 
	unknown_9:
??

unknown_10:	?

unknown_11:	?@

unknown_12:@

unknown_13:
??

unknown_14:	?

unknown_15:
??

unknown_16:	?

unknown_17:	?

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmovehandhistoryunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_model_1_layer_call_and_return_conditional_losses_1879666642o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????4:?????????4:?????????@4: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
+
_output_shapes
:?????????4

_user_specified_namemove:QM
+
_output_shapes
:?????????4

_user_specified_namehand:XT
/
_output_shapes
:?????????@4
!
_user_specified_name	history
?

?
G__inference_dense_7_layer_call_and_return_conditional_losses_1879667551

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
G__inference_model_1_layer_call_and_return_conditional_losses_1879667286
inputs_0
inputs_1
inputs_2M
7hand_layer1_conv1d_expanddims_1_readvariableop_resource:L
>hand_layer1_squeeze_batch_dims_biasadd_readvariableop_resource:?
-hand_layer2_tensordot_readvariableop_resource:@9
+hand_layer2_biasadd_readvariableop_resource:@J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_2_biasadd_readvariableop_resource:J
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_3_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:@ 6
(conv2d_1_biasadd_readvariableop_resource: :
&dense_6_matmul_readvariableop_resource:
??6
'dense_6_biasadd_readvariableop_resource:	?9
&dense_5_matmul_readvariableop_resource:	?@5
'dense_5_biasadd_readvariableop_resource:@:
&dense_7_matmul_readvariableop_resource:
??6
'dense_7_biasadd_readvariableop_resource:	?:
&dense_8_matmul_readvariableop_resource:
??6
'dense_8_biasadd_readvariableop_resource:	?9
&dense_9_matmul_readvariableop_resource:	?5
'dense_9_biasadd_readvariableop_resource:
identity??conv1d_2/BiasAdd/ReadVariableOp?+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp?conv1d_3/BiasAdd/ReadVariableOp?+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?.hand_layer1/Conv1D/ExpandDims_1/ReadVariableOp?5hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOp?"hand_layer2/BiasAdd/ReadVariableOp?$hand_layer2/Tensordot/ReadVariableOpl
!hand_layer1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
hand_layer1/Conv1D/ExpandDims
ExpandDimsinputs_2*hand_layer1/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:?????????@4?
.hand_layer1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp7hand_layer1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0e
#hand_layer1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
hand_layer1/Conv1D/ExpandDims_1
ExpandDims6hand_layer1/Conv1D/ExpandDims_1/ReadVariableOp:value:0,hand_layer1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n
hand_layer1/Conv1D/ShapeShape&hand_layer1/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:p
&hand_layer1/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(hand_layer1/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(hand_layer1/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 hand_layer1/Conv1D/strided_sliceStridedSlice!hand_layer1/Conv1D/Shape:output:0/hand_layer1/Conv1D/strided_slice/stack:output:01hand_layer1/Conv1D/strided_slice/stack_1:output:01hand_layer1/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 hand_layer1/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   4      ?
hand_layer1/Conv1D/ReshapeReshape&hand_layer1/Conv1D/ExpandDims:output:0)hand_layer1/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????4?
hand_layer1/Conv1D/Conv2DConv2D#hand_layer1/Conv1D/Reshape:output:0(hand_layer1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
w
"hand_layer1/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         i
hand_layer1/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
hand_layer1/Conv1D/concatConcatV2)hand_layer1/Conv1D/strided_slice:output:0+hand_layer1/Conv1D/concat/values_1:output:0'hand_layer1/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
hand_layer1/Conv1D/Reshape_1Reshape"hand_layer1/Conv1D/Conv2D:output:0"hand_layer1/Conv1D/concat:output:0*
T0*3
_output_shapes!
:?????????@?
hand_layer1/Conv1D/SqueezeSqueeze%hand_layer1/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:?????????@*
squeeze_dims

?????????w
$hand_layer1/squeeze_batch_dims/ShapeShape#hand_layer1/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:|
2hand_layer1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
4hand_layer1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????~
4hand_layer1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,hand_layer1/squeeze_batch_dims/strided_sliceStridedSlice-hand_layer1/squeeze_batch_dims/Shape:output:0;hand_layer1/squeeze_batch_dims/strided_slice/stack:output:0=hand_layer1/squeeze_batch_dims/strided_slice/stack_1:output:0=hand_layer1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
,hand_layer1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      ?
&hand_layer1/squeeze_batch_dims/ReshapeReshape#hand_layer1/Conv1D/Squeeze:output:05hand_layer1/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:??????????
5hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp>hand_layer1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
&hand_layer1/squeeze_batch_dims/BiasAddBiasAdd/hand_layer1/squeeze_batch_dims/Reshape:output:0=hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????
.hand_layer1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      u
*hand_layer1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%hand_layer1/squeeze_batch_dims/concatConcatV25hand_layer1/squeeze_batch_dims/strided_slice:output:07hand_layer1/squeeze_batch_dims/concat/values_1:output:03hand_layer1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
(hand_layer1/squeeze_batch_dims/Reshape_1Reshape/hand_layer1/squeeze_batch_dims/BiasAdd:output:0.hand_layer1/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????@?
hand_layer1/ReluRelu1hand_layer1/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????@?
$hand_layer2/Tensordot/ReadVariableOpReadVariableOp-hand_layer2_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0d
hand_layer2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:o
hand_layer2/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          i
hand_layer2/Tensordot/ShapeShapehand_layer1/Relu:activations:0*
T0*
_output_shapes
:e
#hand_layer2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
hand_layer2/Tensordot/GatherV2GatherV2$hand_layer2/Tensordot/Shape:output:0#hand_layer2/Tensordot/free:output:0,hand_layer2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
%hand_layer2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
 hand_layer2/Tensordot/GatherV2_1GatherV2$hand_layer2/Tensordot/Shape:output:0#hand_layer2/Tensordot/axes:output:0.hand_layer2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
hand_layer2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
hand_layer2/Tensordot/ProdProd'hand_layer2/Tensordot/GatherV2:output:0$hand_layer2/Tensordot/Const:output:0*
T0*
_output_shapes
: g
hand_layer2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
hand_layer2/Tensordot/Prod_1Prod)hand_layer2/Tensordot/GatherV2_1:output:0&hand_layer2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: c
!hand_layer2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
hand_layer2/Tensordot/concatConcatV2#hand_layer2/Tensordot/free:output:0#hand_layer2/Tensordot/axes:output:0*hand_layer2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
hand_layer2/Tensordot/stackPack#hand_layer2/Tensordot/Prod:output:0%hand_layer2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
hand_layer2/Tensordot/transpose	Transposehand_layer1/Relu:activations:0%hand_layer2/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????@?
hand_layer2/Tensordot/ReshapeReshape#hand_layer2/Tensordot/transpose:y:0$hand_layer2/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
hand_layer2/Tensordot/MatMulMatMul&hand_layer2/Tensordot/Reshape:output:0,hand_layer2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@g
hand_layer2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@e
#hand_layer2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
hand_layer2/Tensordot/concat_1ConcatV2'hand_layer2/Tensordot/GatherV2:output:0&hand_layer2/Tensordot/Const_2:output:0,hand_layer2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
hand_layer2/TensordotReshape&hand_layer2/Tensordot/MatMul:product:0'hand_layer2/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????@@?
"hand_layer2/BiasAdd/ReadVariableOpReadVariableOp+hand_layer2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
hand_layer2/BiasAddBiasAddhand_layer2/Tensordot:output:0*hand_layer2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@p
hand_layer2/ReluReluhand_layer2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@i
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d_2/Conv1D/ExpandDims
ExpandDimsinputs_1'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????4?
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????f
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????i
conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d_3/Conv1D/ExpandDims
ExpandDimsinputs_0'conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????4?
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_3/Conv1D/ExpandDims_1
ExpandDims3conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
conv1d_3/Conv1DConv2D#conv1d_3/Conv1D/ExpandDims:output:0%conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv1d_3/Conv1D/SqueezeSqueezeconv1d_3/Conv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1d_3/BiasAddBiasAdd conv1d_3/Conv1D/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????f
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:??????????
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
conv2d_1/Conv2DConv2Dhand_layer2/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? `
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
flatten_4/ReshapeReshapeconv1d_2/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????`
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
flatten_5/ReshapeReshapeconv1d_3/Relu:activations:0flatten_5/Const:output:0*
T0*(
_output_shapes
:???????????
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_6/MatMulMatMulflatten_5/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense_5/MatMulMatMulflatten_4/Reshape:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@`
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_3/ReshapeReshapeconv2d_1/Relu:activations:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_1/concatConcatV2dense_6/Relu:activations:0dense_5/Relu:activations:0flatten_3/Reshape:output:0"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_7/MatMulMatMulconcatenate_1/concat:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_1/dropout/MulMuldense_7/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????a
dropout_1/dropout/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_8/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp/^hand_layer1/Conv1D/ExpandDims_1/ReadVariableOp6^hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOp#^hand_layer2/BiasAdd/ReadVariableOp%^hand_layer2/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????4:?????????4:?????????@4: : : : : : : : : : : : : : : : : : : : 2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2`
.hand_layer1/Conv1D/ExpandDims_1/ReadVariableOp.hand_layer1/Conv1D/ExpandDims_1/ReadVariableOp2n
5hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOp5hand_layer1/squeeze_batch_dims/BiasAdd/ReadVariableOp2H
"hand_layer2/BiasAdd/ReadVariableOp"hand_layer2/BiasAdd/ReadVariableOp2L
$hand_layer2/Tensordot/ReadVariableOp$hand_layer2/Tensordot/ReadVariableOp:U Q
+
_output_shapes
:?????????4
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????4
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:?????????@4
"
_user_specified_name
inputs/2
?

?
G__inference_dense_6_layer_call_and_return_conditional_losses_1879667485

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_conv2d_1_layer_call_and_return_conditional_losses_1879666206

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
g
.__inference_dropout_1_layer_call_fn_1879667561

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_dropout_1_layer_call_and_return_conditional_losses_1879666421p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_flatten_5_layer_call_fn_1879667428

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_flatten_5_layer_call_and_return_conditional_losses_1879666226a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
h
I__inference_dropout_1_layer_call_and_return_conditional_losses_1879667578

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_conv1d_2_layer_call_and_return_conditional_losses_1879666167

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????4?
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????4
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
9
hand1
serving_default_hand:0?????????4
C
history8
serving_default_history:0?????????@4
9
move1
serving_default_move:0?????????4;
dense_90
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer-13
layer_with_weights-7
layer-14
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
1	variables
2trainable_variables
3regularization_losses
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

]kernel
^bias
_	variables
`trainable_variables
aregularization_losses
b	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
"
	optimizer
?
0
1
2
 3
%4
&5
+6
,7
98
:9
?10
@11
E12
F13
S14
T15
]16
^17
c18
d19"
trackable_list_wrapper
?
0
1
2
 3
%4
&5
+6
,7
98
:9
?10
@11
E12
F13
S14
T15
]16
^17
c18
d19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
(:&2hand_layer1/kernel
:2hand_layer1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#2conv1d_3/kernel
:2conv1d_3/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
!	variables
"trainable_variables
#regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#2conv1d_2/kernel
:2conv1d_2/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
'	variables
(trainable_variables
)regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"@2hand_layer2/kernel
:@2hand_layer2/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
1	variables
2trainable_variables
3regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@ 2conv2d_1/kernel
: 2conv2d_1/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_6/kernel
:?2dense_6/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?@2dense_5/kernel
:@2dense_5/bias
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_7/kernel
:?2dense_7/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_8/kernel
:?2dense_8/bias
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
_	variables
`trainable_variables
aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_9/kernel
:2dense_9/bias
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
e	variables
ftrainable_variables
gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
?2?
,__inference_model_1_layer_call_fn_1879666381
,__inference_model_1_layer_call_fn_1879666950
,__inference_model_1_layer_call_fn_1879666997
,__inference_model_1_layer_call_fn_1879666732?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_model_1_layer_call_and_return_conditional_losses_1879667138
G__inference_model_1_layer_call_and_return_conditional_losses_1879667286
G__inference_model_1_layer_call_and_return_conditional_losses_1879666793
G__inference_model_1_layer_call_and_return_conditional_losses_1879666854?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference__wrapped_model_1879666059movehandhistory"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_hand_layer1_layer_call_fn_1879667295?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_hand_layer1_layer_call_and_return_conditional_losses_1879667333?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_conv1d_3_layer_call_fn_1879667342?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv1d_3_layer_call_and_return_conditional_losses_1879667358?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_conv1d_2_layer_call_fn_1879667367?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv1d_2_layer_call_and_return_conditional_losses_1879667383?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_hand_layer2_layer_call_fn_1879667392?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_hand_layer2_layer_call_and_return_conditional_losses_1879667423?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_flatten_5_layer_call_fn_1879667428?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_flatten_5_layer_call_and_return_conditional_losses_1879667434?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_flatten_4_layer_call_fn_1879667439?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_flatten_4_layer_call_and_return_conditional_losses_1879667445?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_conv2d_1_layer_call_fn_1879667454?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_1_layer_call_and_return_conditional_losses_1879667465?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_6_layer_call_fn_1879667474?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_6_layer_call_and_return_conditional_losses_1879667485?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_5_layer_call_fn_1879667494?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_5_layer_call_and_return_conditional_losses_1879667505?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_flatten_3_layer_call_fn_1879667510?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_flatten_3_layer_call_and_return_conditional_losses_1879667516?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_concatenate_1_layer_call_fn_1879667523?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_concatenate_1_layer_call_and_return_conditional_losses_1879667531?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_7_layer_call_fn_1879667540?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_7_layer_call_and_return_conditional_losses_1879667551?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_dropout_1_layer_call_fn_1879667556
.__inference_dropout_1_layer_call_fn_1879667561?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_dropout_1_layer_call_and_return_conditional_losses_1879667566
I__inference_dropout_1_layer_call_and_return_conditional_losses_1879667578?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_dense_8_layer_call_fn_1879667587?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_8_layer_call_and_return_conditional_losses_1879667598?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_9_layer_call_fn_1879667607?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_9_layer_call_and_return_conditional_losses_1879667617?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
(__inference_signature_wrapper_1879666903handhistorymove"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
%__inference__wrapped_model_1879666059?+,%& 9:?@EFST]^cd???
{?x
v?s
"?
move?????????4
"?
hand?????????4
)?&
history?????????@4
? "1?.
,
dense_9!?
dense_9??????????
M__inference_concatenate_1_layer_call_and_return_conditional_losses_1879667531???}
v?s
q?n
#? 
inputs/0??????????
"?
inputs/1?????????@
#? 
inputs/2??????????
? "&?#
?
0??????????
? ?
2__inference_concatenate_1_layer_call_fn_1879667523???}
v?s
q?n
#? 
inputs/0??????????
"?
inputs/1?????????@
#? 
inputs/2??????????
? "????????????
H__inference_conv1d_2_layer_call_and_return_conditional_losses_1879667383d%&3?0
)?&
$?!
inputs?????????4
? ")?&
?
0?????????
? ?
-__inference_conv1d_2_layer_call_fn_1879667367W%&3?0
)?&
$?!
inputs?????????4
? "???????????
H__inference_conv1d_3_layer_call_and_return_conditional_losses_1879667358d 3?0
)?&
$?!
inputs?????????4
? ")?&
?
0?????????
? ?
-__inference_conv1d_3_layer_call_fn_1879667342W 3?0
)?&
$?!
inputs?????????4
? "???????????
H__inference_conv2d_1_layer_call_and_return_conditional_losses_1879667465l9:7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0????????? 
? ?
-__inference_conv2d_1_layer_call_fn_1879667454_9:7?4
-?*
(?%
inputs?????????@@
? " ?????????? ?
G__inference_dense_5_layer_call_and_return_conditional_losses_1879667505]EF0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? ?
,__inference_dense_5_layer_call_fn_1879667494PEF0?-
&?#
!?
inputs??????????
? "??????????@?
G__inference_dense_6_layer_call_and_return_conditional_losses_1879667485^?@0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_6_layer_call_fn_1879667474Q?@0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dense_7_layer_call_and_return_conditional_losses_1879667551^ST0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_7_layer_call_fn_1879667540QST0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dense_8_layer_call_and_return_conditional_losses_1879667598^]^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_8_layer_call_fn_1879667587Q]^0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dense_9_layer_call_and_return_conditional_losses_1879667617]cd0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
,__inference_dense_9_layer_call_fn_1879667607Pcd0?-
&?#
!?
inputs??????????
? "???????????
I__inference_dropout_1_layer_call_and_return_conditional_losses_1879667566^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
I__inference_dropout_1_layer_call_and_return_conditional_losses_1879667578^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
.__inference_dropout_1_layer_call_fn_1879667556Q4?1
*?'
!?
inputs??????????
p 
? "????????????
.__inference_dropout_1_layer_call_fn_1879667561Q4?1
*?'
!?
inputs??????????
p
? "????????????
I__inference_flatten_3_layer_call_and_return_conditional_losses_1879667516a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????
? ?
.__inference_flatten_3_layer_call_fn_1879667510T7?4
-?*
(?%
inputs????????? 
? "????????????
I__inference_flatten_4_layer_call_and_return_conditional_losses_1879667445]3?0
)?&
$?!
inputs?????????
? "&?#
?
0??????????
? ?
.__inference_flatten_4_layer_call_fn_1879667439P3?0
)?&
$?!
inputs?????????
? "????????????
I__inference_flatten_5_layer_call_and_return_conditional_losses_1879667434]3?0
)?&
$?!
inputs?????????
? "&?#
?
0??????????
? ?
.__inference_flatten_5_layer_call_fn_1879667428P3?0
)?&
$?!
inputs?????????
? "????????????
K__inference_hand_layer1_layer_call_and_return_conditional_losses_1879667333l7?4
-?*
(?%
inputs?????????@4
? "-?*
#? 
0?????????@
? ?
0__inference_hand_layer1_layer_call_fn_1879667295_7?4
-?*
(?%
inputs?????????@4
? " ??????????@?
K__inference_hand_layer2_layer_call_and_return_conditional_losses_1879667423l+,7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@@
? ?
0__inference_hand_layer2_layer_call_fn_1879667392_+,7?4
-?*
(?%
inputs?????????@
? " ??????????@@?
G__inference_model_1_layer_call_and_return_conditional_losses_1879666793?+,%& 9:?@EFST]^cd???
???
v?s
"?
move?????????4
"?
hand?????????4
)?&
history?????????@4
p 

 
? "%?"
?
0?????????
? ?
G__inference_model_1_layer_call_and_return_conditional_losses_1879666854?+,%& 9:?@EFST]^cd???
???
v?s
"?
move?????????4
"?
hand?????????4
)?&
history?????????@4
p

 
? "%?"
?
0?????????
? ?
G__inference_model_1_layer_call_and_return_conditional_losses_1879667138?+,%& 9:?@EFST]^cd???
???
?|
&?#
inputs/0?????????4
&?#
inputs/1?????????4
*?'
inputs/2?????????@4
p 

 
? "%?"
?
0?????????
? ?
G__inference_model_1_layer_call_and_return_conditional_losses_1879667286?+,%& 9:?@EFST]^cd???
???
?|
&?#
inputs/0?????????4
&?#
inputs/1?????????4
*?'
inputs/2?????????@4
p

 
? "%?"
?
0?????????
? ?
,__inference_model_1_layer_call_fn_1879666381?+,%& 9:?@EFST]^cd???
???
v?s
"?
move?????????4
"?
hand?????????4
)?&
history?????????@4
p 

 
? "???????????
,__inference_model_1_layer_call_fn_1879666732?+,%& 9:?@EFST]^cd???
???
v?s
"?
move?????????4
"?
hand?????????4
)?&
history?????????@4
p

 
? "???????????
,__inference_model_1_layer_call_fn_1879666950?+,%& 9:?@EFST]^cd???
???
?|
&?#
inputs/0?????????4
&?#
inputs/1?????????4
*?'
inputs/2?????????@4
p 

 
? "???????????
,__inference_model_1_layer_call_fn_1879666997?+,%& 9:?@EFST]^cd???
???
?|
&?#
inputs/0?????????4
&?#
inputs/1?????????4
*?'
inputs/2?????????@4
p

 
? "???????????
(__inference_signature_wrapper_1879666903?+,%& 9:?@EFST]^cd???
? 
???
*
hand"?
hand?????????4
4
history)?&
history?????????@4
*
move"?
move?????????4"1?.
,
dense_9!?
dense_9?????????