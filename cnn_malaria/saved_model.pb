??
??
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
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
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
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8??
?
conv_model_1/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv_model_1/conv1/kernel
?
-conv_model_1/conv1/kernel/Read/ReadVariableOpReadVariableOpconv_model_1/conv1/kernel*&
_output_shapes
: *
dtype0
?
conv_model_1/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv_model_1/conv1/bias

+conv_model_1/conv1/bias/Read/ReadVariableOpReadVariableOpconv_model_1/conv1/bias*
_output_shapes
: *
dtype0
?
conv_model_1/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameconv_model_1/conv2/kernel
?
-conv_model_1/conv2/kernel/Read/ReadVariableOpReadVariableOpconv_model_1/conv2/kernel*&
_output_shapes
: @*
dtype0
?
conv_model_1/conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameconv_model_1/conv2/bias

+conv_model_1/conv2/bias/Read/ReadVariableOpReadVariableOpconv_model_1/conv2/bias*
_output_shapes
:@*
dtype0
?
conv_model_1/conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?**
shared_nameconv_model_1/conv3/kernel
?
-conv_model_1/conv3/kernel/Read/ReadVariableOpReadVariableOpconv_model_1/conv3/kernel*'
_output_shapes
:@?*
dtype0
?
conv_model_1/conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameconv_model_1/conv3/bias
?
+conv_model_1/conv3/bias/Read/ReadVariableOpReadVariableOpconv_model_1/conv3/bias*
_output_shapes	
:?*
dtype0
?
conv_model_1/d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?*'
shared_nameconv_model_1/d1/kernel
?
*conv_model_1/d1/kernel/Read/ReadVariableOpReadVariableOpconv_model_1/d1/kernel* 
_output_shapes
:
?	?*
dtype0
?
conv_model_1/d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameconv_model_1/d1/bias
z
(conv_model_1/d1/bias/Read/ReadVariableOpReadVariableOpconv_model_1/d1/bias*
_output_shapes	
:?*
dtype0
?
conv_model_1/d2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameconv_model_1/d2/kernel
?
*conv_model_1/d2/kernel/Read/ReadVariableOpReadVariableOpconv_model_1/d2/kernel* 
_output_shapes
:
??*
dtype0
?
conv_model_1/d2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameconv_model_1/d2/bias
z
(conv_model_1/d2/bias/Read/ReadVariableOpReadVariableOpconv_model_1/d2/bias*
_output_shapes	
:?*
dtype0
?
conv_model_1/output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*+
shared_nameconv_model_1/output/kernel
?
.conv_model_1/output/kernel/Read/ReadVariableOpReadVariableOpconv_model_1/output/kernel*
_output_shapes
:	?*
dtype0
?
conv_model_1/output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv_model_1/output/bias
?
,conv_model_1/output/bias/Read/ReadVariableOpReadVariableOpconv_model_1/output/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?'
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?'
value?'B?' B?'
?
	conv1
	pool1
	conv2
	pool2
	conv3
flatten
d1
dop1
	d2

dop2
out
trainable_variables
	variables
regularization_losses
	keras_api

signatures
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
R
!trainable_variables
"	variables
#regularization_losses
$	keras_api
h

%kernel
&bias
'trainable_variables
(	variables
)regularization_losses
*	keras_api
R
+trainable_variables
,	variables
-regularization_losses
.	keras_api
h

/kernel
0bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
R
5trainable_variables
6	variables
7regularization_losses
8	keras_api
h

9kernel
:bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
R
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
h

Ckernel
Dbias
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
V
0
1
2
3
%4
&5
/6
07
98
:9
C10
D11
V
0
1
2
3
%4
&5
/6
07
98
:9
C10
D11
 
?
Inon_trainable_variables

Jlayers
trainable_variables
Kmetrics
Llayer_metrics
Mlayer_regularization_losses
	variables
regularization_losses
 
VT
VARIABLE_VALUEconv_model_1/conv1/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEconv_model_1/conv1/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Nnon_trainable_variables

Olayers
trainable_variables
Pmetrics
Qlayer_metrics
Rlayer_regularization_losses
	variables
regularization_losses
 
 
 
?
Snon_trainable_variables

Tlayers
trainable_variables
Umetrics
Vlayer_metrics
Wlayer_regularization_losses
	variables
regularization_losses
VT
VARIABLE_VALUEconv_model_1/conv2/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEconv_model_1/conv2/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Xnon_trainable_variables

Ylayers
trainable_variables
Zmetrics
[layer_metrics
\layer_regularization_losses
	variables
regularization_losses
 
 
 
?
]non_trainable_variables

^layers
!trainable_variables
_metrics
`layer_metrics
alayer_regularization_losses
"	variables
#regularization_losses
VT
VARIABLE_VALUEconv_model_1/conv3/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEconv_model_1/conv3/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1

%0
&1
 
?
bnon_trainable_variables

clayers
'trainable_variables
dmetrics
elayer_metrics
flayer_regularization_losses
(	variables
)regularization_losses
 
 
 
?
gnon_trainable_variables

hlayers
+trainable_variables
imetrics
jlayer_metrics
klayer_regularization_losses
,	variables
-regularization_losses
PN
VARIABLE_VALUEconv_model_1/d1/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv_model_1/d1/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01

/0
01
 
?
lnon_trainable_variables

mlayers
1trainable_variables
nmetrics
olayer_metrics
player_regularization_losses
2	variables
3regularization_losses
 
 
 
?
qnon_trainable_variables

rlayers
5trainable_variables
smetrics
tlayer_metrics
ulayer_regularization_losses
6	variables
7regularization_losses
PN
VARIABLE_VALUEconv_model_1/d2/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv_model_1/d2/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1

90
:1
 
?
vnon_trainable_variables

wlayers
;trainable_variables
xmetrics
ylayer_metrics
zlayer_regularization_losses
<	variables
=regularization_losses
 
 
 
?
{non_trainable_variables

|layers
?trainable_variables
}metrics
~layer_metrics
layer_regularization_losses
@	variables
Aregularization_losses
US
VARIABLE_VALUEconv_model_1/output/kernel%out/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEconv_model_1/output/bias#out/bias/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

C0
D1
 
?
?non_trainable_variables
?layers
Etrainable_variables
?metrics
?layer_metrics
 ?layer_regularization_losses
F	variables
Gregularization_losses
 
N
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
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv_model_1/conv1/kernelconv_model_1/conv1/biasconv_model_1/conv2/kernelconv_model_1/conv2/biasconv_model_1/conv3/kernelconv_model_1/conv3/biasconv_model_1/d1/kernelconv_model_1/d1/biasconv_model_1/d2/kernelconv_model_1/d2/biasconv_model_1/output/kernelconv_model_1/output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_10068
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-conv_model_1/conv1/kernel/Read/ReadVariableOp+conv_model_1/conv1/bias/Read/ReadVariableOp-conv_model_1/conv2/kernel/Read/ReadVariableOp+conv_model_1/conv2/bias/Read/ReadVariableOp-conv_model_1/conv3/kernel/Read/ReadVariableOp+conv_model_1/conv3/bias/Read/ReadVariableOp*conv_model_1/d1/kernel/Read/ReadVariableOp(conv_model_1/d1/bias/Read/ReadVariableOp*conv_model_1/d2/kernel/Read/ReadVariableOp(conv_model_1/d2/bias/Read/ReadVariableOp.conv_model_1/output/kernel/Read/ReadVariableOp,conv_model_1/output/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_10488
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_model_1/conv1/kernelconv_model_1/conv1/biasconv_model_1/conv2/kernelconv_model_1/conv2/biasconv_model_1/conv3/kernelconv_model_1/conv3/biasconv_model_1/d1/kernelconv_model_1/d1/biasconv_model_1/d2/kernelconv_model_1/d2/biasconv_model_1/output/kernelconv_model_1/output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_10534??
?P
?
G__inference_conv_model_1_layer_call_and_return_conditional_losses_10134	
image(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource%
!d1_matmul_readvariableop_resource&
"d1_biasadd_readvariableop_resource%
!d2_matmul_readvariableop_resource&
"d2_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?d1/BiasAdd/ReadVariableOp?d1/MatMul/ReadVariableOp?d2/BiasAdd/ReadVariableOp?d2/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dimage#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1/BiasAdd/ReadVariableOp?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2

conv1/Relu?
max_pooling2d_2/MaxPoolMaxPoolconv1/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv2/Conv2D/ReadVariableOp?
conv2/Conv2DConv2D max_pooling2d_2/MaxPool:output:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2/Conv2D?
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2/BiasAdd/ReadVariableOp?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2

conv2/Relu?
max_pooling2d_3/MaxPoolMaxPoolconv2/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool?
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
conv3/Conv2D/ReadVariableOp?
conv3/Conv2DConv2D max_pooling2d_3/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv3/Conv2D?
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv3/BiasAdd/ReadVariableOp?
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv3/BiasAdds

conv3/ReluReluconv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2

conv3/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapeconv3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten/Reshape?
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
d1/MatMul/ReadVariableOp?
	d1/MatMulMatMulflatten/Reshape:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	d1/MatMul?
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
d1/BiasAdd/ReadVariableOp?

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

d1/BiasAddb
d1/ReluRelud1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
d1/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/dropout/Const?
dropout_2/dropout/MulMuld1/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_2/dropout/Mulw
dropout_2/dropout/ShapeShaped1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_2/dropout/Mul_1?
d2/MatMul/ReadVariableOpReadVariableOp!d2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
d2/MatMul/ReadVariableOp?
	d2/MatMulMatMuldropout_2/dropout/Mul_1:z:0 d2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	d2/MatMul?
d2/BiasAdd/ReadVariableOpReadVariableOp"d2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
d2/BiasAdd/ReadVariableOp?

d2/BiasAddBiasAddd2/MatMul:product:0!d2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

d2/BiasAddb
d2/ReluRelud2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
d2/Reluw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_3/dropout/Const?
dropout_3/dropout/MulMuld2/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_3/dropout/Mulw
dropout_3/dropout/ShapeShaped2/Relu:activations:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform?
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_3/dropout/GreaterEqual/y?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_3/dropout/Cast?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_3/dropout/Mul_1?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMuldropout_3/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
output/Softmax?
IdentityIdentityoutput/Softmax:softmax:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d2/BiasAdd/ReadVariableOp^d2/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d2/BiasAdd/ReadVariableOpd2/BiasAdd/ReadVariableOp24
d2/MatMul/ReadVariableOpd2/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:V R
/
_output_shapes
:?????????

_user_specified_nameimage
?
C
'__inference_flatten_layer_call_fn_10315

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
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_97502
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?=
?
G__inference_conv_model_1_layer_call_and_return_conditional_losses_10186	
image(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource%
!d1_matmul_readvariableop_resource&
"d1_biasadd_readvariableop_resource%
!d2_matmul_readvariableop_resource&
"d2_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?d1/BiasAdd/ReadVariableOp?d1/MatMul/ReadVariableOp?d2/BiasAdd/ReadVariableOp?d2/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dimage#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1/BiasAdd/ReadVariableOp?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2

conv1/Relu?
max_pooling2d_2/MaxPoolMaxPoolconv1/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv2/Conv2D/ReadVariableOp?
conv2/Conv2DConv2D max_pooling2d_2/MaxPool:output:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2/Conv2D?
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2/BiasAdd/ReadVariableOp?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2

conv2/Relu?
max_pooling2d_3/MaxPoolMaxPoolconv2/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool?
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
conv3/Conv2D/ReadVariableOp?
conv3/Conv2DConv2D max_pooling2d_3/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv3/Conv2D?
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv3/BiasAdd/ReadVariableOp?
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv3/BiasAdds

conv3/ReluReluconv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2

conv3/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapeconv3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten/Reshape?
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
d1/MatMul/ReadVariableOp?
	d1/MatMulMatMulflatten/Reshape:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	d1/MatMul?
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
d1/BiasAdd/ReadVariableOp?

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

d1/BiasAddb
d1/ReluRelud1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
d1/Relu~
dropout_2/IdentityIdentityd1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_2/Identity?
d2/MatMul/ReadVariableOpReadVariableOp!d2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
d2/MatMul/ReadVariableOp?
	d2/MatMulMatMuldropout_2/Identity:output:0 d2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	d2/MatMul?
d2/BiasAdd/ReadVariableOpReadVariableOp"d2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
d2/BiasAdd/ReadVariableOp?

d2/BiasAddBiasAddd2/MatMul:product:0!d2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

d2/BiasAddb
d2/ReluRelud2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
d2/Relu~
dropout_3/IdentityIdentityd2/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_3/Identity?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMuldropout_3/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
output/Softmax?
IdentityIdentityoutput/Softmax:softmax:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d2/BiasAdd/ReadVariableOp^d2/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d2/BiasAdd/ReadVariableOpd2/BiasAdd/ReadVariableOp24
d2/MatMul/ReadVariableOpd2/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:V R
/
_output_shapes
:?????????

_user_specified_nameimage
?0
?
F__inference_conv_model_1_layer_call_and_return_conditional_losses_9900
input_1

conv1_9683

conv1_9685

conv2_9711

conv2_9713

conv3_9739

conv3_9741
d1_9780
d1_9782
d2_9837
d2_9839
output_9894
output_9896
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?d1/StatefulPartitionedCall?d2/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?output/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1
conv1_9683
conv1_9685*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_conv1_layer_call_and_return_conditional_losses_96722
conv1/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_96392!
max_pooling2d_2/PartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0
conv2_9711
conv2_9713*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_conv2_layer_call_and_return_conditional_losses_97002
conv2/StatefulPartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_96512!
max_pooling2d_3/PartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0
conv3_9739
conv3_9741*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_conv3_layer_call_and_return_conditional_losses_97282
conv3/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_97502
flatten/PartitionedCall?
d1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0d1_9780d1_9782*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_d1_layer_call_and_return_conditional_losses_97692
d1/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_97972#
!dropout_2/StatefulPartitionedCall?
d2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0d2_9837d2_9839*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_d2_layer_call_and_return_conditional_losses_98262
d2/StatefulPartitionedCall?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall#d2/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_98542#
!dropout_3/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0output_9894output_9896*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_98832 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_9802

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_10352

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
@__inference_conv1_layer_call_and_return_conditional_losses_10255

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
=__inference_d2_layer_call_and_return_conditional_losses_10373

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_10347

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
,__inference_conv_model_1_layer_call_fn_10215	
image
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallimageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv_model_1_layer_call_and_return_conditional_losses_99812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameimage
?
w
"__inference_d2_layer_call_fn_10382

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_d2_layer_call_and_return_conditional_losses_98262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?%
?
__inference__traced_save_10488
file_prefix8
4savev2_conv_model_1_conv1_kernel_read_readvariableop6
2savev2_conv_model_1_conv1_bias_read_readvariableop8
4savev2_conv_model_1_conv2_kernel_read_readvariableop6
2savev2_conv_model_1_conv2_bias_read_readvariableop8
4savev2_conv_model_1_conv3_kernel_read_readvariableop6
2savev2_conv_model_1_conv3_bias_read_readvariableop5
1savev2_conv_model_1_d1_kernel_read_readvariableop3
/savev2_conv_model_1_d1_bias_read_readvariableop5
1savev2_conv_model_1_d2_kernel_read_readvariableop3
/savev2_conv_model_1_d2_bias_read_readvariableop9
5savev2_conv_model_1_output_kernel_read_readvariableop7
3savev2_conv_model_1_output_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB%out/kernel/.ATTRIBUTES/VARIABLE_VALUEB#out/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_conv_model_1_conv1_kernel_read_readvariableop2savev2_conv_model_1_conv1_bias_read_readvariableop4savev2_conv_model_1_conv2_kernel_read_readvariableop2savev2_conv_model_1_conv2_bias_read_readvariableop4savev2_conv_model_1_conv3_kernel_read_readvariableop2savev2_conv_model_1_conv3_bias_read_readvariableop1savev2_conv_model_1_d1_kernel_read_readvariableop/savev2_conv_model_1_d1_bias_read_readvariableop1savev2_conv_model_1_d2_kernel_read_readvariableop/savev2_conv_model_1_d2_bias_read_readvariableop5savev2_conv_model_1_output_kernel_read_readvariableop3savev2_conv_model_1_output_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : @:@:@?:?:
?	?:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:&"
 
_output_shapes
:
?	?:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
?	
?
,__inference_conv_model_1_layer_call_fn_10037
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv_model_1_layer_call_and_return_conditional_losses_99812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
?__inference_conv2_layer_call_and_return_conditional_losses_9700

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
<__inference_d1_layer_call_and_return_conditional_losses_9769

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?

?
?__inference_conv3_layer_call_and_return_conditional_losses_9728

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
b
C__inference_dropout_3_layer_call_and_return_conditional_losses_9854

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
@__inference_output_layer_call_and_return_conditional_losses_9883

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
z
%__inference_conv3_layer_call_fn_10304

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_conv3_layer_call_and_return_conditional_losses_97282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
E
)__inference_dropout_2_layer_call_fn_10362

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_98022
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_10310

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
A__inference_output_layer_call_and_return_conditional_losses_10420

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_3_layer_call_fn_10404

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_98542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_10394

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
,__inference_conv_model_1_layer_call_fn_10008
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv_model_1_layer_call_and_return_conditional_losses_99812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_9750

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_2_layer_call_fn_9645

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_96392
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
@__inference_conv2_layer_call_and_return_conditional_losses_10275

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
?__inference_conv1_layer_call_and_return_conditional_losses_9672

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_3_layer_call_fn_10409

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_98592
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
<__inference_d2_layer_call_and_return_conditional_losses_9826

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
#__inference_signature_wrapper_10068
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_96332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
=__inference_d1_layer_call_and_return_conditional_losses_10326

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?-
?
F__inference_conv_model_1_layer_call_and_return_conditional_losses_9981	
image

conv1_9945

conv1_9947

conv2_9951

conv2_9953

conv3_9957

conv3_9959
d1_9963
d1_9965
d2_9969
d2_9971
output_9975
output_9977
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?d1/StatefulPartitionedCall?d2/StatefulPartitionedCall?output/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallimage
conv1_9945
conv1_9947*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_conv1_layer_call_and_return_conditional_losses_96722
conv1/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_96392!
max_pooling2d_2/PartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0
conv2_9951
conv2_9953*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_conv2_layer_call_and_return_conditional_losses_97002
conv2/StatefulPartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_96512!
max_pooling2d_3/PartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0
conv3_9957
conv3_9959*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_conv3_layer_call_and_return_conditional_losses_97282
conv3/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_97502
flatten/PartitionedCall?
d1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0d1_9963d1_9965*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_d1_layer_call_and_return_conditional_losses_97692
d1/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_98022
dropout_2/PartitionedCall?
d2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0d2_9969d2_9971*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_d2_layer_call_and_return_conditional_losses_98262
d2/StatefulPartitionedCall?
dropout_3/PartitionedCallPartitionedCall#d2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_98592
dropout_3/PartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0output_9975output_9977*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_98832 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameimage
?
a
C__inference_dropout_3_layer_call_and_return_conditional_losses_9859

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
z
%__inference_conv1_layer_call_fn_10264

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_conv1_layer_call_and_return_conditional_losses_96722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
w
"__inference_d1_layer_call_fn_10335

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_d1_layer_call_and_return_conditional_losses_97692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
b
)__inference_dropout_2_layer_call_fn_10357

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_97972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
{
&__inference_output_layer_call_fn_10429

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_98832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_9639

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_9797

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?O
?	
__inference__wrapped_model_9633
input_15
1conv_model_1_conv1_conv2d_readvariableop_resource6
2conv_model_1_conv1_biasadd_readvariableop_resource5
1conv_model_1_conv2_conv2d_readvariableop_resource6
2conv_model_1_conv2_biasadd_readvariableop_resource5
1conv_model_1_conv3_conv2d_readvariableop_resource6
2conv_model_1_conv3_biasadd_readvariableop_resource2
.conv_model_1_d1_matmul_readvariableop_resource3
/conv_model_1_d1_biasadd_readvariableop_resource2
.conv_model_1_d2_matmul_readvariableop_resource3
/conv_model_1_d2_biasadd_readvariableop_resource6
2conv_model_1_output_matmul_readvariableop_resource7
3conv_model_1_output_biasadd_readvariableop_resource
identity??)conv_model_1/conv1/BiasAdd/ReadVariableOp?(conv_model_1/conv1/Conv2D/ReadVariableOp?)conv_model_1/conv2/BiasAdd/ReadVariableOp?(conv_model_1/conv2/Conv2D/ReadVariableOp?)conv_model_1/conv3/BiasAdd/ReadVariableOp?(conv_model_1/conv3/Conv2D/ReadVariableOp?&conv_model_1/d1/BiasAdd/ReadVariableOp?%conv_model_1/d1/MatMul/ReadVariableOp?&conv_model_1/d2/BiasAdd/ReadVariableOp?%conv_model_1/d2/MatMul/ReadVariableOp?*conv_model_1/output/BiasAdd/ReadVariableOp?)conv_model_1/output/MatMul/ReadVariableOp?
(conv_model_1/conv1/Conv2D/ReadVariableOpReadVariableOp1conv_model_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02*
(conv_model_1/conv1/Conv2D/ReadVariableOp?
conv_model_1/conv1/Conv2DConv2Dinput_10conv_model_1/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv_model_1/conv1/Conv2D?
)conv_model_1/conv1/BiasAdd/ReadVariableOpReadVariableOp2conv_model_1_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv_model_1/conv1/BiasAdd/ReadVariableOp?
conv_model_1/conv1/BiasAddBiasAdd"conv_model_1/conv1/Conv2D:output:01conv_model_1/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv_model_1/conv1/BiasAdd?
conv_model_1/conv1/ReluRelu#conv_model_1/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv_model_1/conv1/Relu?
$conv_model_1/max_pooling2d_2/MaxPoolMaxPool%conv_model_1/conv1/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2&
$conv_model_1/max_pooling2d_2/MaxPool?
(conv_model_1/conv2/Conv2D/ReadVariableOpReadVariableOp1conv_model_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02*
(conv_model_1/conv2/Conv2D/ReadVariableOp?
conv_model_1/conv2/Conv2DConv2D-conv_model_1/max_pooling2d_2/MaxPool:output:00conv_model_1/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv_model_1/conv2/Conv2D?
)conv_model_1/conv2/BiasAdd/ReadVariableOpReadVariableOp2conv_model_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv_model_1/conv2/BiasAdd/ReadVariableOp?
conv_model_1/conv2/BiasAddBiasAdd"conv_model_1/conv2/Conv2D:output:01conv_model_1/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv_model_1/conv2/BiasAdd?
conv_model_1/conv2/ReluRelu#conv_model_1/conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv_model_1/conv2/Relu?
$conv_model_1/max_pooling2d_3/MaxPoolMaxPool%conv_model_1/conv2/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2&
$conv_model_1/max_pooling2d_3/MaxPool?
(conv_model_1/conv3/Conv2D/ReadVariableOpReadVariableOp1conv_model_1_conv3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02*
(conv_model_1/conv3/Conv2D/ReadVariableOp?
conv_model_1/conv3/Conv2DConv2D-conv_model_1/max_pooling2d_3/MaxPool:output:00conv_model_1/conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv_model_1/conv3/Conv2D?
)conv_model_1/conv3/BiasAdd/ReadVariableOpReadVariableOp2conv_model_1_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)conv_model_1/conv3/BiasAdd/ReadVariableOp?
conv_model_1/conv3/BiasAddBiasAdd"conv_model_1/conv3/Conv2D:output:01conv_model_1/conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv_model_1/conv3/BiasAdd?
conv_model_1/conv3/ReluRelu#conv_model_1/conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv_model_1/conv3/Relu?
conv_model_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
conv_model_1/flatten/Const?
conv_model_1/flatten/ReshapeReshape%conv_model_1/conv3/Relu:activations:0#conv_model_1/flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
conv_model_1/flatten/Reshape?
%conv_model_1/d1/MatMul/ReadVariableOpReadVariableOp.conv_model_1_d1_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02'
%conv_model_1/d1/MatMul/ReadVariableOp?
conv_model_1/d1/MatMulMatMul%conv_model_1/flatten/Reshape:output:0-conv_model_1/d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
conv_model_1/d1/MatMul?
&conv_model_1/d1/BiasAdd/ReadVariableOpReadVariableOp/conv_model_1_d1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&conv_model_1/d1/BiasAdd/ReadVariableOp?
conv_model_1/d1/BiasAddBiasAdd conv_model_1/d1/MatMul:product:0.conv_model_1/d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
conv_model_1/d1/BiasAdd?
conv_model_1/d1/ReluRelu conv_model_1/d1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
conv_model_1/d1/Relu?
conv_model_1/dropout_2/IdentityIdentity"conv_model_1/d1/Relu:activations:0*
T0*(
_output_shapes
:??????????2!
conv_model_1/dropout_2/Identity?
%conv_model_1/d2/MatMul/ReadVariableOpReadVariableOp.conv_model_1_d2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%conv_model_1/d2/MatMul/ReadVariableOp?
conv_model_1/d2/MatMulMatMul(conv_model_1/dropout_2/Identity:output:0-conv_model_1/d2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
conv_model_1/d2/MatMul?
&conv_model_1/d2/BiasAdd/ReadVariableOpReadVariableOp/conv_model_1_d2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&conv_model_1/d2/BiasAdd/ReadVariableOp?
conv_model_1/d2/BiasAddBiasAdd conv_model_1/d2/MatMul:product:0.conv_model_1/d2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
conv_model_1/d2/BiasAdd?
conv_model_1/d2/ReluRelu conv_model_1/d2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
conv_model_1/d2/Relu?
conv_model_1/dropout_3/IdentityIdentity"conv_model_1/d2/Relu:activations:0*
T0*(
_output_shapes
:??????????2!
conv_model_1/dropout_3/Identity?
)conv_model_1/output/MatMul/ReadVariableOpReadVariableOp2conv_model_1_output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)conv_model_1/output/MatMul/ReadVariableOp?
conv_model_1/output/MatMulMatMul(conv_model_1/dropout_3/Identity:output:01conv_model_1/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
conv_model_1/output/MatMul?
*conv_model_1/output/BiasAdd/ReadVariableOpReadVariableOp3conv_model_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv_model_1/output/BiasAdd/ReadVariableOp?
conv_model_1/output/BiasAddBiasAdd$conv_model_1/output/MatMul:product:02conv_model_1/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
conv_model_1/output/BiasAdd?
conv_model_1/output/SoftmaxSoftmax$conv_model_1/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
conv_model_1/output/Softmax?
IdentityIdentity%conv_model_1/output/Softmax:softmax:0*^conv_model_1/conv1/BiasAdd/ReadVariableOp)^conv_model_1/conv1/Conv2D/ReadVariableOp*^conv_model_1/conv2/BiasAdd/ReadVariableOp)^conv_model_1/conv2/Conv2D/ReadVariableOp*^conv_model_1/conv3/BiasAdd/ReadVariableOp)^conv_model_1/conv3/Conv2D/ReadVariableOp'^conv_model_1/d1/BiasAdd/ReadVariableOp&^conv_model_1/d1/MatMul/ReadVariableOp'^conv_model_1/d2/BiasAdd/ReadVariableOp&^conv_model_1/d2/MatMul/ReadVariableOp+^conv_model_1/output/BiasAdd/ReadVariableOp*^conv_model_1/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::2V
)conv_model_1/conv1/BiasAdd/ReadVariableOp)conv_model_1/conv1/BiasAdd/ReadVariableOp2T
(conv_model_1/conv1/Conv2D/ReadVariableOp(conv_model_1/conv1/Conv2D/ReadVariableOp2V
)conv_model_1/conv2/BiasAdd/ReadVariableOp)conv_model_1/conv2/BiasAdd/ReadVariableOp2T
(conv_model_1/conv2/Conv2D/ReadVariableOp(conv_model_1/conv2/Conv2D/ReadVariableOp2V
)conv_model_1/conv3/BiasAdd/ReadVariableOp)conv_model_1/conv3/BiasAdd/ReadVariableOp2T
(conv_model_1/conv3/Conv2D/ReadVariableOp(conv_model_1/conv3/Conv2D/ReadVariableOp2P
&conv_model_1/d1/BiasAdd/ReadVariableOp&conv_model_1/d1/BiasAdd/ReadVariableOp2N
%conv_model_1/d1/MatMul/ReadVariableOp%conv_model_1/d1/MatMul/ReadVariableOp2P
&conv_model_1/d2/BiasAdd/ReadVariableOp&conv_model_1/d2/BiasAdd/ReadVariableOp2N
%conv_model_1/d2/MatMul/ReadVariableOp%conv_model_1/d2/MatMul/ReadVariableOp2X
*conv_model_1/output/BiasAdd/ReadVariableOp*conv_model_1/output/BiasAdd/ReadVariableOp2V
)conv_model_1/output/MatMul/ReadVariableOp)conv_model_1/output/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?-
?
F__inference_conv_model_1_layer_call_and_return_conditional_losses_9939
input_1

conv1_9903

conv1_9905

conv2_9909

conv2_9911

conv3_9915

conv3_9917
d1_9921
d1_9923
d2_9927
d2_9929
output_9933
output_9935
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?d1/StatefulPartitionedCall?d2/StatefulPartitionedCall?output/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1
conv1_9903
conv1_9905*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_conv1_layer_call_and_return_conditional_losses_96722
conv1/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_96392!
max_pooling2d_2/PartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0
conv2_9909
conv2_9911*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_conv2_layer_call_and_return_conditional_losses_97002
conv2/StatefulPartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_96512!
max_pooling2d_3/PartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0
conv3_9915
conv3_9917*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_conv3_layer_call_and_return_conditional_losses_97282
conv3/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_97502
flatten/PartitionedCall?
d1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0d1_9921d1_9923*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_d1_layer_call_and_return_conditional_losses_97692
d1/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_98022
dropout_2/PartitionedCall?
d2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0d2_9927d2_9929*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_d2_layer_call_and_return_conditional_losses_98262
d2/StatefulPartitionedCall?
dropout_3/PartitionedCallPartitionedCall#d2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_98592
dropout_3/PartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0output_9933output_9935*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_98832 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
,__inference_conv_model_1_layer_call_fn_10244	
image
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallimageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv_model_1_layer_call_and_return_conditional_losses_99812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameimage
?5
?
!__inference__traced_restore_10534
file_prefix.
*assignvariableop_conv_model_1_conv1_kernel.
*assignvariableop_1_conv_model_1_conv1_bias0
,assignvariableop_2_conv_model_1_conv2_kernel.
*assignvariableop_3_conv_model_1_conv2_bias0
,assignvariableop_4_conv_model_1_conv3_kernel.
*assignvariableop_5_conv_model_1_conv3_bias-
)assignvariableop_6_conv_model_1_d1_kernel+
'assignvariableop_7_conv_model_1_d1_bias-
)assignvariableop_8_conv_model_1_d2_kernel+
'assignvariableop_9_conv_model_1_d2_bias2
.assignvariableop_10_conv_model_1_output_kernel0
,assignvariableop_11_conv_model_1_output_bias
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB%out/kernel/.ATTRIBUTES/VARIABLE_VALUEB#out/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp*assignvariableop_conv_model_1_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp*assignvariableop_1_conv_model_1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_conv_model_1_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp*assignvariableop_3_conv_model_1_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp,assignvariableop_4_conv_model_1_conv3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp*assignvariableop_5_conv_model_1_conv3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp)assignvariableop_6_conv_model_1_d1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp'assignvariableop_7_conv_model_1_d1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp)assignvariableop_8_conv_model_1_d2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp'assignvariableop_9_conv_model_1_d2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp.assignvariableop_10_conv_model_1_output_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp,assignvariableop_11_conv_model_1_output_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12?
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*E
_input_shapes4
2: ::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
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
?
z
%__inference_conv2_layer_call_fn_10284

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_conv2_layer_call_and_return_conditional_losses_97002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_3_layer_call_fn_9657

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_96512
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
@__inference_conv3_layer_call_and_return_conditional_losses_10295

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_10399

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_9651

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
	conv1
	pool1
	conv2
	pool2
	conv3
flatten
d1
dop1
	d2

dop2
out
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?
_tf_keras_model?{"class_name": "ConvModel", "name": "conv_model_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ConvModel"}}
?	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 3]}}
?
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 32]}}
?
!trainable_variables
"	variables
#regularization_losses
$	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

%kernel
&bias
'trainable_variables
(	variables
)regularization_losses
*	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 64]}}
?
+trainable_variables
,	variables
-regularization_losses
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

/kernel
0bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "d1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "d1", "trainable": true, "dtype": "float32", "units": 300, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}
?
5trainable_variables
6	variables
7regularization_losses
8	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

9kernel
:bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "d2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "d2", "trainable": true, "dtype": "float32", "units": 300, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300]}}
?
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

Ckernel
Dbias
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300]}}
v
0
1
2
3
%4
&5
/6
07
98
:9
C10
D11"
trackable_list_wrapper
v
0
1
2
3
%4
&5
/6
07
98
:9
C10
D11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Inon_trainable_variables

Jlayers
trainable_variables
Kmetrics
Llayer_metrics
Mlayer_regularization_losses
	variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
3:1 2conv_model_1/conv1/kernel
%:# 2conv_model_1/conv1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Nnon_trainable_variables

Olayers
trainable_variables
Pmetrics
Qlayer_metrics
Rlayer_regularization_losses
	variables
regularization_losses
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
Snon_trainable_variables

Tlayers
trainable_variables
Umetrics
Vlayer_metrics
Wlayer_regularization_losses
	variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
3:1 @2conv_model_1/conv2/kernel
%:#@2conv_model_1/conv2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xnon_trainable_variables

Ylayers
trainable_variables
Zmetrics
[layer_metrics
\layer_regularization_losses
	variables
regularization_losses
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
]non_trainable_variables

^layers
!trainable_variables
_metrics
`layer_metrics
alayer_regularization_losses
"	variables
#regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:2@?2conv_model_1/conv3/kernel
&:$?2conv_model_1/conv3/bias
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
bnon_trainable_variables

clayers
'trainable_variables
dmetrics
elayer_metrics
flayer_regularization_losses
(	variables
)regularization_losses
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
gnon_trainable_variables

hlayers
+trainable_variables
imetrics
jlayer_metrics
klayer_regularization_losses
,	variables
-regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(
?	?2conv_model_1/d1/kernel
#:!?2conv_model_1/d1/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
lnon_trainable_variables

mlayers
1trainable_variables
nmetrics
olayer_metrics
player_regularization_losses
2	variables
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
qnon_trainable_variables

rlayers
5trainable_variables
smetrics
tlayer_metrics
ulayer_regularization_losses
6	variables
7regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(
??2conv_model_1/d2/kernel
#:!?2conv_model_1/d2/bias
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
vnon_trainable_variables

wlayers
;trainable_variables
xmetrics
ylayer_metrics
zlayer_regularization_losses
<	variables
=regularization_losses
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
{non_trainable_variables

|layers
?trainable_variables
}metrics
~layer_metrics
layer_regularization_losses
@	variables
Aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+	?2conv_model_1/output/kernel
&:$2conv_model_1/output/bias
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
Etrainable_variables
?metrics
?layer_metrics
 ?layer_regularization_losses
F	variables
Gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
n
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
10"
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
?2?
F__inference_conv_model_1_layer_call_and_return_conditional_losses_9900
G__inference_conv_model_1_layer_call_and_return_conditional_losses_10186
F__inference_conv_model_1_layer_call_and_return_conditional_losses_9939
G__inference_conv_model_1_layer_call_and_return_conditional_losses_10134?
???
FullArgSpec(
args ?
jself
jimage

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv_model_1_layer_call_fn_10215
,__inference_conv_model_1_layer_call_fn_10008
,__inference_conv_model_1_layer_call_fn_10037
,__inference_conv_model_1_layer_call_fn_10244?
???
FullArgSpec(
args ?
jself
jimage

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__wrapped_model_9633?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????
?2?
@__inference_conv1_layer_call_and_return_conditional_losses_10255?
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
%__inference_conv1_layer_call_fn_10264?
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
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_9639?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
.__inference_max_pooling2d_2_layer_call_fn_9645?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
@__inference_conv2_layer_call_and_return_conditional_losses_10275?
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
%__inference_conv2_layer_call_fn_10284?
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
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_9651?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
.__inference_max_pooling2d_3_layer_call_fn_9657?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
@__inference_conv3_layer_call_and_return_conditional_losses_10295?
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
%__inference_conv3_layer_call_fn_10304?
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
B__inference_flatten_layer_call_and_return_conditional_losses_10310?
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
'__inference_flatten_layer_call_fn_10315?
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
=__inference_d1_layer_call_and_return_conditional_losses_10326?
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
"__inference_d1_layer_call_fn_10335?
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
D__inference_dropout_2_layer_call_and_return_conditional_losses_10352
D__inference_dropout_2_layer_call_and_return_conditional_losses_10347?
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
)__inference_dropout_2_layer_call_fn_10357
)__inference_dropout_2_layer_call_fn_10362?
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
=__inference_d2_layer_call_and_return_conditional_losses_10373?
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
"__inference_d2_layer_call_fn_10382?
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
D__inference_dropout_3_layer_call_and_return_conditional_losses_10399
D__inference_dropout_3_layer_call_and_return_conditional_losses_10394?
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
)__inference_dropout_3_layer_call_fn_10404
)__inference_dropout_3_layer_call_fn_10409?
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
A__inference_output_layer_call_and_return_conditional_losses_10420?
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
&__inference_output_layer_call_fn_10429?
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
#__inference_signature_wrapper_10068input_1"?
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
__inference__wrapped_model_9633}%&/09:CD8?5
.?+
)?&
input_1?????????
? "3?0
.
output_1"?
output_1??????????
@__inference_conv1_layer_call_and_return_conditional_losses_10255l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
%__inference_conv1_layer_call_fn_10264_7?4
-?*
(?%
inputs?????????
? " ?????????? ?
@__inference_conv2_layer_call_and_return_conditional_losses_10275l7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
%__inference_conv2_layer_call_fn_10284_7?4
-?*
(?%
inputs????????? 
? " ??????????@?
@__inference_conv3_layer_call_and_return_conditional_losses_10295m%&7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
%__inference_conv3_layer_call_fn_10304`%&7?4
-?*
(?%
inputs?????????@
? "!????????????
G__inference_conv_model_1_layer_call_and_return_conditional_losses_10134q%&/09:CD:?7
0?-
'?$
image?????????
p
? "%?"
?
0?????????
? ?
G__inference_conv_model_1_layer_call_and_return_conditional_losses_10186q%&/09:CD:?7
0?-
'?$
image?????????
p 
? "%?"
?
0?????????
? ?
F__inference_conv_model_1_layer_call_and_return_conditional_losses_9900s%&/09:CD<?9
2?/
)?&
input_1?????????
p
? "%?"
?
0?????????
? ?
F__inference_conv_model_1_layer_call_and_return_conditional_losses_9939s%&/09:CD<?9
2?/
)?&
input_1?????????
p 
? "%?"
?
0?????????
? ?
,__inference_conv_model_1_layer_call_fn_10008f%&/09:CD<?9
2?/
)?&
input_1?????????
p
? "???????????
,__inference_conv_model_1_layer_call_fn_10037f%&/09:CD<?9
2?/
)?&
input_1?????????
p 
? "???????????
,__inference_conv_model_1_layer_call_fn_10215d%&/09:CD:?7
0?-
'?$
image?????????
p
? "???????????
,__inference_conv_model_1_layer_call_fn_10244d%&/09:CD:?7
0?-
'?$
image?????????
p 
? "???????????
=__inference_d1_layer_call_and_return_conditional_losses_10326^/00?-
&?#
!?
inputs??????????	
? "&?#
?
0??????????
? w
"__inference_d1_layer_call_fn_10335Q/00?-
&?#
!?
inputs??????????	
? "????????????
=__inference_d2_layer_call_and_return_conditional_losses_10373^9:0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? w
"__inference_d2_layer_call_fn_10382Q9:0?-
&?#
!?
inputs??????????
? "????????????
D__inference_dropout_2_layer_call_and_return_conditional_losses_10347^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_10352^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ~
)__inference_dropout_2_layer_call_fn_10357Q4?1
*?'
!?
inputs??????????
p
? "???????????~
)__inference_dropout_2_layer_call_fn_10362Q4?1
*?'
!?
inputs??????????
p 
? "????????????
D__inference_dropout_3_layer_call_and_return_conditional_losses_10394^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
D__inference_dropout_3_layer_call_and_return_conditional_losses_10399^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ~
)__inference_dropout_3_layer_call_fn_10404Q4?1
*?'
!?
inputs??????????
p
? "???????????~
)__inference_dropout_3_layer_call_fn_10409Q4?1
*?'
!?
inputs??????????
p 
? "????????????
B__inference_flatten_layer_call_and_return_conditional_losses_10310b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????	
? ?
'__inference_flatten_layer_call_fn_10315U8?5
.?+
)?&
inputs??????????
? "???????????	?
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_9639?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_max_pooling2d_2_layer_call_fn_9645?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_9651?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_max_pooling2d_3_layer_call_fn_9657?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
A__inference_output_layer_call_and_return_conditional_losses_10420]CD0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? z
&__inference_output_layer_call_fn_10429PCD0?-
&?#
!?
inputs??????????
? "???????????
#__inference_signature_wrapper_10068?%&/09:CDC?@
? 
9?6
4
input_1)?&
input_1?????????"3?0
.
output_1"?
output_1?????????