??7
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
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
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
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
 ?"serve*2.4.12unknown8??*
|
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
:@*
dtype0
j
	bn1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	bn1/gamma
c
bn1/gamma/Read/ReadVariableOpReadVariableOp	bn1/gamma*
_output_shapes
:@*
dtype0
h
bn1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
bn1/beta
a
bn1/beta/Read/ReadVariableOpReadVariableOpbn1/beta*
_output_shapes
:@*
dtype0
v
bn1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_namebn1/moving_mean
o
#bn1/moving_mean/Read/ReadVariableOpReadVariableOpbn1/moving_mean*
_output_shapes
:@*
dtype0
~
bn1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namebn1/moving_variance
w
'bn1/moving_variance/Read/ReadVariableOpReadVariableOpbn1/moving_variance*
_output_shapes
:@*
dtype0
?
layer1.0.conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*&
shared_namelayer1.0.conv1/kernel
?
)layer1.0.conv1/kernel/Read/ReadVariableOpReadVariableOplayer1.0.conv1/kernel*&
_output_shapes
:@@*
dtype0
|
layer1.0.bn1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namelayer1.0.bn1/gamma
u
&layer1.0.bn1/gamma/Read/ReadVariableOpReadVariableOplayer1.0.bn1/gamma*
_output_shapes
:@*
dtype0
z
layer1.0.bn1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namelayer1.0.bn1/beta
s
%layer1.0.bn1/beta/Read/ReadVariableOpReadVariableOplayer1.0.bn1/beta*
_output_shapes
:@*
dtype0
?
layer1.0.bn1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namelayer1.0.bn1/moving_mean
?
,layer1.0.bn1/moving_mean/Read/ReadVariableOpReadVariableOplayer1.0.bn1/moving_mean*
_output_shapes
:@*
dtype0
?
layer1.0.bn1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namelayer1.0.bn1/moving_variance
?
0layer1.0.bn1/moving_variance/Read/ReadVariableOpReadVariableOplayer1.0.bn1/moving_variance*
_output_shapes
:@*
dtype0
?
layer1.0.conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*&
shared_namelayer1.0.conv2/kernel
?
)layer1.0.conv2/kernel/Read/ReadVariableOpReadVariableOplayer1.0.conv2/kernel*&
_output_shapes
:@@*
dtype0
|
layer1.0.bn2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namelayer1.0.bn2/gamma
u
&layer1.0.bn2/gamma/Read/ReadVariableOpReadVariableOplayer1.0.bn2/gamma*
_output_shapes
:@*
dtype0
z
layer1.0.bn2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namelayer1.0.bn2/beta
s
%layer1.0.bn2/beta/Read/ReadVariableOpReadVariableOplayer1.0.bn2/beta*
_output_shapes
:@*
dtype0
?
layer1.0.bn2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namelayer1.0.bn2/moving_mean
?
,layer1.0.bn2/moving_mean/Read/ReadVariableOpReadVariableOplayer1.0.bn2/moving_mean*
_output_shapes
:@*
dtype0
?
layer1.0.bn2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namelayer1.0.bn2/moving_variance
?
0layer1.0.bn2/moving_variance/Read/ReadVariableOpReadVariableOplayer1.0.bn2/moving_variance*
_output_shapes
:@*
dtype0
?
layer1.1.conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*&
shared_namelayer1.1.conv1/kernel
?
)layer1.1.conv1/kernel/Read/ReadVariableOpReadVariableOplayer1.1.conv1/kernel*&
_output_shapes
:@@*
dtype0
|
layer1.1.bn1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namelayer1.1.bn1/gamma
u
&layer1.1.bn1/gamma/Read/ReadVariableOpReadVariableOplayer1.1.bn1/gamma*
_output_shapes
:@*
dtype0
z
layer1.1.bn1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namelayer1.1.bn1/beta
s
%layer1.1.bn1/beta/Read/ReadVariableOpReadVariableOplayer1.1.bn1/beta*
_output_shapes
:@*
dtype0
?
layer1.1.bn1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namelayer1.1.bn1/moving_mean
?
,layer1.1.bn1/moving_mean/Read/ReadVariableOpReadVariableOplayer1.1.bn1/moving_mean*
_output_shapes
:@*
dtype0
?
layer1.1.bn1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namelayer1.1.bn1/moving_variance
?
0layer1.1.bn1/moving_variance/Read/ReadVariableOpReadVariableOplayer1.1.bn1/moving_variance*
_output_shapes
:@*
dtype0
?
layer1.1.conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*&
shared_namelayer1.1.conv2/kernel
?
)layer1.1.conv2/kernel/Read/ReadVariableOpReadVariableOplayer1.1.conv2/kernel*&
_output_shapes
:@@*
dtype0
|
layer1.1.bn2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namelayer1.1.bn2/gamma
u
&layer1.1.bn2/gamma/Read/ReadVariableOpReadVariableOplayer1.1.bn2/gamma*
_output_shapes
:@*
dtype0
z
layer1.1.bn2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namelayer1.1.bn2/beta
s
%layer1.1.bn2/beta/Read/ReadVariableOpReadVariableOplayer1.1.bn2/beta*
_output_shapes
:@*
dtype0
?
layer1.1.bn2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namelayer1.1.bn2/moving_mean
?
,layer1.1.bn2/moving_mean/Read/ReadVariableOpReadVariableOplayer1.1.bn2/moving_mean*
_output_shapes
:@*
dtype0
?
layer1.1.bn2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namelayer1.1.bn2/moving_variance
?
0layer1.1.bn2/moving_variance/Read/ReadVariableOpReadVariableOplayer1.1.bn2/moving_variance*
_output_shapes
:@*
dtype0
?
layer2.0.conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*&
shared_namelayer2.0.conv1/kernel
?
)layer2.0.conv1/kernel/Read/ReadVariableOpReadVariableOplayer2.0.conv1/kernel*'
_output_shapes
:@?*
dtype0
}
layer2.0.bn1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_namelayer2.0.bn1/gamma
v
&layer2.0.bn1/gamma/Read/ReadVariableOpReadVariableOplayer2.0.bn1/gamma*
_output_shapes	
:?*
dtype0
{
layer2.0.bn1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namelayer2.0.bn1/beta
t
%layer2.0.bn1/beta/Read/ReadVariableOpReadVariableOplayer2.0.bn1/beta*
_output_shapes	
:?*
dtype0
?
layer2.0.bn1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namelayer2.0.bn1/moving_mean
?
,layer2.0.bn1/moving_mean/Read/ReadVariableOpReadVariableOplayer2.0.bn1/moving_mean*
_output_shapes	
:?*
dtype0
?
layer2.0.bn1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namelayer2.0.bn1/moving_variance
?
0layer2.0.bn1/moving_variance/Read/ReadVariableOpReadVariableOplayer2.0.bn1/moving_variance*
_output_shapes	
:?*
dtype0
?
layer2.0.downsample.0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*-
shared_namelayer2.0.downsample.0/kernel
?
0layer2.0.downsample.0/kernel/Read/ReadVariableOpReadVariableOplayer2.0.downsample.0/kernel*'
_output_shapes
:@?*
dtype0
?
layer2.0.conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*&
shared_namelayer2.0.conv2/kernel
?
)layer2.0.conv2/kernel/Read/ReadVariableOpReadVariableOplayer2.0.conv2/kernel*(
_output_shapes
:??*
dtype0
?
layer2.0.downsample.1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namelayer2.0.downsample.1/gamma
?
/layer2.0.downsample.1/gamma/Read/ReadVariableOpReadVariableOplayer2.0.downsample.1/gamma*
_output_shapes	
:?*
dtype0
?
layer2.0.downsample.1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namelayer2.0.downsample.1/beta
?
.layer2.0.downsample.1/beta/Read/ReadVariableOpReadVariableOplayer2.0.downsample.1/beta*
_output_shapes	
:?*
dtype0
?
!layer2.0.downsample.1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!layer2.0.downsample.1/moving_mean
?
5layer2.0.downsample.1/moving_mean/Read/ReadVariableOpReadVariableOp!layer2.0.downsample.1/moving_mean*
_output_shapes	
:?*
dtype0
?
%layer2.0.downsample.1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%layer2.0.downsample.1/moving_variance
?
9layer2.0.downsample.1/moving_variance/Read/ReadVariableOpReadVariableOp%layer2.0.downsample.1/moving_variance*
_output_shapes	
:?*
dtype0
}
layer2.0.bn2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_namelayer2.0.bn2/gamma
v
&layer2.0.bn2/gamma/Read/ReadVariableOpReadVariableOplayer2.0.bn2/gamma*
_output_shapes	
:?*
dtype0
{
layer2.0.bn2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namelayer2.0.bn2/beta
t
%layer2.0.bn2/beta/Read/ReadVariableOpReadVariableOplayer2.0.bn2/beta*
_output_shapes	
:?*
dtype0
?
layer2.0.bn2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namelayer2.0.bn2/moving_mean
?
,layer2.0.bn2/moving_mean/Read/ReadVariableOpReadVariableOplayer2.0.bn2/moving_mean*
_output_shapes	
:?*
dtype0
?
layer2.0.bn2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namelayer2.0.bn2/moving_variance
?
0layer2.0.bn2/moving_variance/Read/ReadVariableOpReadVariableOplayer2.0.bn2/moving_variance*
_output_shapes	
:?*
dtype0
?
layer2.1.conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*&
shared_namelayer2.1.conv1/kernel
?
)layer2.1.conv1/kernel/Read/ReadVariableOpReadVariableOplayer2.1.conv1/kernel*(
_output_shapes
:??*
dtype0
}
layer2.1.bn1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_namelayer2.1.bn1/gamma
v
&layer2.1.bn1/gamma/Read/ReadVariableOpReadVariableOplayer2.1.bn1/gamma*
_output_shapes	
:?*
dtype0
{
layer2.1.bn1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namelayer2.1.bn1/beta
t
%layer2.1.bn1/beta/Read/ReadVariableOpReadVariableOplayer2.1.bn1/beta*
_output_shapes	
:?*
dtype0
?
layer2.1.bn1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namelayer2.1.bn1/moving_mean
?
,layer2.1.bn1/moving_mean/Read/ReadVariableOpReadVariableOplayer2.1.bn1/moving_mean*
_output_shapes	
:?*
dtype0
?
layer2.1.bn1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namelayer2.1.bn1/moving_variance
?
0layer2.1.bn1/moving_variance/Read/ReadVariableOpReadVariableOplayer2.1.bn1/moving_variance*
_output_shapes	
:?*
dtype0
?
layer2.1.conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*&
shared_namelayer2.1.conv2/kernel
?
)layer2.1.conv2/kernel/Read/ReadVariableOpReadVariableOplayer2.1.conv2/kernel*(
_output_shapes
:??*
dtype0
}
layer2.1.bn2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_namelayer2.1.bn2/gamma
v
&layer2.1.bn2/gamma/Read/ReadVariableOpReadVariableOplayer2.1.bn2/gamma*
_output_shapes	
:?*
dtype0
{
layer2.1.bn2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namelayer2.1.bn2/beta
t
%layer2.1.bn2/beta/Read/ReadVariableOpReadVariableOplayer2.1.bn2/beta*
_output_shapes	
:?*
dtype0
?
layer2.1.bn2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namelayer2.1.bn2/moving_mean
?
,layer2.1.bn2/moving_mean/Read/ReadVariableOpReadVariableOplayer2.1.bn2/moving_mean*
_output_shapes	
:?*
dtype0
?
layer2.1.bn2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namelayer2.1.bn2/moving_variance
?
0layer2.1.bn2/moving_variance/Read/ReadVariableOpReadVariableOplayer2.1.bn2/moving_variance*
_output_shapes	
:?*
dtype0
?
layer3.0.conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*&
shared_namelayer3.0.conv1/kernel
?
)layer3.0.conv1/kernel/Read/ReadVariableOpReadVariableOplayer3.0.conv1/kernel*(
_output_shapes
:??*
dtype0
}
layer3.0.bn1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_namelayer3.0.bn1/gamma
v
&layer3.0.bn1/gamma/Read/ReadVariableOpReadVariableOplayer3.0.bn1/gamma*
_output_shapes	
:?*
dtype0
{
layer3.0.bn1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namelayer3.0.bn1/beta
t
%layer3.0.bn1/beta/Read/ReadVariableOpReadVariableOplayer3.0.bn1/beta*
_output_shapes	
:?*
dtype0
?
layer3.0.bn1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namelayer3.0.bn1/moving_mean
?
,layer3.0.bn1/moving_mean/Read/ReadVariableOpReadVariableOplayer3.0.bn1/moving_mean*
_output_shapes	
:?*
dtype0
?
layer3.0.bn1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namelayer3.0.bn1/moving_variance
?
0layer3.0.bn1/moving_variance/Read/ReadVariableOpReadVariableOplayer3.0.bn1/moving_variance*
_output_shapes	
:?*
dtype0
?
layer3.0.downsample.0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*-
shared_namelayer3.0.downsample.0/kernel
?
0layer3.0.downsample.0/kernel/Read/ReadVariableOpReadVariableOplayer3.0.downsample.0/kernel*(
_output_shapes
:??*
dtype0
?
layer3.0.conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*&
shared_namelayer3.0.conv2/kernel
?
)layer3.0.conv2/kernel/Read/ReadVariableOpReadVariableOplayer3.0.conv2/kernel*(
_output_shapes
:??*
dtype0
?
layer3.0.downsample.1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namelayer3.0.downsample.1/gamma
?
/layer3.0.downsample.1/gamma/Read/ReadVariableOpReadVariableOplayer3.0.downsample.1/gamma*
_output_shapes	
:?*
dtype0
?
layer3.0.downsample.1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namelayer3.0.downsample.1/beta
?
.layer3.0.downsample.1/beta/Read/ReadVariableOpReadVariableOplayer3.0.downsample.1/beta*
_output_shapes	
:?*
dtype0
?
!layer3.0.downsample.1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!layer3.0.downsample.1/moving_mean
?
5layer3.0.downsample.1/moving_mean/Read/ReadVariableOpReadVariableOp!layer3.0.downsample.1/moving_mean*
_output_shapes	
:?*
dtype0
?
%layer3.0.downsample.1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%layer3.0.downsample.1/moving_variance
?
9layer3.0.downsample.1/moving_variance/Read/ReadVariableOpReadVariableOp%layer3.0.downsample.1/moving_variance*
_output_shapes	
:?*
dtype0
}
layer3.0.bn2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_namelayer3.0.bn2/gamma
v
&layer3.0.bn2/gamma/Read/ReadVariableOpReadVariableOplayer3.0.bn2/gamma*
_output_shapes	
:?*
dtype0
{
layer3.0.bn2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namelayer3.0.bn2/beta
t
%layer3.0.bn2/beta/Read/ReadVariableOpReadVariableOplayer3.0.bn2/beta*
_output_shapes	
:?*
dtype0
?
layer3.0.bn2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namelayer3.0.bn2/moving_mean
?
,layer3.0.bn2/moving_mean/Read/ReadVariableOpReadVariableOplayer3.0.bn2/moving_mean*
_output_shapes	
:?*
dtype0
?
layer3.0.bn2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namelayer3.0.bn2/moving_variance
?
0layer3.0.bn2/moving_variance/Read/ReadVariableOpReadVariableOplayer3.0.bn2/moving_variance*
_output_shapes	
:?*
dtype0
?
layer3.1.conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*&
shared_namelayer3.1.conv1/kernel
?
)layer3.1.conv1/kernel/Read/ReadVariableOpReadVariableOplayer3.1.conv1/kernel*(
_output_shapes
:??*
dtype0
}
layer3.1.bn1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_namelayer3.1.bn1/gamma
v
&layer3.1.bn1/gamma/Read/ReadVariableOpReadVariableOplayer3.1.bn1/gamma*
_output_shapes	
:?*
dtype0
{
layer3.1.bn1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namelayer3.1.bn1/beta
t
%layer3.1.bn1/beta/Read/ReadVariableOpReadVariableOplayer3.1.bn1/beta*
_output_shapes	
:?*
dtype0
?
layer3.1.bn1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namelayer3.1.bn1/moving_mean
?
,layer3.1.bn1/moving_mean/Read/ReadVariableOpReadVariableOplayer3.1.bn1/moving_mean*
_output_shapes	
:?*
dtype0
?
layer3.1.bn1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namelayer3.1.bn1/moving_variance
?
0layer3.1.bn1/moving_variance/Read/ReadVariableOpReadVariableOplayer3.1.bn1/moving_variance*
_output_shapes	
:?*
dtype0
?
layer3.1.conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*&
shared_namelayer3.1.conv2/kernel
?
)layer3.1.conv2/kernel/Read/ReadVariableOpReadVariableOplayer3.1.conv2/kernel*(
_output_shapes
:??*
dtype0
}
layer3.1.bn2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_namelayer3.1.bn2/gamma
v
&layer3.1.bn2/gamma/Read/ReadVariableOpReadVariableOplayer3.1.bn2/gamma*
_output_shapes	
:?*
dtype0
{
layer3.1.bn2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namelayer3.1.bn2/beta
t
%layer3.1.bn2/beta/Read/ReadVariableOpReadVariableOplayer3.1.bn2/beta*
_output_shapes	
:?*
dtype0
?
layer3.1.bn2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namelayer3.1.bn2/moving_mean
?
,layer3.1.bn2/moving_mean/Read/ReadVariableOpReadVariableOplayer3.1.bn2/moving_mean*
_output_shapes	
:?*
dtype0
?
layer3.1.bn2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namelayer3.1.bn2/moving_variance
?
0layer3.1.bn2/moving_variance/Read/ReadVariableOpReadVariableOplayer3.1.bn2/moving_variance*
_output_shapes	
:?*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
layer_with_weights-9
layer-18
layer-19
layer-20
layer-21
layer_with_weights-10
layer-22
layer_with_weights-11
layer-23
layer-24
layer_with_weights-12
layer-25
layer_with_weights-13
layer-26
layer_with_weights-14
layer-27
layer_with_weights-15
layer-28
layer-29
layer-30
 layer_with_weights-16
 layer-31
!layer_with_weights-17
!layer-32
"layer-33
#layer_with_weights-18
#layer-34
$layer_with_weights-19
$layer-35
%layer-36
&layer-37
'layer-38
(layer_with_weights-20
(layer-39
)layer_with_weights-21
)layer-40
*layer-41
+layer_with_weights-22
+layer-42
,layer_with_weights-23
,layer-43
-layer_with_weights-24
-layer-44
.layer_with_weights-25
.layer-45
/layer-46
0layer-47
1layer_with_weights-26
1layer-48
2layer_with_weights-27
2layer-49
3layer-50
4layer_with_weights-28
4layer-51
5layer_with_weights-29
5layer-52
6layer-53
7layer-54
8regularization_losses
9trainable_variables
:	variables
;	keras_api
<
signatures
%
#=_self_saveable_object_factories
w
#>_self_saveable_object_factories
?regularization_losses
@trainable_variables
A	variables
B	keras_api
?

Ckernel
#D_self_saveable_object_factories
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
?
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
#N_self_saveable_object_factories
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
w
#S_self_saveable_object_factories
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
w
#X_self_saveable_object_factories
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
w
#]_self_saveable_object_factories
^regularization_losses
_trainable_variables
`	variables
a	keras_api
?

bkernel
#c_self_saveable_object_factories
dregularization_losses
etrainable_variables
f	variables
g	keras_api
?
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance
#m_self_saveable_object_factories
nregularization_losses
otrainable_variables
p	variables
q	keras_api
w
#r_self_saveable_object_factories
sregularization_losses
ttrainable_variables
u	variables
v	keras_api
?

wkernel
#x_self_saveable_object_factories
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
?
}axis
	~gamma
beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
?
C0
J1
K2
b3
i4
j5
w6
~7
8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?
C0
J1
K2
L3
M4
b5
i6
j7
k8
l9
w10
~11
12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
?58
?59
?60
?61
?62
?63
?64
?65
?66
?67
?68
?69
?70
?71
?72
?73
?74
?
?layer_metrics
8regularization_losses
?non_trainable_variables
9trainable_variables
?layers
 ?layer_regularization_losses
:	variables
?metrics
 
 
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
@trainable_variables
?layers
 ?layer_regularization_losses
A	variables
?metrics
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

C0

C0
?
?layer_metrics
Eregularization_losses
?non_trainable_variables
Ftrainable_variables
?layers
 ?layer_regularization_losses
G	variables
?metrics
 
TR
VARIABLE_VALUE	bn1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEbn1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbn1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbn1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

J0
K1

J0
K1
L2
M3
?
?layer_metrics
Oregularization_losses
?non_trainable_variables
Ptrainable_variables
?layers
 ?layer_regularization_losses
Q	variables
?metrics
 
 
 
 
?
?layer_metrics
Tregularization_losses
?non_trainable_variables
Utrainable_variables
?layers
 ?layer_regularization_losses
V	variables
?metrics
 
 
 
 
?
?layer_metrics
Yregularization_losses
?non_trainable_variables
Ztrainable_variables
?layers
 ?layer_regularization_losses
[	variables
?metrics
 
 
 
 
?
?layer_metrics
^regularization_losses
?non_trainable_variables
_trainable_variables
?layers
 ?layer_regularization_losses
`	variables
?metrics
a_
VARIABLE_VALUElayer1.0.conv1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

b0

b0
?
?layer_metrics
dregularization_losses
?non_trainable_variables
etrainable_variables
?layers
 ?layer_regularization_losses
f	variables
?metrics
 
][
VARIABLE_VALUElayer1.0.bn1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElayer1.0.bn1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUElayer1.0.bn1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUElayer1.0.bn1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

i0
j1

i0
j1
k2
l3
?
?layer_metrics
nregularization_losses
?non_trainable_variables
otrainable_variables
?layers
 ?layer_regularization_losses
p	variables
?metrics
 
 
 
 
?
?layer_metrics
sregularization_losses
?non_trainable_variables
ttrainable_variables
?layers
 ?layer_regularization_losses
u	variables
?metrics
a_
VARIABLE_VALUElayer1.0.conv2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

w0

w0
?
?layer_metrics
yregularization_losses
?non_trainable_variables
ztrainable_variables
?layers
 ?layer_regularization_losses
{	variables
?metrics
 
][
VARIABLE_VALUElayer1.0.bn2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElayer1.0.bn2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUElayer1.0.bn2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUElayer1.0.bn2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

~0
1

~0
1
?2
?3
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
a_
VARIABLE_VALUElayer1.1.conv1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0

?0
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
][
VARIABLE_VALUElayer1.1.bn1/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElayer1.1.bn1/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUElayer1.1.bn1/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUElayer1.1.bn1/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?0
?1
?2
?3
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
a_
VARIABLE_VALUElayer1.1.conv2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0

?0
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
][
VARIABLE_VALUElayer1.1.bn2/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElayer1.1.bn2/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUElayer1.1.bn2/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUElayer1.1.bn2/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?0
?1
?2
?3
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
b`
VARIABLE_VALUElayer2.0.conv1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0

?0
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
^\
VARIABLE_VALUElayer2.0.bn1/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUElayer2.0.bn1/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUElayer2.0.bn1/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUElayer2.0.bn1/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?0
?1
?2
?3
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
ig
VARIABLE_VALUElayer2.0.downsample.0/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0

?0
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
b`
VARIABLE_VALUElayer2.0.conv2/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0

?0
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
ge
VARIABLE_VALUElayer2.0.downsample.1/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUElayer2.0.downsample.1/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!layer2.0.downsample.1/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%layer2.0.downsample.1/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?0
?1
?2
?3
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
^\
VARIABLE_VALUElayer2.0.bn2/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUElayer2.0.bn2/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUElayer2.0.bn2/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUElayer2.0.bn2/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?0
?1
?2
?3
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
b`
VARIABLE_VALUElayer2.1.conv1/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0

?0
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
^\
VARIABLE_VALUElayer2.1.bn1/gamma6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUElayer2.1.bn1/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUElayer2.1.bn1/moving_mean<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUElayer2.1.bn1/moving_variance@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?0
?1
?2
?3
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
b`
VARIABLE_VALUElayer2.1.conv2/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0

?0
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
^\
VARIABLE_VALUElayer2.1.bn2/gamma6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUElayer2.1.bn2/beta5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUElayer2.1.bn2/moving_mean<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUElayer2.1.bn2/moving_variance@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?0
?1
?2
?3
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
b`
VARIABLE_VALUElayer3.0.conv1/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0

?0
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
^\
VARIABLE_VALUElayer3.0.bn1/gamma6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUElayer3.0.bn1/beta5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUElayer3.0.bn1/moving_mean<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUElayer3.0.bn1/moving_variance@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?0
?1
?2
?3
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
ig
VARIABLE_VALUElayer3.0.downsample.0/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0

?0
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
b`
VARIABLE_VALUElayer3.0.conv2/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0

?0
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
ge
VARIABLE_VALUElayer3.0.downsample.1/gamma6layer_with_weights-24/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUElayer3.0.downsample.1/beta5layer_with_weights-24/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!layer3.0.downsample.1/moving_mean<layer_with_weights-24/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%layer3.0.downsample.1/moving_variance@layer_with_weights-24/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?0
?1
?2
?3
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
^\
VARIABLE_VALUElayer3.0.bn2/gamma6layer_with_weights-25/gamma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUElayer3.0.bn2/beta5layer_with_weights-25/beta/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUElayer3.0.bn2/moving_mean<layer_with_weights-25/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUElayer3.0.bn2/moving_variance@layer_with_weights-25/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?0
?1
?2
?3
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
b`
VARIABLE_VALUElayer3.1.conv1/kernel7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0

?0
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
^\
VARIABLE_VALUElayer3.1.bn1/gamma6layer_with_weights-27/gamma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUElayer3.1.bn1/beta5layer_with_weights-27/beta/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUElayer3.1.bn1/moving_mean<layer_with_weights-27/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUElayer3.1.bn1/moving_variance@layer_with_weights-27/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?0
?1
?2
?3
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
b`
VARIABLE_VALUElayer3.1.conv2/kernel7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0

?0
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
^\
VARIABLE_VALUElayer3.1.bn2/gamma6layer_with_weights-29/gamma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUElayer3.1.bn2/beta5layer_with_weights-29/beta/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUElayer3.1.bn2/moving_mean<layer_with_weights-29/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUElayer3.1.bn2/moving_variance@layer_with_weights-29/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?0
?1
?2
?3
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
 
 
 
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
 
?
L0
M1
k2
l3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?
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
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
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

L0
M1
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

k0
l1
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

?0
?1
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

?0
?1
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

?0
?1
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

?0
?1
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

?0
?1
 
 
 
 

?0
?1
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

?0
?1
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

?0
?1
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

?0
?1
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

?0
?1
 
 
 
 

?0
?1
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

?0
?1
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

?0
?1
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
serving_default_input_1Placeholder*A
_output_shapes/
-:+???????????????????????????*
dtype0*6
shape-:+???????????????????????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv1/kernel	bn1/gammabn1/betabn1/moving_meanbn1/moving_variancelayer1.0.conv1/kernellayer1.0.bn1/gammalayer1.0.bn1/betalayer1.0.bn1/moving_meanlayer1.0.bn1/moving_variancelayer1.0.conv2/kernellayer1.0.bn2/gammalayer1.0.bn2/betalayer1.0.bn2/moving_meanlayer1.0.bn2/moving_variancelayer1.1.conv1/kernellayer1.1.bn1/gammalayer1.1.bn1/betalayer1.1.bn1/moving_meanlayer1.1.bn1/moving_variancelayer1.1.conv2/kernellayer1.1.bn2/gammalayer1.1.bn2/betalayer1.1.bn2/moving_meanlayer1.1.bn2/moving_variancelayer2.0.conv1/kernellayer2.0.bn1/gammalayer2.0.bn1/betalayer2.0.bn1/moving_meanlayer2.0.bn1/moving_variancelayer2.0.conv2/kernellayer2.0.downsample.0/kernellayer2.0.downsample.1/gammalayer2.0.downsample.1/beta!layer2.0.downsample.1/moving_mean%layer2.0.downsample.1/moving_variancelayer2.0.bn2/gammalayer2.0.bn2/betalayer2.0.bn2/moving_meanlayer2.0.bn2/moving_variancelayer2.1.conv1/kernellayer2.1.bn1/gammalayer2.1.bn1/betalayer2.1.bn1/moving_meanlayer2.1.bn1/moving_variancelayer2.1.conv2/kernellayer2.1.bn2/gammalayer2.1.bn2/betalayer2.1.bn2/moving_meanlayer2.1.bn2/moving_variancelayer3.0.conv1/kernellayer3.0.bn1/gammalayer3.0.bn1/betalayer3.0.bn1/moving_meanlayer3.0.bn1/moving_variancelayer3.0.conv2/kernellayer3.0.downsample.0/kernellayer3.0.downsample.1/gammalayer3.0.downsample.1/beta!layer3.0.downsample.1/moving_mean%layer3.0.downsample.1/moving_variancelayer3.0.bn2/gammalayer3.0.bn2/betalayer3.0.bn2/moving_meanlayer3.0.bn2/moving_variancelayer3.1.conv1/kernellayer3.1.bn1/gammalayer3.1.bn1/betalayer3.1.bn1/moving_meanlayer3.1.bn1/moving_variancelayer3.1.conv2/kernellayer3.1.bn2/gammalayer3.1.bn2/betalayer3.1.bn2/moving_meanlayer3.1.bn2/moving_variance*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*m
_read_only_resource_inputsO
MK	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJK*2
config_proto" 

CPU

GPU2 *0J 8? *,
f'R%
#__inference_signature_wrapper_12796
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpbn1/gamma/Read/ReadVariableOpbn1/beta/Read/ReadVariableOp#bn1/moving_mean/Read/ReadVariableOp'bn1/moving_variance/Read/ReadVariableOp)layer1.0.conv1/kernel/Read/ReadVariableOp&layer1.0.bn1/gamma/Read/ReadVariableOp%layer1.0.bn1/beta/Read/ReadVariableOp,layer1.0.bn1/moving_mean/Read/ReadVariableOp0layer1.0.bn1/moving_variance/Read/ReadVariableOp)layer1.0.conv2/kernel/Read/ReadVariableOp&layer1.0.bn2/gamma/Read/ReadVariableOp%layer1.0.bn2/beta/Read/ReadVariableOp,layer1.0.bn2/moving_mean/Read/ReadVariableOp0layer1.0.bn2/moving_variance/Read/ReadVariableOp)layer1.1.conv1/kernel/Read/ReadVariableOp&layer1.1.bn1/gamma/Read/ReadVariableOp%layer1.1.bn1/beta/Read/ReadVariableOp,layer1.1.bn1/moving_mean/Read/ReadVariableOp0layer1.1.bn1/moving_variance/Read/ReadVariableOp)layer1.1.conv2/kernel/Read/ReadVariableOp&layer1.1.bn2/gamma/Read/ReadVariableOp%layer1.1.bn2/beta/Read/ReadVariableOp,layer1.1.bn2/moving_mean/Read/ReadVariableOp0layer1.1.bn2/moving_variance/Read/ReadVariableOp)layer2.0.conv1/kernel/Read/ReadVariableOp&layer2.0.bn1/gamma/Read/ReadVariableOp%layer2.0.bn1/beta/Read/ReadVariableOp,layer2.0.bn1/moving_mean/Read/ReadVariableOp0layer2.0.bn1/moving_variance/Read/ReadVariableOp0layer2.0.downsample.0/kernel/Read/ReadVariableOp)layer2.0.conv2/kernel/Read/ReadVariableOp/layer2.0.downsample.1/gamma/Read/ReadVariableOp.layer2.0.downsample.1/beta/Read/ReadVariableOp5layer2.0.downsample.1/moving_mean/Read/ReadVariableOp9layer2.0.downsample.1/moving_variance/Read/ReadVariableOp&layer2.0.bn2/gamma/Read/ReadVariableOp%layer2.0.bn2/beta/Read/ReadVariableOp,layer2.0.bn2/moving_mean/Read/ReadVariableOp0layer2.0.bn2/moving_variance/Read/ReadVariableOp)layer2.1.conv1/kernel/Read/ReadVariableOp&layer2.1.bn1/gamma/Read/ReadVariableOp%layer2.1.bn1/beta/Read/ReadVariableOp,layer2.1.bn1/moving_mean/Read/ReadVariableOp0layer2.1.bn1/moving_variance/Read/ReadVariableOp)layer2.1.conv2/kernel/Read/ReadVariableOp&layer2.1.bn2/gamma/Read/ReadVariableOp%layer2.1.bn2/beta/Read/ReadVariableOp,layer2.1.bn2/moving_mean/Read/ReadVariableOp0layer2.1.bn2/moving_variance/Read/ReadVariableOp)layer3.0.conv1/kernel/Read/ReadVariableOp&layer3.0.bn1/gamma/Read/ReadVariableOp%layer3.0.bn1/beta/Read/ReadVariableOp,layer3.0.bn1/moving_mean/Read/ReadVariableOp0layer3.0.bn1/moving_variance/Read/ReadVariableOp0layer3.0.downsample.0/kernel/Read/ReadVariableOp)layer3.0.conv2/kernel/Read/ReadVariableOp/layer3.0.downsample.1/gamma/Read/ReadVariableOp.layer3.0.downsample.1/beta/Read/ReadVariableOp5layer3.0.downsample.1/moving_mean/Read/ReadVariableOp9layer3.0.downsample.1/moving_variance/Read/ReadVariableOp&layer3.0.bn2/gamma/Read/ReadVariableOp%layer3.0.bn2/beta/Read/ReadVariableOp,layer3.0.bn2/moving_mean/Read/ReadVariableOp0layer3.0.bn2/moving_variance/Read/ReadVariableOp)layer3.1.conv1/kernel/Read/ReadVariableOp&layer3.1.bn1/gamma/Read/ReadVariableOp%layer3.1.bn1/beta/Read/ReadVariableOp,layer3.1.bn1/moving_mean/Read/ReadVariableOp0layer3.1.bn1/moving_variance/Read/ReadVariableOp)layer3.1.conv2/kernel/Read/ReadVariableOp&layer3.1.bn2/gamma/Read/ReadVariableOp%layer3.1.bn2/beta/Read/ReadVariableOp,layer3.1.bn2/moving_mean/Read/ReadVariableOp0layer3.1.bn2/moving_variance/Read/ReadVariableOpConst*X
TinQ
O2M*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *'
f"R 
__inference__traced_save_15330
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel	bn1/gammabn1/betabn1/moving_meanbn1/moving_variancelayer1.0.conv1/kernellayer1.0.bn1/gammalayer1.0.bn1/betalayer1.0.bn1/moving_meanlayer1.0.bn1/moving_variancelayer1.0.conv2/kernellayer1.0.bn2/gammalayer1.0.bn2/betalayer1.0.bn2/moving_meanlayer1.0.bn2/moving_variancelayer1.1.conv1/kernellayer1.1.bn1/gammalayer1.1.bn1/betalayer1.1.bn1/moving_meanlayer1.1.bn1/moving_variancelayer1.1.conv2/kernellayer1.1.bn2/gammalayer1.1.bn2/betalayer1.1.bn2/moving_meanlayer1.1.bn2/moving_variancelayer2.0.conv1/kernellayer2.0.bn1/gammalayer2.0.bn1/betalayer2.0.bn1/moving_meanlayer2.0.bn1/moving_variancelayer2.0.downsample.0/kernellayer2.0.conv2/kernellayer2.0.downsample.1/gammalayer2.0.downsample.1/beta!layer2.0.downsample.1/moving_mean%layer2.0.downsample.1/moving_variancelayer2.0.bn2/gammalayer2.0.bn2/betalayer2.0.bn2/moving_meanlayer2.0.bn2/moving_variancelayer2.1.conv1/kernellayer2.1.bn1/gammalayer2.1.bn1/betalayer2.1.bn1/moving_meanlayer2.1.bn1/moving_variancelayer2.1.conv2/kernellayer2.1.bn2/gammalayer2.1.bn2/betalayer2.1.bn2/moving_meanlayer2.1.bn2/moving_variancelayer3.0.conv1/kernellayer3.0.bn1/gammalayer3.0.bn1/betalayer3.0.bn1/moving_meanlayer3.0.bn1/moving_variancelayer3.0.downsample.0/kernellayer3.0.conv2/kernellayer3.0.downsample.1/gammalayer3.0.downsample.1/beta!layer3.0.downsample.1/moving_mean%layer3.0.downsample.1/moving_variancelayer3.0.bn2/gammalayer3.0.bn2/betalayer3.0.bn2/moving_meanlayer3.0.bn2/moving_variancelayer3.1.conv1/kernellayer3.1.bn1/gammalayer3.1.bn1/betalayer3.1.bn1/moving_meanlayer3.1.bn1/moving_variancelayer3.1.conv2/kernellayer3.1.bn2/gammalayer3.1.bn2/betalayer3.1.bn2/moving_meanlayer3.1.bn2/moving_variance*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? **
f%R#
!__inference__traced_restore_15565˪&
?
j
@__inference_add_2_layer_call_and_return_conditional_losses_11179

inputs
inputs_1
identityr
addAddV2inputsinputs_1*
T0*B
_output_shapes0
.:,????????????????????????????2
addv
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:,????????????????????????????:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:jf
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
t
.__inference_layer3.0.conv2_layer_call_fn_14744

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.0.conv2_layer_call_and_return_conditional_losses_114252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?)
!__inference__traced_restore_15565
file_prefix!
assignvariableop_conv1_kernel 
assignvariableop_1_bn1_gamma
assignvariableop_2_bn1_beta&
"assignvariableop_3_bn1_moving_mean*
&assignvariableop_4_bn1_moving_variance,
(assignvariableop_5_layer1_0_conv1_kernel)
%assignvariableop_6_layer1_0_bn1_gamma(
$assignvariableop_7_layer1_0_bn1_beta/
+assignvariableop_8_layer1_0_bn1_moving_mean3
/assignvariableop_9_layer1_0_bn1_moving_variance-
)assignvariableop_10_layer1_0_conv2_kernel*
&assignvariableop_11_layer1_0_bn2_gamma)
%assignvariableop_12_layer1_0_bn2_beta0
,assignvariableop_13_layer1_0_bn2_moving_mean4
0assignvariableop_14_layer1_0_bn2_moving_variance-
)assignvariableop_15_layer1_1_conv1_kernel*
&assignvariableop_16_layer1_1_bn1_gamma)
%assignvariableop_17_layer1_1_bn1_beta0
,assignvariableop_18_layer1_1_bn1_moving_mean4
0assignvariableop_19_layer1_1_bn1_moving_variance-
)assignvariableop_20_layer1_1_conv2_kernel*
&assignvariableop_21_layer1_1_bn2_gamma)
%assignvariableop_22_layer1_1_bn2_beta0
,assignvariableop_23_layer1_1_bn2_moving_mean4
0assignvariableop_24_layer1_1_bn2_moving_variance-
)assignvariableop_25_layer2_0_conv1_kernel*
&assignvariableop_26_layer2_0_bn1_gamma)
%assignvariableop_27_layer2_0_bn1_beta0
,assignvariableop_28_layer2_0_bn1_moving_mean4
0assignvariableop_29_layer2_0_bn1_moving_variance4
0assignvariableop_30_layer2_0_downsample_0_kernel-
)assignvariableop_31_layer2_0_conv2_kernel3
/assignvariableop_32_layer2_0_downsample_1_gamma2
.assignvariableop_33_layer2_0_downsample_1_beta9
5assignvariableop_34_layer2_0_downsample_1_moving_mean=
9assignvariableop_35_layer2_0_downsample_1_moving_variance*
&assignvariableop_36_layer2_0_bn2_gamma)
%assignvariableop_37_layer2_0_bn2_beta0
,assignvariableop_38_layer2_0_bn2_moving_mean4
0assignvariableop_39_layer2_0_bn2_moving_variance-
)assignvariableop_40_layer2_1_conv1_kernel*
&assignvariableop_41_layer2_1_bn1_gamma)
%assignvariableop_42_layer2_1_bn1_beta0
,assignvariableop_43_layer2_1_bn1_moving_mean4
0assignvariableop_44_layer2_1_bn1_moving_variance-
)assignvariableop_45_layer2_1_conv2_kernel*
&assignvariableop_46_layer2_1_bn2_gamma)
%assignvariableop_47_layer2_1_bn2_beta0
,assignvariableop_48_layer2_1_bn2_moving_mean4
0assignvariableop_49_layer2_1_bn2_moving_variance-
)assignvariableop_50_layer3_0_conv1_kernel*
&assignvariableop_51_layer3_0_bn1_gamma)
%assignvariableop_52_layer3_0_bn1_beta0
,assignvariableop_53_layer3_0_bn1_moving_mean4
0assignvariableop_54_layer3_0_bn1_moving_variance4
0assignvariableop_55_layer3_0_downsample_0_kernel-
)assignvariableop_56_layer3_0_conv2_kernel3
/assignvariableop_57_layer3_0_downsample_1_gamma2
.assignvariableop_58_layer3_0_downsample_1_beta9
5assignvariableop_59_layer3_0_downsample_1_moving_mean=
9assignvariableop_60_layer3_0_downsample_1_moving_variance*
&assignvariableop_61_layer3_0_bn2_gamma)
%assignvariableop_62_layer3_0_bn2_beta0
,assignvariableop_63_layer3_0_bn2_moving_mean4
0assignvariableop_64_layer3_0_bn2_moving_variance-
)assignvariableop_65_layer3_1_conv1_kernel*
&assignvariableop_66_layer3_1_bn1_gamma)
%assignvariableop_67_layer3_1_bn1_beta0
,assignvariableop_68_layer3_1_bn1_moving_mean4
0assignvariableop_69_layer3_1_bn1_moving_variance-
)assignvariableop_70_layer3_1_conv2_kernel*
&assignvariableop_71_layer3_1_bn2_gamma)
%assignvariableop_72_layer3_1_bn2_beta0
,assignvariableop_73_layer3_1_bn2_moving_mean4
0assignvariableop_74_layer3_1_bn2_moving_variance
identity_76??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_8?AssignVariableOp_9?#
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?"
value?"B?"LB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-24/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-24/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-24/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-25/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-25/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-25/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-27/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-27/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-27/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-29/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-29/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-29/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?
value?B?LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_bn1_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_bn1_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_bn1_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp&assignvariableop_4_bn1_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp(assignvariableop_5_layer1_0_conv1_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_layer1_0_bn1_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp$assignvariableop_7_layer1_0_bn1_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp+assignvariableop_8_layer1_0_bn1_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp/assignvariableop_9_layer1_0_bn1_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp)assignvariableop_10_layer1_0_conv2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp&assignvariableop_11_layer1_0_bn2_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_layer1_0_bn2_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp,assignvariableop_13_layer1_0_bn2_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_layer1_0_bn2_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp)assignvariableop_15_layer1_1_conv1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_layer1_1_bn1_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp%assignvariableop_17_layer1_1_bn1_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp,assignvariableop_18_layer1_1_bn1_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp0assignvariableop_19_layer1_1_bn1_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_layer1_1_conv2_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp&assignvariableop_21_layer1_1_bn2_gammaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp%assignvariableop_22_layer1_1_bn2_betaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp,assignvariableop_23_layer1_1_bn2_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp0assignvariableop_24_layer1_1_bn2_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_layer2_0_conv1_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_layer2_0_bn1_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp%assignvariableop_27_layer2_0_bn1_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp,assignvariableop_28_layer2_0_bn1_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp0assignvariableop_29_layer2_0_bn1_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp0assignvariableop_30_layer2_0_downsample_0_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_layer2_0_conv2_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp/assignvariableop_32_layer2_0_downsample_1_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp.assignvariableop_33_layer2_0_downsample_1_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp5assignvariableop_34_layer2_0_downsample_1_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp9assignvariableop_35_layer2_0_downsample_1_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp&assignvariableop_36_layer2_0_bn2_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp%assignvariableop_37_layer2_0_bn2_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp,assignvariableop_38_layer2_0_bn2_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp0assignvariableop_39_layer2_0_bn2_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_layer2_1_conv1_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp&assignvariableop_41_layer2_1_bn1_gammaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp%assignvariableop_42_layer2_1_bn1_betaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp,assignvariableop_43_layer2_1_bn1_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp0assignvariableop_44_layer2_1_bn1_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp)assignvariableop_45_layer2_1_conv2_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp&assignvariableop_46_layer2_1_bn2_gammaIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp%assignvariableop_47_layer2_1_bn2_betaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp,assignvariableop_48_layer2_1_bn2_moving_meanIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp0assignvariableop_49_layer2_1_bn2_moving_varianceIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp)assignvariableop_50_layer3_0_conv1_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp&assignvariableop_51_layer3_0_bn1_gammaIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp%assignvariableop_52_layer3_0_bn1_betaIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp,assignvariableop_53_layer3_0_bn1_moving_meanIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp0assignvariableop_54_layer3_0_bn1_moving_varianceIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp0assignvariableop_55_layer3_0_downsample_0_kernelIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp)assignvariableop_56_layer3_0_conv2_kernelIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp/assignvariableop_57_layer3_0_downsample_1_gammaIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp.assignvariableop_58_layer3_0_downsample_1_betaIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp5assignvariableop_59_layer3_0_downsample_1_moving_meanIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp9assignvariableop_60_layer3_0_downsample_1_moving_varianceIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp&assignvariableop_61_layer3_0_bn2_gammaIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp%assignvariableop_62_layer3_0_bn2_betaIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp,assignvariableop_63_layer3_0_bn2_moving_meanIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp0assignvariableop_64_layer3_0_bn2_moving_varianceIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp)assignvariableop_65_layer3_1_conv1_kernelIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp&assignvariableop_66_layer3_1_bn1_gammaIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp%assignvariableop_67_layer3_1_bn1_betaIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp,assignvariableop_68_layer3_1_bn1_moving_meanIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp0assignvariableop_69_layer3_1_bn1_moving_varianceIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp)assignvariableop_70_layer3_1_conv2_kernelIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp&assignvariableop_71_layer3_1_bn2_gammaIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp%assignvariableop_72_layer3_1_bn2_betaIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp,assignvariableop_73_layer3_1_bn2_moving_meanIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp0assignvariableop_74_layer3_1_bn2_moving_varianceIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_749
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_75?
Identity_76IdentityIdentity_75:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_76"#
identity_76Identity_76:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
F__inference_layer1.0.bn2_layer_call_and_return_conditional_losses_9340

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
d
H__inference_activation_10_layer_call_and_return_conditional_losses_11613

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
Q
%__inference_add_2_layer_call_fn_14430
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_111792
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:,????????????????????????????:,????????????????????????????:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/0:lh
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/1
??
?
@__inference_model_layer_call_and_return_conditional_losses_12123

inputs
conv1_11919
	bn1_11922
	bn1_11924
	bn1_11926
	bn1_11928
layer1_0_conv1_11934
layer1_0_bn1_11937
layer1_0_bn1_11939
layer1_0_bn1_11941
layer1_0_bn1_11943
layer1_0_conv2_11947
layer1_0_bn2_11950
layer1_0_bn2_11952
layer1_0_bn2_11954
layer1_0_bn2_11956
layer1_1_conv1_11961
layer1_1_bn1_11964
layer1_1_bn1_11966
layer1_1_bn1_11968
layer1_1_bn1_11970
layer1_1_conv2_11974
layer1_1_bn2_11977
layer1_1_bn2_11979
layer1_1_bn2_11981
layer1_1_bn2_11983
layer2_0_conv1_11989
layer2_0_bn1_11992
layer2_0_bn1_11994
layer2_0_bn1_11996
layer2_0_bn1_11998
layer2_0_conv2_12002
layer2_0_downsample_0_12005
layer2_0_downsample_1_12008
layer2_0_downsample_1_12010
layer2_0_downsample_1_12012
layer2_0_downsample_1_12014
layer2_0_bn2_12017
layer2_0_bn2_12019
layer2_0_bn2_12021
layer2_0_bn2_12023
layer2_1_conv1_12028
layer2_1_bn1_12031
layer2_1_bn1_12033
layer2_1_bn1_12035
layer2_1_bn1_12037
layer2_1_conv2_12041
layer2_1_bn2_12044
layer2_1_bn2_12046
layer2_1_bn2_12048
layer2_1_bn2_12050
layer3_0_conv1_12056
layer3_0_bn1_12059
layer3_0_bn1_12061
layer3_0_bn1_12063
layer3_0_bn1_12065
layer3_0_conv2_12069
layer3_0_downsample_0_12072
layer3_0_downsample_1_12075
layer3_0_downsample_1_12077
layer3_0_downsample_1_12079
layer3_0_downsample_1_12081
layer3_0_bn2_12084
layer3_0_bn2_12086
layer3_0_bn2_12088
layer3_0_bn2_12090
layer3_1_conv1_12095
layer3_1_bn1_12098
layer3_1_bn1_12100
layer3_1_bn1_12102
layer3_1_bn1_12104
layer3_1_conv2_12108
layer3_1_bn2_12111
layer3_1_bn2_12113
layer3_1_bn2_12115
layer3_1_bn2_12117
identity??bn1/StatefulPartitionedCall?conv1/StatefulPartitionedCall?$layer1.0.bn1/StatefulPartitionedCall?$layer1.0.bn2/StatefulPartitionedCall?&layer1.0.conv1/StatefulPartitionedCall?&layer1.0.conv2/StatefulPartitionedCall?$layer1.1.bn1/StatefulPartitionedCall?$layer1.1.bn2/StatefulPartitionedCall?&layer1.1.conv1/StatefulPartitionedCall?&layer1.1.conv2/StatefulPartitionedCall?$layer2.0.bn1/StatefulPartitionedCall?$layer2.0.bn2/StatefulPartitionedCall?&layer2.0.conv1/StatefulPartitionedCall?&layer2.0.conv2/StatefulPartitionedCall?-layer2.0.downsample.0/StatefulPartitionedCall?-layer2.0.downsample.1/StatefulPartitionedCall?$layer2.1.bn1/StatefulPartitionedCall?$layer2.1.bn2/StatefulPartitionedCall?&layer2.1.conv1/StatefulPartitionedCall?&layer2.1.conv2/StatefulPartitionedCall?$layer3.0.bn1/StatefulPartitionedCall?$layer3.0.bn2/StatefulPartitionedCall?&layer3.0.conv1/StatefulPartitionedCall?&layer3.0.conv2/StatefulPartitionedCall?-layer3.0.downsample.0/StatefulPartitionedCall?-layer3.0.downsample.1/StatefulPartitionedCall?$layer3.1.bn1/StatefulPartitionedCall?$layer3.1.bn2/StatefulPartitionedCall?&layer3.1.conv1/StatefulPartitionedCall?&layer3.1.conv2/StatefulPartitionedCall?
pad/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *F
fAR?
=__inference_pad_layer_call_and_return_conditional_losses_90082
pad/PartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallpad/PartitionedCall:output:0conv1_11919*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_106372
conv1/StatefulPartitionedCall?
bn1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0	bn1_11922	bn1_11924	bn1_11926	bn1_11928*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *F
fAR?
=__inference_bn1_layer_call_and_return_conditional_losses_90762
bn1/StatefulPartitionedCall?
relu/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_relu_layer_call_and_return_conditional_losses_106892
relu/PartitionedCall?
pad1/PartitionedCallPartitionedCallrelu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *G
fBR@
>__inference_pad1_layer_call_and_return_conditional_losses_91252
pad1/PartitionedCall?
maxpool/PartitionedCallPartitionedCallpad1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_maxpool_layer_call_and_return_conditional_losses_91372
maxpool/PartitionedCall?
&layer1.0.conv1/StatefulPartitionedCallStatefulPartitionedCall maxpool/PartitionedCall:output:0layer1_0_conv1_11934*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.0.conv1_layer_call_and_return_conditional_losses_107062(
&layer1.0.conv1/StatefulPartitionedCall?
$layer1.0.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer1.0.conv1/StatefulPartitionedCall:output:0layer1_0_bn1_11937layer1_0_bn1_11939layer1_0_bn1_11941layer1_0_bn1_11943*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.0.bn1_layer_call_and_return_conditional_losses_92052&
$layer1.0.bn1/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall-layer1.0.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_107582
activation/PartitionedCall?
&layer1.0.conv2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0layer1_0_conv2_11947*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.0.conv2_layer_call_and_return_conditional_losses_107732(
&layer1.0.conv2/StatefulPartitionedCall?
$layer1.0.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer1.0.conv2/StatefulPartitionedCall:output:0layer1_0_bn2_11950layer1_0_bn2_11952layer1_0_bn2_11954layer1_0_bn2_11956*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.0.bn2_layer_call_and_return_conditional_losses_93092&
$layer1.0.bn2/StatefulPartitionedCall?
add/PartitionedCallPartitionedCall maxpool/PartitionedCall:output:0-layer1.0.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_108262
add/PartitionedCall?
activation_1/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_108402
activation_1/PartitionedCall?
&layer1.1.conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0layer1_1_conv1_11961*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.1.conv1_layer_call_and_return_conditional_losses_108552(
&layer1.1.conv1/StatefulPartitionedCall?
$layer1.1.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer1.1.conv1/StatefulPartitionedCall:output:0layer1_1_bn1_11964layer1_1_bn1_11966layer1_1_bn1_11968layer1_1_bn1_11970*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.1.bn1_layer_call_and_return_conditional_losses_94132&
$layer1.1.bn1/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall-layer1.1.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_109072
activation_2/PartitionedCall?
&layer1.1.conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0layer1_1_conv2_11974*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.1.conv2_layer_call_and_return_conditional_losses_109222(
&layer1.1.conv2/StatefulPartitionedCall?
$layer1.1.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer1.1.conv2/StatefulPartitionedCall:output:0layer1_1_bn2_11977layer1_1_bn2_11979layer1_1_bn2_11981layer1_1_bn2_11983*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.1.bn2_layer_call_and_return_conditional_losses_95172&
$layer1.1.bn2/StatefulPartitionedCall?
add_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0-layer1.1.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_109752
add_1/PartitionedCall?
activation_3/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_109892
activation_3/PartitionedCall?
layer2.0.pad/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.0.pad_layer_call_and_return_conditional_losses_95662
layer2.0.pad/PartitionedCall?
&layer2.0.conv1/StatefulPartitionedCallStatefulPartitionedCall%layer2.0.pad/PartitionedCall:output:0layer2_0_conv1_11989*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.0.conv1_layer_call_and_return_conditional_losses_110052(
&layer2.0.conv1/StatefulPartitionedCall?
$layer2.0.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer2.0.conv1/StatefulPartitionedCall:output:0layer2_0_bn1_11992layer2_0_bn1_11994layer2_0_bn1_11996layer2_0_bn1_11998*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.0.bn1_layer_call_and_return_conditional_losses_96342&
$layer2.0.bn1/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall-layer2.0.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_110572
activation_4/PartitionedCall?
&layer2.0.conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0layer2_0_conv2_12002*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.0.conv2_layer_call_and_return_conditional_losses_110722(
&layer2.0.conv2/StatefulPartitionedCall?
-layer2.0.downsample.0/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0layer2_0_downsample_0_12005*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_layer2.0.downsample.0_layer_call_and_return_conditional_losses_110912/
-layer2.0.downsample.0/StatefulPartitionedCall?
-layer2.0.downsample.1/StatefulPartitionedCallStatefulPartitionedCall6layer2.0.downsample.0/StatefulPartitionedCall:output:0layer2_0_downsample_1_12008layer2_0_downsample_1_12010layer2_0_downsample_1_12012layer2_0_downsample_1_12014*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_layer2.0.downsample.1_layer_call_and_return_conditional_losses_97382/
-layer2.0.downsample.1/StatefulPartitionedCall?
$layer2.0.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer2.0.conv2/StatefulPartitionedCall:output:0layer2_0_bn2_12017layer2_0_bn2_12019layer2_0_bn2_12021layer2_0_bn2_12023*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.0.bn2_layer_call_and_return_conditional_losses_98422&
$layer2.0.bn2/StatefulPartitionedCall?
add_2/PartitionedCallPartitionedCall6layer2.0.downsample.1/StatefulPartitionedCall:output:0-layer2.0.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_111792
add_2/PartitionedCall?
activation_5/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_111932
activation_5/PartitionedCall?
&layer2.1.conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0layer2_1_conv1_12028*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.1.conv1_layer_call_and_return_conditional_losses_112082(
&layer2.1.conv1/StatefulPartitionedCall?
$layer2.1.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer2.1.conv1/StatefulPartitionedCall:output:0layer2_1_bn1_12031layer2_1_bn1_12033layer2_1_bn1_12035layer2_1_bn1_12037*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.1.bn1_layer_call_and_return_conditional_losses_99462&
$layer2.1.bn1/StatefulPartitionedCall?
activation_6/PartitionedCallPartitionedCall-layer2.1.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_112602
activation_6/PartitionedCall?
&layer2.1.conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0layer2_1_conv2_12041*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.1.conv2_layer_call_and_return_conditional_losses_112752(
&layer2.1.conv2/StatefulPartitionedCall?
$layer2.1.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer2.1.conv2/StatefulPartitionedCall:output:0layer2_1_bn2_12044layer2_1_bn2_12046layer2_1_bn2_12048layer2_1_bn2_12050*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer2.1.bn2_layer_call_and_return_conditional_losses_100502&
$layer2.1.bn2/StatefulPartitionedCall?
add_3/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0-layer2.1.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_113282
add_3/PartitionedCall?
activation_7/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_7_layer_call_and_return_conditional_losses_113422
activation_7/PartitionedCall?
layer3.0.pad/PartitionedCallPartitionedCall%activation_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.0.pad_layer_call_and_return_conditional_losses_100992
layer3.0.pad/PartitionedCall?
&layer3.0.conv1/StatefulPartitionedCallStatefulPartitionedCall%layer3.0.pad/PartitionedCall:output:0layer3_0_conv1_12056*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.0.conv1_layer_call_and_return_conditional_losses_113582(
&layer3.0.conv1/StatefulPartitionedCall?
$layer3.0.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer3.0.conv1/StatefulPartitionedCall:output:0layer3_0_bn1_12059layer3_0_bn1_12061layer3_0_bn1_12063layer3_0_bn1_12065*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.0.bn1_layer_call_and_return_conditional_losses_101672&
$layer3.0.bn1/StatefulPartitionedCall?
activation_8/PartitionedCallPartitionedCall-layer3.0.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_8_layer_call_and_return_conditional_losses_114102
activation_8/PartitionedCall?
&layer3.0.conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_8/PartitionedCall:output:0layer3_0_conv2_12069*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.0.conv2_layer_call_and_return_conditional_losses_114252(
&layer3.0.conv2/StatefulPartitionedCall?
-layer3.0.downsample.0/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0layer3_0_downsample_0_12072*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_layer3.0.downsample.0_layer_call_and_return_conditional_losses_114442/
-layer3.0.downsample.0/StatefulPartitionedCall?
-layer3.0.downsample.1/StatefulPartitionedCallStatefulPartitionedCall6layer3.0.downsample.0/StatefulPartitionedCall:output:0layer3_0_downsample_1_12075layer3_0_downsample_1_12077layer3_0_downsample_1_12079layer3_0_downsample_1_12081*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_layer3.0.downsample.1_layer_call_and_return_conditional_losses_102712/
-layer3.0.downsample.1/StatefulPartitionedCall?
$layer3.0.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer3.0.conv2/StatefulPartitionedCall:output:0layer3_0_bn2_12084layer3_0_bn2_12086layer3_0_bn2_12088layer3_0_bn2_12090*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.0.bn2_layer_call_and_return_conditional_losses_103752&
$layer3.0.bn2/StatefulPartitionedCall?
add_4/PartitionedCallPartitionedCall6layer3.0.downsample.1/StatefulPartitionedCall:output:0-layer3.0.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_4_layer_call_and_return_conditional_losses_115322
add_4/PartitionedCall?
activation_9/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_9_layer_call_and_return_conditional_losses_115462
activation_9/PartitionedCall?
&layer3.1.conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_9/PartitionedCall:output:0layer3_1_conv1_12095*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.1.conv1_layer_call_and_return_conditional_losses_115612(
&layer3.1.conv1/StatefulPartitionedCall?
$layer3.1.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer3.1.conv1/StatefulPartitionedCall:output:0layer3_1_bn1_12098layer3_1_bn1_12100layer3_1_bn1_12102layer3_1_bn1_12104*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.1.bn1_layer_call_and_return_conditional_losses_104792&
$layer3.1.bn1/StatefulPartitionedCall?
activation_10/PartitionedCallPartitionedCall-layer3.1.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_10_layer_call_and_return_conditional_losses_116132
activation_10/PartitionedCall?
&layer3.1.conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_10/PartitionedCall:output:0layer3_1_conv2_12108*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.1.conv2_layer_call_and_return_conditional_losses_116282(
&layer3.1.conv2/StatefulPartitionedCall?
$layer3.1.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer3.1.conv2/StatefulPartitionedCall:output:0layer3_1_bn2_12111layer3_1_bn2_12113layer3_1_bn2_12115layer3_1_bn2_12117*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.1.bn2_layer_call_and_return_conditional_losses_105832&
$layer3.1.bn2/StatefulPartitionedCall?
add_5/PartitionedCallPartitionedCall%activation_9/PartitionedCall:output:0-layer3.1.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_5_layer_call_and_return_conditional_losses_116812
add_5/PartitionedCall?
activation_11/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_11_layer_call_and_return_conditional_losses_116952
activation_11/PartitionedCall?

IdentityIdentity&activation_11/PartitionedCall:output:0^bn1/StatefulPartitionedCall^conv1/StatefulPartitionedCall%^layer1.0.bn1/StatefulPartitionedCall%^layer1.0.bn2/StatefulPartitionedCall'^layer1.0.conv1/StatefulPartitionedCall'^layer1.0.conv2/StatefulPartitionedCall%^layer1.1.bn1/StatefulPartitionedCall%^layer1.1.bn2/StatefulPartitionedCall'^layer1.1.conv1/StatefulPartitionedCall'^layer1.1.conv2/StatefulPartitionedCall%^layer2.0.bn1/StatefulPartitionedCall%^layer2.0.bn2/StatefulPartitionedCall'^layer2.0.conv1/StatefulPartitionedCall'^layer2.0.conv2/StatefulPartitionedCall.^layer2.0.downsample.0/StatefulPartitionedCall.^layer2.0.downsample.1/StatefulPartitionedCall%^layer2.1.bn1/StatefulPartitionedCall%^layer2.1.bn2/StatefulPartitionedCall'^layer2.1.conv1/StatefulPartitionedCall'^layer2.1.conv2/StatefulPartitionedCall%^layer3.0.bn1/StatefulPartitionedCall%^layer3.0.bn2/StatefulPartitionedCall'^layer3.0.conv1/StatefulPartitionedCall'^layer3.0.conv2/StatefulPartitionedCall.^layer3.0.downsample.0/StatefulPartitionedCall.^layer3.0.downsample.1/StatefulPartitionedCall%^layer3.1.bn1/StatefulPartitionedCall%^layer3.1.bn2/StatefulPartitionedCall'^layer3.1.conv1/StatefulPartitionedCall'^layer3.1.conv2/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:+???????????????????????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2L
$layer1.0.bn1/StatefulPartitionedCall$layer1.0.bn1/StatefulPartitionedCall2L
$layer1.0.bn2/StatefulPartitionedCall$layer1.0.bn2/StatefulPartitionedCall2P
&layer1.0.conv1/StatefulPartitionedCall&layer1.0.conv1/StatefulPartitionedCall2P
&layer1.0.conv2/StatefulPartitionedCall&layer1.0.conv2/StatefulPartitionedCall2L
$layer1.1.bn1/StatefulPartitionedCall$layer1.1.bn1/StatefulPartitionedCall2L
$layer1.1.bn2/StatefulPartitionedCall$layer1.1.bn2/StatefulPartitionedCall2P
&layer1.1.conv1/StatefulPartitionedCall&layer1.1.conv1/StatefulPartitionedCall2P
&layer1.1.conv2/StatefulPartitionedCall&layer1.1.conv2/StatefulPartitionedCall2L
$layer2.0.bn1/StatefulPartitionedCall$layer2.0.bn1/StatefulPartitionedCall2L
$layer2.0.bn2/StatefulPartitionedCall$layer2.0.bn2/StatefulPartitionedCall2P
&layer2.0.conv1/StatefulPartitionedCall&layer2.0.conv1/StatefulPartitionedCall2P
&layer2.0.conv2/StatefulPartitionedCall&layer2.0.conv2/StatefulPartitionedCall2^
-layer2.0.downsample.0/StatefulPartitionedCall-layer2.0.downsample.0/StatefulPartitionedCall2^
-layer2.0.downsample.1/StatefulPartitionedCall-layer2.0.downsample.1/StatefulPartitionedCall2L
$layer2.1.bn1/StatefulPartitionedCall$layer2.1.bn1/StatefulPartitionedCall2L
$layer2.1.bn2/StatefulPartitionedCall$layer2.1.bn2/StatefulPartitionedCall2P
&layer2.1.conv1/StatefulPartitionedCall&layer2.1.conv1/StatefulPartitionedCall2P
&layer2.1.conv2/StatefulPartitionedCall&layer2.1.conv2/StatefulPartitionedCall2L
$layer3.0.bn1/StatefulPartitionedCall$layer3.0.bn1/StatefulPartitionedCall2L
$layer3.0.bn2/StatefulPartitionedCall$layer3.0.bn2/StatefulPartitionedCall2P
&layer3.0.conv1/StatefulPartitionedCall&layer3.0.conv1/StatefulPartitionedCall2P
&layer3.0.conv2/StatefulPartitionedCall&layer3.0.conv2/StatefulPartitionedCall2^
-layer3.0.downsample.0/StatefulPartitionedCall-layer3.0.downsample.0/StatefulPartitionedCall2^
-layer3.0.downsample.1/StatefulPartitionedCall-layer3.0.downsample.1/StatefulPartitionedCall2L
$layer3.1.bn1/StatefulPartitionedCall$layer3.1.bn1/StatefulPartitionedCall2L
$layer3.1.bn2/StatefulPartitionedCall$layer3.1.bn2/StatefulPartitionedCall2P
&layer3.1.conv1/StatefulPartitionedCall&layer3.1.conv1/StatefulPartitionedCall2P
&layer3.1.conv2/StatefulPartitionedCall&layer3.1.conv2/StatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_activation_5_layer_call_fn_14440

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_111932
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_layer2.0.downsample.1_layer_call_and_return_conditional_losses_14328

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_activation_7_layer_call_and_return_conditional_losses_14623

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
[
?__inference_relu_layer_call_and_return_conditional_losses_13793

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
G__inference_layer2.1.bn2_layer_call_and_return_conditional_losses_14562

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_activation_2_layer_call_fn_14074

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_109072
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
>
"__inference_pad_layer_call_fn_9014

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
 *2
config_proto" 

CPU

GPU2 *0J 8? *F
fAR?
=__inference_pad_layer_call_and_return_conditional_losses_90082
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
?
?
,__inference_layer2.1.bn2_layer_call_fn_14606

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer2.1.bn2_layer_call_and_return_conditional_losses_100812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_layer1.1.conv1_layer_call_and_return_conditional_losses_10855

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
G__inference_activation_2_layer_call_and_return_conditional_losses_14069

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
t
.__inference_layer3.0.conv1_layer_call_fn_14642

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.0.conv1_layer_call_and_return_conditional_losses_113582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_layer1.0.bn2_layer_call_and_return_conditional_losses_9309

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
,__inference_layer3.0.bn1_layer_call_fn_14706

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.0.bn1_layer_call_and_return_conditional_losses_101982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?

%__inference_model_layer_call_fn_12639
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

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*m
_read_only_resource_inputsO
MK	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJK*2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_124862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:+???????????????????????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+???????????????????????????
!
_user_specified_name	input_1
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_13744

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
G__inference_activation_1_layer_call_and_return_conditional_losses_13981

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?

%__inference_model_layer_call_fn_12276
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

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73
identity??StatefulPartitionedCall?

StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*O
_read_only_resource_inputs1
/- !"%&)*+./034589:;>?BCDGHI*2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_121232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:+???????????????????????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+???????????????????????????
!
_user_specified_name	input_1
?
c
G__inference_activation_5_layer_call_and_return_conditional_losses_14435

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_layer3.0.downsample.1_layer_call_and_return_conditional_losses_14782

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer3.0.bn1_layer_call_and_return_conditional_losses_10167

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_layer3.0.conv2_layer_call_and_return_conditional_losses_14737

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer2.1.bn2_layer_call_and_return_conditional_losses_10050

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer1.0.bn1_layer_call_and_return_conditional_losses_13850

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
G__inference_activation_6_layer_call_and_return_conditional_losses_14523

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
I
-__inference_activation_10_layer_call_fn_14982

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_10_layer_call_and_return_conditional_losses_116132
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
I
-__inference_activation_11_layer_call_fn_15082

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_11_layer_call_and_return_conditional_losses_116952
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_layer1.1.bn1_layer_call_and_return_conditional_losses_9413

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
#__inference_pad1_layer_call_fn_9131

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
 *2
config_proto" 

CPU

GPU2 *0J 8? *G
fBR@
>__inference_pad1_layer_call_and_return_conditional_losses_91252
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
?
{
5__inference_layer3.0.downsample.0_layer_call_fn_14730

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_layer3.0.downsample.0_layer_call_and_return_conditional_losses_114442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_activation_3_layer_call_fn_14174

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_109892
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
=__inference_bn1_layer_call_and_return_conditional_losses_9076

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
=__inference_bn1_layer_call_and_return_conditional_losses_9107

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
I__inference_layer1.1.conv2_layer_call_and_return_conditional_losses_14081

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
G__inference_activation_9_layer_call_and_return_conditional_losses_14889

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
5__inference_layer2.0.downsample.1_layer_call_fn_14341

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_layer2.0.downsample.1_layer_call_and_return_conditional_losses_97382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_activation_layer_call_and_return_conditional_losses_10758

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
G__inference_activation_2_layer_call_and_return_conditional_losses_10907

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
,__inference_layer3.1.bn1_layer_call_fn_14972

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.1.bn1_layer_call_and_return_conditional_losses_105102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_layer3.0.bn2_layer_call_fn_14859

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.0.bn2_layer_call_and_return_conditional_losses_103752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_layer2.1.bn1_layer_call_fn_14518

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.1.bn1_layer_call_and_return_conditional_losses_99772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_layer3.0.downsample.1_layer_call_and_return_conditional_losses_10302

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer2.1.bn2_layer_call_and_return_conditional_losses_14580

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
t
.__inference_layer2.1.conv1_layer_call_fn_14454

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.1.conv1_layer_call_and_return_conditional_losses_112082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_layer3.0.downsample.0_layer_call_and_return_conditional_losses_11444

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_layer1.1.bn2_layer_call_and_return_conditional_losses_9517

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_layer2.0.downsample.1_layer_call_and_return_conditional_losses_9769

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_activation_3_layer_call_and_return_conditional_losses_14169

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?

%__inference_model_layer_call_fn_13710

inputs
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

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*m
_read_only_resource_inputsO
MK	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJK*2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_124862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:+???????????????????????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_layer3.0.downsample.0_layer_call_and_return_conditional_losses_14723

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
O
#__inference_add_layer_call_fn_13976
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_108262
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:+???????????????????????????@:+???????????????????????????@:k g
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/1
?
?
I__inference_layer2.1.conv2_layer_call_and_return_conditional_losses_14535

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_layer3.0.bn1_layer_call_fn_14693

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.0.bn1_layer_call_and_return_conditional_losses_101672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_layer1.1.bn1_layer_call_fn_14051

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.1.bn1_layer_call_and_return_conditional_losses_94132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?

%__inference_model_layer_call_fn_13555

inputs
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

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73
identity??StatefulPartitionedCall?

StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*O
_read_only_resource_inputs1
/- !"%&)*+./034589:;>?BCDGHI*2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_121232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:+???????????????????????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_layer2.0.pad_layer_call_fn_9572

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
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.0.pad_layer_call_and_return_conditional_losses_95662
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
?
?
F__inference_layer2.0.bn2_layer_call_and_return_conditional_losses_9842

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_layer3.0.conv2_layer_call_and_return_conditional_losses_11425

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_activation_4_layer_call_and_return_conditional_losses_14257

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_layer1.1.bn2_layer_call_fn_14152

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.1.bn2_layer_call_and_return_conditional_losses_95482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?6
@__inference_model_layer_call_and_return_conditional_losses_13400

inputs(
$conv1_conv2d_readvariableop_resource
bn1_readvariableop_resource!
bn1_readvariableop_1_resource0
,bn1_fusedbatchnormv3_readvariableop_resource2
.bn1_fusedbatchnormv3_readvariableop_1_resource1
-layer1_0_conv1_conv2d_readvariableop_resource(
$layer1_0_bn1_readvariableop_resource*
&layer1_0_bn1_readvariableop_1_resource9
5layer1_0_bn1_fusedbatchnormv3_readvariableop_resource;
7layer1_0_bn1_fusedbatchnormv3_readvariableop_1_resource1
-layer1_0_conv2_conv2d_readvariableop_resource(
$layer1_0_bn2_readvariableop_resource*
&layer1_0_bn2_readvariableop_1_resource9
5layer1_0_bn2_fusedbatchnormv3_readvariableop_resource;
7layer1_0_bn2_fusedbatchnormv3_readvariableop_1_resource1
-layer1_1_conv1_conv2d_readvariableop_resource(
$layer1_1_bn1_readvariableop_resource*
&layer1_1_bn1_readvariableop_1_resource9
5layer1_1_bn1_fusedbatchnormv3_readvariableop_resource;
7layer1_1_bn1_fusedbatchnormv3_readvariableop_1_resource1
-layer1_1_conv2_conv2d_readvariableop_resource(
$layer1_1_bn2_readvariableop_resource*
&layer1_1_bn2_readvariableop_1_resource9
5layer1_1_bn2_fusedbatchnormv3_readvariableop_resource;
7layer1_1_bn2_fusedbatchnormv3_readvariableop_1_resource1
-layer2_0_conv1_conv2d_readvariableop_resource(
$layer2_0_bn1_readvariableop_resource*
&layer2_0_bn1_readvariableop_1_resource9
5layer2_0_bn1_fusedbatchnormv3_readvariableop_resource;
7layer2_0_bn1_fusedbatchnormv3_readvariableop_1_resource1
-layer2_0_conv2_conv2d_readvariableop_resource8
4layer2_0_downsample_0_conv2d_readvariableop_resource1
-layer2_0_downsample_1_readvariableop_resource3
/layer2_0_downsample_1_readvariableop_1_resourceB
>layer2_0_downsample_1_fusedbatchnormv3_readvariableop_resourceD
@layer2_0_downsample_1_fusedbatchnormv3_readvariableop_1_resource(
$layer2_0_bn2_readvariableop_resource*
&layer2_0_bn2_readvariableop_1_resource9
5layer2_0_bn2_fusedbatchnormv3_readvariableop_resource;
7layer2_0_bn2_fusedbatchnormv3_readvariableop_1_resource1
-layer2_1_conv1_conv2d_readvariableop_resource(
$layer2_1_bn1_readvariableop_resource*
&layer2_1_bn1_readvariableop_1_resource9
5layer2_1_bn1_fusedbatchnormv3_readvariableop_resource;
7layer2_1_bn1_fusedbatchnormv3_readvariableop_1_resource1
-layer2_1_conv2_conv2d_readvariableop_resource(
$layer2_1_bn2_readvariableop_resource*
&layer2_1_bn2_readvariableop_1_resource9
5layer2_1_bn2_fusedbatchnormv3_readvariableop_resource;
7layer2_1_bn2_fusedbatchnormv3_readvariableop_1_resource1
-layer3_0_conv1_conv2d_readvariableop_resource(
$layer3_0_bn1_readvariableop_resource*
&layer3_0_bn1_readvariableop_1_resource9
5layer3_0_bn1_fusedbatchnormv3_readvariableop_resource;
7layer3_0_bn1_fusedbatchnormv3_readvariableop_1_resource1
-layer3_0_conv2_conv2d_readvariableop_resource8
4layer3_0_downsample_0_conv2d_readvariableop_resource1
-layer3_0_downsample_1_readvariableop_resource3
/layer3_0_downsample_1_readvariableop_1_resourceB
>layer3_0_downsample_1_fusedbatchnormv3_readvariableop_resourceD
@layer3_0_downsample_1_fusedbatchnormv3_readvariableop_1_resource(
$layer3_0_bn2_readvariableop_resource*
&layer3_0_bn2_readvariableop_1_resource9
5layer3_0_bn2_fusedbatchnormv3_readvariableop_resource;
7layer3_0_bn2_fusedbatchnormv3_readvariableop_1_resource1
-layer3_1_conv1_conv2d_readvariableop_resource(
$layer3_1_bn1_readvariableop_resource*
&layer3_1_bn1_readvariableop_1_resource9
5layer3_1_bn1_fusedbatchnormv3_readvariableop_resource;
7layer3_1_bn1_fusedbatchnormv3_readvariableop_1_resource1
-layer3_1_conv2_conv2d_readvariableop_resource(
$layer3_1_bn2_readvariableop_resource*
&layer3_1_bn2_readvariableop_1_resource9
5layer3_1_bn2_fusedbatchnormv3_readvariableop_resource;
7layer3_1_bn2_fusedbatchnormv3_readvariableop_1_resource
identity??#bn1/FusedBatchNormV3/ReadVariableOp?%bn1/FusedBatchNormV3/ReadVariableOp_1?bn1/ReadVariableOp?bn1/ReadVariableOp_1?conv1/Conv2D/ReadVariableOp?,layer1.0.bn1/FusedBatchNormV3/ReadVariableOp?.layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1?layer1.0.bn1/ReadVariableOp?layer1.0.bn1/ReadVariableOp_1?,layer1.0.bn2/FusedBatchNormV3/ReadVariableOp?.layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1?layer1.0.bn2/ReadVariableOp?layer1.0.bn2/ReadVariableOp_1?$layer1.0.conv1/Conv2D/ReadVariableOp?$layer1.0.conv2/Conv2D/ReadVariableOp?,layer1.1.bn1/FusedBatchNormV3/ReadVariableOp?.layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1?layer1.1.bn1/ReadVariableOp?layer1.1.bn1/ReadVariableOp_1?,layer1.1.bn2/FusedBatchNormV3/ReadVariableOp?.layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1?layer1.1.bn2/ReadVariableOp?layer1.1.bn2/ReadVariableOp_1?$layer1.1.conv1/Conv2D/ReadVariableOp?$layer1.1.conv2/Conv2D/ReadVariableOp?,layer2.0.bn1/FusedBatchNormV3/ReadVariableOp?.layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1?layer2.0.bn1/ReadVariableOp?layer2.0.bn1/ReadVariableOp_1?,layer2.0.bn2/FusedBatchNormV3/ReadVariableOp?.layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1?layer2.0.bn2/ReadVariableOp?layer2.0.bn2/ReadVariableOp_1?$layer2.0.conv1/Conv2D/ReadVariableOp?$layer2.0.conv2/Conv2D/ReadVariableOp?+layer2.0.downsample.0/Conv2D/ReadVariableOp?5layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp?7layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1?$layer2.0.downsample.1/ReadVariableOp?&layer2.0.downsample.1/ReadVariableOp_1?,layer2.1.bn1/FusedBatchNormV3/ReadVariableOp?.layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1?layer2.1.bn1/ReadVariableOp?layer2.1.bn1/ReadVariableOp_1?,layer2.1.bn2/FusedBatchNormV3/ReadVariableOp?.layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1?layer2.1.bn2/ReadVariableOp?layer2.1.bn2/ReadVariableOp_1?$layer2.1.conv1/Conv2D/ReadVariableOp?$layer2.1.conv2/Conv2D/ReadVariableOp?,layer3.0.bn1/FusedBatchNormV3/ReadVariableOp?.layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1?layer3.0.bn1/ReadVariableOp?layer3.0.bn1/ReadVariableOp_1?,layer3.0.bn2/FusedBatchNormV3/ReadVariableOp?.layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1?layer3.0.bn2/ReadVariableOp?layer3.0.bn2/ReadVariableOp_1?$layer3.0.conv1/Conv2D/ReadVariableOp?$layer3.0.conv2/Conv2D/ReadVariableOp?+layer3.0.downsample.0/Conv2D/ReadVariableOp?5layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp?7layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1?$layer3.0.downsample.1/ReadVariableOp?&layer3.0.downsample.1/ReadVariableOp_1?,layer3.1.bn1/FusedBatchNormV3/ReadVariableOp?.layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1?layer3.1.bn1/ReadVariableOp?layer3.1.bn1/ReadVariableOp_1?,layer3.1.bn2/FusedBatchNormV3/ReadVariableOp?.layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1?layer3.1.bn2/ReadVariableOp?layer3.1.bn2/ReadVariableOp_1?$layer3.1.conv1/Conv2D/ReadVariableOp?$layer3.1.conv2/Conv2D/ReadVariableOp?
pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
pad/Pad/paddings?
pad/PadPadinputspad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
pad/Pad?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dpad/Pad:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
conv1/Conv2D?
bn1/ReadVariableOpReadVariableOpbn1_readvariableop_resource*
_output_shapes
:@*
dtype02
bn1/ReadVariableOp?
bn1/ReadVariableOp_1ReadVariableOpbn1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
bn1/ReadVariableOp_1?
#bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02%
#bn1/FusedBatchNormV3/ReadVariableOp?
%bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02'
%bn1/FusedBatchNormV3/ReadVariableOp_1?
bn1/FusedBatchNormV3FusedBatchNormV3conv1/Conv2D:output:0bn1/ReadVariableOp:value:0bn1/ReadVariableOp_1:value:0+bn1/FusedBatchNormV3/ReadVariableOp:value:0-bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
bn1/FusedBatchNormV3?
	relu/ReluRelubn1/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
	relu/Relu?
pad1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
pad1/Pad/paddings?
pad1/PadPadrelu/Relu:activations:0pad1/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

pad1/Pad?
maxpool/MaxPoolMaxPoolpad1/Pad:output:0*A
_output_shapes/
-:+???????????????????????????@*
ksize
*
paddingVALID*
strides
2
maxpool/MaxPool?
$layer1.0.conv1/Conv2D/ReadVariableOpReadVariableOp-layer1_0_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$layer1.0.conv1/Conv2D/ReadVariableOp?
layer1.0.conv1/Conv2DConv2Dmaxpool/MaxPool:output:0,layer1.0.conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
layer1.0.conv1/Conv2D?
layer1.0.bn1/ReadVariableOpReadVariableOp$layer1_0_bn1_readvariableop_resource*
_output_shapes
:@*
dtype02
layer1.0.bn1/ReadVariableOp?
layer1.0.bn1/ReadVariableOp_1ReadVariableOp&layer1_0_bn1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
layer1.0.bn1/ReadVariableOp_1?
,layer1.0.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer1_0_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer1.0.bn1/FusedBatchNormV3/ReadVariableOp?
.layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer1_0_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1?
layer1.0.bn1/FusedBatchNormV3FusedBatchNormV3layer1.0.conv1/Conv2D:output:0#layer1.0.bn1/ReadVariableOp:value:0%layer1.0.bn1/ReadVariableOp_1:value:04layer1.0.bn1/FusedBatchNormV3/ReadVariableOp:value:06layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
layer1.0.bn1/FusedBatchNormV3?
activation/ReluRelu!layer1.0.bn1/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
activation/Relu?
$layer1.0.conv2/Conv2D/ReadVariableOpReadVariableOp-layer1_0_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$layer1.0.conv2/Conv2D/ReadVariableOp?
layer1.0.conv2/Conv2DConv2Dactivation/Relu:activations:0,layer1.0.conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
layer1.0.conv2/Conv2D?
layer1.0.bn2/ReadVariableOpReadVariableOp$layer1_0_bn2_readvariableop_resource*
_output_shapes
:@*
dtype02
layer1.0.bn2/ReadVariableOp?
layer1.0.bn2/ReadVariableOp_1ReadVariableOp&layer1_0_bn2_readvariableop_1_resource*
_output_shapes
:@*
dtype02
layer1.0.bn2/ReadVariableOp_1?
,layer1.0.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer1_0_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer1.0.bn2/FusedBatchNormV3/ReadVariableOp?
.layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer1_0_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1?
layer1.0.bn2/FusedBatchNormV3FusedBatchNormV3layer1.0.conv2/Conv2D:output:0#layer1.0.bn2/ReadVariableOp:value:0%layer1.0.bn2/ReadVariableOp_1:value:04layer1.0.bn2/FusedBatchNormV3/ReadVariableOp:value:06layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
layer1.0.bn2/FusedBatchNormV3?
add/addAddV2maxpool/MaxPool:output:0!layer1.0.bn2/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
add/add?
activation_1/ReluReluadd/add:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
activation_1/Relu?
$layer1.1.conv1/Conv2D/ReadVariableOpReadVariableOp-layer1_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$layer1.1.conv1/Conv2D/ReadVariableOp?
layer1.1.conv1/Conv2DConv2Dactivation_1/Relu:activations:0,layer1.1.conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
layer1.1.conv1/Conv2D?
layer1.1.bn1/ReadVariableOpReadVariableOp$layer1_1_bn1_readvariableop_resource*
_output_shapes
:@*
dtype02
layer1.1.bn1/ReadVariableOp?
layer1.1.bn1/ReadVariableOp_1ReadVariableOp&layer1_1_bn1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
layer1.1.bn1/ReadVariableOp_1?
,layer1.1.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer1_1_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer1.1.bn1/FusedBatchNormV3/ReadVariableOp?
.layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer1_1_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1?
layer1.1.bn1/FusedBatchNormV3FusedBatchNormV3layer1.1.conv1/Conv2D:output:0#layer1.1.bn1/ReadVariableOp:value:0%layer1.1.bn1/ReadVariableOp_1:value:04layer1.1.bn1/FusedBatchNormV3/ReadVariableOp:value:06layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
layer1.1.bn1/FusedBatchNormV3?
activation_2/ReluRelu!layer1.1.bn1/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
activation_2/Relu?
$layer1.1.conv2/Conv2D/ReadVariableOpReadVariableOp-layer1_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$layer1.1.conv2/Conv2D/ReadVariableOp?
layer1.1.conv2/Conv2DConv2Dactivation_2/Relu:activations:0,layer1.1.conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
layer1.1.conv2/Conv2D?
layer1.1.bn2/ReadVariableOpReadVariableOp$layer1_1_bn2_readvariableop_resource*
_output_shapes
:@*
dtype02
layer1.1.bn2/ReadVariableOp?
layer1.1.bn2/ReadVariableOp_1ReadVariableOp&layer1_1_bn2_readvariableop_1_resource*
_output_shapes
:@*
dtype02
layer1.1.bn2/ReadVariableOp_1?
,layer1.1.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer1_1_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer1.1.bn2/FusedBatchNormV3/ReadVariableOp?
.layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer1_1_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1?
layer1.1.bn2/FusedBatchNormV3FusedBatchNormV3layer1.1.conv2/Conv2D:output:0#layer1.1.bn2/ReadVariableOp:value:0%layer1.1.bn2/ReadVariableOp_1:value:04layer1.1.bn2/FusedBatchNormV3/ReadVariableOp:value:06layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
layer1.1.bn2/FusedBatchNormV3?
	add_1/addAddV2activation_1/Relu:activations:0!layer1.1.bn2/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
	add_1/add?
activation_3/ReluReluadd_1/add:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
activation_3/Relu?
layer2.0.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
layer2.0.pad/Pad/paddings?
layer2.0.pad/PadPadactivation_3/Relu:activations:0"layer2.0.pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
layer2.0.pad/Pad?
$layer2.0.conv1/Conv2D/ReadVariableOpReadVariableOp-layer2_0_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02&
$layer2.0.conv1/Conv2D/ReadVariableOp?
layer2.0.conv1/Conv2DConv2Dlayer2.0.pad/Pad:output:0,layer2.0.conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
layer2.0.conv1/Conv2D?
layer2.0.bn1/ReadVariableOpReadVariableOp$layer2_0_bn1_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer2.0.bn1/ReadVariableOp?
layer2.0.bn1/ReadVariableOp_1ReadVariableOp&layer2_0_bn1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
layer2.0.bn1/ReadVariableOp_1?
,layer2.0.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer2_0_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,layer2.0.bn1/FusedBatchNormV3/ReadVariableOp?
.layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer2_0_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1?
layer2.0.bn1/FusedBatchNormV3FusedBatchNormV3layer2.0.conv1/Conv2D:output:0#layer2.0.bn1/ReadVariableOp:value:0%layer2.0.bn1/ReadVariableOp_1:value:04layer2.0.bn1/FusedBatchNormV3/ReadVariableOp:value:06layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
layer2.0.bn1/FusedBatchNormV3?
activation_4/ReluRelu!layer2.0.bn1/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
activation_4/Relu?
$layer2.0.conv2/Conv2D/ReadVariableOpReadVariableOp-layer2_0_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$layer2.0.conv2/Conv2D/ReadVariableOp?
layer2.0.conv2/Conv2DConv2Dactivation_4/Relu:activations:0,layer2.0.conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
layer2.0.conv2/Conv2D?
+layer2.0.downsample.0/Conv2D/ReadVariableOpReadVariableOp4layer2_0_downsample_0_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02-
+layer2.0.downsample.0/Conv2D/ReadVariableOp?
layer2.0.downsample.0/Conv2DConv2Dactivation_3/Relu:activations:03layer2.0.downsample.0/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
layer2.0.downsample.0/Conv2D?
$layer2.0.downsample.1/ReadVariableOpReadVariableOp-layer2_0_downsample_1_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$layer2.0.downsample.1/ReadVariableOp?
&layer2.0.downsample.1/ReadVariableOp_1ReadVariableOp/layer2_0_downsample_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&layer2.0.downsample.1/ReadVariableOp_1?
5layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOpReadVariableOp>layer2_0_downsample_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp?
7layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@layer2_0_downsample_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1?
&layer2.0.downsample.1/FusedBatchNormV3FusedBatchNormV3%layer2.0.downsample.0/Conv2D:output:0,layer2.0.downsample.1/ReadVariableOp:value:0.layer2.0.downsample.1/ReadVariableOp_1:value:0=layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp:value:0?layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2(
&layer2.0.downsample.1/FusedBatchNormV3?
layer2.0.bn2/ReadVariableOpReadVariableOp$layer2_0_bn2_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer2.0.bn2/ReadVariableOp?
layer2.0.bn2/ReadVariableOp_1ReadVariableOp&layer2_0_bn2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
layer2.0.bn2/ReadVariableOp_1?
,layer2.0.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer2_0_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,layer2.0.bn2/FusedBatchNormV3/ReadVariableOp?
.layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer2_0_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1?
layer2.0.bn2/FusedBatchNormV3FusedBatchNormV3layer2.0.conv2/Conv2D:output:0#layer2.0.bn2/ReadVariableOp:value:0%layer2.0.bn2/ReadVariableOp_1:value:04layer2.0.bn2/FusedBatchNormV3/ReadVariableOp:value:06layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
layer2.0.bn2/FusedBatchNormV3?
	add_2/addAddV2*layer2.0.downsample.1/FusedBatchNormV3:y:0!layer2.0.bn2/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
	add_2/add?
activation_5/ReluReluadd_2/add:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
activation_5/Relu?
$layer2.1.conv1/Conv2D/ReadVariableOpReadVariableOp-layer2_1_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$layer2.1.conv1/Conv2D/ReadVariableOp?
layer2.1.conv1/Conv2DConv2Dactivation_5/Relu:activations:0,layer2.1.conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
layer2.1.conv1/Conv2D?
layer2.1.bn1/ReadVariableOpReadVariableOp$layer2_1_bn1_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer2.1.bn1/ReadVariableOp?
layer2.1.bn1/ReadVariableOp_1ReadVariableOp&layer2_1_bn1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
layer2.1.bn1/ReadVariableOp_1?
,layer2.1.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer2_1_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,layer2.1.bn1/FusedBatchNormV3/ReadVariableOp?
.layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer2_1_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1?
layer2.1.bn1/FusedBatchNormV3FusedBatchNormV3layer2.1.conv1/Conv2D:output:0#layer2.1.bn1/ReadVariableOp:value:0%layer2.1.bn1/ReadVariableOp_1:value:04layer2.1.bn1/FusedBatchNormV3/ReadVariableOp:value:06layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
layer2.1.bn1/FusedBatchNormV3?
activation_6/ReluRelu!layer2.1.bn1/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
activation_6/Relu?
$layer2.1.conv2/Conv2D/ReadVariableOpReadVariableOp-layer2_1_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$layer2.1.conv2/Conv2D/ReadVariableOp?
layer2.1.conv2/Conv2DConv2Dactivation_6/Relu:activations:0,layer2.1.conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
layer2.1.conv2/Conv2D?
layer2.1.bn2/ReadVariableOpReadVariableOp$layer2_1_bn2_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer2.1.bn2/ReadVariableOp?
layer2.1.bn2/ReadVariableOp_1ReadVariableOp&layer2_1_bn2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
layer2.1.bn2/ReadVariableOp_1?
,layer2.1.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer2_1_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,layer2.1.bn2/FusedBatchNormV3/ReadVariableOp?
.layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer2_1_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1?
layer2.1.bn2/FusedBatchNormV3FusedBatchNormV3layer2.1.conv2/Conv2D:output:0#layer2.1.bn2/ReadVariableOp:value:0%layer2.1.bn2/ReadVariableOp_1:value:04layer2.1.bn2/FusedBatchNormV3/ReadVariableOp:value:06layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
layer2.1.bn2/FusedBatchNormV3?
	add_3/addAddV2activation_5/Relu:activations:0!layer2.1.bn2/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
	add_3/add?
activation_7/ReluReluadd_3/add:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
activation_7/Relu?
layer3.0.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
layer3.0.pad/Pad/paddings?
layer3.0.pad/PadPadactivation_7/Relu:activations:0"layer3.0.pad/Pad/paddings:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer3.0.pad/Pad?
$layer3.0.conv1/Conv2D/ReadVariableOpReadVariableOp-layer3_0_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$layer3.0.conv1/Conv2D/ReadVariableOp?
layer3.0.conv1/Conv2DConv2Dlayer3.0.pad/Pad:output:0,layer3.0.conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
layer3.0.conv1/Conv2D?
layer3.0.bn1/ReadVariableOpReadVariableOp$layer3_0_bn1_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer3.0.bn1/ReadVariableOp?
layer3.0.bn1/ReadVariableOp_1ReadVariableOp&layer3_0_bn1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
layer3.0.bn1/ReadVariableOp_1?
,layer3.0.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer3_0_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,layer3.0.bn1/FusedBatchNormV3/ReadVariableOp?
.layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer3_0_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1?
layer3.0.bn1/FusedBatchNormV3FusedBatchNormV3layer3.0.conv1/Conv2D:output:0#layer3.0.bn1/ReadVariableOp:value:0%layer3.0.bn1/ReadVariableOp_1:value:04layer3.0.bn1/FusedBatchNormV3/ReadVariableOp:value:06layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
layer3.0.bn1/FusedBatchNormV3?
activation_8/ReluRelu!layer3.0.bn1/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
activation_8/Relu?
$layer3.0.conv2/Conv2D/ReadVariableOpReadVariableOp-layer3_0_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$layer3.0.conv2/Conv2D/ReadVariableOp?
layer3.0.conv2/Conv2DConv2Dactivation_8/Relu:activations:0,layer3.0.conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
layer3.0.conv2/Conv2D?
+layer3.0.downsample.0/Conv2D/ReadVariableOpReadVariableOp4layer3_0_downsample_0_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02-
+layer3.0.downsample.0/Conv2D/ReadVariableOp?
layer3.0.downsample.0/Conv2DConv2Dactivation_7/Relu:activations:03layer3.0.downsample.0/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
layer3.0.downsample.0/Conv2D?
$layer3.0.downsample.1/ReadVariableOpReadVariableOp-layer3_0_downsample_1_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$layer3.0.downsample.1/ReadVariableOp?
&layer3.0.downsample.1/ReadVariableOp_1ReadVariableOp/layer3_0_downsample_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&layer3.0.downsample.1/ReadVariableOp_1?
5layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOpReadVariableOp>layer3_0_downsample_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp?
7layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@layer3_0_downsample_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1?
&layer3.0.downsample.1/FusedBatchNormV3FusedBatchNormV3%layer3.0.downsample.0/Conv2D:output:0,layer3.0.downsample.1/ReadVariableOp:value:0.layer3.0.downsample.1/ReadVariableOp_1:value:0=layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp:value:0?layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2(
&layer3.0.downsample.1/FusedBatchNormV3?
layer3.0.bn2/ReadVariableOpReadVariableOp$layer3_0_bn2_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer3.0.bn2/ReadVariableOp?
layer3.0.bn2/ReadVariableOp_1ReadVariableOp&layer3_0_bn2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
layer3.0.bn2/ReadVariableOp_1?
,layer3.0.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer3_0_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,layer3.0.bn2/FusedBatchNormV3/ReadVariableOp?
.layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer3_0_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1?
layer3.0.bn2/FusedBatchNormV3FusedBatchNormV3layer3.0.conv2/Conv2D:output:0#layer3.0.bn2/ReadVariableOp:value:0%layer3.0.bn2/ReadVariableOp_1:value:04layer3.0.bn2/FusedBatchNormV3/ReadVariableOp:value:06layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
layer3.0.bn2/FusedBatchNormV3?
	add_4/addAddV2*layer3.0.downsample.1/FusedBatchNormV3:y:0!layer3.0.bn2/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
	add_4/add?
activation_9/ReluReluadd_4/add:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
activation_9/Relu?
$layer3.1.conv1/Conv2D/ReadVariableOpReadVariableOp-layer3_1_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$layer3.1.conv1/Conv2D/ReadVariableOp?
layer3.1.conv1/Conv2DConv2Dactivation_9/Relu:activations:0,layer3.1.conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
layer3.1.conv1/Conv2D?
layer3.1.bn1/ReadVariableOpReadVariableOp$layer3_1_bn1_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer3.1.bn1/ReadVariableOp?
layer3.1.bn1/ReadVariableOp_1ReadVariableOp&layer3_1_bn1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
layer3.1.bn1/ReadVariableOp_1?
,layer3.1.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer3_1_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,layer3.1.bn1/FusedBatchNormV3/ReadVariableOp?
.layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer3_1_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1?
layer3.1.bn1/FusedBatchNormV3FusedBatchNormV3layer3.1.conv1/Conv2D:output:0#layer3.1.bn1/ReadVariableOp:value:0%layer3.1.bn1/ReadVariableOp_1:value:04layer3.1.bn1/FusedBatchNormV3/ReadVariableOp:value:06layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
layer3.1.bn1/FusedBatchNormV3?
activation_10/ReluRelu!layer3.1.bn1/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
activation_10/Relu?
$layer3.1.conv2/Conv2D/ReadVariableOpReadVariableOp-layer3_1_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$layer3.1.conv2/Conv2D/ReadVariableOp?
layer3.1.conv2/Conv2DConv2D activation_10/Relu:activations:0,layer3.1.conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
layer3.1.conv2/Conv2D?
layer3.1.bn2/ReadVariableOpReadVariableOp$layer3_1_bn2_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer3.1.bn2/ReadVariableOp?
layer3.1.bn2/ReadVariableOp_1ReadVariableOp&layer3_1_bn2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
layer3.1.bn2/ReadVariableOp_1?
,layer3.1.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer3_1_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,layer3.1.bn2/FusedBatchNormV3/ReadVariableOp?
.layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer3_1_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1?
layer3.1.bn2/FusedBatchNormV3FusedBatchNormV3layer3.1.conv2/Conv2D:output:0#layer3.1.bn2/ReadVariableOp:value:0%layer3.1.bn2/ReadVariableOp_1:value:04layer3.1.bn2/FusedBatchNormV3/ReadVariableOp:value:06layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
layer3.1.bn2/FusedBatchNormV3?
	add_5/addAddV2activation_9/Relu:activations:0!layer3.1.bn2/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
	add_5/add?
activation_11/ReluReluadd_5/add:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
activation_11/Relu?
IdentityIdentity activation_11/Relu:activations:0$^bn1/FusedBatchNormV3/ReadVariableOp&^bn1/FusedBatchNormV3/ReadVariableOp_1^bn1/ReadVariableOp^bn1/ReadVariableOp_1^conv1/Conv2D/ReadVariableOp-^layer1.0.bn1/FusedBatchNormV3/ReadVariableOp/^layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1^layer1.0.bn1/ReadVariableOp^layer1.0.bn1/ReadVariableOp_1-^layer1.0.bn2/FusedBatchNormV3/ReadVariableOp/^layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1^layer1.0.bn2/ReadVariableOp^layer1.0.bn2/ReadVariableOp_1%^layer1.0.conv1/Conv2D/ReadVariableOp%^layer1.0.conv2/Conv2D/ReadVariableOp-^layer1.1.bn1/FusedBatchNormV3/ReadVariableOp/^layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1^layer1.1.bn1/ReadVariableOp^layer1.1.bn1/ReadVariableOp_1-^layer1.1.bn2/FusedBatchNormV3/ReadVariableOp/^layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1^layer1.1.bn2/ReadVariableOp^layer1.1.bn2/ReadVariableOp_1%^layer1.1.conv1/Conv2D/ReadVariableOp%^layer1.1.conv2/Conv2D/ReadVariableOp-^layer2.0.bn1/FusedBatchNormV3/ReadVariableOp/^layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1^layer2.0.bn1/ReadVariableOp^layer2.0.bn1/ReadVariableOp_1-^layer2.0.bn2/FusedBatchNormV3/ReadVariableOp/^layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1^layer2.0.bn2/ReadVariableOp^layer2.0.bn2/ReadVariableOp_1%^layer2.0.conv1/Conv2D/ReadVariableOp%^layer2.0.conv2/Conv2D/ReadVariableOp,^layer2.0.downsample.0/Conv2D/ReadVariableOp6^layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp8^layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1%^layer2.0.downsample.1/ReadVariableOp'^layer2.0.downsample.1/ReadVariableOp_1-^layer2.1.bn1/FusedBatchNormV3/ReadVariableOp/^layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1^layer2.1.bn1/ReadVariableOp^layer2.1.bn1/ReadVariableOp_1-^layer2.1.bn2/FusedBatchNormV3/ReadVariableOp/^layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1^layer2.1.bn2/ReadVariableOp^layer2.1.bn2/ReadVariableOp_1%^layer2.1.conv1/Conv2D/ReadVariableOp%^layer2.1.conv2/Conv2D/ReadVariableOp-^layer3.0.bn1/FusedBatchNormV3/ReadVariableOp/^layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1^layer3.0.bn1/ReadVariableOp^layer3.0.bn1/ReadVariableOp_1-^layer3.0.bn2/FusedBatchNormV3/ReadVariableOp/^layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1^layer3.0.bn2/ReadVariableOp^layer3.0.bn2/ReadVariableOp_1%^layer3.0.conv1/Conv2D/ReadVariableOp%^layer3.0.conv2/Conv2D/ReadVariableOp,^layer3.0.downsample.0/Conv2D/ReadVariableOp6^layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp8^layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1%^layer3.0.downsample.1/ReadVariableOp'^layer3.0.downsample.1/ReadVariableOp_1-^layer3.1.bn1/FusedBatchNormV3/ReadVariableOp/^layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1^layer3.1.bn1/ReadVariableOp^layer3.1.bn1/ReadVariableOp_1-^layer3.1.bn2/FusedBatchNormV3/ReadVariableOp/^layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1^layer3.1.bn2/ReadVariableOp^layer3.1.bn2/ReadVariableOp_1%^layer3.1.conv1/Conv2D/ReadVariableOp%^layer3.1.conv2/Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:+???????????????????????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2J
#bn1/FusedBatchNormV3/ReadVariableOp#bn1/FusedBatchNormV3/ReadVariableOp2N
%bn1/FusedBatchNormV3/ReadVariableOp_1%bn1/FusedBatchNormV3/ReadVariableOp_12(
bn1/ReadVariableOpbn1/ReadVariableOp2,
bn1/ReadVariableOp_1bn1/ReadVariableOp_12:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2\
,layer1.0.bn1/FusedBatchNormV3/ReadVariableOp,layer1.0.bn1/FusedBatchNormV3/ReadVariableOp2`
.layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1.layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_12:
layer1.0.bn1/ReadVariableOplayer1.0.bn1/ReadVariableOp2>
layer1.0.bn1/ReadVariableOp_1layer1.0.bn1/ReadVariableOp_12\
,layer1.0.bn2/FusedBatchNormV3/ReadVariableOp,layer1.0.bn2/FusedBatchNormV3/ReadVariableOp2`
.layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1.layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_12:
layer1.0.bn2/ReadVariableOplayer1.0.bn2/ReadVariableOp2>
layer1.0.bn2/ReadVariableOp_1layer1.0.bn2/ReadVariableOp_12L
$layer1.0.conv1/Conv2D/ReadVariableOp$layer1.0.conv1/Conv2D/ReadVariableOp2L
$layer1.0.conv2/Conv2D/ReadVariableOp$layer1.0.conv2/Conv2D/ReadVariableOp2\
,layer1.1.bn1/FusedBatchNormV3/ReadVariableOp,layer1.1.bn1/FusedBatchNormV3/ReadVariableOp2`
.layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1.layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_12:
layer1.1.bn1/ReadVariableOplayer1.1.bn1/ReadVariableOp2>
layer1.1.bn1/ReadVariableOp_1layer1.1.bn1/ReadVariableOp_12\
,layer1.1.bn2/FusedBatchNormV3/ReadVariableOp,layer1.1.bn2/FusedBatchNormV3/ReadVariableOp2`
.layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1.layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_12:
layer1.1.bn2/ReadVariableOplayer1.1.bn2/ReadVariableOp2>
layer1.1.bn2/ReadVariableOp_1layer1.1.bn2/ReadVariableOp_12L
$layer1.1.conv1/Conv2D/ReadVariableOp$layer1.1.conv1/Conv2D/ReadVariableOp2L
$layer1.1.conv2/Conv2D/ReadVariableOp$layer1.1.conv2/Conv2D/ReadVariableOp2\
,layer2.0.bn1/FusedBatchNormV3/ReadVariableOp,layer2.0.bn1/FusedBatchNormV3/ReadVariableOp2`
.layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1.layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_12:
layer2.0.bn1/ReadVariableOplayer2.0.bn1/ReadVariableOp2>
layer2.0.bn1/ReadVariableOp_1layer2.0.bn1/ReadVariableOp_12\
,layer2.0.bn2/FusedBatchNormV3/ReadVariableOp,layer2.0.bn2/FusedBatchNormV3/ReadVariableOp2`
.layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1.layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_12:
layer2.0.bn2/ReadVariableOplayer2.0.bn2/ReadVariableOp2>
layer2.0.bn2/ReadVariableOp_1layer2.0.bn2/ReadVariableOp_12L
$layer2.0.conv1/Conv2D/ReadVariableOp$layer2.0.conv1/Conv2D/ReadVariableOp2L
$layer2.0.conv2/Conv2D/ReadVariableOp$layer2.0.conv2/Conv2D/ReadVariableOp2Z
+layer2.0.downsample.0/Conv2D/ReadVariableOp+layer2.0.downsample.0/Conv2D/ReadVariableOp2n
5layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp5layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp2r
7layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_17layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_12L
$layer2.0.downsample.1/ReadVariableOp$layer2.0.downsample.1/ReadVariableOp2P
&layer2.0.downsample.1/ReadVariableOp_1&layer2.0.downsample.1/ReadVariableOp_12\
,layer2.1.bn1/FusedBatchNormV3/ReadVariableOp,layer2.1.bn1/FusedBatchNormV3/ReadVariableOp2`
.layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1.layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_12:
layer2.1.bn1/ReadVariableOplayer2.1.bn1/ReadVariableOp2>
layer2.1.bn1/ReadVariableOp_1layer2.1.bn1/ReadVariableOp_12\
,layer2.1.bn2/FusedBatchNormV3/ReadVariableOp,layer2.1.bn2/FusedBatchNormV3/ReadVariableOp2`
.layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1.layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_12:
layer2.1.bn2/ReadVariableOplayer2.1.bn2/ReadVariableOp2>
layer2.1.bn2/ReadVariableOp_1layer2.1.bn2/ReadVariableOp_12L
$layer2.1.conv1/Conv2D/ReadVariableOp$layer2.1.conv1/Conv2D/ReadVariableOp2L
$layer2.1.conv2/Conv2D/ReadVariableOp$layer2.1.conv2/Conv2D/ReadVariableOp2\
,layer3.0.bn1/FusedBatchNormV3/ReadVariableOp,layer3.0.bn1/FusedBatchNormV3/ReadVariableOp2`
.layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1.layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_12:
layer3.0.bn1/ReadVariableOplayer3.0.bn1/ReadVariableOp2>
layer3.0.bn1/ReadVariableOp_1layer3.0.bn1/ReadVariableOp_12\
,layer3.0.bn2/FusedBatchNormV3/ReadVariableOp,layer3.0.bn2/FusedBatchNormV3/ReadVariableOp2`
.layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1.layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_12:
layer3.0.bn2/ReadVariableOplayer3.0.bn2/ReadVariableOp2>
layer3.0.bn2/ReadVariableOp_1layer3.0.bn2/ReadVariableOp_12L
$layer3.0.conv1/Conv2D/ReadVariableOp$layer3.0.conv1/Conv2D/ReadVariableOp2L
$layer3.0.conv2/Conv2D/ReadVariableOp$layer3.0.conv2/Conv2D/ReadVariableOp2Z
+layer3.0.downsample.0/Conv2D/ReadVariableOp+layer3.0.downsample.0/Conv2D/ReadVariableOp2n
5layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp5layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp2r
7layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_17layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_12L
$layer3.0.downsample.1/ReadVariableOp$layer3.0.downsample.1/ReadVariableOp2P
&layer3.0.downsample.1/ReadVariableOp_1&layer3.0.downsample.1/ReadVariableOp_12\
,layer3.1.bn1/FusedBatchNormV3/ReadVariableOp,layer3.1.bn1/FusedBatchNormV3/ReadVariableOp2`
.layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1.layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_12:
layer3.1.bn1/ReadVariableOplayer3.1.bn1/ReadVariableOp2>
layer3.1.bn1/ReadVariableOp_1layer3.1.bn1/ReadVariableOp_12\
,layer3.1.bn2/FusedBatchNormV3/ReadVariableOp,layer3.1.bn2/FusedBatchNormV3/ReadVariableOp2`
.layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1.layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_12:
layer3.1.bn2/ReadVariableOplayer3.1.bn2/ReadVariableOp2>
layer3.1.bn2/ReadVariableOp_1layer3.1.bn2/ReadVariableOp_12L
$layer3.1.conv1/Conv2D/ReadVariableOp$layer3.1.conv1/Conv2D/ReadVariableOp2L
$layer3.1.conv2/Conv2D/ReadVariableOp$layer3.1.conv2/Conv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
Q
%__inference_add_4_layer_call_fn_14884
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_4_layer_call_and_return_conditional_losses_115322
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:,????????????????????????????:,????????????????????????????:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/0:lh
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/1
?
?
I__inference_layer3.1.conv2_layer_call_and_return_conditional_losses_14989

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
l
@__inference_add_5_layer_call_and_return_conditional_losses_15066
inputs_0
inputs_1
identityt
addAddV2inputs_0inputs_1*
T0*B
_output_shapes0
.:,????????????????????????????2
addv
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:,????????????????????????????:,????????????????????????????:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/0:lh
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/1
?
?
G__inference_layer2.0.bn1_layer_call_and_return_conditional_losses_14208

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_activation_layer_call_and_return_conditional_losses_13881

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
t
.__inference_layer3.1.conv1_layer_call_fn_14908

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.1.conv1_layer_call_and_return_conditional_losses_115612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_layer1.0.conv1_layer_call_and_return_conditional_losses_10706

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
,__inference_layer1.1.bn1_layer_call_fn_14064

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.1.bn1_layer_call_and_return_conditional_losses_94442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
F__inference_layer2.0.bn1_layer_call_and_return_conditional_losses_9665

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer3.1.bn1_layer_call_and_return_conditional_losses_14928

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_layer1.0.conv2_layer_call_and_return_conditional_losses_10773

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
5__inference_layer2.0.downsample.1_layer_call_fn_14354

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_layer2.0.downsample.1_layer_call_and_return_conditional_losses_97692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
l
@__inference_add_2_layer_call_and_return_conditional_losses_14424
inputs_0
inputs_1
identityt
addAddV2inputs_0inputs_1*
T0*B
_output_shapes0
.:,????????????????????????????2
addv
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:,????????????????????????????:,????????????????????????????:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/0:lh
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/1
?
?
5__inference_layer3.0.downsample.1_layer_call_fn_14795

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_layer3.0.downsample.1_layer_call_and_return_conditional_losses_102712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
[
?__inference_relu_layer_call_and_return_conditional_losses_10689

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
G__inference_layer1.1.bn1_layer_call_and_return_conditional_losses_14020

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
t
.__inference_layer3.1.conv2_layer_call_fn_14996

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.1.conv2_layer_call_and_return_conditional_losses_116282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_layer2.1.bn1_layer_call_and_return_conditional_losses_9977

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_layer1.0.bn2_layer_call_fn_13951

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.0.bn2_layer_call_and_return_conditional_losses_93092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
F
*__inference_activation_layer_call_fn_13886

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_107582
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
,__inference_layer2.0.bn1_layer_call_fn_14252

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.0.bn1_layer_call_and_return_conditional_losses_96652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
{
5__inference_layer2.0.downsample.0_layer_call_fn_14276

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_layer2.0.downsample.0_layer_call_and_return_conditional_losses_110912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
G__inference_layer3.1.bn2_layer_call_and_return_conditional_losses_15016

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_layer2.0.conv2_layer_call_and_return_conditional_losses_11072

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_bn1_layer_call_fn_13775

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *F
fAR?
=__inference_bn1_layer_call_and_return_conditional_losses_90762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
G__inference_layer3.1.bn1_layer_call_and_return_conditional_losses_14946

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
Q
%__inference_add_3_layer_call_fn_14618
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_113282
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:,????????????????????????????:,????????????????????????????:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/0:lh
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/1
?
Y
=__inference_pad_layer_call_and_return_conditional_losses_9008

inputs
identity?
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Pad/paddings?
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Pad?
IdentityIdentityPad:output:0*
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
?
?
G__inference_layer3.0.bn2_layer_call_and_return_conditional_losses_14846

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_layer2.0.conv2_layer_call_and_return_conditional_losses_14283

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
l
@__inference_add_3_layer_call_and_return_conditional_losses_14612
inputs_0
inputs_1
identityt
addAddV2inputs_0inputs_1*
T0*B
_output_shapes0
.:,????????????????????????????2
addv
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:,????????????????????????????:,????????????????????????????:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/0:lh
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/1
?
?

#__inference_signature_wrapper_12796
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

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73
identity??StatefulPartitionedCall?

StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*m
_read_only_resource_inputsO
MK	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJK*2
config_proto" 

CPU

GPU2 *0J 8? *(
f#R!
__inference__wrapped_model_90012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:+???????????????????????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+???????????????????????????
!
_user_specified_name	input_1
?
?
,__inference_layer2.0.bn2_layer_call_fn_14418

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.0.bn2_layer_call_and_return_conditional_losses_98732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_activation_8_layer_call_and_return_conditional_losses_14711

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?=
@__inference_model_layer_call_and_return_conditional_losses_13113

inputs(
$conv1_conv2d_readvariableop_resource
bn1_readvariableop_resource!
bn1_readvariableop_1_resource0
,bn1_fusedbatchnormv3_readvariableop_resource2
.bn1_fusedbatchnormv3_readvariableop_1_resource1
-layer1_0_conv1_conv2d_readvariableop_resource(
$layer1_0_bn1_readvariableop_resource*
&layer1_0_bn1_readvariableop_1_resource9
5layer1_0_bn1_fusedbatchnormv3_readvariableop_resource;
7layer1_0_bn1_fusedbatchnormv3_readvariableop_1_resource1
-layer1_0_conv2_conv2d_readvariableop_resource(
$layer1_0_bn2_readvariableop_resource*
&layer1_0_bn2_readvariableop_1_resource9
5layer1_0_bn2_fusedbatchnormv3_readvariableop_resource;
7layer1_0_bn2_fusedbatchnormv3_readvariableop_1_resource1
-layer1_1_conv1_conv2d_readvariableop_resource(
$layer1_1_bn1_readvariableop_resource*
&layer1_1_bn1_readvariableop_1_resource9
5layer1_1_bn1_fusedbatchnormv3_readvariableop_resource;
7layer1_1_bn1_fusedbatchnormv3_readvariableop_1_resource1
-layer1_1_conv2_conv2d_readvariableop_resource(
$layer1_1_bn2_readvariableop_resource*
&layer1_1_bn2_readvariableop_1_resource9
5layer1_1_bn2_fusedbatchnormv3_readvariableop_resource;
7layer1_1_bn2_fusedbatchnormv3_readvariableop_1_resource1
-layer2_0_conv1_conv2d_readvariableop_resource(
$layer2_0_bn1_readvariableop_resource*
&layer2_0_bn1_readvariableop_1_resource9
5layer2_0_bn1_fusedbatchnormv3_readvariableop_resource;
7layer2_0_bn1_fusedbatchnormv3_readvariableop_1_resource1
-layer2_0_conv2_conv2d_readvariableop_resource8
4layer2_0_downsample_0_conv2d_readvariableop_resource1
-layer2_0_downsample_1_readvariableop_resource3
/layer2_0_downsample_1_readvariableop_1_resourceB
>layer2_0_downsample_1_fusedbatchnormv3_readvariableop_resourceD
@layer2_0_downsample_1_fusedbatchnormv3_readvariableop_1_resource(
$layer2_0_bn2_readvariableop_resource*
&layer2_0_bn2_readvariableop_1_resource9
5layer2_0_bn2_fusedbatchnormv3_readvariableop_resource;
7layer2_0_bn2_fusedbatchnormv3_readvariableop_1_resource1
-layer2_1_conv1_conv2d_readvariableop_resource(
$layer2_1_bn1_readvariableop_resource*
&layer2_1_bn1_readvariableop_1_resource9
5layer2_1_bn1_fusedbatchnormv3_readvariableop_resource;
7layer2_1_bn1_fusedbatchnormv3_readvariableop_1_resource1
-layer2_1_conv2_conv2d_readvariableop_resource(
$layer2_1_bn2_readvariableop_resource*
&layer2_1_bn2_readvariableop_1_resource9
5layer2_1_bn2_fusedbatchnormv3_readvariableop_resource;
7layer2_1_bn2_fusedbatchnormv3_readvariableop_1_resource1
-layer3_0_conv1_conv2d_readvariableop_resource(
$layer3_0_bn1_readvariableop_resource*
&layer3_0_bn1_readvariableop_1_resource9
5layer3_0_bn1_fusedbatchnormv3_readvariableop_resource;
7layer3_0_bn1_fusedbatchnormv3_readvariableop_1_resource1
-layer3_0_conv2_conv2d_readvariableop_resource8
4layer3_0_downsample_0_conv2d_readvariableop_resource1
-layer3_0_downsample_1_readvariableop_resource3
/layer3_0_downsample_1_readvariableop_1_resourceB
>layer3_0_downsample_1_fusedbatchnormv3_readvariableop_resourceD
@layer3_0_downsample_1_fusedbatchnormv3_readvariableop_1_resource(
$layer3_0_bn2_readvariableop_resource*
&layer3_0_bn2_readvariableop_1_resource9
5layer3_0_bn2_fusedbatchnormv3_readvariableop_resource;
7layer3_0_bn2_fusedbatchnormv3_readvariableop_1_resource1
-layer3_1_conv1_conv2d_readvariableop_resource(
$layer3_1_bn1_readvariableop_resource*
&layer3_1_bn1_readvariableop_1_resource9
5layer3_1_bn1_fusedbatchnormv3_readvariableop_resource;
7layer3_1_bn1_fusedbatchnormv3_readvariableop_1_resource1
-layer3_1_conv2_conv2d_readvariableop_resource(
$layer3_1_bn2_readvariableop_resource*
&layer3_1_bn2_readvariableop_1_resource9
5layer3_1_bn2_fusedbatchnormv3_readvariableop_resource;
7layer3_1_bn2_fusedbatchnormv3_readvariableop_1_resource
identity??bn1/AssignNewValue?bn1/AssignNewValue_1?#bn1/FusedBatchNormV3/ReadVariableOp?%bn1/FusedBatchNormV3/ReadVariableOp_1?bn1/ReadVariableOp?bn1/ReadVariableOp_1?conv1/Conv2D/ReadVariableOp?layer1.0.bn1/AssignNewValue?layer1.0.bn1/AssignNewValue_1?,layer1.0.bn1/FusedBatchNormV3/ReadVariableOp?.layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1?layer1.0.bn1/ReadVariableOp?layer1.0.bn1/ReadVariableOp_1?layer1.0.bn2/AssignNewValue?layer1.0.bn2/AssignNewValue_1?,layer1.0.bn2/FusedBatchNormV3/ReadVariableOp?.layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1?layer1.0.bn2/ReadVariableOp?layer1.0.bn2/ReadVariableOp_1?$layer1.0.conv1/Conv2D/ReadVariableOp?$layer1.0.conv2/Conv2D/ReadVariableOp?layer1.1.bn1/AssignNewValue?layer1.1.bn1/AssignNewValue_1?,layer1.1.bn1/FusedBatchNormV3/ReadVariableOp?.layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1?layer1.1.bn1/ReadVariableOp?layer1.1.bn1/ReadVariableOp_1?layer1.1.bn2/AssignNewValue?layer1.1.bn2/AssignNewValue_1?,layer1.1.bn2/FusedBatchNormV3/ReadVariableOp?.layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1?layer1.1.bn2/ReadVariableOp?layer1.1.bn2/ReadVariableOp_1?$layer1.1.conv1/Conv2D/ReadVariableOp?$layer1.1.conv2/Conv2D/ReadVariableOp?layer2.0.bn1/AssignNewValue?layer2.0.bn1/AssignNewValue_1?,layer2.0.bn1/FusedBatchNormV3/ReadVariableOp?.layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1?layer2.0.bn1/ReadVariableOp?layer2.0.bn1/ReadVariableOp_1?layer2.0.bn2/AssignNewValue?layer2.0.bn2/AssignNewValue_1?,layer2.0.bn2/FusedBatchNormV3/ReadVariableOp?.layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1?layer2.0.bn2/ReadVariableOp?layer2.0.bn2/ReadVariableOp_1?$layer2.0.conv1/Conv2D/ReadVariableOp?$layer2.0.conv2/Conv2D/ReadVariableOp?+layer2.0.downsample.0/Conv2D/ReadVariableOp?$layer2.0.downsample.1/AssignNewValue?&layer2.0.downsample.1/AssignNewValue_1?5layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp?7layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1?$layer2.0.downsample.1/ReadVariableOp?&layer2.0.downsample.1/ReadVariableOp_1?layer2.1.bn1/AssignNewValue?layer2.1.bn1/AssignNewValue_1?,layer2.1.bn1/FusedBatchNormV3/ReadVariableOp?.layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1?layer2.1.bn1/ReadVariableOp?layer2.1.bn1/ReadVariableOp_1?layer2.1.bn2/AssignNewValue?layer2.1.bn2/AssignNewValue_1?,layer2.1.bn2/FusedBatchNormV3/ReadVariableOp?.layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1?layer2.1.bn2/ReadVariableOp?layer2.1.bn2/ReadVariableOp_1?$layer2.1.conv1/Conv2D/ReadVariableOp?$layer2.1.conv2/Conv2D/ReadVariableOp?layer3.0.bn1/AssignNewValue?layer3.0.bn1/AssignNewValue_1?,layer3.0.bn1/FusedBatchNormV3/ReadVariableOp?.layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1?layer3.0.bn1/ReadVariableOp?layer3.0.bn1/ReadVariableOp_1?layer3.0.bn2/AssignNewValue?layer3.0.bn2/AssignNewValue_1?,layer3.0.bn2/FusedBatchNormV3/ReadVariableOp?.layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1?layer3.0.bn2/ReadVariableOp?layer3.0.bn2/ReadVariableOp_1?$layer3.0.conv1/Conv2D/ReadVariableOp?$layer3.0.conv2/Conv2D/ReadVariableOp?+layer3.0.downsample.0/Conv2D/ReadVariableOp?$layer3.0.downsample.1/AssignNewValue?&layer3.0.downsample.1/AssignNewValue_1?5layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp?7layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1?$layer3.0.downsample.1/ReadVariableOp?&layer3.0.downsample.1/ReadVariableOp_1?layer3.1.bn1/AssignNewValue?layer3.1.bn1/AssignNewValue_1?,layer3.1.bn1/FusedBatchNormV3/ReadVariableOp?.layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1?layer3.1.bn1/ReadVariableOp?layer3.1.bn1/ReadVariableOp_1?layer3.1.bn2/AssignNewValue?layer3.1.bn2/AssignNewValue_1?,layer3.1.bn2/FusedBatchNormV3/ReadVariableOp?.layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1?layer3.1.bn2/ReadVariableOp?layer3.1.bn2/ReadVariableOp_1?$layer3.1.conv1/Conv2D/ReadVariableOp?$layer3.1.conv2/Conv2D/ReadVariableOp?
pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
pad/Pad/paddings?
pad/PadPadinputspad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
pad/Pad?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dpad/Pad:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
conv1/Conv2D?
bn1/ReadVariableOpReadVariableOpbn1_readvariableop_resource*
_output_shapes
:@*
dtype02
bn1/ReadVariableOp?
bn1/ReadVariableOp_1ReadVariableOpbn1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
bn1/ReadVariableOp_1?
#bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02%
#bn1/FusedBatchNormV3/ReadVariableOp?
%bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02'
%bn1/FusedBatchNormV3/ReadVariableOp_1?
bn1/FusedBatchNormV3FusedBatchNormV3conv1/Conv2D:output:0bn1/ReadVariableOp:value:0bn1/ReadVariableOp_1:value:0+bn1/FusedBatchNormV3/ReadVariableOp:value:0-bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%fff?2
bn1/FusedBatchNormV3?
bn1/AssignNewValueAssignVariableOp,bn1_fusedbatchnormv3_readvariableop_resource!bn1/FusedBatchNormV3:batch_mean:0$^bn1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@bn1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
bn1/AssignNewValue?
bn1/AssignNewValue_1AssignVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource%bn1/FusedBatchNormV3:batch_variance:0&^bn1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@bn1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
bn1/AssignNewValue_1?
	relu/ReluRelubn1/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
	relu/Relu?
pad1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
pad1/Pad/paddings?
pad1/PadPadrelu/Relu:activations:0pad1/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

pad1/Pad?
maxpool/MaxPoolMaxPoolpad1/Pad:output:0*A
_output_shapes/
-:+???????????????????????????@*
ksize
*
paddingVALID*
strides
2
maxpool/MaxPool?
$layer1.0.conv1/Conv2D/ReadVariableOpReadVariableOp-layer1_0_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$layer1.0.conv1/Conv2D/ReadVariableOp?
layer1.0.conv1/Conv2DConv2Dmaxpool/MaxPool:output:0,layer1.0.conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
layer1.0.conv1/Conv2D?
layer1.0.bn1/ReadVariableOpReadVariableOp$layer1_0_bn1_readvariableop_resource*
_output_shapes
:@*
dtype02
layer1.0.bn1/ReadVariableOp?
layer1.0.bn1/ReadVariableOp_1ReadVariableOp&layer1_0_bn1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
layer1.0.bn1/ReadVariableOp_1?
,layer1.0.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer1_0_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer1.0.bn1/FusedBatchNormV3/ReadVariableOp?
.layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer1_0_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1?
layer1.0.bn1/FusedBatchNormV3FusedBatchNormV3layer1.0.conv1/Conv2D:output:0#layer1.0.bn1/ReadVariableOp:value:0%layer1.0.bn1/ReadVariableOp_1:value:04layer1.0.bn1/FusedBatchNormV3/ReadVariableOp:value:06layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%fff?2
layer1.0.bn1/FusedBatchNormV3?
layer1.0.bn1/AssignNewValueAssignVariableOp5layer1_0_bn1_fusedbatchnormv3_readvariableop_resource*layer1.0.bn1/FusedBatchNormV3:batch_mean:0-^layer1.0.bn1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*H
_class>
<:loc:@layer1.0.bn1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
layer1.0.bn1/AssignNewValue?
layer1.0.bn1/AssignNewValue_1AssignVariableOp7layer1_0_bn1_fusedbatchnormv3_readvariableop_1_resource.layer1.0.bn1/FusedBatchNormV3:batch_variance:0/^layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*J
_class@
><loc:@layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
layer1.0.bn1/AssignNewValue_1?
activation/ReluRelu!layer1.0.bn1/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
activation/Relu?
$layer1.0.conv2/Conv2D/ReadVariableOpReadVariableOp-layer1_0_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$layer1.0.conv2/Conv2D/ReadVariableOp?
layer1.0.conv2/Conv2DConv2Dactivation/Relu:activations:0,layer1.0.conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
layer1.0.conv2/Conv2D?
layer1.0.bn2/ReadVariableOpReadVariableOp$layer1_0_bn2_readvariableop_resource*
_output_shapes
:@*
dtype02
layer1.0.bn2/ReadVariableOp?
layer1.0.bn2/ReadVariableOp_1ReadVariableOp&layer1_0_bn2_readvariableop_1_resource*
_output_shapes
:@*
dtype02
layer1.0.bn2/ReadVariableOp_1?
,layer1.0.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer1_0_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer1.0.bn2/FusedBatchNormV3/ReadVariableOp?
.layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer1_0_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1?
layer1.0.bn2/FusedBatchNormV3FusedBatchNormV3layer1.0.conv2/Conv2D:output:0#layer1.0.bn2/ReadVariableOp:value:0%layer1.0.bn2/ReadVariableOp_1:value:04layer1.0.bn2/FusedBatchNormV3/ReadVariableOp:value:06layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%fff?2
layer1.0.bn2/FusedBatchNormV3?
layer1.0.bn2/AssignNewValueAssignVariableOp5layer1_0_bn2_fusedbatchnormv3_readvariableop_resource*layer1.0.bn2/FusedBatchNormV3:batch_mean:0-^layer1.0.bn2/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*H
_class>
<:loc:@layer1.0.bn2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
layer1.0.bn2/AssignNewValue?
layer1.0.bn2/AssignNewValue_1AssignVariableOp7layer1_0_bn2_fusedbatchnormv3_readvariableop_1_resource.layer1.0.bn2/FusedBatchNormV3:batch_variance:0/^layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*J
_class@
><loc:@layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
layer1.0.bn2/AssignNewValue_1?
add/addAddV2maxpool/MaxPool:output:0!layer1.0.bn2/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
add/add?
activation_1/ReluReluadd/add:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
activation_1/Relu?
$layer1.1.conv1/Conv2D/ReadVariableOpReadVariableOp-layer1_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$layer1.1.conv1/Conv2D/ReadVariableOp?
layer1.1.conv1/Conv2DConv2Dactivation_1/Relu:activations:0,layer1.1.conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
layer1.1.conv1/Conv2D?
layer1.1.bn1/ReadVariableOpReadVariableOp$layer1_1_bn1_readvariableop_resource*
_output_shapes
:@*
dtype02
layer1.1.bn1/ReadVariableOp?
layer1.1.bn1/ReadVariableOp_1ReadVariableOp&layer1_1_bn1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
layer1.1.bn1/ReadVariableOp_1?
,layer1.1.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer1_1_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer1.1.bn1/FusedBatchNormV3/ReadVariableOp?
.layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer1_1_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1?
layer1.1.bn1/FusedBatchNormV3FusedBatchNormV3layer1.1.conv1/Conv2D:output:0#layer1.1.bn1/ReadVariableOp:value:0%layer1.1.bn1/ReadVariableOp_1:value:04layer1.1.bn1/FusedBatchNormV3/ReadVariableOp:value:06layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%fff?2
layer1.1.bn1/FusedBatchNormV3?
layer1.1.bn1/AssignNewValueAssignVariableOp5layer1_1_bn1_fusedbatchnormv3_readvariableop_resource*layer1.1.bn1/FusedBatchNormV3:batch_mean:0-^layer1.1.bn1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*H
_class>
<:loc:@layer1.1.bn1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
layer1.1.bn1/AssignNewValue?
layer1.1.bn1/AssignNewValue_1AssignVariableOp7layer1_1_bn1_fusedbatchnormv3_readvariableop_1_resource.layer1.1.bn1/FusedBatchNormV3:batch_variance:0/^layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*J
_class@
><loc:@layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
layer1.1.bn1/AssignNewValue_1?
activation_2/ReluRelu!layer1.1.bn1/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
activation_2/Relu?
$layer1.1.conv2/Conv2D/ReadVariableOpReadVariableOp-layer1_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$layer1.1.conv2/Conv2D/ReadVariableOp?
layer1.1.conv2/Conv2DConv2Dactivation_2/Relu:activations:0,layer1.1.conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
layer1.1.conv2/Conv2D?
layer1.1.bn2/ReadVariableOpReadVariableOp$layer1_1_bn2_readvariableop_resource*
_output_shapes
:@*
dtype02
layer1.1.bn2/ReadVariableOp?
layer1.1.bn2/ReadVariableOp_1ReadVariableOp&layer1_1_bn2_readvariableop_1_resource*
_output_shapes
:@*
dtype02
layer1.1.bn2/ReadVariableOp_1?
,layer1.1.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer1_1_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer1.1.bn2/FusedBatchNormV3/ReadVariableOp?
.layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer1_1_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1?
layer1.1.bn2/FusedBatchNormV3FusedBatchNormV3layer1.1.conv2/Conv2D:output:0#layer1.1.bn2/ReadVariableOp:value:0%layer1.1.bn2/ReadVariableOp_1:value:04layer1.1.bn2/FusedBatchNormV3/ReadVariableOp:value:06layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%fff?2
layer1.1.bn2/FusedBatchNormV3?
layer1.1.bn2/AssignNewValueAssignVariableOp5layer1_1_bn2_fusedbatchnormv3_readvariableop_resource*layer1.1.bn2/FusedBatchNormV3:batch_mean:0-^layer1.1.bn2/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*H
_class>
<:loc:@layer1.1.bn2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
layer1.1.bn2/AssignNewValue?
layer1.1.bn2/AssignNewValue_1AssignVariableOp7layer1_1_bn2_fusedbatchnormv3_readvariableop_1_resource.layer1.1.bn2/FusedBatchNormV3:batch_variance:0/^layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*J
_class@
><loc:@layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
layer1.1.bn2/AssignNewValue_1?
	add_1/addAddV2activation_1/Relu:activations:0!layer1.1.bn2/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
	add_1/add?
activation_3/ReluReluadd_1/add:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
activation_3/Relu?
layer2.0.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
layer2.0.pad/Pad/paddings?
layer2.0.pad/PadPadactivation_3/Relu:activations:0"layer2.0.pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
layer2.0.pad/Pad?
$layer2.0.conv1/Conv2D/ReadVariableOpReadVariableOp-layer2_0_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02&
$layer2.0.conv1/Conv2D/ReadVariableOp?
layer2.0.conv1/Conv2DConv2Dlayer2.0.pad/Pad:output:0,layer2.0.conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
layer2.0.conv1/Conv2D?
layer2.0.bn1/ReadVariableOpReadVariableOp$layer2_0_bn1_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer2.0.bn1/ReadVariableOp?
layer2.0.bn1/ReadVariableOp_1ReadVariableOp&layer2_0_bn1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
layer2.0.bn1/ReadVariableOp_1?
,layer2.0.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer2_0_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,layer2.0.bn1/FusedBatchNormV3/ReadVariableOp?
.layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer2_0_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1?
layer2.0.bn1/FusedBatchNormV3FusedBatchNormV3layer2.0.conv1/Conv2D:output:0#layer2.0.bn1/ReadVariableOp:value:0%layer2.0.bn1/ReadVariableOp_1:value:04layer2.0.bn1/FusedBatchNormV3/ReadVariableOp:value:06layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
layer2.0.bn1/FusedBatchNormV3?
layer2.0.bn1/AssignNewValueAssignVariableOp5layer2_0_bn1_fusedbatchnormv3_readvariableop_resource*layer2.0.bn1/FusedBatchNormV3:batch_mean:0-^layer2.0.bn1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*H
_class>
<:loc:@layer2.0.bn1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
layer2.0.bn1/AssignNewValue?
layer2.0.bn1/AssignNewValue_1AssignVariableOp7layer2_0_bn1_fusedbatchnormv3_readvariableop_1_resource.layer2.0.bn1/FusedBatchNormV3:batch_variance:0/^layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*J
_class@
><loc:@layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
layer2.0.bn1/AssignNewValue_1?
activation_4/ReluRelu!layer2.0.bn1/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
activation_4/Relu?
$layer2.0.conv2/Conv2D/ReadVariableOpReadVariableOp-layer2_0_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$layer2.0.conv2/Conv2D/ReadVariableOp?
layer2.0.conv2/Conv2DConv2Dactivation_4/Relu:activations:0,layer2.0.conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
layer2.0.conv2/Conv2D?
+layer2.0.downsample.0/Conv2D/ReadVariableOpReadVariableOp4layer2_0_downsample_0_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02-
+layer2.0.downsample.0/Conv2D/ReadVariableOp?
layer2.0.downsample.0/Conv2DConv2Dactivation_3/Relu:activations:03layer2.0.downsample.0/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
layer2.0.downsample.0/Conv2D?
$layer2.0.downsample.1/ReadVariableOpReadVariableOp-layer2_0_downsample_1_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$layer2.0.downsample.1/ReadVariableOp?
&layer2.0.downsample.1/ReadVariableOp_1ReadVariableOp/layer2_0_downsample_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&layer2.0.downsample.1/ReadVariableOp_1?
5layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOpReadVariableOp>layer2_0_downsample_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp?
7layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@layer2_0_downsample_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1?
&layer2.0.downsample.1/FusedBatchNormV3FusedBatchNormV3%layer2.0.downsample.0/Conv2D:output:0,layer2.0.downsample.1/ReadVariableOp:value:0.layer2.0.downsample.1/ReadVariableOp_1:value:0=layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp:value:0?layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2(
&layer2.0.downsample.1/FusedBatchNormV3?
$layer2.0.downsample.1/AssignNewValueAssignVariableOp>layer2_0_downsample_1_fusedbatchnormv3_readvariableop_resource3layer2.0.downsample.1/FusedBatchNormV3:batch_mean:06^layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$layer2.0.downsample.1/AssignNewValue?
&layer2.0.downsample.1/AssignNewValue_1AssignVariableOp@layer2_0_downsample_1_fusedbatchnormv3_readvariableop_1_resource7layer2.0.downsample.1/FusedBatchNormV3:batch_variance:08^layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&layer2.0.downsample.1/AssignNewValue_1?
layer2.0.bn2/ReadVariableOpReadVariableOp$layer2_0_bn2_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer2.0.bn2/ReadVariableOp?
layer2.0.bn2/ReadVariableOp_1ReadVariableOp&layer2_0_bn2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
layer2.0.bn2/ReadVariableOp_1?
,layer2.0.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer2_0_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,layer2.0.bn2/FusedBatchNormV3/ReadVariableOp?
.layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer2_0_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1?
layer2.0.bn2/FusedBatchNormV3FusedBatchNormV3layer2.0.conv2/Conv2D:output:0#layer2.0.bn2/ReadVariableOp:value:0%layer2.0.bn2/ReadVariableOp_1:value:04layer2.0.bn2/FusedBatchNormV3/ReadVariableOp:value:06layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
layer2.0.bn2/FusedBatchNormV3?
layer2.0.bn2/AssignNewValueAssignVariableOp5layer2_0_bn2_fusedbatchnormv3_readvariableop_resource*layer2.0.bn2/FusedBatchNormV3:batch_mean:0-^layer2.0.bn2/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*H
_class>
<:loc:@layer2.0.bn2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
layer2.0.bn2/AssignNewValue?
layer2.0.bn2/AssignNewValue_1AssignVariableOp7layer2_0_bn2_fusedbatchnormv3_readvariableop_1_resource.layer2.0.bn2/FusedBatchNormV3:batch_variance:0/^layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*J
_class@
><loc:@layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
layer2.0.bn2/AssignNewValue_1?
	add_2/addAddV2*layer2.0.downsample.1/FusedBatchNormV3:y:0!layer2.0.bn2/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
	add_2/add?
activation_5/ReluReluadd_2/add:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
activation_5/Relu?
$layer2.1.conv1/Conv2D/ReadVariableOpReadVariableOp-layer2_1_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$layer2.1.conv1/Conv2D/ReadVariableOp?
layer2.1.conv1/Conv2DConv2Dactivation_5/Relu:activations:0,layer2.1.conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
layer2.1.conv1/Conv2D?
layer2.1.bn1/ReadVariableOpReadVariableOp$layer2_1_bn1_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer2.1.bn1/ReadVariableOp?
layer2.1.bn1/ReadVariableOp_1ReadVariableOp&layer2_1_bn1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
layer2.1.bn1/ReadVariableOp_1?
,layer2.1.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer2_1_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,layer2.1.bn1/FusedBatchNormV3/ReadVariableOp?
.layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer2_1_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1?
layer2.1.bn1/FusedBatchNormV3FusedBatchNormV3layer2.1.conv1/Conv2D:output:0#layer2.1.bn1/ReadVariableOp:value:0%layer2.1.bn1/ReadVariableOp_1:value:04layer2.1.bn1/FusedBatchNormV3/ReadVariableOp:value:06layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
layer2.1.bn1/FusedBatchNormV3?
layer2.1.bn1/AssignNewValueAssignVariableOp5layer2_1_bn1_fusedbatchnormv3_readvariableop_resource*layer2.1.bn1/FusedBatchNormV3:batch_mean:0-^layer2.1.bn1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*H
_class>
<:loc:@layer2.1.bn1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
layer2.1.bn1/AssignNewValue?
layer2.1.bn1/AssignNewValue_1AssignVariableOp7layer2_1_bn1_fusedbatchnormv3_readvariableop_1_resource.layer2.1.bn1/FusedBatchNormV3:batch_variance:0/^layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*J
_class@
><loc:@layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
layer2.1.bn1/AssignNewValue_1?
activation_6/ReluRelu!layer2.1.bn1/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
activation_6/Relu?
$layer2.1.conv2/Conv2D/ReadVariableOpReadVariableOp-layer2_1_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$layer2.1.conv2/Conv2D/ReadVariableOp?
layer2.1.conv2/Conv2DConv2Dactivation_6/Relu:activations:0,layer2.1.conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
layer2.1.conv2/Conv2D?
layer2.1.bn2/ReadVariableOpReadVariableOp$layer2_1_bn2_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer2.1.bn2/ReadVariableOp?
layer2.1.bn2/ReadVariableOp_1ReadVariableOp&layer2_1_bn2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
layer2.1.bn2/ReadVariableOp_1?
,layer2.1.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer2_1_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,layer2.1.bn2/FusedBatchNormV3/ReadVariableOp?
.layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer2_1_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1?
layer2.1.bn2/FusedBatchNormV3FusedBatchNormV3layer2.1.conv2/Conv2D:output:0#layer2.1.bn2/ReadVariableOp:value:0%layer2.1.bn2/ReadVariableOp_1:value:04layer2.1.bn2/FusedBatchNormV3/ReadVariableOp:value:06layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
layer2.1.bn2/FusedBatchNormV3?
layer2.1.bn2/AssignNewValueAssignVariableOp5layer2_1_bn2_fusedbatchnormv3_readvariableop_resource*layer2.1.bn2/FusedBatchNormV3:batch_mean:0-^layer2.1.bn2/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*H
_class>
<:loc:@layer2.1.bn2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
layer2.1.bn2/AssignNewValue?
layer2.1.bn2/AssignNewValue_1AssignVariableOp7layer2_1_bn2_fusedbatchnormv3_readvariableop_1_resource.layer2.1.bn2/FusedBatchNormV3:batch_variance:0/^layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*J
_class@
><loc:@layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
layer2.1.bn2/AssignNewValue_1?
	add_3/addAddV2activation_5/Relu:activations:0!layer2.1.bn2/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
	add_3/add?
activation_7/ReluReluadd_3/add:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
activation_7/Relu?
layer3.0.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
layer3.0.pad/Pad/paddings?
layer3.0.pad/PadPadactivation_7/Relu:activations:0"layer3.0.pad/Pad/paddings:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer3.0.pad/Pad?
$layer3.0.conv1/Conv2D/ReadVariableOpReadVariableOp-layer3_0_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$layer3.0.conv1/Conv2D/ReadVariableOp?
layer3.0.conv1/Conv2DConv2Dlayer3.0.pad/Pad:output:0,layer3.0.conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
layer3.0.conv1/Conv2D?
layer3.0.bn1/ReadVariableOpReadVariableOp$layer3_0_bn1_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer3.0.bn1/ReadVariableOp?
layer3.0.bn1/ReadVariableOp_1ReadVariableOp&layer3_0_bn1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
layer3.0.bn1/ReadVariableOp_1?
,layer3.0.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer3_0_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,layer3.0.bn1/FusedBatchNormV3/ReadVariableOp?
.layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer3_0_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1?
layer3.0.bn1/FusedBatchNormV3FusedBatchNormV3layer3.0.conv1/Conv2D:output:0#layer3.0.bn1/ReadVariableOp:value:0%layer3.0.bn1/ReadVariableOp_1:value:04layer3.0.bn1/FusedBatchNormV3/ReadVariableOp:value:06layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
layer3.0.bn1/FusedBatchNormV3?
layer3.0.bn1/AssignNewValueAssignVariableOp5layer3_0_bn1_fusedbatchnormv3_readvariableop_resource*layer3.0.bn1/FusedBatchNormV3:batch_mean:0-^layer3.0.bn1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*H
_class>
<:loc:@layer3.0.bn1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
layer3.0.bn1/AssignNewValue?
layer3.0.bn1/AssignNewValue_1AssignVariableOp7layer3_0_bn1_fusedbatchnormv3_readvariableop_1_resource.layer3.0.bn1/FusedBatchNormV3:batch_variance:0/^layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*J
_class@
><loc:@layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
layer3.0.bn1/AssignNewValue_1?
activation_8/ReluRelu!layer3.0.bn1/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
activation_8/Relu?
$layer3.0.conv2/Conv2D/ReadVariableOpReadVariableOp-layer3_0_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$layer3.0.conv2/Conv2D/ReadVariableOp?
layer3.0.conv2/Conv2DConv2Dactivation_8/Relu:activations:0,layer3.0.conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
layer3.0.conv2/Conv2D?
+layer3.0.downsample.0/Conv2D/ReadVariableOpReadVariableOp4layer3_0_downsample_0_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02-
+layer3.0.downsample.0/Conv2D/ReadVariableOp?
layer3.0.downsample.0/Conv2DConv2Dactivation_7/Relu:activations:03layer3.0.downsample.0/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
layer3.0.downsample.0/Conv2D?
$layer3.0.downsample.1/ReadVariableOpReadVariableOp-layer3_0_downsample_1_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$layer3.0.downsample.1/ReadVariableOp?
&layer3.0.downsample.1/ReadVariableOp_1ReadVariableOp/layer3_0_downsample_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&layer3.0.downsample.1/ReadVariableOp_1?
5layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOpReadVariableOp>layer3_0_downsample_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp?
7layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@layer3_0_downsample_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1?
&layer3.0.downsample.1/FusedBatchNormV3FusedBatchNormV3%layer3.0.downsample.0/Conv2D:output:0,layer3.0.downsample.1/ReadVariableOp:value:0.layer3.0.downsample.1/ReadVariableOp_1:value:0=layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp:value:0?layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2(
&layer3.0.downsample.1/FusedBatchNormV3?
$layer3.0.downsample.1/AssignNewValueAssignVariableOp>layer3_0_downsample_1_fusedbatchnormv3_readvariableop_resource3layer3.0.downsample.1/FusedBatchNormV3:batch_mean:06^layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$layer3.0.downsample.1/AssignNewValue?
&layer3.0.downsample.1/AssignNewValue_1AssignVariableOp@layer3_0_downsample_1_fusedbatchnormv3_readvariableop_1_resource7layer3.0.downsample.1/FusedBatchNormV3:batch_variance:08^layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&layer3.0.downsample.1/AssignNewValue_1?
layer3.0.bn2/ReadVariableOpReadVariableOp$layer3_0_bn2_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer3.0.bn2/ReadVariableOp?
layer3.0.bn2/ReadVariableOp_1ReadVariableOp&layer3_0_bn2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
layer3.0.bn2/ReadVariableOp_1?
,layer3.0.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer3_0_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,layer3.0.bn2/FusedBatchNormV3/ReadVariableOp?
.layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer3_0_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1?
layer3.0.bn2/FusedBatchNormV3FusedBatchNormV3layer3.0.conv2/Conv2D:output:0#layer3.0.bn2/ReadVariableOp:value:0%layer3.0.bn2/ReadVariableOp_1:value:04layer3.0.bn2/FusedBatchNormV3/ReadVariableOp:value:06layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
layer3.0.bn2/FusedBatchNormV3?
layer3.0.bn2/AssignNewValueAssignVariableOp5layer3_0_bn2_fusedbatchnormv3_readvariableop_resource*layer3.0.bn2/FusedBatchNormV3:batch_mean:0-^layer3.0.bn2/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*H
_class>
<:loc:@layer3.0.bn2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
layer3.0.bn2/AssignNewValue?
layer3.0.bn2/AssignNewValue_1AssignVariableOp7layer3_0_bn2_fusedbatchnormv3_readvariableop_1_resource.layer3.0.bn2/FusedBatchNormV3:batch_variance:0/^layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*J
_class@
><loc:@layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
layer3.0.bn2/AssignNewValue_1?
	add_4/addAddV2*layer3.0.downsample.1/FusedBatchNormV3:y:0!layer3.0.bn2/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
	add_4/add?
activation_9/ReluReluadd_4/add:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
activation_9/Relu?
$layer3.1.conv1/Conv2D/ReadVariableOpReadVariableOp-layer3_1_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$layer3.1.conv1/Conv2D/ReadVariableOp?
layer3.1.conv1/Conv2DConv2Dactivation_9/Relu:activations:0,layer3.1.conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
layer3.1.conv1/Conv2D?
layer3.1.bn1/ReadVariableOpReadVariableOp$layer3_1_bn1_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer3.1.bn1/ReadVariableOp?
layer3.1.bn1/ReadVariableOp_1ReadVariableOp&layer3_1_bn1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
layer3.1.bn1/ReadVariableOp_1?
,layer3.1.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer3_1_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,layer3.1.bn1/FusedBatchNormV3/ReadVariableOp?
.layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer3_1_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1?
layer3.1.bn1/FusedBatchNormV3FusedBatchNormV3layer3.1.conv1/Conv2D:output:0#layer3.1.bn1/ReadVariableOp:value:0%layer3.1.bn1/ReadVariableOp_1:value:04layer3.1.bn1/FusedBatchNormV3/ReadVariableOp:value:06layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
layer3.1.bn1/FusedBatchNormV3?
layer3.1.bn1/AssignNewValueAssignVariableOp5layer3_1_bn1_fusedbatchnormv3_readvariableop_resource*layer3.1.bn1/FusedBatchNormV3:batch_mean:0-^layer3.1.bn1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*H
_class>
<:loc:@layer3.1.bn1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
layer3.1.bn1/AssignNewValue?
layer3.1.bn1/AssignNewValue_1AssignVariableOp7layer3_1_bn1_fusedbatchnormv3_readvariableop_1_resource.layer3.1.bn1/FusedBatchNormV3:batch_variance:0/^layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*J
_class@
><loc:@layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
layer3.1.bn1/AssignNewValue_1?
activation_10/ReluRelu!layer3.1.bn1/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
activation_10/Relu?
$layer3.1.conv2/Conv2D/ReadVariableOpReadVariableOp-layer3_1_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$layer3.1.conv2/Conv2D/ReadVariableOp?
layer3.1.conv2/Conv2DConv2D activation_10/Relu:activations:0,layer3.1.conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
layer3.1.conv2/Conv2D?
layer3.1.bn2/ReadVariableOpReadVariableOp$layer3_1_bn2_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer3.1.bn2/ReadVariableOp?
layer3.1.bn2/ReadVariableOp_1ReadVariableOp&layer3_1_bn2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
layer3.1.bn2/ReadVariableOp_1?
,layer3.1.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp5layer3_1_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,layer3.1.bn2/FusedBatchNormV3/ReadVariableOp?
.layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7layer3_1_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1?
layer3.1.bn2/FusedBatchNormV3FusedBatchNormV3layer3.1.conv2/Conv2D:output:0#layer3.1.bn2/ReadVariableOp:value:0%layer3.1.bn2/ReadVariableOp_1:value:04layer3.1.bn2/FusedBatchNormV3/ReadVariableOp:value:06layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
layer3.1.bn2/FusedBatchNormV3?
layer3.1.bn2/AssignNewValueAssignVariableOp5layer3_1_bn2_fusedbatchnormv3_readvariableop_resource*layer3.1.bn2/FusedBatchNormV3:batch_mean:0-^layer3.1.bn2/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*H
_class>
<:loc:@layer3.1.bn2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
layer3.1.bn2/AssignNewValue?
layer3.1.bn2/AssignNewValue_1AssignVariableOp7layer3_1_bn2_fusedbatchnormv3_readvariableop_1_resource.layer3.1.bn2/FusedBatchNormV3:batch_variance:0/^layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*J
_class@
><loc:@layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
layer3.1.bn2/AssignNewValue_1?
	add_5/addAddV2activation_9/Relu:activations:0!layer3.1.bn2/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
	add_5/add?
activation_11/ReluReluadd_5/add:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
activation_11/Relu?
IdentityIdentity activation_11/Relu:activations:0^bn1/AssignNewValue^bn1/AssignNewValue_1$^bn1/FusedBatchNormV3/ReadVariableOp&^bn1/FusedBatchNormV3/ReadVariableOp_1^bn1/ReadVariableOp^bn1/ReadVariableOp_1^conv1/Conv2D/ReadVariableOp^layer1.0.bn1/AssignNewValue^layer1.0.bn1/AssignNewValue_1-^layer1.0.bn1/FusedBatchNormV3/ReadVariableOp/^layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1^layer1.0.bn1/ReadVariableOp^layer1.0.bn1/ReadVariableOp_1^layer1.0.bn2/AssignNewValue^layer1.0.bn2/AssignNewValue_1-^layer1.0.bn2/FusedBatchNormV3/ReadVariableOp/^layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1^layer1.0.bn2/ReadVariableOp^layer1.0.bn2/ReadVariableOp_1%^layer1.0.conv1/Conv2D/ReadVariableOp%^layer1.0.conv2/Conv2D/ReadVariableOp^layer1.1.bn1/AssignNewValue^layer1.1.bn1/AssignNewValue_1-^layer1.1.bn1/FusedBatchNormV3/ReadVariableOp/^layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1^layer1.1.bn1/ReadVariableOp^layer1.1.bn1/ReadVariableOp_1^layer1.1.bn2/AssignNewValue^layer1.1.bn2/AssignNewValue_1-^layer1.1.bn2/FusedBatchNormV3/ReadVariableOp/^layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1^layer1.1.bn2/ReadVariableOp^layer1.1.bn2/ReadVariableOp_1%^layer1.1.conv1/Conv2D/ReadVariableOp%^layer1.1.conv2/Conv2D/ReadVariableOp^layer2.0.bn1/AssignNewValue^layer2.0.bn1/AssignNewValue_1-^layer2.0.bn1/FusedBatchNormV3/ReadVariableOp/^layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1^layer2.0.bn1/ReadVariableOp^layer2.0.bn1/ReadVariableOp_1^layer2.0.bn2/AssignNewValue^layer2.0.bn2/AssignNewValue_1-^layer2.0.bn2/FusedBatchNormV3/ReadVariableOp/^layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1^layer2.0.bn2/ReadVariableOp^layer2.0.bn2/ReadVariableOp_1%^layer2.0.conv1/Conv2D/ReadVariableOp%^layer2.0.conv2/Conv2D/ReadVariableOp,^layer2.0.downsample.0/Conv2D/ReadVariableOp%^layer2.0.downsample.1/AssignNewValue'^layer2.0.downsample.1/AssignNewValue_16^layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp8^layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1%^layer2.0.downsample.1/ReadVariableOp'^layer2.0.downsample.1/ReadVariableOp_1^layer2.1.bn1/AssignNewValue^layer2.1.bn1/AssignNewValue_1-^layer2.1.bn1/FusedBatchNormV3/ReadVariableOp/^layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1^layer2.1.bn1/ReadVariableOp^layer2.1.bn1/ReadVariableOp_1^layer2.1.bn2/AssignNewValue^layer2.1.bn2/AssignNewValue_1-^layer2.1.bn2/FusedBatchNormV3/ReadVariableOp/^layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1^layer2.1.bn2/ReadVariableOp^layer2.1.bn2/ReadVariableOp_1%^layer2.1.conv1/Conv2D/ReadVariableOp%^layer2.1.conv2/Conv2D/ReadVariableOp^layer3.0.bn1/AssignNewValue^layer3.0.bn1/AssignNewValue_1-^layer3.0.bn1/FusedBatchNormV3/ReadVariableOp/^layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1^layer3.0.bn1/ReadVariableOp^layer3.0.bn1/ReadVariableOp_1^layer3.0.bn2/AssignNewValue^layer3.0.bn2/AssignNewValue_1-^layer3.0.bn2/FusedBatchNormV3/ReadVariableOp/^layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1^layer3.0.bn2/ReadVariableOp^layer3.0.bn2/ReadVariableOp_1%^layer3.0.conv1/Conv2D/ReadVariableOp%^layer3.0.conv2/Conv2D/ReadVariableOp,^layer3.0.downsample.0/Conv2D/ReadVariableOp%^layer3.0.downsample.1/AssignNewValue'^layer3.0.downsample.1/AssignNewValue_16^layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp8^layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1%^layer3.0.downsample.1/ReadVariableOp'^layer3.0.downsample.1/ReadVariableOp_1^layer3.1.bn1/AssignNewValue^layer3.1.bn1/AssignNewValue_1-^layer3.1.bn1/FusedBatchNormV3/ReadVariableOp/^layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1^layer3.1.bn1/ReadVariableOp^layer3.1.bn1/ReadVariableOp_1^layer3.1.bn2/AssignNewValue^layer3.1.bn2/AssignNewValue_1-^layer3.1.bn2/FusedBatchNormV3/ReadVariableOp/^layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1^layer3.1.bn2/ReadVariableOp^layer3.1.bn2/ReadVariableOp_1%^layer3.1.conv1/Conv2D/ReadVariableOp%^layer3.1.conv2/Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:+???????????????????????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2(
bn1/AssignNewValuebn1/AssignNewValue2,
bn1/AssignNewValue_1bn1/AssignNewValue_12J
#bn1/FusedBatchNormV3/ReadVariableOp#bn1/FusedBatchNormV3/ReadVariableOp2N
%bn1/FusedBatchNormV3/ReadVariableOp_1%bn1/FusedBatchNormV3/ReadVariableOp_12(
bn1/ReadVariableOpbn1/ReadVariableOp2,
bn1/ReadVariableOp_1bn1/ReadVariableOp_12:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2:
layer1.0.bn1/AssignNewValuelayer1.0.bn1/AssignNewValue2>
layer1.0.bn1/AssignNewValue_1layer1.0.bn1/AssignNewValue_12\
,layer1.0.bn1/FusedBatchNormV3/ReadVariableOp,layer1.0.bn1/FusedBatchNormV3/ReadVariableOp2`
.layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1.layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_12:
layer1.0.bn1/ReadVariableOplayer1.0.bn1/ReadVariableOp2>
layer1.0.bn1/ReadVariableOp_1layer1.0.bn1/ReadVariableOp_12:
layer1.0.bn2/AssignNewValuelayer1.0.bn2/AssignNewValue2>
layer1.0.bn2/AssignNewValue_1layer1.0.bn2/AssignNewValue_12\
,layer1.0.bn2/FusedBatchNormV3/ReadVariableOp,layer1.0.bn2/FusedBatchNormV3/ReadVariableOp2`
.layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1.layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_12:
layer1.0.bn2/ReadVariableOplayer1.0.bn2/ReadVariableOp2>
layer1.0.bn2/ReadVariableOp_1layer1.0.bn2/ReadVariableOp_12L
$layer1.0.conv1/Conv2D/ReadVariableOp$layer1.0.conv1/Conv2D/ReadVariableOp2L
$layer1.0.conv2/Conv2D/ReadVariableOp$layer1.0.conv2/Conv2D/ReadVariableOp2:
layer1.1.bn1/AssignNewValuelayer1.1.bn1/AssignNewValue2>
layer1.1.bn1/AssignNewValue_1layer1.1.bn1/AssignNewValue_12\
,layer1.1.bn1/FusedBatchNormV3/ReadVariableOp,layer1.1.bn1/FusedBatchNormV3/ReadVariableOp2`
.layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1.layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_12:
layer1.1.bn1/ReadVariableOplayer1.1.bn1/ReadVariableOp2>
layer1.1.bn1/ReadVariableOp_1layer1.1.bn1/ReadVariableOp_12:
layer1.1.bn2/AssignNewValuelayer1.1.bn2/AssignNewValue2>
layer1.1.bn2/AssignNewValue_1layer1.1.bn2/AssignNewValue_12\
,layer1.1.bn2/FusedBatchNormV3/ReadVariableOp,layer1.1.bn2/FusedBatchNormV3/ReadVariableOp2`
.layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1.layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_12:
layer1.1.bn2/ReadVariableOplayer1.1.bn2/ReadVariableOp2>
layer1.1.bn2/ReadVariableOp_1layer1.1.bn2/ReadVariableOp_12L
$layer1.1.conv1/Conv2D/ReadVariableOp$layer1.1.conv1/Conv2D/ReadVariableOp2L
$layer1.1.conv2/Conv2D/ReadVariableOp$layer1.1.conv2/Conv2D/ReadVariableOp2:
layer2.0.bn1/AssignNewValuelayer2.0.bn1/AssignNewValue2>
layer2.0.bn1/AssignNewValue_1layer2.0.bn1/AssignNewValue_12\
,layer2.0.bn1/FusedBatchNormV3/ReadVariableOp,layer2.0.bn1/FusedBatchNormV3/ReadVariableOp2`
.layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1.layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_12:
layer2.0.bn1/ReadVariableOplayer2.0.bn1/ReadVariableOp2>
layer2.0.bn1/ReadVariableOp_1layer2.0.bn1/ReadVariableOp_12:
layer2.0.bn2/AssignNewValuelayer2.0.bn2/AssignNewValue2>
layer2.0.bn2/AssignNewValue_1layer2.0.bn2/AssignNewValue_12\
,layer2.0.bn2/FusedBatchNormV3/ReadVariableOp,layer2.0.bn2/FusedBatchNormV3/ReadVariableOp2`
.layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1.layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_12:
layer2.0.bn2/ReadVariableOplayer2.0.bn2/ReadVariableOp2>
layer2.0.bn2/ReadVariableOp_1layer2.0.bn2/ReadVariableOp_12L
$layer2.0.conv1/Conv2D/ReadVariableOp$layer2.0.conv1/Conv2D/ReadVariableOp2L
$layer2.0.conv2/Conv2D/ReadVariableOp$layer2.0.conv2/Conv2D/ReadVariableOp2Z
+layer2.0.downsample.0/Conv2D/ReadVariableOp+layer2.0.downsample.0/Conv2D/ReadVariableOp2L
$layer2.0.downsample.1/AssignNewValue$layer2.0.downsample.1/AssignNewValue2P
&layer2.0.downsample.1/AssignNewValue_1&layer2.0.downsample.1/AssignNewValue_12n
5layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp5layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp2r
7layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_17layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_12L
$layer2.0.downsample.1/ReadVariableOp$layer2.0.downsample.1/ReadVariableOp2P
&layer2.0.downsample.1/ReadVariableOp_1&layer2.0.downsample.1/ReadVariableOp_12:
layer2.1.bn1/AssignNewValuelayer2.1.bn1/AssignNewValue2>
layer2.1.bn1/AssignNewValue_1layer2.1.bn1/AssignNewValue_12\
,layer2.1.bn1/FusedBatchNormV3/ReadVariableOp,layer2.1.bn1/FusedBatchNormV3/ReadVariableOp2`
.layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1.layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_12:
layer2.1.bn1/ReadVariableOplayer2.1.bn1/ReadVariableOp2>
layer2.1.bn1/ReadVariableOp_1layer2.1.bn1/ReadVariableOp_12:
layer2.1.bn2/AssignNewValuelayer2.1.bn2/AssignNewValue2>
layer2.1.bn2/AssignNewValue_1layer2.1.bn2/AssignNewValue_12\
,layer2.1.bn2/FusedBatchNormV3/ReadVariableOp,layer2.1.bn2/FusedBatchNormV3/ReadVariableOp2`
.layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1.layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_12:
layer2.1.bn2/ReadVariableOplayer2.1.bn2/ReadVariableOp2>
layer2.1.bn2/ReadVariableOp_1layer2.1.bn2/ReadVariableOp_12L
$layer2.1.conv1/Conv2D/ReadVariableOp$layer2.1.conv1/Conv2D/ReadVariableOp2L
$layer2.1.conv2/Conv2D/ReadVariableOp$layer2.1.conv2/Conv2D/ReadVariableOp2:
layer3.0.bn1/AssignNewValuelayer3.0.bn1/AssignNewValue2>
layer3.0.bn1/AssignNewValue_1layer3.0.bn1/AssignNewValue_12\
,layer3.0.bn1/FusedBatchNormV3/ReadVariableOp,layer3.0.bn1/FusedBatchNormV3/ReadVariableOp2`
.layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1.layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_12:
layer3.0.bn1/ReadVariableOplayer3.0.bn1/ReadVariableOp2>
layer3.0.bn1/ReadVariableOp_1layer3.0.bn1/ReadVariableOp_12:
layer3.0.bn2/AssignNewValuelayer3.0.bn2/AssignNewValue2>
layer3.0.bn2/AssignNewValue_1layer3.0.bn2/AssignNewValue_12\
,layer3.0.bn2/FusedBatchNormV3/ReadVariableOp,layer3.0.bn2/FusedBatchNormV3/ReadVariableOp2`
.layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1.layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_12:
layer3.0.bn2/ReadVariableOplayer3.0.bn2/ReadVariableOp2>
layer3.0.bn2/ReadVariableOp_1layer3.0.bn2/ReadVariableOp_12L
$layer3.0.conv1/Conv2D/ReadVariableOp$layer3.0.conv1/Conv2D/ReadVariableOp2L
$layer3.0.conv2/Conv2D/ReadVariableOp$layer3.0.conv2/Conv2D/ReadVariableOp2Z
+layer3.0.downsample.0/Conv2D/ReadVariableOp+layer3.0.downsample.0/Conv2D/ReadVariableOp2L
$layer3.0.downsample.1/AssignNewValue$layer3.0.downsample.1/AssignNewValue2P
&layer3.0.downsample.1/AssignNewValue_1&layer3.0.downsample.1/AssignNewValue_12n
5layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp5layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp2r
7layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_17layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_12L
$layer3.0.downsample.1/ReadVariableOp$layer3.0.downsample.1/ReadVariableOp2P
&layer3.0.downsample.1/ReadVariableOp_1&layer3.0.downsample.1/ReadVariableOp_12:
layer3.1.bn1/AssignNewValuelayer3.1.bn1/AssignNewValue2>
layer3.1.bn1/AssignNewValue_1layer3.1.bn1/AssignNewValue_12\
,layer3.1.bn1/FusedBatchNormV3/ReadVariableOp,layer3.1.bn1/FusedBatchNormV3/ReadVariableOp2`
.layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1.layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_12:
layer3.1.bn1/ReadVariableOplayer3.1.bn1/ReadVariableOp2>
layer3.1.bn1/ReadVariableOp_1layer3.1.bn1/ReadVariableOp_12:
layer3.1.bn2/AssignNewValuelayer3.1.bn2/AssignNewValue2>
layer3.1.bn2/AssignNewValue_1layer3.1.bn2/AssignNewValue_12\
,layer3.1.bn2/FusedBatchNormV3/ReadVariableOp,layer3.1.bn2/FusedBatchNormV3/ReadVariableOp2`
.layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1.layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_12:
layer3.1.bn2/ReadVariableOplayer3.1.bn2/ReadVariableOp2>
layer3.1.bn2/ReadVariableOp_1layer3.1.bn2/ReadVariableOp_12L
$layer3.1.conv1/Conv2D/ReadVariableOp$layer3.1.conv1/Conv2D/ReadVariableOp2L
$layer3.1.conv2/Conv2D/ReadVariableOp$layer3.1.conv2/Conv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer1.1.bn2_layer_call_and_return_conditional_losses_14126

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
]
A__inference_maxpool_layer_call_and_return_conditional_losses_9137

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
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
?
?
I__inference_layer2.1.conv1_layer_call_and_return_conditional_losses_14447

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_layer3.0.conv1_layer_call_and_return_conditional_losses_11358

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_layer1.0.bn1_layer_call_fn_13876

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.0.bn1_layer_call_and_return_conditional_losses_92362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
t
.__inference_layer1.1.conv1_layer_call_fn_14000

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.1.conv1_layer_call_and_return_conditional_losses_108552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
G__inference_activation_7_layer_call_and_return_conditional_losses_11342

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_layer2.0.bn2_layer_call_fn_14405

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.0.bn2_layer_call_and_return_conditional_losses_98422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_layer3.1.conv2_layer_call_and_return_conditional_losses_11628

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_activation_6_layer_call_fn_14528

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_112602
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_activation_6_layer_call_and_return_conditional_losses_11260

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer1.0.bn2_layer_call_and_return_conditional_losses_13920

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
G__inference_layer3.1.bn2_layer_call_and_return_conditional_losses_15034

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer3.0.bn1_layer_call_and_return_conditional_losses_14680

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_layer1.0.bn1_layer_call_and_return_conditional_losses_9205

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_layer2.0.downsample.1_layer_call_and_return_conditional_losses_9738

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
@__inference_conv1_layer_call_and_return_conditional_losses_10637

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_layer1.1.conv1_layer_call_and_return_conditional_losses_13993

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
G__inference_layer2.0.bn1_layer_call_and_return_conditional_losses_14226

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_layer2.0.conv1_layer_call_and_return_conditional_losses_11005

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
G__inference_layer3.1.bn2_layer_call_and_return_conditional_losses_10583

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_activation_1_layer_call_fn_13986

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_108402
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
F__inference_layer1.0.bn1_layer_call_and_return_conditional_losses_9236

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
,__inference_layer2.1.bn2_layer_call_fn_14593

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer2.1.bn2_layer_call_and_return_conditional_losses_100502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_layer3.1.conv1_layer_call_and_return_conditional_losses_14901

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
@__inference_conv1_layer_call_and_return_conditional_losses_13717

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_activation_7_layer_call_fn_14628

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_7_layer_call_and_return_conditional_losses_113422
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer2.1.bn2_layer_call_and_return_conditional_losses_10081

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_11912
input_1
conv1_11708
	bn1_11711
	bn1_11713
	bn1_11715
	bn1_11717
layer1_0_conv1_11723
layer1_0_bn1_11726
layer1_0_bn1_11728
layer1_0_bn1_11730
layer1_0_bn1_11732
layer1_0_conv2_11736
layer1_0_bn2_11739
layer1_0_bn2_11741
layer1_0_bn2_11743
layer1_0_bn2_11745
layer1_1_conv1_11750
layer1_1_bn1_11753
layer1_1_bn1_11755
layer1_1_bn1_11757
layer1_1_bn1_11759
layer1_1_conv2_11763
layer1_1_bn2_11766
layer1_1_bn2_11768
layer1_1_bn2_11770
layer1_1_bn2_11772
layer2_0_conv1_11778
layer2_0_bn1_11781
layer2_0_bn1_11783
layer2_0_bn1_11785
layer2_0_bn1_11787
layer2_0_conv2_11791
layer2_0_downsample_0_11794
layer2_0_downsample_1_11797
layer2_0_downsample_1_11799
layer2_0_downsample_1_11801
layer2_0_downsample_1_11803
layer2_0_bn2_11806
layer2_0_bn2_11808
layer2_0_bn2_11810
layer2_0_bn2_11812
layer2_1_conv1_11817
layer2_1_bn1_11820
layer2_1_bn1_11822
layer2_1_bn1_11824
layer2_1_bn1_11826
layer2_1_conv2_11830
layer2_1_bn2_11833
layer2_1_bn2_11835
layer2_1_bn2_11837
layer2_1_bn2_11839
layer3_0_conv1_11845
layer3_0_bn1_11848
layer3_0_bn1_11850
layer3_0_bn1_11852
layer3_0_bn1_11854
layer3_0_conv2_11858
layer3_0_downsample_0_11861
layer3_0_downsample_1_11864
layer3_0_downsample_1_11866
layer3_0_downsample_1_11868
layer3_0_downsample_1_11870
layer3_0_bn2_11873
layer3_0_bn2_11875
layer3_0_bn2_11877
layer3_0_bn2_11879
layer3_1_conv1_11884
layer3_1_bn1_11887
layer3_1_bn1_11889
layer3_1_bn1_11891
layer3_1_bn1_11893
layer3_1_conv2_11897
layer3_1_bn2_11900
layer3_1_bn2_11902
layer3_1_bn2_11904
layer3_1_bn2_11906
identity??bn1/StatefulPartitionedCall?conv1/StatefulPartitionedCall?$layer1.0.bn1/StatefulPartitionedCall?$layer1.0.bn2/StatefulPartitionedCall?&layer1.0.conv1/StatefulPartitionedCall?&layer1.0.conv2/StatefulPartitionedCall?$layer1.1.bn1/StatefulPartitionedCall?$layer1.1.bn2/StatefulPartitionedCall?&layer1.1.conv1/StatefulPartitionedCall?&layer1.1.conv2/StatefulPartitionedCall?$layer2.0.bn1/StatefulPartitionedCall?$layer2.0.bn2/StatefulPartitionedCall?&layer2.0.conv1/StatefulPartitionedCall?&layer2.0.conv2/StatefulPartitionedCall?-layer2.0.downsample.0/StatefulPartitionedCall?-layer2.0.downsample.1/StatefulPartitionedCall?$layer2.1.bn1/StatefulPartitionedCall?$layer2.1.bn2/StatefulPartitionedCall?&layer2.1.conv1/StatefulPartitionedCall?&layer2.1.conv2/StatefulPartitionedCall?$layer3.0.bn1/StatefulPartitionedCall?$layer3.0.bn2/StatefulPartitionedCall?&layer3.0.conv1/StatefulPartitionedCall?&layer3.0.conv2/StatefulPartitionedCall?-layer3.0.downsample.0/StatefulPartitionedCall?-layer3.0.downsample.1/StatefulPartitionedCall?$layer3.1.bn1/StatefulPartitionedCall?$layer3.1.bn2/StatefulPartitionedCall?&layer3.1.conv1/StatefulPartitionedCall?&layer3.1.conv2/StatefulPartitionedCall?
pad/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *F
fAR?
=__inference_pad_layer_call_and_return_conditional_losses_90082
pad/PartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallpad/PartitionedCall:output:0conv1_11708*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_106372
conv1/StatefulPartitionedCall?
bn1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0	bn1_11711	bn1_11713	bn1_11715	bn1_11717*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *F
fAR?
=__inference_bn1_layer_call_and_return_conditional_losses_91072
bn1/StatefulPartitionedCall?
relu/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_relu_layer_call_and_return_conditional_losses_106892
relu/PartitionedCall?
pad1/PartitionedCallPartitionedCallrelu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *G
fBR@
>__inference_pad1_layer_call_and_return_conditional_losses_91252
pad1/PartitionedCall?
maxpool/PartitionedCallPartitionedCallpad1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_maxpool_layer_call_and_return_conditional_losses_91372
maxpool/PartitionedCall?
&layer1.0.conv1/StatefulPartitionedCallStatefulPartitionedCall maxpool/PartitionedCall:output:0layer1_0_conv1_11723*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.0.conv1_layer_call_and_return_conditional_losses_107062(
&layer1.0.conv1/StatefulPartitionedCall?
$layer1.0.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer1.0.conv1/StatefulPartitionedCall:output:0layer1_0_bn1_11726layer1_0_bn1_11728layer1_0_bn1_11730layer1_0_bn1_11732*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.0.bn1_layer_call_and_return_conditional_losses_92362&
$layer1.0.bn1/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall-layer1.0.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_107582
activation/PartitionedCall?
&layer1.0.conv2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0layer1_0_conv2_11736*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.0.conv2_layer_call_and_return_conditional_losses_107732(
&layer1.0.conv2/StatefulPartitionedCall?
$layer1.0.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer1.0.conv2/StatefulPartitionedCall:output:0layer1_0_bn2_11739layer1_0_bn2_11741layer1_0_bn2_11743layer1_0_bn2_11745*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.0.bn2_layer_call_and_return_conditional_losses_93402&
$layer1.0.bn2/StatefulPartitionedCall?
add/PartitionedCallPartitionedCall maxpool/PartitionedCall:output:0-layer1.0.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_108262
add/PartitionedCall?
activation_1/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_108402
activation_1/PartitionedCall?
&layer1.1.conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0layer1_1_conv1_11750*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.1.conv1_layer_call_and_return_conditional_losses_108552(
&layer1.1.conv1/StatefulPartitionedCall?
$layer1.1.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer1.1.conv1/StatefulPartitionedCall:output:0layer1_1_bn1_11753layer1_1_bn1_11755layer1_1_bn1_11757layer1_1_bn1_11759*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.1.bn1_layer_call_and_return_conditional_losses_94442&
$layer1.1.bn1/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall-layer1.1.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_109072
activation_2/PartitionedCall?
&layer1.1.conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0layer1_1_conv2_11763*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.1.conv2_layer_call_and_return_conditional_losses_109222(
&layer1.1.conv2/StatefulPartitionedCall?
$layer1.1.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer1.1.conv2/StatefulPartitionedCall:output:0layer1_1_bn2_11766layer1_1_bn2_11768layer1_1_bn2_11770layer1_1_bn2_11772*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.1.bn2_layer_call_and_return_conditional_losses_95482&
$layer1.1.bn2/StatefulPartitionedCall?
add_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0-layer1.1.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_109752
add_1/PartitionedCall?
activation_3/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_109892
activation_3/PartitionedCall?
layer2.0.pad/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.0.pad_layer_call_and_return_conditional_losses_95662
layer2.0.pad/PartitionedCall?
&layer2.0.conv1/StatefulPartitionedCallStatefulPartitionedCall%layer2.0.pad/PartitionedCall:output:0layer2_0_conv1_11778*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.0.conv1_layer_call_and_return_conditional_losses_110052(
&layer2.0.conv1/StatefulPartitionedCall?
$layer2.0.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer2.0.conv1/StatefulPartitionedCall:output:0layer2_0_bn1_11781layer2_0_bn1_11783layer2_0_bn1_11785layer2_0_bn1_11787*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.0.bn1_layer_call_and_return_conditional_losses_96652&
$layer2.0.bn1/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall-layer2.0.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_110572
activation_4/PartitionedCall?
&layer2.0.conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0layer2_0_conv2_11791*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.0.conv2_layer_call_and_return_conditional_losses_110722(
&layer2.0.conv2/StatefulPartitionedCall?
-layer2.0.downsample.0/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0layer2_0_downsample_0_11794*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_layer2.0.downsample.0_layer_call_and_return_conditional_losses_110912/
-layer2.0.downsample.0/StatefulPartitionedCall?
-layer2.0.downsample.1/StatefulPartitionedCallStatefulPartitionedCall6layer2.0.downsample.0/StatefulPartitionedCall:output:0layer2_0_downsample_1_11797layer2_0_downsample_1_11799layer2_0_downsample_1_11801layer2_0_downsample_1_11803*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_layer2.0.downsample.1_layer_call_and_return_conditional_losses_97692/
-layer2.0.downsample.1/StatefulPartitionedCall?
$layer2.0.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer2.0.conv2/StatefulPartitionedCall:output:0layer2_0_bn2_11806layer2_0_bn2_11808layer2_0_bn2_11810layer2_0_bn2_11812*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.0.bn2_layer_call_and_return_conditional_losses_98732&
$layer2.0.bn2/StatefulPartitionedCall?
add_2/PartitionedCallPartitionedCall6layer2.0.downsample.1/StatefulPartitionedCall:output:0-layer2.0.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_111792
add_2/PartitionedCall?
activation_5/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_111932
activation_5/PartitionedCall?
&layer2.1.conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0layer2_1_conv1_11817*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.1.conv1_layer_call_and_return_conditional_losses_112082(
&layer2.1.conv1/StatefulPartitionedCall?
$layer2.1.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer2.1.conv1/StatefulPartitionedCall:output:0layer2_1_bn1_11820layer2_1_bn1_11822layer2_1_bn1_11824layer2_1_bn1_11826*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.1.bn1_layer_call_and_return_conditional_losses_99772&
$layer2.1.bn1/StatefulPartitionedCall?
activation_6/PartitionedCallPartitionedCall-layer2.1.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_112602
activation_6/PartitionedCall?
&layer2.1.conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0layer2_1_conv2_11830*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.1.conv2_layer_call_and_return_conditional_losses_112752(
&layer2.1.conv2/StatefulPartitionedCall?
$layer2.1.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer2.1.conv2/StatefulPartitionedCall:output:0layer2_1_bn2_11833layer2_1_bn2_11835layer2_1_bn2_11837layer2_1_bn2_11839*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer2.1.bn2_layer_call_and_return_conditional_losses_100812&
$layer2.1.bn2/StatefulPartitionedCall?
add_3/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0-layer2.1.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_113282
add_3/PartitionedCall?
activation_7/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_7_layer_call_and_return_conditional_losses_113422
activation_7/PartitionedCall?
layer3.0.pad/PartitionedCallPartitionedCall%activation_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.0.pad_layer_call_and_return_conditional_losses_100992
layer3.0.pad/PartitionedCall?
&layer3.0.conv1/StatefulPartitionedCallStatefulPartitionedCall%layer3.0.pad/PartitionedCall:output:0layer3_0_conv1_11845*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.0.conv1_layer_call_and_return_conditional_losses_113582(
&layer3.0.conv1/StatefulPartitionedCall?
$layer3.0.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer3.0.conv1/StatefulPartitionedCall:output:0layer3_0_bn1_11848layer3_0_bn1_11850layer3_0_bn1_11852layer3_0_bn1_11854*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.0.bn1_layer_call_and_return_conditional_losses_101982&
$layer3.0.bn1/StatefulPartitionedCall?
activation_8/PartitionedCallPartitionedCall-layer3.0.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_8_layer_call_and_return_conditional_losses_114102
activation_8/PartitionedCall?
&layer3.0.conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_8/PartitionedCall:output:0layer3_0_conv2_11858*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.0.conv2_layer_call_and_return_conditional_losses_114252(
&layer3.0.conv2/StatefulPartitionedCall?
-layer3.0.downsample.0/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0layer3_0_downsample_0_11861*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_layer3.0.downsample.0_layer_call_and_return_conditional_losses_114442/
-layer3.0.downsample.0/StatefulPartitionedCall?
-layer3.0.downsample.1/StatefulPartitionedCallStatefulPartitionedCall6layer3.0.downsample.0/StatefulPartitionedCall:output:0layer3_0_downsample_1_11864layer3_0_downsample_1_11866layer3_0_downsample_1_11868layer3_0_downsample_1_11870*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_layer3.0.downsample.1_layer_call_and_return_conditional_losses_103022/
-layer3.0.downsample.1/StatefulPartitionedCall?
$layer3.0.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer3.0.conv2/StatefulPartitionedCall:output:0layer3_0_bn2_11873layer3_0_bn2_11875layer3_0_bn2_11877layer3_0_bn2_11879*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.0.bn2_layer_call_and_return_conditional_losses_104062&
$layer3.0.bn2/StatefulPartitionedCall?
add_4/PartitionedCallPartitionedCall6layer3.0.downsample.1/StatefulPartitionedCall:output:0-layer3.0.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_4_layer_call_and_return_conditional_losses_115322
add_4/PartitionedCall?
activation_9/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_9_layer_call_and_return_conditional_losses_115462
activation_9/PartitionedCall?
&layer3.1.conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_9/PartitionedCall:output:0layer3_1_conv1_11884*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.1.conv1_layer_call_and_return_conditional_losses_115612(
&layer3.1.conv1/StatefulPartitionedCall?
$layer3.1.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer3.1.conv1/StatefulPartitionedCall:output:0layer3_1_bn1_11887layer3_1_bn1_11889layer3_1_bn1_11891layer3_1_bn1_11893*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.1.bn1_layer_call_and_return_conditional_losses_105102&
$layer3.1.bn1/StatefulPartitionedCall?
activation_10/PartitionedCallPartitionedCall-layer3.1.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_10_layer_call_and_return_conditional_losses_116132
activation_10/PartitionedCall?
&layer3.1.conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_10/PartitionedCall:output:0layer3_1_conv2_11897*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.1.conv2_layer_call_and_return_conditional_losses_116282(
&layer3.1.conv2/StatefulPartitionedCall?
$layer3.1.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer3.1.conv2/StatefulPartitionedCall:output:0layer3_1_bn2_11900layer3_1_bn2_11902layer3_1_bn2_11904layer3_1_bn2_11906*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.1.bn2_layer_call_and_return_conditional_losses_106142&
$layer3.1.bn2/StatefulPartitionedCall?
add_5/PartitionedCallPartitionedCall%activation_9/PartitionedCall:output:0-layer3.1.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_5_layer_call_and_return_conditional_losses_116812
add_5/PartitionedCall?
activation_11/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_11_layer_call_and_return_conditional_losses_116952
activation_11/PartitionedCall?

IdentityIdentity&activation_11/PartitionedCall:output:0^bn1/StatefulPartitionedCall^conv1/StatefulPartitionedCall%^layer1.0.bn1/StatefulPartitionedCall%^layer1.0.bn2/StatefulPartitionedCall'^layer1.0.conv1/StatefulPartitionedCall'^layer1.0.conv2/StatefulPartitionedCall%^layer1.1.bn1/StatefulPartitionedCall%^layer1.1.bn2/StatefulPartitionedCall'^layer1.1.conv1/StatefulPartitionedCall'^layer1.1.conv2/StatefulPartitionedCall%^layer2.0.bn1/StatefulPartitionedCall%^layer2.0.bn2/StatefulPartitionedCall'^layer2.0.conv1/StatefulPartitionedCall'^layer2.0.conv2/StatefulPartitionedCall.^layer2.0.downsample.0/StatefulPartitionedCall.^layer2.0.downsample.1/StatefulPartitionedCall%^layer2.1.bn1/StatefulPartitionedCall%^layer2.1.bn2/StatefulPartitionedCall'^layer2.1.conv1/StatefulPartitionedCall'^layer2.1.conv2/StatefulPartitionedCall%^layer3.0.bn1/StatefulPartitionedCall%^layer3.0.bn2/StatefulPartitionedCall'^layer3.0.conv1/StatefulPartitionedCall'^layer3.0.conv2/StatefulPartitionedCall.^layer3.0.downsample.0/StatefulPartitionedCall.^layer3.0.downsample.1/StatefulPartitionedCall%^layer3.1.bn1/StatefulPartitionedCall%^layer3.1.bn2/StatefulPartitionedCall'^layer3.1.conv1/StatefulPartitionedCall'^layer3.1.conv2/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:+???????????????????????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2L
$layer1.0.bn1/StatefulPartitionedCall$layer1.0.bn1/StatefulPartitionedCall2L
$layer1.0.bn2/StatefulPartitionedCall$layer1.0.bn2/StatefulPartitionedCall2P
&layer1.0.conv1/StatefulPartitionedCall&layer1.0.conv1/StatefulPartitionedCall2P
&layer1.0.conv2/StatefulPartitionedCall&layer1.0.conv2/StatefulPartitionedCall2L
$layer1.1.bn1/StatefulPartitionedCall$layer1.1.bn1/StatefulPartitionedCall2L
$layer1.1.bn2/StatefulPartitionedCall$layer1.1.bn2/StatefulPartitionedCall2P
&layer1.1.conv1/StatefulPartitionedCall&layer1.1.conv1/StatefulPartitionedCall2P
&layer1.1.conv2/StatefulPartitionedCall&layer1.1.conv2/StatefulPartitionedCall2L
$layer2.0.bn1/StatefulPartitionedCall$layer2.0.bn1/StatefulPartitionedCall2L
$layer2.0.bn2/StatefulPartitionedCall$layer2.0.bn2/StatefulPartitionedCall2P
&layer2.0.conv1/StatefulPartitionedCall&layer2.0.conv1/StatefulPartitionedCall2P
&layer2.0.conv2/StatefulPartitionedCall&layer2.0.conv2/StatefulPartitionedCall2^
-layer2.0.downsample.0/StatefulPartitionedCall-layer2.0.downsample.0/StatefulPartitionedCall2^
-layer2.0.downsample.1/StatefulPartitionedCall-layer2.0.downsample.1/StatefulPartitionedCall2L
$layer2.1.bn1/StatefulPartitionedCall$layer2.1.bn1/StatefulPartitionedCall2L
$layer2.1.bn2/StatefulPartitionedCall$layer2.1.bn2/StatefulPartitionedCall2P
&layer2.1.conv1/StatefulPartitionedCall&layer2.1.conv1/StatefulPartitionedCall2P
&layer2.1.conv2/StatefulPartitionedCall&layer2.1.conv2/StatefulPartitionedCall2L
$layer3.0.bn1/StatefulPartitionedCall$layer3.0.bn1/StatefulPartitionedCall2L
$layer3.0.bn2/StatefulPartitionedCall$layer3.0.bn2/StatefulPartitionedCall2P
&layer3.0.conv1/StatefulPartitionedCall&layer3.0.conv1/StatefulPartitionedCall2P
&layer3.0.conv2/StatefulPartitionedCall&layer3.0.conv2/StatefulPartitionedCall2^
-layer3.0.downsample.0/StatefulPartitionedCall-layer3.0.downsample.0/StatefulPartitionedCall2^
-layer3.0.downsample.1/StatefulPartitionedCall-layer3.0.downsample.1/StatefulPartitionedCall2L
$layer3.1.bn1/StatefulPartitionedCall$layer3.1.bn1/StatefulPartitionedCall2L
$layer3.1.bn2/StatefulPartitionedCall$layer3.1.bn2/StatefulPartitionedCall2P
&layer3.1.conv1/StatefulPartitionedCall&layer3.1.conv1/StatefulPartitionedCall2P
&layer3.1.conv2/StatefulPartitionedCall&layer3.1.conv2/StatefulPartitionedCall:j f
A
_output_shapes/
-:+???????????????????????????
!
_user_specified_name	input_1
?
?
I__inference_layer2.1.conv2_layer_call_and_return_conditional_losses_11275

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
Z
>__inference_pad1_layer_call_and_return_conditional_losses_9125

inputs
identity?
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Pad/paddings?
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Pad?
IdentityIdentityPad:output:0*
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
?
?
P__inference_layer2.0.downsample.0_layer_call_and_return_conditional_losses_11091

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
I__inference_layer1.0.conv2_layer_call_and_return_conditional_losses_13893

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
G__inference_activation_1_layer_call_and_return_conditional_losses_10840

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
j
>__inference_add_layer_call_and_return_conditional_losses_13970
inputs_0
inputs_1
identitys
addAddV2inputs_0inputs_1*
T0*A
_output_shapes/
-:+???????????????????????????@2
addu
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:+???????????????????????????@:+???????????????????????????@:k g
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/1
?
Q
%__inference_add_1_layer_call_fn_14164
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_109752
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:+???????????????????????????@:+???????????????????????????@:k g
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/1
?
H
,__inference_activation_8_layer_call_fn_14716

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_8_layer_call_and_return_conditional_losses_114102
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer1.0.bn2_layer_call_and_return_conditional_losses_13938

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
H
,__inference_layer3.0.pad_layer_call_fn_10105

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
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.0.pad_layer_call_and_return_conditional_losses_100992
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
?
?
F__inference_layer2.0.bn1_layer_call_and_return_conditional_losses_9634

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_activation_4_layer_call_fn_14262

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_110572
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_activation_8_layer_call_and_return_conditional_losses_11410

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
t
.__inference_layer1.0.conv2_layer_call_fn_13900

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.0.conv2_layer_call_and_return_conditional_losses_107732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
P__inference_layer2.0.downsample.1_layer_call_and_return_conditional_losses_14310

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_11_layer_call_and_return_conditional_losses_15077

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_layer1.0.bn1_layer_call_fn_13863

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.0.bn1_layer_call_and_return_conditional_losses_92052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
d
H__inference_activation_11_layer_call_and_return_conditional_losses_11695

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer3.0.bn2_layer_call_and_return_conditional_losses_10406

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer3.0.bn2_layer_call_and_return_conditional_losses_10375

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
B
&__inference_maxpool_layer_call_fn_9143

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
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_maxpool_layer_call_and_return_conditional_losses_91372
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
?
t
.__inference_layer2.0.conv2_layer_call_fn_14290

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.0.conv2_layer_call_and_return_conditional_losses_110722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer3.1.bn2_layer_call_and_return_conditional_losses_10614

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_layer3.0.bn2_layer_call_fn_14872

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.0.bn2_layer_call_and_return_conditional_losses_104062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_layer2.1.bn1_layer_call_fn_14505

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.1.bn1_layer_call_and_return_conditional_losses_99462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_layer1.0.bn2_layer_call_fn_13964

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.0.bn2_layer_call_and_return_conditional_losses_93402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
I__inference_layer2.0.conv1_layer_call_and_return_conditional_losses_14181

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
G__inference_activation_3_layer_call_and_return_conditional_losses_10989

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
l
@__inference_add_4_layer_call_and_return_conditional_losses_14878
inputs_0
inputs_1
identityt
addAddV2inputs_0inputs_1*
T0*B
_output_shapes0
.:,????????????????????????????2
addv
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:,????????????????????????????:,????????????????????????????:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/0:lh
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/1
?
c
G__inference_layer3.0.pad_layer_call_and_return_conditional_losses_10099

inputs
identity?
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Pad/paddings?
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Pad?
IdentityIdentityPad:output:0*
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
?
?
I__inference_layer1.1.conv2_layer_call_and_return_conditional_losses_10922

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
t
.__inference_layer1.0.conv1_layer_call_fn_13812

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.0.conv1_layer_call_and_return_conditional_losses_107062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
Q
%__inference_add_5_layer_call_fn_15072
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_5_layer_call_and_return_conditional_losses_116812
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:,????????????????????????????:,????????????????????????????:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/0:lh
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/1
?
?
G__inference_layer2.1.bn1_layer_call_and_return_conditional_losses_14492

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer1.1.bn2_layer_call_and_return_conditional_losses_14108

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
F__inference_layer2.0.bn2_layer_call_and_return_conditional_losses_9873

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_activation_5_layer_call_and_return_conditional_losses_11193

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_activation_9_layer_call_and_return_conditional_losses_11546

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
t
.__inference_layer2.0.conv1_layer_call_fn_14188

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.0.conv1_layer_call_and_return_conditional_losses_110052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
j
@__inference_add_3_layer_call_and_return_conditional_losses_11328

inputs
inputs_1
identityr
addAddV2inputsinputs_1*
T0*B
_output_shapes0
.:,????????????????????????????2
addv
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:,????????????????????????????:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:jf
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
5__inference_layer3.0.downsample.1_layer_call_fn_14808

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_layer3.0.downsample.1_layer_call_and_return_conditional_losses_103022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_bn1_layer_call_fn_13788

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *F
fAR?
=__inference_bn1_layer_call_and_return_conditional_losses_91072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
P__inference_layer2.0.downsample.0_layer_call_and_return_conditional_losses_14269

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
I__inference_layer3.1.conv1_layer_call_and_return_conditional_losses_11561

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_12486

inputs
conv1_12282
	bn1_12285
	bn1_12287
	bn1_12289
	bn1_12291
layer1_0_conv1_12297
layer1_0_bn1_12300
layer1_0_bn1_12302
layer1_0_bn1_12304
layer1_0_bn1_12306
layer1_0_conv2_12310
layer1_0_bn2_12313
layer1_0_bn2_12315
layer1_0_bn2_12317
layer1_0_bn2_12319
layer1_1_conv1_12324
layer1_1_bn1_12327
layer1_1_bn1_12329
layer1_1_bn1_12331
layer1_1_bn1_12333
layer1_1_conv2_12337
layer1_1_bn2_12340
layer1_1_bn2_12342
layer1_1_bn2_12344
layer1_1_bn2_12346
layer2_0_conv1_12352
layer2_0_bn1_12355
layer2_0_bn1_12357
layer2_0_bn1_12359
layer2_0_bn1_12361
layer2_0_conv2_12365
layer2_0_downsample_0_12368
layer2_0_downsample_1_12371
layer2_0_downsample_1_12373
layer2_0_downsample_1_12375
layer2_0_downsample_1_12377
layer2_0_bn2_12380
layer2_0_bn2_12382
layer2_0_bn2_12384
layer2_0_bn2_12386
layer2_1_conv1_12391
layer2_1_bn1_12394
layer2_1_bn1_12396
layer2_1_bn1_12398
layer2_1_bn1_12400
layer2_1_conv2_12404
layer2_1_bn2_12407
layer2_1_bn2_12409
layer2_1_bn2_12411
layer2_1_bn2_12413
layer3_0_conv1_12419
layer3_0_bn1_12422
layer3_0_bn1_12424
layer3_0_bn1_12426
layer3_0_bn1_12428
layer3_0_conv2_12432
layer3_0_downsample_0_12435
layer3_0_downsample_1_12438
layer3_0_downsample_1_12440
layer3_0_downsample_1_12442
layer3_0_downsample_1_12444
layer3_0_bn2_12447
layer3_0_bn2_12449
layer3_0_bn2_12451
layer3_0_bn2_12453
layer3_1_conv1_12458
layer3_1_bn1_12461
layer3_1_bn1_12463
layer3_1_bn1_12465
layer3_1_bn1_12467
layer3_1_conv2_12471
layer3_1_bn2_12474
layer3_1_bn2_12476
layer3_1_bn2_12478
layer3_1_bn2_12480
identity??bn1/StatefulPartitionedCall?conv1/StatefulPartitionedCall?$layer1.0.bn1/StatefulPartitionedCall?$layer1.0.bn2/StatefulPartitionedCall?&layer1.0.conv1/StatefulPartitionedCall?&layer1.0.conv2/StatefulPartitionedCall?$layer1.1.bn1/StatefulPartitionedCall?$layer1.1.bn2/StatefulPartitionedCall?&layer1.1.conv1/StatefulPartitionedCall?&layer1.1.conv2/StatefulPartitionedCall?$layer2.0.bn1/StatefulPartitionedCall?$layer2.0.bn2/StatefulPartitionedCall?&layer2.0.conv1/StatefulPartitionedCall?&layer2.0.conv2/StatefulPartitionedCall?-layer2.0.downsample.0/StatefulPartitionedCall?-layer2.0.downsample.1/StatefulPartitionedCall?$layer2.1.bn1/StatefulPartitionedCall?$layer2.1.bn2/StatefulPartitionedCall?&layer2.1.conv1/StatefulPartitionedCall?&layer2.1.conv2/StatefulPartitionedCall?$layer3.0.bn1/StatefulPartitionedCall?$layer3.0.bn2/StatefulPartitionedCall?&layer3.0.conv1/StatefulPartitionedCall?&layer3.0.conv2/StatefulPartitionedCall?-layer3.0.downsample.0/StatefulPartitionedCall?-layer3.0.downsample.1/StatefulPartitionedCall?$layer3.1.bn1/StatefulPartitionedCall?$layer3.1.bn2/StatefulPartitionedCall?&layer3.1.conv1/StatefulPartitionedCall?&layer3.1.conv2/StatefulPartitionedCall?
pad/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *F
fAR?
=__inference_pad_layer_call_and_return_conditional_losses_90082
pad/PartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallpad/PartitionedCall:output:0conv1_12282*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_106372
conv1/StatefulPartitionedCall?
bn1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0	bn1_12285	bn1_12287	bn1_12289	bn1_12291*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *F
fAR?
=__inference_bn1_layer_call_and_return_conditional_losses_91072
bn1/StatefulPartitionedCall?
relu/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_relu_layer_call_and_return_conditional_losses_106892
relu/PartitionedCall?
pad1/PartitionedCallPartitionedCallrelu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *G
fBR@
>__inference_pad1_layer_call_and_return_conditional_losses_91252
pad1/PartitionedCall?
maxpool/PartitionedCallPartitionedCallpad1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_maxpool_layer_call_and_return_conditional_losses_91372
maxpool/PartitionedCall?
&layer1.0.conv1/StatefulPartitionedCallStatefulPartitionedCall maxpool/PartitionedCall:output:0layer1_0_conv1_12297*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.0.conv1_layer_call_and_return_conditional_losses_107062(
&layer1.0.conv1/StatefulPartitionedCall?
$layer1.0.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer1.0.conv1/StatefulPartitionedCall:output:0layer1_0_bn1_12300layer1_0_bn1_12302layer1_0_bn1_12304layer1_0_bn1_12306*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.0.bn1_layer_call_and_return_conditional_losses_92362&
$layer1.0.bn1/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall-layer1.0.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_107582
activation/PartitionedCall?
&layer1.0.conv2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0layer1_0_conv2_12310*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.0.conv2_layer_call_and_return_conditional_losses_107732(
&layer1.0.conv2/StatefulPartitionedCall?
$layer1.0.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer1.0.conv2/StatefulPartitionedCall:output:0layer1_0_bn2_12313layer1_0_bn2_12315layer1_0_bn2_12317layer1_0_bn2_12319*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.0.bn2_layer_call_and_return_conditional_losses_93402&
$layer1.0.bn2/StatefulPartitionedCall?
add/PartitionedCallPartitionedCall maxpool/PartitionedCall:output:0-layer1.0.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_108262
add/PartitionedCall?
activation_1/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_108402
activation_1/PartitionedCall?
&layer1.1.conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0layer1_1_conv1_12324*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.1.conv1_layer_call_and_return_conditional_losses_108552(
&layer1.1.conv1/StatefulPartitionedCall?
$layer1.1.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer1.1.conv1/StatefulPartitionedCall:output:0layer1_1_bn1_12327layer1_1_bn1_12329layer1_1_bn1_12331layer1_1_bn1_12333*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.1.bn1_layer_call_and_return_conditional_losses_94442&
$layer1.1.bn1/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall-layer1.1.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_109072
activation_2/PartitionedCall?
&layer1.1.conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0layer1_1_conv2_12337*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.1.conv2_layer_call_and_return_conditional_losses_109222(
&layer1.1.conv2/StatefulPartitionedCall?
$layer1.1.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer1.1.conv2/StatefulPartitionedCall:output:0layer1_1_bn2_12340layer1_1_bn2_12342layer1_1_bn2_12344layer1_1_bn2_12346*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.1.bn2_layer_call_and_return_conditional_losses_95482&
$layer1.1.bn2/StatefulPartitionedCall?
add_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0-layer1.1.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_109752
add_1/PartitionedCall?
activation_3/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_109892
activation_3/PartitionedCall?
layer2.0.pad/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.0.pad_layer_call_and_return_conditional_losses_95662
layer2.0.pad/PartitionedCall?
&layer2.0.conv1/StatefulPartitionedCallStatefulPartitionedCall%layer2.0.pad/PartitionedCall:output:0layer2_0_conv1_12352*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.0.conv1_layer_call_and_return_conditional_losses_110052(
&layer2.0.conv1/StatefulPartitionedCall?
$layer2.0.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer2.0.conv1/StatefulPartitionedCall:output:0layer2_0_bn1_12355layer2_0_bn1_12357layer2_0_bn1_12359layer2_0_bn1_12361*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.0.bn1_layer_call_and_return_conditional_losses_96652&
$layer2.0.bn1/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall-layer2.0.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_110572
activation_4/PartitionedCall?
&layer2.0.conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0layer2_0_conv2_12365*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.0.conv2_layer_call_and_return_conditional_losses_110722(
&layer2.0.conv2/StatefulPartitionedCall?
-layer2.0.downsample.0/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0layer2_0_downsample_0_12368*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_layer2.0.downsample.0_layer_call_and_return_conditional_losses_110912/
-layer2.0.downsample.0/StatefulPartitionedCall?
-layer2.0.downsample.1/StatefulPartitionedCallStatefulPartitionedCall6layer2.0.downsample.0/StatefulPartitionedCall:output:0layer2_0_downsample_1_12371layer2_0_downsample_1_12373layer2_0_downsample_1_12375layer2_0_downsample_1_12377*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_layer2.0.downsample.1_layer_call_and_return_conditional_losses_97692/
-layer2.0.downsample.1/StatefulPartitionedCall?
$layer2.0.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer2.0.conv2/StatefulPartitionedCall:output:0layer2_0_bn2_12380layer2_0_bn2_12382layer2_0_bn2_12384layer2_0_bn2_12386*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.0.bn2_layer_call_and_return_conditional_losses_98732&
$layer2.0.bn2/StatefulPartitionedCall?
add_2/PartitionedCallPartitionedCall6layer2.0.downsample.1/StatefulPartitionedCall:output:0-layer2.0.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_111792
add_2/PartitionedCall?
activation_5/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_111932
activation_5/PartitionedCall?
&layer2.1.conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0layer2_1_conv1_12391*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.1.conv1_layer_call_and_return_conditional_losses_112082(
&layer2.1.conv1/StatefulPartitionedCall?
$layer2.1.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer2.1.conv1/StatefulPartitionedCall:output:0layer2_1_bn1_12394layer2_1_bn1_12396layer2_1_bn1_12398layer2_1_bn1_12400*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.1.bn1_layer_call_and_return_conditional_losses_99772&
$layer2.1.bn1/StatefulPartitionedCall?
activation_6/PartitionedCallPartitionedCall-layer2.1.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_112602
activation_6/PartitionedCall?
&layer2.1.conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0layer2_1_conv2_12404*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.1.conv2_layer_call_and_return_conditional_losses_112752(
&layer2.1.conv2/StatefulPartitionedCall?
$layer2.1.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer2.1.conv2/StatefulPartitionedCall:output:0layer2_1_bn2_12407layer2_1_bn2_12409layer2_1_bn2_12411layer2_1_bn2_12413*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer2.1.bn2_layer_call_and_return_conditional_losses_100812&
$layer2.1.bn2/StatefulPartitionedCall?
add_3/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0-layer2.1.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_113282
add_3/PartitionedCall?
activation_7/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_7_layer_call_and_return_conditional_losses_113422
activation_7/PartitionedCall?
layer3.0.pad/PartitionedCallPartitionedCall%activation_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.0.pad_layer_call_and_return_conditional_losses_100992
layer3.0.pad/PartitionedCall?
&layer3.0.conv1/StatefulPartitionedCallStatefulPartitionedCall%layer3.0.pad/PartitionedCall:output:0layer3_0_conv1_12419*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.0.conv1_layer_call_and_return_conditional_losses_113582(
&layer3.0.conv1/StatefulPartitionedCall?
$layer3.0.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer3.0.conv1/StatefulPartitionedCall:output:0layer3_0_bn1_12422layer3_0_bn1_12424layer3_0_bn1_12426layer3_0_bn1_12428*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.0.bn1_layer_call_and_return_conditional_losses_101982&
$layer3.0.bn1/StatefulPartitionedCall?
activation_8/PartitionedCallPartitionedCall-layer3.0.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_8_layer_call_and_return_conditional_losses_114102
activation_8/PartitionedCall?
&layer3.0.conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_8/PartitionedCall:output:0layer3_0_conv2_12432*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.0.conv2_layer_call_and_return_conditional_losses_114252(
&layer3.0.conv2/StatefulPartitionedCall?
-layer3.0.downsample.0/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0layer3_0_downsample_0_12435*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_layer3.0.downsample.0_layer_call_and_return_conditional_losses_114442/
-layer3.0.downsample.0/StatefulPartitionedCall?
-layer3.0.downsample.1/StatefulPartitionedCallStatefulPartitionedCall6layer3.0.downsample.0/StatefulPartitionedCall:output:0layer3_0_downsample_1_12438layer3_0_downsample_1_12440layer3_0_downsample_1_12442layer3_0_downsample_1_12444*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_layer3.0.downsample.1_layer_call_and_return_conditional_losses_103022/
-layer3.0.downsample.1/StatefulPartitionedCall?
$layer3.0.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer3.0.conv2/StatefulPartitionedCall:output:0layer3_0_bn2_12447layer3_0_bn2_12449layer3_0_bn2_12451layer3_0_bn2_12453*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.0.bn2_layer_call_and_return_conditional_losses_104062&
$layer3.0.bn2/StatefulPartitionedCall?
add_4/PartitionedCallPartitionedCall6layer3.0.downsample.1/StatefulPartitionedCall:output:0-layer3.0.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_4_layer_call_and_return_conditional_losses_115322
add_4/PartitionedCall?
activation_9/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_9_layer_call_and_return_conditional_losses_115462
activation_9/PartitionedCall?
&layer3.1.conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_9/PartitionedCall:output:0layer3_1_conv1_12458*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.1.conv1_layer_call_and_return_conditional_losses_115612(
&layer3.1.conv1/StatefulPartitionedCall?
$layer3.1.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer3.1.conv1/StatefulPartitionedCall:output:0layer3_1_bn1_12461layer3_1_bn1_12463layer3_1_bn1_12465layer3_1_bn1_12467*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.1.bn1_layer_call_and_return_conditional_losses_105102&
$layer3.1.bn1/StatefulPartitionedCall?
activation_10/PartitionedCallPartitionedCall-layer3.1.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_10_layer_call_and_return_conditional_losses_116132
activation_10/PartitionedCall?
&layer3.1.conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_10/PartitionedCall:output:0layer3_1_conv2_12471*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.1.conv2_layer_call_and_return_conditional_losses_116282(
&layer3.1.conv2/StatefulPartitionedCall?
$layer3.1.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer3.1.conv2/StatefulPartitionedCall:output:0layer3_1_bn2_12474layer3_1_bn2_12476layer3_1_bn2_12478layer3_1_bn2_12480*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.1.bn2_layer_call_and_return_conditional_losses_106142&
$layer3.1.bn2/StatefulPartitionedCall?
add_5/PartitionedCallPartitionedCall%activation_9/PartitionedCall:output:0-layer3.1.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_5_layer_call_and_return_conditional_losses_116812
add_5/PartitionedCall?
activation_11/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_11_layer_call_and_return_conditional_losses_116952
activation_11/PartitionedCall?

IdentityIdentity&activation_11/PartitionedCall:output:0^bn1/StatefulPartitionedCall^conv1/StatefulPartitionedCall%^layer1.0.bn1/StatefulPartitionedCall%^layer1.0.bn2/StatefulPartitionedCall'^layer1.0.conv1/StatefulPartitionedCall'^layer1.0.conv2/StatefulPartitionedCall%^layer1.1.bn1/StatefulPartitionedCall%^layer1.1.bn2/StatefulPartitionedCall'^layer1.1.conv1/StatefulPartitionedCall'^layer1.1.conv2/StatefulPartitionedCall%^layer2.0.bn1/StatefulPartitionedCall%^layer2.0.bn2/StatefulPartitionedCall'^layer2.0.conv1/StatefulPartitionedCall'^layer2.0.conv2/StatefulPartitionedCall.^layer2.0.downsample.0/StatefulPartitionedCall.^layer2.0.downsample.1/StatefulPartitionedCall%^layer2.1.bn1/StatefulPartitionedCall%^layer2.1.bn2/StatefulPartitionedCall'^layer2.1.conv1/StatefulPartitionedCall'^layer2.1.conv2/StatefulPartitionedCall%^layer3.0.bn1/StatefulPartitionedCall%^layer3.0.bn2/StatefulPartitionedCall'^layer3.0.conv1/StatefulPartitionedCall'^layer3.0.conv2/StatefulPartitionedCall.^layer3.0.downsample.0/StatefulPartitionedCall.^layer3.0.downsample.1/StatefulPartitionedCall%^layer3.1.bn1/StatefulPartitionedCall%^layer3.1.bn2/StatefulPartitionedCall'^layer3.1.conv1/StatefulPartitionedCall'^layer3.1.conv2/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:+???????????????????????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2L
$layer1.0.bn1/StatefulPartitionedCall$layer1.0.bn1/StatefulPartitionedCall2L
$layer1.0.bn2/StatefulPartitionedCall$layer1.0.bn2/StatefulPartitionedCall2P
&layer1.0.conv1/StatefulPartitionedCall&layer1.0.conv1/StatefulPartitionedCall2P
&layer1.0.conv2/StatefulPartitionedCall&layer1.0.conv2/StatefulPartitionedCall2L
$layer1.1.bn1/StatefulPartitionedCall$layer1.1.bn1/StatefulPartitionedCall2L
$layer1.1.bn2/StatefulPartitionedCall$layer1.1.bn2/StatefulPartitionedCall2P
&layer1.1.conv1/StatefulPartitionedCall&layer1.1.conv1/StatefulPartitionedCall2P
&layer1.1.conv2/StatefulPartitionedCall&layer1.1.conv2/StatefulPartitionedCall2L
$layer2.0.bn1/StatefulPartitionedCall$layer2.0.bn1/StatefulPartitionedCall2L
$layer2.0.bn2/StatefulPartitionedCall$layer2.0.bn2/StatefulPartitionedCall2P
&layer2.0.conv1/StatefulPartitionedCall&layer2.0.conv1/StatefulPartitionedCall2P
&layer2.0.conv2/StatefulPartitionedCall&layer2.0.conv2/StatefulPartitionedCall2^
-layer2.0.downsample.0/StatefulPartitionedCall-layer2.0.downsample.0/StatefulPartitionedCall2^
-layer2.0.downsample.1/StatefulPartitionedCall-layer2.0.downsample.1/StatefulPartitionedCall2L
$layer2.1.bn1/StatefulPartitionedCall$layer2.1.bn1/StatefulPartitionedCall2L
$layer2.1.bn2/StatefulPartitionedCall$layer2.1.bn2/StatefulPartitionedCall2P
&layer2.1.conv1/StatefulPartitionedCall&layer2.1.conv1/StatefulPartitionedCall2P
&layer2.1.conv2/StatefulPartitionedCall&layer2.1.conv2/StatefulPartitionedCall2L
$layer3.0.bn1/StatefulPartitionedCall$layer3.0.bn1/StatefulPartitionedCall2L
$layer3.0.bn2/StatefulPartitionedCall$layer3.0.bn2/StatefulPartitionedCall2P
&layer3.0.conv1/StatefulPartitionedCall&layer3.0.conv1/StatefulPartitionedCall2P
&layer3.0.conv2/StatefulPartitionedCall&layer3.0.conv2/StatefulPartitionedCall2^
-layer3.0.downsample.0/StatefulPartitionedCall-layer3.0.downsample.0/StatefulPartitionedCall2^
-layer3.0.downsample.1/StatefulPartitionedCall-layer3.0.downsample.1/StatefulPartitionedCall2L
$layer3.1.bn1/StatefulPartitionedCall$layer3.1.bn1/StatefulPartitionedCall2L
$layer3.1.bn2/StatefulPartitionedCall$layer3.1.bn2/StatefulPartitionedCall2P
&layer3.1.conv1/StatefulPartitionedCall&layer3.1.conv1/StatefulPartitionedCall2P
&layer3.1.conv2/StatefulPartitionedCall&layer3.1.conv2/StatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer3.1.bn1_layer_call_and_return_conditional_losses_10479

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer2.0.bn2_layer_call_and_return_conditional_losses_14392

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer3.0.bn2_layer_call_and_return_conditional_losses_14828

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?=
__inference__wrapped_model_9001
input_1.
*model_conv1_conv2d_readvariableop_resource%
!model_bn1_readvariableop_resource'
#model_bn1_readvariableop_1_resource6
2model_bn1_fusedbatchnormv3_readvariableop_resource8
4model_bn1_fusedbatchnormv3_readvariableop_1_resource7
3model_layer1_0_conv1_conv2d_readvariableop_resource.
*model_layer1_0_bn1_readvariableop_resource0
,model_layer1_0_bn1_readvariableop_1_resource?
;model_layer1_0_bn1_fusedbatchnormv3_readvariableop_resourceA
=model_layer1_0_bn1_fusedbatchnormv3_readvariableop_1_resource7
3model_layer1_0_conv2_conv2d_readvariableop_resource.
*model_layer1_0_bn2_readvariableop_resource0
,model_layer1_0_bn2_readvariableop_1_resource?
;model_layer1_0_bn2_fusedbatchnormv3_readvariableop_resourceA
=model_layer1_0_bn2_fusedbatchnormv3_readvariableop_1_resource7
3model_layer1_1_conv1_conv2d_readvariableop_resource.
*model_layer1_1_bn1_readvariableop_resource0
,model_layer1_1_bn1_readvariableop_1_resource?
;model_layer1_1_bn1_fusedbatchnormv3_readvariableop_resourceA
=model_layer1_1_bn1_fusedbatchnormv3_readvariableop_1_resource7
3model_layer1_1_conv2_conv2d_readvariableop_resource.
*model_layer1_1_bn2_readvariableop_resource0
,model_layer1_1_bn2_readvariableop_1_resource?
;model_layer1_1_bn2_fusedbatchnormv3_readvariableop_resourceA
=model_layer1_1_bn2_fusedbatchnormv3_readvariableop_1_resource7
3model_layer2_0_conv1_conv2d_readvariableop_resource.
*model_layer2_0_bn1_readvariableop_resource0
,model_layer2_0_bn1_readvariableop_1_resource?
;model_layer2_0_bn1_fusedbatchnormv3_readvariableop_resourceA
=model_layer2_0_bn1_fusedbatchnormv3_readvariableop_1_resource7
3model_layer2_0_conv2_conv2d_readvariableop_resource>
:model_layer2_0_downsample_0_conv2d_readvariableop_resource7
3model_layer2_0_downsample_1_readvariableop_resource9
5model_layer2_0_downsample_1_readvariableop_1_resourceH
Dmodel_layer2_0_downsample_1_fusedbatchnormv3_readvariableop_resourceJ
Fmodel_layer2_0_downsample_1_fusedbatchnormv3_readvariableop_1_resource.
*model_layer2_0_bn2_readvariableop_resource0
,model_layer2_0_bn2_readvariableop_1_resource?
;model_layer2_0_bn2_fusedbatchnormv3_readvariableop_resourceA
=model_layer2_0_bn2_fusedbatchnormv3_readvariableop_1_resource7
3model_layer2_1_conv1_conv2d_readvariableop_resource.
*model_layer2_1_bn1_readvariableop_resource0
,model_layer2_1_bn1_readvariableop_1_resource?
;model_layer2_1_bn1_fusedbatchnormv3_readvariableop_resourceA
=model_layer2_1_bn1_fusedbatchnormv3_readvariableop_1_resource7
3model_layer2_1_conv2_conv2d_readvariableop_resource.
*model_layer2_1_bn2_readvariableop_resource0
,model_layer2_1_bn2_readvariableop_1_resource?
;model_layer2_1_bn2_fusedbatchnormv3_readvariableop_resourceA
=model_layer2_1_bn2_fusedbatchnormv3_readvariableop_1_resource7
3model_layer3_0_conv1_conv2d_readvariableop_resource.
*model_layer3_0_bn1_readvariableop_resource0
,model_layer3_0_bn1_readvariableop_1_resource?
;model_layer3_0_bn1_fusedbatchnormv3_readvariableop_resourceA
=model_layer3_0_bn1_fusedbatchnormv3_readvariableop_1_resource7
3model_layer3_0_conv2_conv2d_readvariableop_resource>
:model_layer3_0_downsample_0_conv2d_readvariableop_resource7
3model_layer3_0_downsample_1_readvariableop_resource9
5model_layer3_0_downsample_1_readvariableop_1_resourceH
Dmodel_layer3_0_downsample_1_fusedbatchnormv3_readvariableop_resourceJ
Fmodel_layer3_0_downsample_1_fusedbatchnormv3_readvariableop_1_resource.
*model_layer3_0_bn2_readvariableop_resource0
,model_layer3_0_bn2_readvariableop_1_resource?
;model_layer3_0_bn2_fusedbatchnormv3_readvariableop_resourceA
=model_layer3_0_bn2_fusedbatchnormv3_readvariableop_1_resource7
3model_layer3_1_conv1_conv2d_readvariableop_resource.
*model_layer3_1_bn1_readvariableop_resource0
,model_layer3_1_bn1_readvariableop_1_resource?
;model_layer3_1_bn1_fusedbatchnormv3_readvariableop_resourceA
=model_layer3_1_bn1_fusedbatchnormv3_readvariableop_1_resource7
3model_layer3_1_conv2_conv2d_readvariableop_resource.
*model_layer3_1_bn2_readvariableop_resource0
,model_layer3_1_bn2_readvariableop_1_resource?
;model_layer3_1_bn2_fusedbatchnormv3_readvariableop_resourceA
=model_layer3_1_bn2_fusedbatchnormv3_readvariableop_1_resource
identity??)model/bn1/FusedBatchNormV3/ReadVariableOp?+model/bn1/FusedBatchNormV3/ReadVariableOp_1?model/bn1/ReadVariableOp?model/bn1/ReadVariableOp_1?!model/conv1/Conv2D/ReadVariableOp?2model/layer1.0.bn1/FusedBatchNormV3/ReadVariableOp?4model/layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1?!model/layer1.0.bn1/ReadVariableOp?#model/layer1.0.bn1/ReadVariableOp_1?2model/layer1.0.bn2/FusedBatchNormV3/ReadVariableOp?4model/layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1?!model/layer1.0.bn2/ReadVariableOp?#model/layer1.0.bn2/ReadVariableOp_1?*model/layer1.0.conv1/Conv2D/ReadVariableOp?*model/layer1.0.conv2/Conv2D/ReadVariableOp?2model/layer1.1.bn1/FusedBatchNormV3/ReadVariableOp?4model/layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1?!model/layer1.1.bn1/ReadVariableOp?#model/layer1.1.bn1/ReadVariableOp_1?2model/layer1.1.bn2/FusedBatchNormV3/ReadVariableOp?4model/layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1?!model/layer1.1.bn2/ReadVariableOp?#model/layer1.1.bn2/ReadVariableOp_1?*model/layer1.1.conv1/Conv2D/ReadVariableOp?*model/layer1.1.conv2/Conv2D/ReadVariableOp?2model/layer2.0.bn1/FusedBatchNormV3/ReadVariableOp?4model/layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1?!model/layer2.0.bn1/ReadVariableOp?#model/layer2.0.bn1/ReadVariableOp_1?2model/layer2.0.bn2/FusedBatchNormV3/ReadVariableOp?4model/layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1?!model/layer2.0.bn2/ReadVariableOp?#model/layer2.0.bn2/ReadVariableOp_1?*model/layer2.0.conv1/Conv2D/ReadVariableOp?*model/layer2.0.conv2/Conv2D/ReadVariableOp?1model/layer2.0.downsample.0/Conv2D/ReadVariableOp?;model/layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp?=model/layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1?*model/layer2.0.downsample.1/ReadVariableOp?,model/layer2.0.downsample.1/ReadVariableOp_1?2model/layer2.1.bn1/FusedBatchNormV3/ReadVariableOp?4model/layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1?!model/layer2.1.bn1/ReadVariableOp?#model/layer2.1.bn1/ReadVariableOp_1?2model/layer2.1.bn2/FusedBatchNormV3/ReadVariableOp?4model/layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1?!model/layer2.1.bn2/ReadVariableOp?#model/layer2.1.bn2/ReadVariableOp_1?*model/layer2.1.conv1/Conv2D/ReadVariableOp?*model/layer2.1.conv2/Conv2D/ReadVariableOp?2model/layer3.0.bn1/FusedBatchNormV3/ReadVariableOp?4model/layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1?!model/layer3.0.bn1/ReadVariableOp?#model/layer3.0.bn1/ReadVariableOp_1?2model/layer3.0.bn2/FusedBatchNormV3/ReadVariableOp?4model/layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1?!model/layer3.0.bn2/ReadVariableOp?#model/layer3.0.bn2/ReadVariableOp_1?*model/layer3.0.conv1/Conv2D/ReadVariableOp?*model/layer3.0.conv2/Conv2D/ReadVariableOp?1model/layer3.0.downsample.0/Conv2D/ReadVariableOp?;model/layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp?=model/layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1?*model/layer3.0.downsample.1/ReadVariableOp?,model/layer3.0.downsample.1/ReadVariableOp_1?2model/layer3.1.bn1/FusedBatchNormV3/ReadVariableOp?4model/layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1?!model/layer3.1.bn1/ReadVariableOp?#model/layer3.1.bn1/ReadVariableOp_1?2model/layer3.1.bn2/FusedBatchNormV3/ReadVariableOp?4model/layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1?!model/layer3.1.bn2/ReadVariableOp?#model/layer3.1.bn2/ReadVariableOp_1?*model/layer3.1.conv1/Conv2D/ReadVariableOp?*model/layer3.1.conv2/Conv2D/ReadVariableOp?
model/pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
model/pad/Pad/paddings?
model/pad/PadPadinput_1model/pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
model/pad/Pad?
!model/conv1/Conv2D/ReadVariableOpReadVariableOp*model_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02#
!model/conv1/Conv2D/ReadVariableOp?
model/conv1/Conv2DConv2Dmodel/pad/Pad:output:0)model/conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
model/conv1/Conv2D?
model/bn1/ReadVariableOpReadVariableOp!model_bn1_readvariableop_resource*
_output_shapes
:@*
dtype02
model/bn1/ReadVariableOp?
model/bn1/ReadVariableOp_1ReadVariableOp#model_bn1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
model/bn1/ReadVariableOp_1?
)model/bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp2model_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model/bn1/FusedBatchNormV3/ReadVariableOp?
+model/bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4model_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02-
+model/bn1/FusedBatchNormV3/ReadVariableOp_1?
model/bn1/FusedBatchNormV3FusedBatchNormV3model/conv1/Conv2D:output:0 model/bn1/ReadVariableOp:value:0"model/bn1/ReadVariableOp_1:value:01model/bn1/FusedBatchNormV3/ReadVariableOp:value:03model/bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
model/bn1/FusedBatchNormV3?
model/relu/ReluRelumodel/bn1/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
model/relu/Relu?
model/pad1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
model/pad1/Pad/paddings?
model/pad1/PadPadmodel/relu/Relu:activations:0 model/pad1/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
model/pad1/Pad?
model/maxpool/MaxPoolMaxPoolmodel/pad1/Pad:output:0*A
_output_shapes/
-:+???????????????????????????@*
ksize
*
paddingVALID*
strides
2
model/maxpool/MaxPool?
*model/layer1.0.conv1/Conv2D/ReadVariableOpReadVariableOp3model_layer1_0_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*model/layer1.0.conv1/Conv2D/ReadVariableOp?
model/layer1.0.conv1/Conv2DConv2Dmodel/maxpool/MaxPool:output:02model/layer1.0.conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
model/layer1.0.conv1/Conv2D?
!model/layer1.0.bn1/ReadVariableOpReadVariableOp*model_layer1_0_bn1_readvariableop_resource*
_output_shapes
:@*
dtype02#
!model/layer1.0.bn1/ReadVariableOp?
#model/layer1.0.bn1/ReadVariableOp_1ReadVariableOp,model_layer1_0_bn1_readvariableop_1_resource*
_output_shapes
:@*
dtype02%
#model/layer1.0.bn1/ReadVariableOp_1?
2model/layer1.0.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp;model_layer1_0_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype024
2model/layer1.0.bn1/FusedBatchNormV3/ReadVariableOp?
4model/layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=model_layer1_0_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4model/layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1?
#model/layer1.0.bn1/FusedBatchNormV3FusedBatchNormV3$model/layer1.0.conv1/Conv2D:output:0)model/layer1.0.bn1/ReadVariableOp:value:0+model/layer1.0.bn1/ReadVariableOp_1:value:0:model/layer1.0.bn1/FusedBatchNormV3/ReadVariableOp:value:0<model/layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2%
#model/layer1.0.bn1/FusedBatchNormV3?
model/activation/ReluRelu'model/layer1.0.bn1/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
model/activation/Relu?
*model/layer1.0.conv2/Conv2D/ReadVariableOpReadVariableOp3model_layer1_0_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*model/layer1.0.conv2/Conv2D/ReadVariableOp?
model/layer1.0.conv2/Conv2DConv2D#model/activation/Relu:activations:02model/layer1.0.conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
model/layer1.0.conv2/Conv2D?
!model/layer1.0.bn2/ReadVariableOpReadVariableOp*model_layer1_0_bn2_readvariableop_resource*
_output_shapes
:@*
dtype02#
!model/layer1.0.bn2/ReadVariableOp?
#model/layer1.0.bn2/ReadVariableOp_1ReadVariableOp,model_layer1_0_bn2_readvariableop_1_resource*
_output_shapes
:@*
dtype02%
#model/layer1.0.bn2/ReadVariableOp_1?
2model/layer1.0.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp;model_layer1_0_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype024
2model/layer1.0.bn2/FusedBatchNormV3/ReadVariableOp?
4model/layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=model_layer1_0_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4model/layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1?
#model/layer1.0.bn2/FusedBatchNormV3FusedBatchNormV3$model/layer1.0.conv2/Conv2D:output:0)model/layer1.0.bn2/ReadVariableOp:value:0+model/layer1.0.bn2/ReadVariableOp_1:value:0:model/layer1.0.bn2/FusedBatchNormV3/ReadVariableOp:value:0<model/layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2%
#model/layer1.0.bn2/FusedBatchNormV3?
model/add/addAddV2model/maxpool/MaxPool:output:0'model/layer1.0.bn2/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
model/add/add?
model/activation_1/ReluRelumodel/add/add:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
model/activation_1/Relu?
*model/layer1.1.conv1/Conv2D/ReadVariableOpReadVariableOp3model_layer1_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*model/layer1.1.conv1/Conv2D/ReadVariableOp?
model/layer1.1.conv1/Conv2DConv2D%model/activation_1/Relu:activations:02model/layer1.1.conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
model/layer1.1.conv1/Conv2D?
!model/layer1.1.bn1/ReadVariableOpReadVariableOp*model_layer1_1_bn1_readvariableop_resource*
_output_shapes
:@*
dtype02#
!model/layer1.1.bn1/ReadVariableOp?
#model/layer1.1.bn1/ReadVariableOp_1ReadVariableOp,model_layer1_1_bn1_readvariableop_1_resource*
_output_shapes
:@*
dtype02%
#model/layer1.1.bn1/ReadVariableOp_1?
2model/layer1.1.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp;model_layer1_1_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype024
2model/layer1.1.bn1/FusedBatchNormV3/ReadVariableOp?
4model/layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=model_layer1_1_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4model/layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1?
#model/layer1.1.bn1/FusedBatchNormV3FusedBatchNormV3$model/layer1.1.conv1/Conv2D:output:0)model/layer1.1.bn1/ReadVariableOp:value:0+model/layer1.1.bn1/ReadVariableOp_1:value:0:model/layer1.1.bn1/FusedBatchNormV3/ReadVariableOp:value:0<model/layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2%
#model/layer1.1.bn1/FusedBatchNormV3?
model/activation_2/ReluRelu'model/layer1.1.bn1/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
model/activation_2/Relu?
*model/layer1.1.conv2/Conv2D/ReadVariableOpReadVariableOp3model_layer1_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*model/layer1.1.conv2/Conv2D/ReadVariableOp?
model/layer1.1.conv2/Conv2DConv2D%model/activation_2/Relu:activations:02model/layer1.1.conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
model/layer1.1.conv2/Conv2D?
!model/layer1.1.bn2/ReadVariableOpReadVariableOp*model_layer1_1_bn2_readvariableop_resource*
_output_shapes
:@*
dtype02#
!model/layer1.1.bn2/ReadVariableOp?
#model/layer1.1.bn2/ReadVariableOp_1ReadVariableOp,model_layer1_1_bn2_readvariableop_1_resource*
_output_shapes
:@*
dtype02%
#model/layer1.1.bn2/ReadVariableOp_1?
2model/layer1.1.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp;model_layer1_1_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype024
2model/layer1.1.bn2/FusedBatchNormV3/ReadVariableOp?
4model/layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=model_layer1_1_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4model/layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1?
#model/layer1.1.bn2/FusedBatchNormV3FusedBatchNormV3$model/layer1.1.conv2/Conv2D:output:0)model/layer1.1.bn2/ReadVariableOp:value:0+model/layer1.1.bn2/ReadVariableOp_1:value:0:model/layer1.1.bn2/FusedBatchNormV3/ReadVariableOp:value:0<model/layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2%
#model/layer1.1.bn2/FusedBatchNormV3?
model/add_1/addAddV2%model/activation_1/Relu:activations:0'model/layer1.1.bn2/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
model/add_1/add?
model/activation_3/ReluRelumodel/add_1/add:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
model/activation_3/Relu?
model/layer2.0.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2!
model/layer2.0.pad/Pad/paddings?
model/layer2.0.pad/PadPad%model/activation_3/Relu:activations:0(model/layer2.0.pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
model/layer2.0.pad/Pad?
*model/layer2.0.conv1/Conv2D/ReadVariableOpReadVariableOp3model_layer2_0_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02,
*model/layer2.0.conv1/Conv2D/ReadVariableOp?
model/layer2.0.conv1/Conv2DConv2Dmodel/layer2.0.pad/Pad:output:02model/layer2.0.conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
model/layer2.0.conv1/Conv2D?
!model/layer2.0.bn1/ReadVariableOpReadVariableOp*model_layer2_0_bn1_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/layer2.0.bn1/ReadVariableOp?
#model/layer2.0.bn1/ReadVariableOp_1ReadVariableOp,model_layer2_0_bn1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02%
#model/layer2.0.bn1/ReadVariableOp_1?
2model/layer2.0.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp;model_layer2_0_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model/layer2.0.bn1/FusedBatchNormV3/ReadVariableOp?
4model/layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=model_layer2_0_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4model/layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1?
#model/layer2.0.bn1/FusedBatchNormV3FusedBatchNormV3$model/layer2.0.conv1/Conv2D:output:0)model/layer2.0.bn1/ReadVariableOp:value:0+model/layer2.0.bn1/ReadVariableOp_1:value:0:model/layer2.0.bn1/FusedBatchNormV3/ReadVariableOp:value:0<model/layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2%
#model/layer2.0.bn1/FusedBatchNormV3?
model/activation_4/ReluRelu'model/layer2.0.bn1/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
model/activation_4/Relu?
*model/layer2.0.conv2/Conv2D/ReadVariableOpReadVariableOp3model_layer2_0_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*model/layer2.0.conv2/Conv2D/ReadVariableOp?
model/layer2.0.conv2/Conv2DConv2D%model/activation_4/Relu:activations:02model/layer2.0.conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
model/layer2.0.conv2/Conv2D?
1model/layer2.0.downsample.0/Conv2D/ReadVariableOpReadVariableOp:model_layer2_0_downsample_0_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype023
1model/layer2.0.downsample.0/Conv2D/ReadVariableOp?
"model/layer2.0.downsample.0/Conv2DConv2D%model/activation_3/Relu:activations:09model/layer2.0.downsample.0/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2$
"model/layer2.0.downsample.0/Conv2D?
*model/layer2.0.downsample.1/ReadVariableOpReadVariableOp3model_layer2_0_downsample_1_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model/layer2.0.downsample.1/ReadVariableOp?
,model/layer2.0.downsample.1/ReadVariableOp_1ReadVariableOp5model_layer2_0_downsample_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02.
,model/layer2.0.downsample.1/ReadVariableOp_1?
;model/layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_layer2_0_downsample_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;model/layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp?
=model/layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_layer2_0_downsample_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02?
=model/layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1?
,model/layer2.0.downsample.1/FusedBatchNormV3FusedBatchNormV3+model/layer2.0.downsample.0/Conv2D:output:02model/layer2.0.downsample.1/ReadVariableOp:value:04model/layer2.0.downsample.1/ReadVariableOp_1:value:0Cmodel/layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2.
,model/layer2.0.downsample.1/FusedBatchNormV3?
!model/layer2.0.bn2/ReadVariableOpReadVariableOp*model_layer2_0_bn2_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/layer2.0.bn2/ReadVariableOp?
#model/layer2.0.bn2/ReadVariableOp_1ReadVariableOp,model_layer2_0_bn2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02%
#model/layer2.0.bn2/ReadVariableOp_1?
2model/layer2.0.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp;model_layer2_0_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model/layer2.0.bn2/FusedBatchNormV3/ReadVariableOp?
4model/layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=model_layer2_0_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4model/layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1?
#model/layer2.0.bn2/FusedBatchNormV3FusedBatchNormV3$model/layer2.0.conv2/Conv2D:output:0)model/layer2.0.bn2/ReadVariableOp:value:0+model/layer2.0.bn2/ReadVariableOp_1:value:0:model/layer2.0.bn2/FusedBatchNormV3/ReadVariableOp:value:0<model/layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2%
#model/layer2.0.bn2/FusedBatchNormV3?
model/add_2/addAddV20model/layer2.0.downsample.1/FusedBatchNormV3:y:0'model/layer2.0.bn2/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
model/add_2/add?
model/activation_5/ReluRelumodel/add_2/add:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
model/activation_5/Relu?
*model/layer2.1.conv1/Conv2D/ReadVariableOpReadVariableOp3model_layer2_1_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*model/layer2.1.conv1/Conv2D/ReadVariableOp?
model/layer2.1.conv1/Conv2DConv2D%model/activation_5/Relu:activations:02model/layer2.1.conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
model/layer2.1.conv1/Conv2D?
!model/layer2.1.bn1/ReadVariableOpReadVariableOp*model_layer2_1_bn1_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/layer2.1.bn1/ReadVariableOp?
#model/layer2.1.bn1/ReadVariableOp_1ReadVariableOp,model_layer2_1_bn1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02%
#model/layer2.1.bn1/ReadVariableOp_1?
2model/layer2.1.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp;model_layer2_1_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model/layer2.1.bn1/FusedBatchNormV3/ReadVariableOp?
4model/layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=model_layer2_1_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4model/layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1?
#model/layer2.1.bn1/FusedBatchNormV3FusedBatchNormV3$model/layer2.1.conv1/Conv2D:output:0)model/layer2.1.bn1/ReadVariableOp:value:0+model/layer2.1.bn1/ReadVariableOp_1:value:0:model/layer2.1.bn1/FusedBatchNormV3/ReadVariableOp:value:0<model/layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2%
#model/layer2.1.bn1/FusedBatchNormV3?
model/activation_6/ReluRelu'model/layer2.1.bn1/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
model/activation_6/Relu?
*model/layer2.1.conv2/Conv2D/ReadVariableOpReadVariableOp3model_layer2_1_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*model/layer2.1.conv2/Conv2D/ReadVariableOp?
model/layer2.1.conv2/Conv2DConv2D%model/activation_6/Relu:activations:02model/layer2.1.conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
model/layer2.1.conv2/Conv2D?
!model/layer2.1.bn2/ReadVariableOpReadVariableOp*model_layer2_1_bn2_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/layer2.1.bn2/ReadVariableOp?
#model/layer2.1.bn2/ReadVariableOp_1ReadVariableOp,model_layer2_1_bn2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02%
#model/layer2.1.bn2/ReadVariableOp_1?
2model/layer2.1.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp;model_layer2_1_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model/layer2.1.bn2/FusedBatchNormV3/ReadVariableOp?
4model/layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=model_layer2_1_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4model/layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1?
#model/layer2.1.bn2/FusedBatchNormV3FusedBatchNormV3$model/layer2.1.conv2/Conv2D:output:0)model/layer2.1.bn2/ReadVariableOp:value:0+model/layer2.1.bn2/ReadVariableOp_1:value:0:model/layer2.1.bn2/FusedBatchNormV3/ReadVariableOp:value:0<model/layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2%
#model/layer2.1.bn2/FusedBatchNormV3?
model/add_3/addAddV2%model/activation_5/Relu:activations:0'model/layer2.1.bn2/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
model/add_3/add?
model/activation_7/ReluRelumodel/add_3/add:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
model/activation_7/Relu?
model/layer3.0.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2!
model/layer3.0.pad/Pad/paddings?
model/layer3.0.pad/PadPad%model/activation_7/Relu:activations:0(model/layer3.0.pad/Pad/paddings:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
model/layer3.0.pad/Pad?
*model/layer3.0.conv1/Conv2D/ReadVariableOpReadVariableOp3model_layer3_0_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*model/layer3.0.conv1/Conv2D/ReadVariableOp?
model/layer3.0.conv1/Conv2DConv2Dmodel/layer3.0.pad/Pad:output:02model/layer3.0.conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
model/layer3.0.conv1/Conv2D?
!model/layer3.0.bn1/ReadVariableOpReadVariableOp*model_layer3_0_bn1_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/layer3.0.bn1/ReadVariableOp?
#model/layer3.0.bn1/ReadVariableOp_1ReadVariableOp,model_layer3_0_bn1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02%
#model/layer3.0.bn1/ReadVariableOp_1?
2model/layer3.0.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp;model_layer3_0_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model/layer3.0.bn1/FusedBatchNormV3/ReadVariableOp?
4model/layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=model_layer3_0_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4model/layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1?
#model/layer3.0.bn1/FusedBatchNormV3FusedBatchNormV3$model/layer3.0.conv1/Conv2D:output:0)model/layer3.0.bn1/ReadVariableOp:value:0+model/layer3.0.bn1/ReadVariableOp_1:value:0:model/layer3.0.bn1/FusedBatchNormV3/ReadVariableOp:value:0<model/layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2%
#model/layer3.0.bn1/FusedBatchNormV3?
model/activation_8/ReluRelu'model/layer3.0.bn1/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
model/activation_8/Relu?
*model/layer3.0.conv2/Conv2D/ReadVariableOpReadVariableOp3model_layer3_0_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*model/layer3.0.conv2/Conv2D/ReadVariableOp?
model/layer3.0.conv2/Conv2DConv2D%model/activation_8/Relu:activations:02model/layer3.0.conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
model/layer3.0.conv2/Conv2D?
1model/layer3.0.downsample.0/Conv2D/ReadVariableOpReadVariableOp:model_layer3_0_downsample_0_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype023
1model/layer3.0.downsample.0/Conv2D/ReadVariableOp?
"model/layer3.0.downsample.0/Conv2DConv2D%model/activation_7/Relu:activations:09model/layer3.0.downsample.0/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2$
"model/layer3.0.downsample.0/Conv2D?
*model/layer3.0.downsample.1/ReadVariableOpReadVariableOp3model_layer3_0_downsample_1_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model/layer3.0.downsample.1/ReadVariableOp?
,model/layer3.0.downsample.1/ReadVariableOp_1ReadVariableOp5model_layer3_0_downsample_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02.
,model/layer3.0.downsample.1/ReadVariableOp_1?
;model/layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_layer3_0_downsample_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;model/layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp?
=model/layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_layer3_0_downsample_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02?
=model/layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1?
,model/layer3.0.downsample.1/FusedBatchNormV3FusedBatchNormV3+model/layer3.0.downsample.0/Conv2D:output:02model/layer3.0.downsample.1/ReadVariableOp:value:04model/layer3.0.downsample.1/ReadVariableOp_1:value:0Cmodel/layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2.
,model/layer3.0.downsample.1/FusedBatchNormV3?
!model/layer3.0.bn2/ReadVariableOpReadVariableOp*model_layer3_0_bn2_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/layer3.0.bn2/ReadVariableOp?
#model/layer3.0.bn2/ReadVariableOp_1ReadVariableOp,model_layer3_0_bn2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02%
#model/layer3.0.bn2/ReadVariableOp_1?
2model/layer3.0.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp;model_layer3_0_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model/layer3.0.bn2/FusedBatchNormV3/ReadVariableOp?
4model/layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=model_layer3_0_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4model/layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1?
#model/layer3.0.bn2/FusedBatchNormV3FusedBatchNormV3$model/layer3.0.conv2/Conv2D:output:0)model/layer3.0.bn2/ReadVariableOp:value:0+model/layer3.0.bn2/ReadVariableOp_1:value:0:model/layer3.0.bn2/FusedBatchNormV3/ReadVariableOp:value:0<model/layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2%
#model/layer3.0.bn2/FusedBatchNormV3?
model/add_4/addAddV20model/layer3.0.downsample.1/FusedBatchNormV3:y:0'model/layer3.0.bn2/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
model/add_4/add?
model/activation_9/ReluRelumodel/add_4/add:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
model/activation_9/Relu?
*model/layer3.1.conv1/Conv2D/ReadVariableOpReadVariableOp3model_layer3_1_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*model/layer3.1.conv1/Conv2D/ReadVariableOp?
model/layer3.1.conv1/Conv2DConv2D%model/activation_9/Relu:activations:02model/layer3.1.conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
model/layer3.1.conv1/Conv2D?
!model/layer3.1.bn1/ReadVariableOpReadVariableOp*model_layer3_1_bn1_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/layer3.1.bn1/ReadVariableOp?
#model/layer3.1.bn1/ReadVariableOp_1ReadVariableOp,model_layer3_1_bn1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02%
#model/layer3.1.bn1/ReadVariableOp_1?
2model/layer3.1.bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp;model_layer3_1_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model/layer3.1.bn1/FusedBatchNormV3/ReadVariableOp?
4model/layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=model_layer3_1_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4model/layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1?
#model/layer3.1.bn1/FusedBatchNormV3FusedBatchNormV3$model/layer3.1.conv1/Conv2D:output:0)model/layer3.1.bn1/ReadVariableOp:value:0+model/layer3.1.bn1/ReadVariableOp_1:value:0:model/layer3.1.bn1/FusedBatchNormV3/ReadVariableOp:value:0<model/layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2%
#model/layer3.1.bn1/FusedBatchNormV3?
model/activation_10/ReluRelu'model/layer3.1.bn1/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
model/activation_10/Relu?
*model/layer3.1.conv2/Conv2D/ReadVariableOpReadVariableOp3model_layer3_1_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*model/layer3.1.conv2/Conv2D/ReadVariableOp?
model/layer3.1.conv2/Conv2DConv2D&model/activation_10/Relu:activations:02model/layer3.1.conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
model/layer3.1.conv2/Conv2D?
!model/layer3.1.bn2/ReadVariableOpReadVariableOp*model_layer3_1_bn2_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/layer3.1.bn2/ReadVariableOp?
#model/layer3.1.bn2/ReadVariableOp_1ReadVariableOp,model_layer3_1_bn2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02%
#model/layer3.1.bn2/ReadVariableOp_1?
2model/layer3.1.bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp;model_layer3_1_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model/layer3.1.bn2/FusedBatchNormV3/ReadVariableOp?
4model/layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=model_layer3_1_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4model/layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1?
#model/layer3.1.bn2/FusedBatchNormV3FusedBatchNormV3$model/layer3.1.conv2/Conv2D:output:0)model/layer3.1.bn2/ReadVariableOp:value:0+model/layer3.1.bn2/ReadVariableOp_1:value:0:model/layer3.1.bn2/FusedBatchNormV3/ReadVariableOp:value:0<model/layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2%
#model/layer3.1.bn2/FusedBatchNormV3?
model/add_5/addAddV2%model/activation_9/Relu:activations:0'model/layer3.1.bn2/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
model/add_5/add?
model/activation_11/ReluRelumodel/add_5/add:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
model/activation_11/Relu?
IdentityIdentity&model/activation_11/Relu:activations:0*^model/bn1/FusedBatchNormV3/ReadVariableOp,^model/bn1/FusedBatchNormV3/ReadVariableOp_1^model/bn1/ReadVariableOp^model/bn1/ReadVariableOp_1"^model/conv1/Conv2D/ReadVariableOp3^model/layer1.0.bn1/FusedBatchNormV3/ReadVariableOp5^model/layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_1"^model/layer1.0.bn1/ReadVariableOp$^model/layer1.0.bn1/ReadVariableOp_13^model/layer1.0.bn2/FusedBatchNormV3/ReadVariableOp5^model/layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_1"^model/layer1.0.bn2/ReadVariableOp$^model/layer1.0.bn2/ReadVariableOp_1+^model/layer1.0.conv1/Conv2D/ReadVariableOp+^model/layer1.0.conv2/Conv2D/ReadVariableOp3^model/layer1.1.bn1/FusedBatchNormV3/ReadVariableOp5^model/layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_1"^model/layer1.1.bn1/ReadVariableOp$^model/layer1.1.bn1/ReadVariableOp_13^model/layer1.1.bn2/FusedBatchNormV3/ReadVariableOp5^model/layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_1"^model/layer1.1.bn2/ReadVariableOp$^model/layer1.1.bn2/ReadVariableOp_1+^model/layer1.1.conv1/Conv2D/ReadVariableOp+^model/layer1.1.conv2/Conv2D/ReadVariableOp3^model/layer2.0.bn1/FusedBatchNormV3/ReadVariableOp5^model/layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_1"^model/layer2.0.bn1/ReadVariableOp$^model/layer2.0.bn1/ReadVariableOp_13^model/layer2.0.bn2/FusedBatchNormV3/ReadVariableOp5^model/layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_1"^model/layer2.0.bn2/ReadVariableOp$^model/layer2.0.bn2/ReadVariableOp_1+^model/layer2.0.conv1/Conv2D/ReadVariableOp+^model/layer2.0.conv2/Conv2D/ReadVariableOp2^model/layer2.0.downsample.0/Conv2D/ReadVariableOp<^model/layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp>^model/layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1+^model/layer2.0.downsample.1/ReadVariableOp-^model/layer2.0.downsample.1/ReadVariableOp_13^model/layer2.1.bn1/FusedBatchNormV3/ReadVariableOp5^model/layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_1"^model/layer2.1.bn1/ReadVariableOp$^model/layer2.1.bn1/ReadVariableOp_13^model/layer2.1.bn2/FusedBatchNormV3/ReadVariableOp5^model/layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_1"^model/layer2.1.bn2/ReadVariableOp$^model/layer2.1.bn2/ReadVariableOp_1+^model/layer2.1.conv1/Conv2D/ReadVariableOp+^model/layer2.1.conv2/Conv2D/ReadVariableOp3^model/layer3.0.bn1/FusedBatchNormV3/ReadVariableOp5^model/layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_1"^model/layer3.0.bn1/ReadVariableOp$^model/layer3.0.bn1/ReadVariableOp_13^model/layer3.0.bn2/FusedBatchNormV3/ReadVariableOp5^model/layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_1"^model/layer3.0.bn2/ReadVariableOp$^model/layer3.0.bn2/ReadVariableOp_1+^model/layer3.0.conv1/Conv2D/ReadVariableOp+^model/layer3.0.conv2/Conv2D/ReadVariableOp2^model/layer3.0.downsample.0/Conv2D/ReadVariableOp<^model/layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp>^model/layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1+^model/layer3.0.downsample.1/ReadVariableOp-^model/layer3.0.downsample.1/ReadVariableOp_13^model/layer3.1.bn1/FusedBatchNormV3/ReadVariableOp5^model/layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_1"^model/layer3.1.bn1/ReadVariableOp$^model/layer3.1.bn1/ReadVariableOp_13^model/layer3.1.bn2/FusedBatchNormV3/ReadVariableOp5^model/layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_1"^model/layer3.1.bn2/ReadVariableOp$^model/layer3.1.bn2/ReadVariableOp_1+^model/layer3.1.conv1/Conv2D/ReadVariableOp+^model/layer3.1.conv2/Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:+???????????????????????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2V
)model/bn1/FusedBatchNormV3/ReadVariableOp)model/bn1/FusedBatchNormV3/ReadVariableOp2Z
+model/bn1/FusedBatchNormV3/ReadVariableOp_1+model/bn1/FusedBatchNormV3/ReadVariableOp_124
model/bn1/ReadVariableOpmodel/bn1/ReadVariableOp28
model/bn1/ReadVariableOp_1model/bn1/ReadVariableOp_12F
!model/conv1/Conv2D/ReadVariableOp!model/conv1/Conv2D/ReadVariableOp2h
2model/layer1.0.bn1/FusedBatchNormV3/ReadVariableOp2model/layer1.0.bn1/FusedBatchNormV3/ReadVariableOp2l
4model/layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_14model/layer1.0.bn1/FusedBatchNormV3/ReadVariableOp_12F
!model/layer1.0.bn1/ReadVariableOp!model/layer1.0.bn1/ReadVariableOp2J
#model/layer1.0.bn1/ReadVariableOp_1#model/layer1.0.bn1/ReadVariableOp_12h
2model/layer1.0.bn2/FusedBatchNormV3/ReadVariableOp2model/layer1.0.bn2/FusedBatchNormV3/ReadVariableOp2l
4model/layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_14model/layer1.0.bn2/FusedBatchNormV3/ReadVariableOp_12F
!model/layer1.0.bn2/ReadVariableOp!model/layer1.0.bn2/ReadVariableOp2J
#model/layer1.0.bn2/ReadVariableOp_1#model/layer1.0.bn2/ReadVariableOp_12X
*model/layer1.0.conv1/Conv2D/ReadVariableOp*model/layer1.0.conv1/Conv2D/ReadVariableOp2X
*model/layer1.0.conv2/Conv2D/ReadVariableOp*model/layer1.0.conv2/Conv2D/ReadVariableOp2h
2model/layer1.1.bn1/FusedBatchNormV3/ReadVariableOp2model/layer1.1.bn1/FusedBatchNormV3/ReadVariableOp2l
4model/layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_14model/layer1.1.bn1/FusedBatchNormV3/ReadVariableOp_12F
!model/layer1.1.bn1/ReadVariableOp!model/layer1.1.bn1/ReadVariableOp2J
#model/layer1.1.bn1/ReadVariableOp_1#model/layer1.1.bn1/ReadVariableOp_12h
2model/layer1.1.bn2/FusedBatchNormV3/ReadVariableOp2model/layer1.1.bn2/FusedBatchNormV3/ReadVariableOp2l
4model/layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_14model/layer1.1.bn2/FusedBatchNormV3/ReadVariableOp_12F
!model/layer1.1.bn2/ReadVariableOp!model/layer1.1.bn2/ReadVariableOp2J
#model/layer1.1.bn2/ReadVariableOp_1#model/layer1.1.bn2/ReadVariableOp_12X
*model/layer1.1.conv1/Conv2D/ReadVariableOp*model/layer1.1.conv1/Conv2D/ReadVariableOp2X
*model/layer1.1.conv2/Conv2D/ReadVariableOp*model/layer1.1.conv2/Conv2D/ReadVariableOp2h
2model/layer2.0.bn1/FusedBatchNormV3/ReadVariableOp2model/layer2.0.bn1/FusedBatchNormV3/ReadVariableOp2l
4model/layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_14model/layer2.0.bn1/FusedBatchNormV3/ReadVariableOp_12F
!model/layer2.0.bn1/ReadVariableOp!model/layer2.0.bn1/ReadVariableOp2J
#model/layer2.0.bn1/ReadVariableOp_1#model/layer2.0.bn1/ReadVariableOp_12h
2model/layer2.0.bn2/FusedBatchNormV3/ReadVariableOp2model/layer2.0.bn2/FusedBatchNormV3/ReadVariableOp2l
4model/layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_14model/layer2.0.bn2/FusedBatchNormV3/ReadVariableOp_12F
!model/layer2.0.bn2/ReadVariableOp!model/layer2.0.bn2/ReadVariableOp2J
#model/layer2.0.bn2/ReadVariableOp_1#model/layer2.0.bn2/ReadVariableOp_12X
*model/layer2.0.conv1/Conv2D/ReadVariableOp*model/layer2.0.conv1/Conv2D/ReadVariableOp2X
*model/layer2.0.conv2/Conv2D/ReadVariableOp*model/layer2.0.conv2/Conv2D/ReadVariableOp2f
1model/layer2.0.downsample.0/Conv2D/ReadVariableOp1model/layer2.0.downsample.0/Conv2D/ReadVariableOp2z
;model/layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp;model/layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp2~
=model/layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1=model/layer2.0.downsample.1/FusedBatchNormV3/ReadVariableOp_12X
*model/layer2.0.downsample.1/ReadVariableOp*model/layer2.0.downsample.1/ReadVariableOp2\
,model/layer2.0.downsample.1/ReadVariableOp_1,model/layer2.0.downsample.1/ReadVariableOp_12h
2model/layer2.1.bn1/FusedBatchNormV3/ReadVariableOp2model/layer2.1.bn1/FusedBatchNormV3/ReadVariableOp2l
4model/layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_14model/layer2.1.bn1/FusedBatchNormV3/ReadVariableOp_12F
!model/layer2.1.bn1/ReadVariableOp!model/layer2.1.bn1/ReadVariableOp2J
#model/layer2.1.bn1/ReadVariableOp_1#model/layer2.1.bn1/ReadVariableOp_12h
2model/layer2.1.bn2/FusedBatchNormV3/ReadVariableOp2model/layer2.1.bn2/FusedBatchNormV3/ReadVariableOp2l
4model/layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_14model/layer2.1.bn2/FusedBatchNormV3/ReadVariableOp_12F
!model/layer2.1.bn2/ReadVariableOp!model/layer2.1.bn2/ReadVariableOp2J
#model/layer2.1.bn2/ReadVariableOp_1#model/layer2.1.bn2/ReadVariableOp_12X
*model/layer2.1.conv1/Conv2D/ReadVariableOp*model/layer2.1.conv1/Conv2D/ReadVariableOp2X
*model/layer2.1.conv2/Conv2D/ReadVariableOp*model/layer2.1.conv2/Conv2D/ReadVariableOp2h
2model/layer3.0.bn1/FusedBatchNormV3/ReadVariableOp2model/layer3.0.bn1/FusedBatchNormV3/ReadVariableOp2l
4model/layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_14model/layer3.0.bn1/FusedBatchNormV3/ReadVariableOp_12F
!model/layer3.0.bn1/ReadVariableOp!model/layer3.0.bn1/ReadVariableOp2J
#model/layer3.0.bn1/ReadVariableOp_1#model/layer3.0.bn1/ReadVariableOp_12h
2model/layer3.0.bn2/FusedBatchNormV3/ReadVariableOp2model/layer3.0.bn2/FusedBatchNormV3/ReadVariableOp2l
4model/layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_14model/layer3.0.bn2/FusedBatchNormV3/ReadVariableOp_12F
!model/layer3.0.bn2/ReadVariableOp!model/layer3.0.bn2/ReadVariableOp2J
#model/layer3.0.bn2/ReadVariableOp_1#model/layer3.0.bn2/ReadVariableOp_12X
*model/layer3.0.conv1/Conv2D/ReadVariableOp*model/layer3.0.conv1/Conv2D/ReadVariableOp2X
*model/layer3.0.conv2/Conv2D/ReadVariableOp*model/layer3.0.conv2/Conv2D/ReadVariableOp2f
1model/layer3.0.downsample.0/Conv2D/ReadVariableOp1model/layer3.0.downsample.0/Conv2D/ReadVariableOp2z
;model/layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp;model/layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp2~
=model/layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_1=model/layer3.0.downsample.1/FusedBatchNormV3/ReadVariableOp_12X
*model/layer3.0.downsample.1/ReadVariableOp*model/layer3.0.downsample.1/ReadVariableOp2\
,model/layer3.0.downsample.1/ReadVariableOp_1,model/layer3.0.downsample.1/ReadVariableOp_12h
2model/layer3.1.bn1/FusedBatchNormV3/ReadVariableOp2model/layer3.1.bn1/FusedBatchNormV3/ReadVariableOp2l
4model/layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_14model/layer3.1.bn1/FusedBatchNormV3/ReadVariableOp_12F
!model/layer3.1.bn1/ReadVariableOp!model/layer3.1.bn1/ReadVariableOp2J
#model/layer3.1.bn1/ReadVariableOp_1#model/layer3.1.bn1/ReadVariableOp_12h
2model/layer3.1.bn2/FusedBatchNormV3/ReadVariableOp2model/layer3.1.bn2/FusedBatchNormV3/ReadVariableOp2l
4model/layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_14model/layer3.1.bn2/FusedBatchNormV3/ReadVariableOp_12F
!model/layer3.1.bn2/ReadVariableOp!model/layer3.1.bn2/ReadVariableOp2J
#model/layer3.1.bn2/ReadVariableOp_1#model/layer3.1.bn2/ReadVariableOp_12X
*model/layer3.1.conv1/Conv2D/ReadVariableOp*model/layer3.1.conv1/Conv2D/ReadVariableOp2X
*model/layer3.1.conv2/Conv2D/ReadVariableOp*model/layer3.1.conv2/Conv2D/ReadVariableOp:j f
A
_output_shapes/
-:+???????????????????????????
!
_user_specified_name	input_1
?
j
@__inference_add_5_layer_call_and_return_conditional_losses_11681

inputs
inputs_1
identityr
addAddV2inputsinputs_1*
T0*B
_output_shapes0
.:,????????????????????????????2
addv
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:,????????????????????????????:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:jf
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
t
.__inference_layer2.1.conv2_layer_call_fn_14542

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.1.conv2_layer_call_and_return_conditional_losses_112752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
h
>__inference_add_layer_call_and_return_conditional_losses_10826

inputs
inputs_1
identityq
addAddV2inputsinputs_1*
T0*A
_output_shapes/
-:+???????????????????????????@2
addu
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:+???????????????????????????@:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
,__inference_layer2.0.bn1_layer_call_fn_14239

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.0.bn1_layer_call_and_return_conditional_losses_96342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer2.1.bn1_layer_call_and_return_conditional_losses_14474

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer2.0.bn2_layer_call_and_return_conditional_losses_14374

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_layer3.0.conv1_layer_call_and_return_conditional_losses_14635

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_layer2.0.pad_layer_call_and_return_conditional_losses_9566

inputs
identity?
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Pad/paddings?
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Pad?
IdentityIdentityPad:output:0*
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
?
j
@__inference_add_4_layer_call_and_return_conditional_losses_11532

inputs
inputs_1
identityr
addAddV2inputsinputs_1*
T0*B
_output_shapes0
.:,????????????????????????????2
addv
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:,????????????????????????????:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:jf
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
t
.__inference_layer1.1.conv2_layer_call_fn_14088

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.1.conv2_layer_call_and_return_conditional_losses_109222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
G__inference_layer3.0.bn1_layer_call_and_return_conditional_losses_14662

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_layer1.1.bn2_layer_call_fn_14139

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.1.bn2_layer_call_and_return_conditional_losses_95172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
d
H__inference_activation_10_layer_call_and_return_conditional_losses_14977

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_layer1.1.bn2_layer_call_and_return_conditional_losses_9548

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
I__inference_layer2.1.conv1_layer_call_and_return_conditional_losses_11208

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,????????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
@
$__inference_relu_layer_call_fn_13798

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_relu_layer_call_and_return_conditional_losses_106892
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_11704
input_1
conv1_10646
	bn1_10675
	bn1_10677
	bn1_10679
	bn1_10681
layer1_0_conv1_10715
layer1_0_bn1_10744
layer1_0_bn1_10746
layer1_0_bn1_10748
layer1_0_bn1_10750
layer1_0_conv2_10782
layer1_0_bn2_10811
layer1_0_bn2_10813
layer1_0_bn2_10815
layer1_0_bn2_10817
layer1_1_conv1_10864
layer1_1_bn1_10893
layer1_1_bn1_10895
layer1_1_bn1_10897
layer1_1_bn1_10899
layer1_1_conv2_10931
layer1_1_bn2_10960
layer1_1_bn2_10962
layer1_1_bn2_10964
layer1_1_bn2_10966
layer2_0_conv1_11014
layer2_0_bn1_11043
layer2_0_bn1_11045
layer2_0_bn1_11047
layer2_0_bn1_11049
layer2_0_conv2_11081
layer2_0_downsample_0_11100
layer2_0_downsample_1_11129
layer2_0_downsample_1_11131
layer2_0_downsample_1_11133
layer2_0_downsample_1_11135
layer2_0_bn2_11164
layer2_0_bn2_11166
layer2_0_bn2_11168
layer2_0_bn2_11170
layer2_1_conv1_11217
layer2_1_bn1_11246
layer2_1_bn1_11248
layer2_1_bn1_11250
layer2_1_bn1_11252
layer2_1_conv2_11284
layer2_1_bn2_11313
layer2_1_bn2_11315
layer2_1_bn2_11317
layer2_1_bn2_11319
layer3_0_conv1_11367
layer3_0_bn1_11396
layer3_0_bn1_11398
layer3_0_bn1_11400
layer3_0_bn1_11402
layer3_0_conv2_11434
layer3_0_downsample_0_11453
layer3_0_downsample_1_11482
layer3_0_downsample_1_11484
layer3_0_downsample_1_11486
layer3_0_downsample_1_11488
layer3_0_bn2_11517
layer3_0_bn2_11519
layer3_0_bn2_11521
layer3_0_bn2_11523
layer3_1_conv1_11570
layer3_1_bn1_11599
layer3_1_bn1_11601
layer3_1_bn1_11603
layer3_1_bn1_11605
layer3_1_conv2_11637
layer3_1_bn2_11666
layer3_1_bn2_11668
layer3_1_bn2_11670
layer3_1_bn2_11672
identity??bn1/StatefulPartitionedCall?conv1/StatefulPartitionedCall?$layer1.0.bn1/StatefulPartitionedCall?$layer1.0.bn2/StatefulPartitionedCall?&layer1.0.conv1/StatefulPartitionedCall?&layer1.0.conv2/StatefulPartitionedCall?$layer1.1.bn1/StatefulPartitionedCall?$layer1.1.bn2/StatefulPartitionedCall?&layer1.1.conv1/StatefulPartitionedCall?&layer1.1.conv2/StatefulPartitionedCall?$layer2.0.bn1/StatefulPartitionedCall?$layer2.0.bn2/StatefulPartitionedCall?&layer2.0.conv1/StatefulPartitionedCall?&layer2.0.conv2/StatefulPartitionedCall?-layer2.0.downsample.0/StatefulPartitionedCall?-layer2.0.downsample.1/StatefulPartitionedCall?$layer2.1.bn1/StatefulPartitionedCall?$layer2.1.bn2/StatefulPartitionedCall?&layer2.1.conv1/StatefulPartitionedCall?&layer2.1.conv2/StatefulPartitionedCall?$layer3.0.bn1/StatefulPartitionedCall?$layer3.0.bn2/StatefulPartitionedCall?&layer3.0.conv1/StatefulPartitionedCall?&layer3.0.conv2/StatefulPartitionedCall?-layer3.0.downsample.0/StatefulPartitionedCall?-layer3.0.downsample.1/StatefulPartitionedCall?$layer3.1.bn1/StatefulPartitionedCall?$layer3.1.bn2/StatefulPartitionedCall?&layer3.1.conv1/StatefulPartitionedCall?&layer3.1.conv2/StatefulPartitionedCall?
pad/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *F
fAR?
=__inference_pad_layer_call_and_return_conditional_losses_90082
pad/PartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallpad/PartitionedCall:output:0conv1_10646*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_106372
conv1/StatefulPartitionedCall?
bn1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0	bn1_10675	bn1_10677	bn1_10679	bn1_10681*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *F
fAR?
=__inference_bn1_layer_call_and_return_conditional_losses_90762
bn1/StatefulPartitionedCall?
relu/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_relu_layer_call_and_return_conditional_losses_106892
relu/PartitionedCall?
pad1/PartitionedCallPartitionedCallrelu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *G
fBR@
>__inference_pad1_layer_call_and_return_conditional_losses_91252
pad1/PartitionedCall?
maxpool/PartitionedCallPartitionedCallpad1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_maxpool_layer_call_and_return_conditional_losses_91372
maxpool/PartitionedCall?
&layer1.0.conv1/StatefulPartitionedCallStatefulPartitionedCall maxpool/PartitionedCall:output:0layer1_0_conv1_10715*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.0.conv1_layer_call_and_return_conditional_losses_107062(
&layer1.0.conv1/StatefulPartitionedCall?
$layer1.0.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer1.0.conv1/StatefulPartitionedCall:output:0layer1_0_bn1_10744layer1_0_bn1_10746layer1_0_bn1_10748layer1_0_bn1_10750*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.0.bn1_layer_call_and_return_conditional_losses_92052&
$layer1.0.bn1/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall-layer1.0.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_107582
activation/PartitionedCall?
&layer1.0.conv2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0layer1_0_conv2_10782*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.0.conv2_layer_call_and_return_conditional_losses_107732(
&layer1.0.conv2/StatefulPartitionedCall?
$layer1.0.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer1.0.conv2/StatefulPartitionedCall:output:0layer1_0_bn2_10811layer1_0_bn2_10813layer1_0_bn2_10815layer1_0_bn2_10817*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.0.bn2_layer_call_and_return_conditional_losses_93092&
$layer1.0.bn2/StatefulPartitionedCall?
add/PartitionedCallPartitionedCall maxpool/PartitionedCall:output:0-layer1.0.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_108262
add/PartitionedCall?
activation_1/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_108402
activation_1/PartitionedCall?
&layer1.1.conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0layer1_1_conv1_10864*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.1.conv1_layer_call_and_return_conditional_losses_108552(
&layer1.1.conv1/StatefulPartitionedCall?
$layer1.1.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer1.1.conv1/StatefulPartitionedCall:output:0layer1_1_bn1_10893layer1_1_bn1_10895layer1_1_bn1_10897layer1_1_bn1_10899*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.1.bn1_layer_call_and_return_conditional_losses_94132&
$layer1.1.bn1/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall-layer1.1.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_109072
activation_2/PartitionedCall?
&layer1.1.conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0layer1_1_conv2_10931*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer1.1.conv2_layer_call_and_return_conditional_losses_109222(
&layer1.1.conv2/StatefulPartitionedCall?
$layer1.1.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer1.1.conv2/StatefulPartitionedCall:output:0layer1_1_bn2_10960layer1_1_bn2_10962layer1_1_bn2_10964layer1_1_bn2_10966*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer1.1.bn2_layer_call_and_return_conditional_losses_95172&
$layer1.1.bn2/StatefulPartitionedCall?
add_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0-layer1.1.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_109752
add_1/PartitionedCall?
activation_3/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_109892
activation_3/PartitionedCall?
layer2.0.pad/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.0.pad_layer_call_and_return_conditional_losses_95662
layer2.0.pad/PartitionedCall?
&layer2.0.conv1/StatefulPartitionedCallStatefulPartitionedCall%layer2.0.pad/PartitionedCall:output:0layer2_0_conv1_11014*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.0.conv1_layer_call_and_return_conditional_losses_110052(
&layer2.0.conv1/StatefulPartitionedCall?
$layer2.0.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer2.0.conv1/StatefulPartitionedCall:output:0layer2_0_bn1_11043layer2_0_bn1_11045layer2_0_bn1_11047layer2_0_bn1_11049*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.0.bn1_layer_call_and_return_conditional_losses_96342&
$layer2.0.bn1/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall-layer2.0.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_110572
activation_4/PartitionedCall?
&layer2.0.conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0layer2_0_conv2_11081*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.0.conv2_layer_call_and_return_conditional_losses_110722(
&layer2.0.conv2/StatefulPartitionedCall?
-layer2.0.downsample.0/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0layer2_0_downsample_0_11100*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_layer2.0.downsample.0_layer_call_and_return_conditional_losses_110912/
-layer2.0.downsample.0/StatefulPartitionedCall?
-layer2.0.downsample.1/StatefulPartitionedCallStatefulPartitionedCall6layer2.0.downsample.0/StatefulPartitionedCall:output:0layer2_0_downsample_1_11129layer2_0_downsample_1_11131layer2_0_downsample_1_11133layer2_0_downsample_1_11135*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_layer2.0.downsample.1_layer_call_and_return_conditional_losses_97382/
-layer2.0.downsample.1/StatefulPartitionedCall?
$layer2.0.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer2.0.conv2/StatefulPartitionedCall:output:0layer2_0_bn2_11164layer2_0_bn2_11166layer2_0_bn2_11168layer2_0_bn2_11170*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.0.bn2_layer_call_and_return_conditional_losses_98422&
$layer2.0.bn2/StatefulPartitionedCall?
add_2/PartitionedCallPartitionedCall6layer2.0.downsample.1/StatefulPartitionedCall:output:0-layer2.0.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_111792
add_2/PartitionedCall?
activation_5/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_111932
activation_5/PartitionedCall?
&layer2.1.conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0layer2_1_conv1_11217*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.1.conv1_layer_call_and_return_conditional_losses_112082(
&layer2.1.conv1/StatefulPartitionedCall?
$layer2.1.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer2.1.conv1/StatefulPartitionedCall:output:0layer2_1_bn1_11246layer2_1_bn1_11248layer2_1_bn1_11250layer2_1_bn1_11252*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_layer2.1.bn1_layer_call_and_return_conditional_losses_99462&
$layer2.1.bn1/StatefulPartitionedCall?
activation_6/PartitionedCallPartitionedCall-layer2.1.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_112602
activation_6/PartitionedCall?
&layer2.1.conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0layer2_1_conv2_11284*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer2.1.conv2_layer_call_and_return_conditional_losses_112752(
&layer2.1.conv2/StatefulPartitionedCall?
$layer2.1.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer2.1.conv2/StatefulPartitionedCall:output:0layer2_1_bn2_11313layer2_1_bn2_11315layer2_1_bn2_11317layer2_1_bn2_11319*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer2.1.bn2_layer_call_and_return_conditional_losses_100502&
$layer2.1.bn2/StatefulPartitionedCall?
add_3/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0-layer2.1.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_113282
add_3/PartitionedCall?
activation_7/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_7_layer_call_and_return_conditional_losses_113422
activation_7/PartitionedCall?
layer3.0.pad/PartitionedCallPartitionedCall%activation_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.0.pad_layer_call_and_return_conditional_losses_100992
layer3.0.pad/PartitionedCall?
&layer3.0.conv1/StatefulPartitionedCallStatefulPartitionedCall%layer3.0.pad/PartitionedCall:output:0layer3_0_conv1_11367*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.0.conv1_layer_call_and_return_conditional_losses_113582(
&layer3.0.conv1/StatefulPartitionedCall?
$layer3.0.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer3.0.conv1/StatefulPartitionedCall:output:0layer3_0_bn1_11396layer3_0_bn1_11398layer3_0_bn1_11400layer3_0_bn1_11402*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.0.bn1_layer_call_and_return_conditional_losses_101672&
$layer3.0.bn1/StatefulPartitionedCall?
activation_8/PartitionedCallPartitionedCall-layer3.0.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_8_layer_call_and_return_conditional_losses_114102
activation_8/PartitionedCall?
&layer3.0.conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_8/PartitionedCall:output:0layer3_0_conv2_11434*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.0.conv2_layer_call_and_return_conditional_losses_114252(
&layer3.0.conv2/StatefulPartitionedCall?
-layer3.0.downsample.0/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0layer3_0_downsample_0_11453*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_layer3.0.downsample.0_layer_call_and_return_conditional_losses_114442/
-layer3.0.downsample.0/StatefulPartitionedCall?
-layer3.0.downsample.1/StatefulPartitionedCallStatefulPartitionedCall6layer3.0.downsample.0/StatefulPartitionedCall:output:0layer3_0_downsample_1_11482layer3_0_downsample_1_11484layer3_0_downsample_1_11486layer3_0_downsample_1_11488*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_layer3.0.downsample.1_layer_call_and_return_conditional_losses_102712/
-layer3.0.downsample.1/StatefulPartitionedCall?
$layer3.0.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer3.0.conv2/StatefulPartitionedCall:output:0layer3_0_bn2_11517layer3_0_bn2_11519layer3_0_bn2_11521layer3_0_bn2_11523*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.0.bn2_layer_call_and_return_conditional_losses_103752&
$layer3.0.bn2/StatefulPartitionedCall?
add_4/PartitionedCallPartitionedCall6layer3.0.downsample.1/StatefulPartitionedCall:output:0-layer3.0.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_4_layer_call_and_return_conditional_losses_115322
add_4/PartitionedCall?
activation_9/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_9_layer_call_and_return_conditional_losses_115462
activation_9/PartitionedCall?
&layer3.1.conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_9/PartitionedCall:output:0layer3_1_conv1_11570*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.1.conv1_layer_call_and_return_conditional_losses_115612(
&layer3.1.conv1/StatefulPartitionedCall?
$layer3.1.bn1/StatefulPartitionedCallStatefulPartitionedCall/layer3.1.conv1/StatefulPartitionedCall:output:0layer3_1_bn1_11599layer3_1_bn1_11601layer3_1_bn1_11603layer3_1_bn1_11605*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.1.bn1_layer_call_and_return_conditional_losses_104792&
$layer3.1.bn1/StatefulPartitionedCall?
activation_10/PartitionedCallPartitionedCall-layer3.1.bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_10_layer_call_and_return_conditional_losses_116132
activation_10/PartitionedCall?
&layer3.1.conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_10/PartitionedCall:output:0layer3_1_conv2_11637*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_layer3.1.conv2_layer_call_and_return_conditional_losses_116282(
&layer3.1.conv2/StatefulPartitionedCall?
$layer3.1.bn2/StatefulPartitionedCallStatefulPartitionedCall/layer3.1.conv2/StatefulPartitionedCall:output:0layer3_1_bn2_11666layer3_1_bn2_11668layer3_1_bn2_11670layer3_1_bn2_11672*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.1.bn2_layer_call_and_return_conditional_losses_105832&
$layer3.1.bn2/StatefulPartitionedCall?
add_5/PartitionedCallPartitionedCall%activation_9/PartitionedCall:output:0-layer3.1.bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_add_5_layer_call_and_return_conditional_losses_116812
add_5/PartitionedCall?
activation_11/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_11_layer_call_and_return_conditional_losses_116952
activation_11/PartitionedCall?

IdentityIdentity&activation_11/PartitionedCall:output:0^bn1/StatefulPartitionedCall^conv1/StatefulPartitionedCall%^layer1.0.bn1/StatefulPartitionedCall%^layer1.0.bn2/StatefulPartitionedCall'^layer1.0.conv1/StatefulPartitionedCall'^layer1.0.conv2/StatefulPartitionedCall%^layer1.1.bn1/StatefulPartitionedCall%^layer1.1.bn2/StatefulPartitionedCall'^layer1.1.conv1/StatefulPartitionedCall'^layer1.1.conv2/StatefulPartitionedCall%^layer2.0.bn1/StatefulPartitionedCall%^layer2.0.bn2/StatefulPartitionedCall'^layer2.0.conv1/StatefulPartitionedCall'^layer2.0.conv2/StatefulPartitionedCall.^layer2.0.downsample.0/StatefulPartitionedCall.^layer2.0.downsample.1/StatefulPartitionedCall%^layer2.1.bn1/StatefulPartitionedCall%^layer2.1.bn2/StatefulPartitionedCall'^layer2.1.conv1/StatefulPartitionedCall'^layer2.1.conv2/StatefulPartitionedCall%^layer3.0.bn1/StatefulPartitionedCall%^layer3.0.bn2/StatefulPartitionedCall'^layer3.0.conv1/StatefulPartitionedCall'^layer3.0.conv2/StatefulPartitionedCall.^layer3.0.downsample.0/StatefulPartitionedCall.^layer3.0.downsample.1/StatefulPartitionedCall%^layer3.1.bn1/StatefulPartitionedCall%^layer3.1.bn2/StatefulPartitionedCall'^layer3.1.conv1/StatefulPartitionedCall'^layer3.1.conv2/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:+???????????????????????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2L
$layer1.0.bn1/StatefulPartitionedCall$layer1.0.bn1/StatefulPartitionedCall2L
$layer1.0.bn2/StatefulPartitionedCall$layer1.0.bn2/StatefulPartitionedCall2P
&layer1.0.conv1/StatefulPartitionedCall&layer1.0.conv1/StatefulPartitionedCall2P
&layer1.0.conv2/StatefulPartitionedCall&layer1.0.conv2/StatefulPartitionedCall2L
$layer1.1.bn1/StatefulPartitionedCall$layer1.1.bn1/StatefulPartitionedCall2L
$layer1.1.bn2/StatefulPartitionedCall$layer1.1.bn2/StatefulPartitionedCall2P
&layer1.1.conv1/StatefulPartitionedCall&layer1.1.conv1/StatefulPartitionedCall2P
&layer1.1.conv2/StatefulPartitionedCall&layer1.1.conv2/StatefulPartitionedCall2L
$layer2.0.bn1/StatefulPartitionedCall$layer2.0.bn1/StatefulPartitionedCall2L
$layer2.0.bn2/StatefulPartitionedCall$layer2.0.bn2/StatefulPartitionedCall2P
&layer2.0.conv1/StatefulPartitionedCall&layer2.0.conv1/StatefulPartitionedCall2P
&layer2.0.conv2/StatefulPartitionedCall&layer2.0.conv2/StatefulPartitionedCall2^
-layer2.0.downsample.0/StatefulPartitionedCall-layer2.0.downsample.0/StatefulPartitionedCall2^
-layer2.0.downsample.1/StatefulPartitionedCall-layer2.0.downsample.1/StatefulPartitionedCall2L
$layer2.1.bn1/StatefulPartitionedCall$layer2.1.bn1/StatefulPartitionedCall2L
$layer2.1.bn2/StatefulPartitionedCall$layer2.1.bn2/StatefulPartitionedCall2P
&layer2.1.conv1/StatefulPartitionedCall&layer2.1.conv1/StatefulPartitionedCall2P
&layer2.1.conv2/StatefulPartitionedCall&layer2.1.conv2/StatefulPartitionedCall2L
$layer3.0.bn1/StatefulPartitionedCall$layer3.0.bn1/StatefulPartitionedCall2L
$layer3.0.bn2/StatefulPartitionedCall$layer3.0.bn2/StatefulPartitionedCall2P
&layer3.0.conv1/StatefulPartitionedCall&layer3.0.conv1/StatefulPartitionedCall2P
&layer3.0.conv2/StatefulPartitionedCall&layer3.0.conv2/StatefulPartitionedCall2^
-layer3.0.downsample.0/StatefulPartitionedCall-layer3.0.downsample.0/StatefulPartitionedCall2^
-layer3.0.downsample.1/StatefulPartitionedCall-layer3.0.downsample.1/StatefulPartitionedCall2L
$layer3.1.bn1/StatefulPartitionedCall$layer3.1.bn1/StatefulPartitionedCall2L
$layer3.1.bn2/StatefulPartitionedCall$layer3.1.bn2/StatefulPartitionedCall2P
&layer3.1.conv1/StatefulPartitionedCall&layer3.1.conv1/StatefulPartitionedCall2P
&layer3.1.conv2/StatefulPartitionedCall&layer3.1.conv2/StatefulPartitionedCall:j f
A
_output_shapes/
-:+???????????????????????????
!
_user_specified_name	input_1
?
?
F__inference_layer1.1.bn1_layer_call_and_return_conditional_losses_9444

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
G__inference_activation_4_layer_call_and_return_conditional_losses_11057

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_layer3.1.bn1_layer_call_fn_14959

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.1.bn1_layer_call_and_return_conditional_losses_104792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_activation_9_layer_call_fn_14894

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_activation_9_layer_call_and_return_conditional_losses_115462
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_layer1.0.conv1_layer_call_and_return_conditional_losses_13805

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
F__inference_layer2.1.bn1_layer_call_and_return_conditional_losses_9946

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_layer3.1.bn2_layer_call_fn_15060

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.1.bn2_layer_call_and_return_conditional_losses_106142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer1.1.bn1_layer_call_and_return_conditional_losses_14038

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
,__inference_layer3.1.bn2_layer_call_fn_15047

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_layer3.1.bn2_layer_call_and_return_conditional_losses_105832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_layer3.0.downsample.1_layer_call_and_return_conditional_losses_14764

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_layer3.0.downsample.1_layer_call_and_return_conditional_losses_10271

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_layer1.0.bn1_layer_call_and_return_conditional_losses_13832

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
j
@__inference_add_1_layer_call_and_return_conditional_losses_10975

inputs
inputs_1
identityq
addAddV2inputsinputs_1*
T0*A
_output_shapes/
-:+???????????????????????????@2
addu
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:+???????????????????????????@:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
G__inference_layer3.1.bn1_layer_call_and_return_conditional_losses_10510

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
k
%__inference_conv1_layer_call_fn_13724

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_106372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
l
@__inference_add_1_layer_call_and_return_conditional_losses_14158
inputs_0
inputs_1
identitys
addAddV2inputs_0inputs_1*
T0*A
_output_shapes/
-:+???????????????????????????@2
addu
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:+???????????????????????????@:+???????????????????????????@:k g
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/1
?
?
G__inference_layer3.0.bn1_layer_call_and_return_conditional_losses_10198

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_13762

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?!
__inference__traced_save_15330
file_prefix+
'savev2_conv1_kernel_read_readvariableop(
$savev2_bn1_gamma_read_readvariableop'
#savev2_bn1_beta_read_readvariableop.
*savev2_bn1_moving_mean_read_readvariableop2
.savev2_bn1_moving_variance_read_readvariableop4
0savev2_layer1_0_conv1_kernel_read_readvariableop1
-savev2_layer1_0_bn1_gamma_read_readvariableop0
,savev2_layer1_0_bn1_beta_read_readvariableop7
3savev2_layer1_0_bn1_moving_mean_read_readvariableop;
7savev2_layer1_0_bn1_moving_variance_read_readvariableop4
0savev2_layer1_0_conv2_kernel_read_readvariableop1
-savev2_layer1_0_bn2_gamma_read_readvariableop0
,savev2_layer1_0_bn2_beta_read_readvariableop7
3savev2_layer1_0_bn2_moving_mean_read_readvariableop;
7savev2_layer1_0_bn2_moving_variance_read_readvariableop4
0savev2_layer1_1_conv1_kernel_read_readvariableop1
-savev2_layer1_1_bn1_gamma_read_readvariableop0
,savev2_layer1_1_bn1_beta_read_readvariableop7
3savev2_layer1_1_bn1_moving_mean_read_readvariableop;
7savev2_layer1_1_bn1_moving_variance_read_readvariableop4
0savev2_layer1_1_conv2_kernel_read_readvariableop1
-savev2_layer1_1_bn2_gamma_read_readvariableop0
,savev2_layer1_1_bn2_beta_read_readvariableop7
3savev2_layer1_1_bn2_moving_mean_read_readvariableop;
7savev2_layer1_1_bn2_moving_variance_read_readvariableop4
0savev2_layer2_0_conv1_kernel_read_readvariableop1
-savev2_layer2_0_bn1_gamma_read_readvariableop0
,savev2_layer2_0_bn1_beta_read_readvariableop7
3savev2_layer2_0_bn1_moving_mean_read_readvariableop;
7savev2_layer2_0_bn1_moving_variance_read_readvariableop;
7savev2_layer2_0_downsample_0_kernel_read_readvariableop4
0savev2_layer2_0_conv2_kernel_read_readvariableop:
6savev2_layer2_0_downsample_1_gamma_read_readvariableop9
5savev2_layer2_0_downsample_1_beta_read_readvariableop@
<savev2_layer2_0_downsample_1_moving_mean_read_readvariableopD
@savev2_layer2_0_downsample_1_moving_variance_read_readvariableop1
-savev2_layer2_0_bn2_gamma_read_readvariableop0
,savev2_layer2_0_bn2_beta_read_readvariableop7
3savev2_layer2_0_bn2_moving_mean_read_readvariableop;
7savev2_layer2_0_bn2_moving_variance_read_readvariableop4
0savev2_layer2_1_conv1_kernel_read_readvariableop1
-savev2_layer2_1_bn1_gamma_read_readvariableop0
,savev2_layer2_1_bn1_beta_read_readvariableop7
3savev2_layer2_1_bn1_moving_mean_read_readvariableop;
7savev2_layer2_1_bn1_moving_variance_read_readvariableop4
0savev2_layer2_1_conv2_kernel_read_readvariableop1
-savev2_layer2_1_bn2_gamma_read_readvariableop0
,savev2_layer2_1_bn2_beta_read_readvariableop7
3savev2_layer2_1_bn2_moving_mean_read_readvariableop;
7savev2_layer2_1_bn2_moving_variance_read_readvariableop4
0savev2_layer3_0_conv1_kernel_read_readvariableop1
-savev2_layer3_0_bn1_gamma_read_readvariableop0
,savev2_layer3_0_bn1_beta_read_readvariableop7
3savev2_layer3_0_bn1_moving_mean_read_readvariableop;
7savev2_layer3_0_bn1_moving_variance_read_readvariableop;
7savev2_layer3_0_downsample_0_kernel_read_readvariableop4
0savev2_layer3_0_conv2_kernel_read_readvariableop:
6savev2_layer3_0_downsample_1_gamma_read_readvariableop9
5savev2_layer3_0_downsample_1_beta_read_readvariableop@
<savev2_layer3_0_downsample_1_moving_mean_read_readvariableopD
@savev2_layer3_0_downsample_1_moving_variance_read_readvariableop1
-savev2_layer3_0_bn2_gamma_read_readvariableop0
,savev2_layer3_0_bn2_beta_read_readvariableop7
3savev2_layer3_0_bn2_moving_mean_read_readvariableop;
7savev2_layer3_0_bn2_moving_variance_read_readvariableop4
0savev2_layer3_1_conv1_kernel_read_readvariableop1
-savev2_layer3_1_bn1_gamma_read_readvariableop0
,savev2_layer3_1_bn1_beta_read_readvariableop7
3savev2_layer3_1_bn1_moving_mean_read_readvariableop;
7savev2_layer3_1_bn1_moving_variance_read_readvariableop4
0savev2_layer3_1_conv2_kernel_read_readvariableop1
-savev2_layer3_1_bn2_gamma_read_readvariableop0
,savev2_layer3_1_bn2_beta_read_readvariableop7
3savev2_layer3_1_bn2_moving_mean_read_readvariableop;
7savev2_layer3_1_bn2_moving_variance_read_readvariableop
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
ShardedFilename?#
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?"
value?"B?"LB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-24/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-24/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-24/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-25/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-25/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-25/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-27/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-27/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-27/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-29/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-29/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-29/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?
value?B?LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices? 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop$savev2_bn1_gamma_read_readvariableop#savev2_bn1_beta_read_readvariableop*savev2_bn1_moving_mean_read_readvariableop.savev2_bn1_moving_variance_read_readvariableop0savev2_layer1_0_conv1_kernel_read_readvariableop-savev2_layer1_0_bn1_gamma_read_readvariableop,savev2_layer1_0_bn1_beta_read_readvariableop3savev2_layer1_0_bn1_moving_mean_read_readvariableop7savev2_layer1_0_bn1_moving_variance_read_readvariableop0savev2_layer1_0_conv2_kernel_read_readvariableop-savev2_layer1_0_bn2_gamma_read_readvariableop,savev2_layer1_0_bn2_beta_read_readvariableop3savev2_layer1_0_bn2_moving_mean_read_readvariableop7savev2_layer1_0_bn2_moving_variance_read_readvariableop0savev2_layer1_1_conv1_kernel_read_readvariableop-savev2_layer1_1_bn1_gamma_read_readvariableop,savev2_layer1_1_bn1_beta_read_readvariableop3savev2_layer1_1_bn1_moving_mean_read_readvariableop7savev2_layer1_1_bn1_moving_variance_read_readvariableop0savev2_layer1_1_conv2_kernel_read_readvariableop-savev2_layer1_1_bn2_gamma_read_readvariableop,savev2_layer1_1_bn2_beta_read_readvariableop3savev2_layer1_1_bn2_moving_mean_read_readvariableop7savev2_layer1_1_bn2_moving_variance_read_readvariableop0savev2_layer2_0_conv1_kernel_read_readvariableop-savev2_layer2_0_bn1_gamma_read_readvariableop,savev2_layer2_0_bn1_beta_read_readvariableop3savev2_layer2_0_bn1_moving_mean_read_readvariableop7savev2_layer2_0_bn1_moving_variance_read_readvariableop7savev2_layer2_0_downsample_0_kernel_read_readvariableop0savev2_layer2_0_conv2_kernel_read_readvariableop6savev2_layer2_0_downsample_1_gamma_read_readvariableop5savev2_layer2_0_downsample_1_beta_read_readvariableop<savev2_layer2_0_downsample_1_moving_mean_read_readvariableop@savev2_layer2_0_downsample_1_moving_variance_read_readvariableop-savev2_layer2_0_bn2_gamma_read_readvariableop,savev2_layer2_0_bn2_beta_read_readvariableop3savev2_layer2_0_bn2_moving_mean_read_readvariableop7savev2_layer2_0_bn2_moving_variance_read_readvariableop0savev2_layer2_1_conv1_kernel_read_readvariableop-savev2_layer2_1_bn1_gamma_read_readvariableop,savev2_layer2_1_bn1_beta_read_readvariableop3savev2_layer2_1_bn1_moving_mean_read_readvariableop7savev2_layer2_1_bn1_moving_variance_read_readvariableop0savev2_layer2_1_conv2_kernel_read_readvariableop-savev2_layer2_1_bn2_gamma_read_readvariableop,savev2_layer2_1_bn2_beta_read_readvariableop3savev2_layer2_1_bn2_moving_mean_read_readvariableop7savev2_layer2_1_bn2_moving_variance_read_readvariableop0savev2_layer3_0_conv1_kernel_read_readvariableop-savev2_layer3_0_bn1_gamma_read_readvariableop,savev2_layer3_0_bn1_beta_read_readvariableop3savev2_layer3_0_bn1_moving_mean_read_readvariableop7savev2_layer3_0_bn1_moving_variance_read_readvariableop7savev2_layer3_0_downsample_0_kernel_read_readvariableop0savev2_layer3_0_conv2_kernel_read_readvariableop6savev2_layer3_0_downsample_1_gamma_read_readvariableop5savev2_layer3_0_downsample_1_beta_read_readvariableop<savev2_layer3_0_downsample_1_moving_mean_read_readvariableop@savev2_layer3_0_downsample_1_moving_variance_read_readvariableop-savev2_layer3_0_bn2_gamma_read_readvariableop,savev2_layer3_0_bn2_beta_read_readvariableop3savev2_layer3_0_bn2_moving_mean_read_readvariableop7savev2_layer3_0_bn2_moving_variance_read_readvariableop0savev2_layer3_1_conv1_kernel_read_readvariableop-savev2_layer3_1_bn1_gamma_read_readvariableop,savev2_layer3_1_bn1_beta_read_readvariableop3savev2_layer3_1_bn1_moving_mean_read_readvariableop7savev2_layer3_1_bn1_moving_variance_read_readvariableop0savev2_layer3_1_conv2_kernel_read_readvariableop-savev2_layer3_1_bn2_gamma_read_readvariableop,savev2_layer3_1_bn2_beta_read_readvariableop3savev2_layer3_1_bn2_moving_mean_read_readvariableop7savev2_layer3_1_bn2_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Z
dtypesP
N2L2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@:@:@:@@:@:@:@:@:@@:@:@:@:@:@@:@:@:@:@:@@:@:@:@:@:@?:?:?:?:?:@?:??:?:?:?:?:?:?:?:?:??:?:?:?:?:??:?:?:?:?:??:?:?:?:?:??:??:?:?:?:?:?:?:?:?:??:?:?:?:?:??:?:?:?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:-)
'
_output_shapes
:@?:. *
(
_output_shapes
:??:!!

_output_shapes	
:?:!"

_output_shapes	
:?:!#

_output_shapes	
:?:!$

_output_shapes	
:?:!%

_output_shapes	
:?:!&

_output_shapes	
:?:!'

_output_shapes	
:?:!(

_output_shapes	
:?:.)*
(
_output_shapes
:??:!*

_output_shapes	
:?:!+

_output_shapes	
:?:!,

_output_shapes	
:?:!-

_output_shapes	
:?:..*
(
_output_shapes
:??:!/

_output_shapes	
:?:!0

_output_shapes	
:?:!1

_output_shapes	
:?:!2

_output_shapes	
:?:.3*
(
_output_shapes
:??:!4

_output_shapes	
:?:!5

_output_shapes	
:?:!6

_output_shapes	
:?:!7

_output_shapes	
:?:.8*
(
_output_shapes
:??:.9*
(
_output_shapes
:??:!:

_output_shapes	
:?:!;

_output_shapes	
:?:!<

_output_shapes	
:?:!=

_output_shapes	
:?:!>

_output_shapes	
:?:!?

_output_shapes	
:?:!@

_output_shapes	
:?:!A

_output_shapes	
:?:.B*
(
_output_shapes
:??:!C

_output_shapes	
:?:!D

_output_shapes	
:?:!E

_output_shapes	
:?:!F

_output_shapes	
:?:.G*
(
_output_shapes
:??:!H

_output_shapes	
:?:!I

_output_shapes	
:?:!J

_output_shapes	
:?:!K

_output_shapes	
:?:L

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
input_1J
serving_default_input_1:0+???????????????????????????\
activation_11K
StatefulPartitionedCall:0,????????????????????????????tensorflow/serving/predict:??
??
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
layer_with_weights-9
layer-18
layer-19
layer-20
layer-21
layer_with_weights-10
layer-22
layer_with_weights-11
layer-23
layer-24
layer_with_weights-12
layer-25
layer_with_weights-13
layer-26
layer_with_weights-14
layer-27
layer_with_weights-15
layer-28
layer-29
layer-30
 layer_with_weights-16
 layer-31
!layer_with_weights-17
!layer-32
"layer-33
#layer_with_weights-18
#layer-34
$layer_with_weights-19
$layer-35
%layer-36
&layer-37
'layer-38
(layer_with_weights-20
(layer-39
)layer_with_weights-21
)layer-40
*layer-41
+layer_with_weights-22
+layer-42
,layer_with_weights-23
,layer-43
-layer_with_weights-24
-layer-44
.layer_with_weights-25
.layer-45
/layer-46
0layer-47
1layer_with_weights-26
1layer-48
2layer_with_weights-27
2layer-49
3layer-50
4layer_with_weights-28
4layer-51
5layer_with_weights-29
5layer-52
6layer-53
7layer-54
8regularization_losses
9trainable_variables
:	variables
;	keras_api
<
signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"??
_tf_keras_network??{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "ZeroPadding2D", "config": {"name": "pad", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [3, 3]}, {"class_name": "__tuple__", "items": [3, 3]}]}, "data_format": "channels_last"}, "name": "pad", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["pad", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn1", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "relu", "inbound_nodes": [[["bn1", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "pad1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "pad1", "inbound_nodes": [[["relu", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool", "inbound_nodes": [[["pad1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer1.0.conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1.0.conv1", "inbound_nodes": [[["maxpool", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer1.0.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer1.0.bn1", "inbound_nodes": [[["layer1.0.conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["layer1.0.bn1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer1.0.conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1.0.conv2", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer1.0.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer1.0.bn2", "inbound_nodes": [[["layer1.0.conv2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["maxpool", 0, 0, {}], ["layer1.0.bn2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer1.1.conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1.1.conv1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer1.1.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer1.1.bn1", "inbound_nodes": [[["layer1.1.conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["layer1.1.bn1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer1.1.conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1.1.conv2", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer1.1.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer1.1.bn2", "inbound_nodes": [[["layer1.1.conv2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["activation_1", 0, 0, {}], ["layer1.1.bn2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "layer2.0.pad", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "layer2.0.pad", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer2.0.conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2.0.conv1", "inbound_nodes": [[["layer2.0.pad", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer2.0.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer2.0.bn1", "inbound_nodes": [[["layer2.0.conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["layer2.0.bn1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer2.0.downsample.0", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2.0.downsample.0", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer2.0.conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2.0.conv2", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer2.0.downsample.1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer2.0.downsample.1", "inbound_nodes": [[["layer2.0.downsample.0", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer2.0.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer2.0.bn2", "inbound_nodes": [[["layer2.0.conv2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["layer2.0.downsample.1", 0, 0, {}], ["layer2.0.bn2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer2.1.conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2.1.conv1", "inbound_nodes": [[["activation_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer2.1.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer2.1.bn1", "inbound_nodes": [[["layer2.1.conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_6", "inbound_nodes": [[["layer2.1.bn1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer2.1.conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2.1.conv2", "inbound_nodes": [[["activation_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer2.1.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer2.1.bn2", "inbound_nodes": [[["layer2.1.conv2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["activation_5", 0, 0, {}], ["layer2.1.bn2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_7", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "layer3.0.pad", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "layer3.0.pad", "inbound_nodes": [[["activation_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer3.0.conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer3.0.conv1", "inbound_nodes": [[["layer3.0.pad", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer3.0.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer3.0.bn1", "inbound_nodes": [[["layer3.0.conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_8", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_8", "inbound_nodes": [[["layer3.0.bn1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer3.0.downsample.0", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer3.0.downsample.0", "inbound_nodes": [[["activation_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer3.0.conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer3.0.conv2", "inbound_nodes": [[["activation_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer3.0.downsample.1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer3.0.downsample.1", "inbound_nodes": [[["layer3.0.downsample.0", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer3.0.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer3.0.bn2", "inbound_nodes": [[["layer3.0.conv2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "name": "add_4", "inbound_nodes": [[["layer3.0.downsample.1", 0, 0, {}], ["layer3.0.bn2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_9", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_9", "inbound_nodes": [[["add_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer3.1.conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer3.1.conv1", "inbound_nodes": [[["activation_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer3.1.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer3.1.bn1", "inbound_nodes": [[["layer3.1.conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_10", "inbound_nodes": [[["layer3.1.bn1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer3.1.conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer3.1.conv2", "inbound_nodes": [[["activation_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer3.1.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer3.1.bn2", "inbound_nodes": [[["layer3.1.conv2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "name": "add_5", "inbound_nodes": [[["activation_9", 0, 0, {}], ["layer3.1.bn2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_11", "inbound_nodes": [[["add_5", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["activation_11", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "ZeroPadding2D", "config": {"name": "pad", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [3, 3]}, {"class_name": "__tuple__", "items": [3, 3]}]}, "data_format": "channels_last"}, "name": "pad", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["pad", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn1", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "relu", "inbound_nodes": [[["bn1", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "pad1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "pad1", "inbound_nodes": [[["relu", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool", "inbound_nodes": [[["pad1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer1.0.conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1.0.conv1", "inbound_nodes": [[["maxpool", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer1.0.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer1.0.bn1", "inbound_nodes": [[["layer1.0.conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["layer1.0.bn1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer1.0.conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1.0.conv2", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer1.0.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer1.0.bn2", "inbound_nodes": [[["layer1.0.conv2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["maxpool", 0, 0, {}], ["layer1.0.bn2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer1.1.conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1.1.conv1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer1.1.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer1.1.bn1", "inbound_nodes": [[["layer1.1.conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["layer1.1.bn1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer1.1.conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1.1.conv2", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer1.1.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer1.1.bn2", "inbound_nodes": [[["layer1.1.conv2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["activation_1", 0, 0, {}], ["layer1.1.bn2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "layer2.0.pad", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "layer2.0.pad", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer2.0.conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2.0.conv1", "inbound_nodes": [[["layer2.0.pad", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer2.0.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer2.0.bn1", "inbound_nodes": [[["layer2.0.conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["layer2.0.bn1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer2.0.downsample.0", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2.0.downsample.0", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer2.0.conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2.0.conv2", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer2.0.downsample.1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer2.0.downsample.1", "inbound_nodes": [[["layer2.0.downsample.0", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer2.0.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer2.0.bn2", "inbound_nodes": [[["layer2.0.conv2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["layer2.0.downsample.1", 0, 0, {}], ["layer2.0.bn2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer2.1.conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2.1.conv1", "inbound_nodes": [[["activation_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer2.1.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer2.1.bn1", "inbound_nodes": [[["layer2.1.conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_6", "inbound_nodes": [[["layer2.1.bn1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer2.1.conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2.1.conv2", "inbound_nodes": [[["activation_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer2.1.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer2.1.bn2", "inbound_nodes": [[["layer2.1.conv2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["activation_5", 0, 0, {}], ["layer2.1.bn2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_7", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "layer3.0.pad", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "layer3.0.pad", "inbound_nodes": [[["activation_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer3.0.conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer3.0.conv1", "inbound_nodes": [[["layer3.0.pad", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer3.0.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer3.0.bn1", "inbound_nodes": [[["layer3.0.conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_8", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_8", "inbound_nodes": [[["layer3.0.bn1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer3.0.downsample.0", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer3.0.downsample.0", "inbound_nodes": [[["activation_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer3.0.conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer3.0.conv2", "inbound_nodes": [[["activation_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer3.0.downsample.1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer3.0.downsample.1", "inbound_nodes": [[["layer3.0.downsample.0", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer3.0.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer3.0.bn2", "inbound_nodes": [[["layer3.0.conv2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "name": "add_4", "inbound_nodes": [[["layer3.0.downsample.1", 0, 0, {}], ["layer3.0.bn2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_9", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_9", "inbound_nodes": [[["add_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer3.1.conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer3.1.conv1", "inbound_nodes": [[["activation_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer3.1.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer3.1.bn1", "inbound_nodes": [[["layer3.1.conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_10", "inbound_nodes": [[["layer3.1.bn1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer3.1.conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer3.1.conv2", "inbound_nodes": [[["activation_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "layer3.1.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer3.1.bn2", "inbound_nodes": [[["layer3.1.conv2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "name": "add_5", "inbound_nodes": [[["activation_9", 0, 0, {}], ["layer3.1.bn2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_11", "inbound_nodes": [[["add_5", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["activation_11", 0, 0]]}}}
?
#=_self_saveable_object_factories"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?
#>_self_saveable_object_factories
?regularization_losses
@trainable_variables
A	variables
B	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ZeroPadding2D", "name": "pad", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pad", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [3, 3]}, {"class_name": "__tuple__", "items": [3, 3]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


Ckernel
#D_self_saveable_object_factories
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}}
?	
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
#N_self_saveable_object_factories
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "bn1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
?
#S_self_saveable_object_factories
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
#X_self_saveable_object_factories
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ZeroPadding2D", "name": "pad1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pad1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
#]_self_saveable_object_factories
^regularization_losses
_trainable_variables
`	variables
a	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "maxpool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


bkernel
#c_self_saveable_object_factories
dregularization_losses
etrainable_variables
f	variables
g	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer1.0.conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer1.0.conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
?	
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance
#m_self_saveable_object_factories
nregularization_losses
otrainable_variables
p	variables
q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "layer1.0.bn1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer1.0.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
?
#r_self_saveable_object_factories
sregularization_losses
ttrainable_variables
u	variables
v	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


wkernel
#x_self_saveable_object_factories
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer1.0.conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer1.0.conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
?	
}axis
	~gamma
beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "layer1.0.bn2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer1.0.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Add", "name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, null, 64]}, {"class_name": "TensorShape", "items": [null, null, null, 64]}]}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
?

?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer1.1.conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer1.1.conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "layer1.1.bn1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer1.1.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
?

?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer1.1.conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer1.1.conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "layer1.1.bn2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer1.1.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Add", "name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, null, 64]}, {"class_name": "TensorShape", "items": [null, null, null, 64]}]}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ZeroPadding2D", "name": "layer2.0.pad", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer2.0.pad", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer2.0.conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer2.0.conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "layer2.0.bn1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer2.0.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 128]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}
?

?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer2.0.downsample.0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer2.0.downsample.0", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 64]}}
?

?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer2.0.conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer2.0.conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 128]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "layer2.0.downsample.1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer2.0.downsample.1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 128]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "layer2.0.bn2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer2.0.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 128]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Add", "name": "add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, null, 128]}, {"class_name": "TensorShape", "items": [null, null, null, 128]}]}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}}
?

?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer2.1.conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer2.1.conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 128]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "layer2.1.bn1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer2.1.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 128]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}}
?

?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer2.1.conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer2.1.conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 128]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "layer2.1.bn2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer2.1.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 128]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Add", "name": "add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, null, 128]}, {"class_name": "TensorShape", "items": [null, null, null, 128]}]}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ZeroPadding2D", "name": "layer3.0.pad", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer3.0.pad", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer3.0.conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer3.0.conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 128]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "layer3.0.bn1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer3.0.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 256]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_8", "trainable": true, "dtype": "float32", "activation": "relu"}}
?

?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer3.0.downsample.0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer3.0.downsample.0", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 128]}}
?

?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer3.0.conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer3.0.conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 256]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "layer3.0.downsample.1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer3.0.downsample.1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 256]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "layer3.0.bn2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer3.0.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 256]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Add", "name": "add_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, null, 256]}, {"class_name": "TensorShape", "items": [null, null, null, 256]}]}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_9", "trainable": true, "dtype": "float32", "activation": "relu"}}
?

?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer3.1.conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer3.1.conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 256]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "layer3.1.bn1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer3.1.bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 256]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "relu"}}
?

?kernel
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer3.1.conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer3.1.conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 256]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "layer3.1.bn2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer3.1.bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 256]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Add", "name": "add_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, null, 256]}, {"class_name": "TensorShape", "items": [null, null, null, 256]}]}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "relu"}}
 "
trackable_list_wrapper
?
C0
J1
K2
b3
i4
j5
w6
~7
8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44"
trackable_list_wrapper
?
C0
J1
K2
L3
M4
b5
i6
j7
k8
l9
w10
~11
12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
?58
?59
?60
?61
?62
?63
?64
?65
?66
?67
?68
?69
?70
?71
?72
?73
?74"
trackable_list_wrapper
?
?layer_metrics
8regularization_losses
?non_trainable_variables
9trainable_variables
?layers
 ?layer_regularization_losses
:	variables
?metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
@trainable_variables
?layers
 ?layer_regularization_losses
A	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@2conv1/kernel
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
C0"
trackable_list_wrapper
'
C0"
trackable_list_wrapper
?
?layer_metrics
Eregularization_losses
?non_trainable_variables
Ftrainable_variables
?layers
 ?layer_regularization_losses
G	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2	bn1/gamma
:@2bn1/beta
:@ (2bn1/moving_mean
#:!@ (2bn1/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
<
J0
K1
L2
M3"
trackable_list_wrapper
?
?layer_metrics
Oregularization_losses
?non_trainable_variables
Ptrainable_variables
?layers
 ?layer_regularization_losses
Q	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
Tregularization_losses
?non_trainable_variables
Utrainable_variables
?layers
 ?layer_regularization_losses
V	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
Yregularization_losses
?non_trainable_variables
Ztrainable_variables
?layers
 ?layer_regularization_losses
[	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
^regularization_losses
?non_trainable_variables
_trainable_variables
?layers
 ?layer_regularization_losses
`	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-@@2layer1.0.conv1/kernel
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
b0"
trackable_list_wrapper
'
b0"
trackable_list_wrapper
?
?layer_metrics
dregularization_losses
?non_trainable_variables
etrainable_variables
?layers
 ?layer_regularization_losses
f	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 :@2layer1.0.bn1/gamma
:@2layer1.0.bn1/beta
(:&@ (2layer1.0.bn1/moving_mean
,:*@ (2layer1.0.bn1/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
<
i0
j1
k2
l3"
trackable_list_wrapper
?
?layer_metrics
nregularization_losses
?non_trainable_variables
otrainable_variables
?layers
 ?layer_regularization_losses
p	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
sregularization_losses
?non_trainable_variables
ttrainable_variables
?layers
 ?layer_regularization_losses
u	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-@@2layer1.0.conv2/kernel
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
w0"
trackable_list_wrapper
'
w0"
trackable_list_wrapper
?
?layer_metrics
yregularization_losses
?non_trainable_variables
ztrainable_variables
?layers
 ?layer_regularization_losses
{	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 :@2layer1.0.bn2/gamma
:@2layer1.0.bn2/beta
(:&@ (2layer1.0.bn2/moving_mean
,:*@ (2layer1.0.bn2/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
>
~0
1
?2
?3"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-@@2layer1.1.conv1/kernel
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 :@2layer1.1.bn1/gamma
:@2layer1.1.bn1/beta
(:&@ (2layer1.1.bn1/moving_mean
,:*@ (2layer1.1.bn1/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-@@2layer1.1.conv2/kernel
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 :@2layer1.1.bn2/gamma
:@2layer1.1.bn2/beta
(:&@ (2layer1.1.bn2/moving_mean
,:*@ (2layer1.1.bn2/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.@?2layer2.0.conv1/kernel
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
!:?2layer2.0.bn1/gamma
 :?2layer2.0.bn1/beta
):'? (2layer2.0.bn1/moving_mean
-:+? (2layer2.0.bn1/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
7:5@?2layer2.0.downsample.0/kernel
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/??2layer2.0.conv2/kernel
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(?2layer2.0.downsample.1/gamma
):'?2layer2.0.downsample.1/beta
2:0? (2!layer2.0.downsample.1/moving_mean
6:4? (2%layer2.0.downsample.1/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
!:?2layer2.0.bn2/gamma
 :?2layer2.0.bn2/beta
):'? (2layer2.0.bn2/moving_mean
-:+? (2layer2.0.bn2/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/??2layer2.1.conv1/kernel
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
!:?2layer2.1.bn1/gamma
 :?2layer2.1.bn1/beta
):'? (2layer2.1.bn1/moving_mean
-:+? (2layer2.1.bn1/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/??2layer2.1.conv2/kernel
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
!:?2layer2.1.bn2/gamma
 :?2layer2.1.bn2/beta
):'? (2layer2.1.bn2/moving_mean
-:+? (2layer2.1.bn2/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/??2layer3.0.conv1/kernel
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
!:?2layer3.0.bn1/gamma
 :?2layer3.0.bn1/beta
):'? (2layer3.0.bn1/moving_mean
-:+? (2layer3.0.bn1/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
8:6??2layer3.0.downsample.0/kernel
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/??2layer3.0.conv2/kernel
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(?2layer3.0.downsample.1/gamma
):'?2layer3.0.downsample.1/beta
2:0? (2!layer3.0.downsample.1/moving_mean
6:4? (2%layer3.0.downsample.1/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
!:?2layer3.0.bn2/gamma
 :?2layer3.0.bn2/beta
):'? (2layer3.0.bn2/moving_mean
-:+? (2layer3.0.bn2/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/??2layer3.1.conv1/kernel
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
!:?2layer3.1.bn1/gamma
 :?2layer3.1.bn1/beta
):'? (2layer3.1.bn1/moving_mean
-:+? (2layer3.1.bn1/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/??2layer3.1.conv2/kernel
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
!:?2layer3.1.bn2/gamma
 :?2layer3.1.bn2/beta
):'? (2layer3.1.bn2/moving_mean
-:+? (2layer3.1.bn2/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?layers
 ?layer_regularization_losses
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
?
L0
M1
k2
l3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29"
trackable_list_wrapper
?
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
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754"
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
.
L0
M1"
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
.
k0
l1"
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
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
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
?2?
@__inference_model_layer_call_and_return_conditional_losses_13113
@__inference_model_layer_call_and_return_conditional_losses_11704
@__inference_model_layer_call_and_return_conditional_losses_11912
@__inference_model_layer_call_and_return_conditional_losses_13400?
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
?2?
%__inference_model_layer_call_fn_12639
%__inference_model_layer_call_fn_13555
%__inference_model_layer_call_fn_12276
%__inference_model_layer_call_fn_13710?
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
?2?
__inference__wrapped_model_9001?
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
annotations? *@?=
;?8
input_1+???????????????????????????
?2?
=__inference_pad_layer_call_and_return_conditional_losses_9008?
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
"__inference_pad_layer_call_fn_9014?
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
@__inference_conv1_layer_call_and_return_conditional_losses_13717?
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
%__inference_conv1_layer_call_fn_13724?
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
>__inference_bn1_layer_call_and_return_conditional_losses_13744
>__inference_bn1_layer_call_and_return_conditional_losses_13762?
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
#__inference_bn1_layer_call_fn_13775
#__inference_bn1_layer_call_fn_13788?
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
?__inference_relu_layer_call_and_return_conditional_losses_13793?
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
$__inference_relu_layer_call_fn_13798?
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
>__inference_pad1_layer_call_and_return_conditional_losses_9125?
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
#__inference_pad1_layer_call_fn_9131?
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
A__inference_maxpool_layer_call_and_return_conditional_losses_9137?
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
&__inference_maxpool_layer_call_fn_9143?
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
I__inference_layer1.0.conv1_layer_call_and_return_conditional_losses_13805?
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
.__inference_layer1.0.conv1_layer_call_fn_13812?
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
G__inference_layer1.0.bn1_layer_call_and_return_conditional_losses_13832
G__inference_layer1.0.bn1_layer_call_and_return_conditional_losses_13850?
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
,__inference_layer1.0.bn1_layer_call_fn_13876
,__inference_layer1.0.bn1_layer_call_fn_13863?
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
E__inference_activation_layer_call_and_return_conditional_losses_13881?
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
*__inference_activation_layer_call_fn_13886?
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
I__inference_layer1.0.conv2_layer_call_and_return_conditional_losses_13893?
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
.__inference_layer1.0.conv2_layer_call_fn_13900?
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
G__inference_layer1.0.bn2_layer_call_and_return_conditional_losses_13920
G__inference_layer1.0.bn2_layer_call_and_return_conditional_losses_13938?
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
,__inference_layer1.0.bn2_layer_call_fn_13951
,__inference_layer1.0.bn2_layer_call_fn_13964?
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
>__inference_add_layer_call_and_return_conditional_losses_13970?
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
#__inference_add_layer_call_fn_13976?
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
G__inference_activation_1_layer_call_and_return_conditional_losses_13981?
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
,__inference_activation_1_layer_call_fn_13986?
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
I__inference_layer1.1.conv1_layer_call_and_return_conditional_losses_13993?
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
.__inference_layer1.1.conv1_layer_call_fn_14000?
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
G__inference_layer1.1.bn1_layer_call_and_return_conditional_losses_14038
G__inference_layer1.1.bn1_layer_call_and_return_conditional_losses_14020?
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
,__inference_layer1.1.bn1_layer_call_fn_14051
,__inference_layer1.1.bn1_layer_call_fn_14064?
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
G__inference_activation_2_layer_call_and_return_conditional_losses_14069?
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
,__inference_activation_2_layer_call_fn_14074?
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
I__inference_layer1.1.conv2_layer_call_and_return_conditional_losses_14081?
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
.__inference_layer1.1.conv2_layer_call_fn_14088?
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
G__inference_layer1.1.bn2_layer_call_and_return_conditional_losses_14108
G__inference_layer1.1.bn2_layer_call_and_return_conditional_losses_14126?
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
,__inference_layer1.1.bn2_layer_call_fn_14152
,__inference_layer1.1.bn2_layer_call_fn_14139?
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
@__inference_add_1_layer_call_and_return_conditional_losses_14158?
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
%__inference_add_1_layer_call_fn_14164?
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
G__inference_activation_3_layer_call_and_return_conditional_losses_14169?
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
,__inference_activation_3_layer_call_fn_14174?
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
F__inference_layer2.0.pad_layer_call_and_return_conditional_losses_9566?
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
+__inference_layer2.0.pad_layer_call_fn_9572?
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
I__inference_layer2.0.conv1_layer_call_and_return_conditional_losses_14181?
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
.__inference_layer2.0.conv1_layer_call_fn_14188?
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
G__inference_layer2.0.bn1_layer_call_and_return_conditional_losses_14208
G__inference_layer2.0.bn1_layer_call_and_return_conditional_losses_14226?
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
,__inference_layer2.0.bn1_layer_call_fn_14239
,__inference_layer2.0.bn1_layer_call_fn_14252?
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
G__inference_activation_4_layer_call_and_return_conditional_losses_14257?
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
,__inference_activation_4_layer_call_fn_14262?
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
P__inference_layer2.0.downsample.0_layer_call_and_return_conditional_losses_14269?
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
5__inference_layer2.0.downsample.0_layer_call_fn_14276?
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
I__inference_layer2.0.conv2_layer_call_and_return_conditional_losses_14283?
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
.__inference_layer2.0.conv2_layer_call_fn_14290?
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
P__inference_layer2.0.downsample.1_layer_call_and_return_conditional_losses_14328
P__inference_layer2.0.downsample.1_layer_call_and_return_conditional_losses_14310?
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
5__inference_layer2.0.downsample.1_layer_call_fn_14354
5__inference_layer2.0.downsample.1_layer_call_fn_14341?
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
G__inference_layer2.0.bn2_layer_call_and_return_conditional_losses_14374
G__inference_layer2.0.bn2_layer_call_and_return_conditional_losses_14392?
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
,__inference_layer2.0.bn2_layer_call_fn_14418
,__inference_layer2.0.bn2_layer_call_fn_14405?
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
@__inference_add_2_layer_call_and_return_conditional_losses_14424?
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
%__inference_add_2_layer_call_fn_14430?
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
G__inference_activation_5_layer_call_and_return_conditional_losses_14435?
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
,__inference_activation_5_layer_call_fn_14440?
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
I__inference_layer2.1.conv1_layer_call_and_return_conditional_losses_14447?
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
.__inference_layer2.1.conv1_layer_call_fn_14454?
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
G__inference_layer2.1.bn1_layer_call_and_return_conditional_losses_14492
G__inference_layer2.1.bn1_layer_call_and_return_conditional_losses_14474?
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
,__inference_layer2.1.bn1_layer_call_fn_14505
,__inference_layer2.1.bn1_layer_call_fn_14518?
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
G__inference_activation_6_layer_call_and_return_conditional_losses_14523?
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
,__inference_activation_6_layer_call_fn_14528?
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
I__inference_layer2.1.conv2_layer_call_and_return_conditional_losses_14535?
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
.__inference_layer2.1.conv2_layer_call_fn_14542?
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
G__inference_layer2.1.bn2_layer_call_and_return_conditional_losses_14580
G__inference_layer2.1.bn2_layer_call_and_return_conditional_losses_14562?
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
,__inference_layer2.1.bn2_layer_call_fn_14606
,__inference_layer2.1.bn2_layer_call_fn_14593?
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
@__inference_add_3_layer_call_and_return_conditional_losses_14612?
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
%__inference_add_3_layer_call_fn_14618?
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
G__inference_activation_7_layer_call_and_return_conditional_losses_14623?
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
,__inference_activation_7_layer_call_fn_14628?
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
G__inference_layer3.0.pad_layer_call_and_return_conditional_losses_10099?
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
,__inference_layer3.0.pad_layer_call_fn_10105?
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
I__inference_layer3.0.conv1_layer_call_and_return_conditional_losses_14635?
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
.__inference_layer3.0.conv1_layer_call_fn_14642?
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
G__inference_layer3.0.bn1_layer_call_and_return_conditional_losses_14662
G__inference_layer3.0.bn1_layer_call_and_return_conditional_losses_14680?
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
,__inference_layer3.0.bn1_layer_call_fn_14706
,__inference_layer3.0.bn1_layer_call_fn_14693?
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
G__inference_activation_8_layer_call_and_return_conditional_losses_14711?
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
,__inference_activation_8_layer_call_fn_14716?
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
P__inference_layer3.0.downsample.0_layer_call_and_return_conditional_losses_14723?
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
5__inference_layer3.0.downsample.0_layer_call_fn_14730?
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
I__inference_layer3.0.conv2_layer_call_and_return_conditional_losses_14737?
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
.__inference_layer3.0.conv2_layer_call_fn_14744?
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
P__inference_layer3.0.downsample.1_layer_call_and_return_conditional_losses_14782
P__inference_layer3.0.downsample.1_layer_call_and_return_conditional_losses_14764?
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
5__inference_layer3.0.downsample.1_layer_call_fn_14808
5__inference_layer3.0.downsample.1_layer_call_fn_14795?
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
G__inference_layer3.0.bn2_layer_call_and_return_conditional_losses_14846
G__inference_layer3.0.bn2_layer_call_and_return_conditional_losses_14828?
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
,__inference_layer3.0.bn2_layer_call_fn_14859
,__inference_layer3.0.bn2_layer_call_fn_14872?
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
@__inference_add_4_layer_call_and_return_conditional_losses_14878?
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
%__inference_add_4_layer_call_fn_14884?
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
G__inference_activation_9_layer_call_and_return_conditional_losses_14889?
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
,__inference_activation_9_layer_call_fn_14894?
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
I__inference_layer3.1.conv1_layer_call_and_return_conditional_losses_14901?
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
.__inference_layer3.1.conv1_layer_call_fn_14908?
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
G__inference_layer3.1.bn1_layer_call_and_return_conditional_losses_14928
G__inference_layer3.1.bn1_layer_call_and_return_conditional_losses_14946?
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
,__inference_layer3.1.bn1_layer_call_fn_14972
,__inference_layer3.1.bn1_layer_call_fn_14959?
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
H__inference_activation_10_layer_call_and_return_conditional_losses_14977?
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
-__inference_activation_10_layer_call_fn_14982?
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
I__inference_layer3.1.conv2_layer_call_and_return_conditional_losses_14989?
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
.__inference_layer3.1.conv2_layer_call_fn_14996?
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
G__inference_layer3.1.bn2_layer_call_and_return_conditional_losses_15034
G__inference_layer3.1.bn2_layer_call_and_return_conditional_losses_15016?
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
,__inference_layer3.1.bn2_layer_call_fn_15060
,__inference_layer3.1.bn2_layer_call_fn_15047?
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
@__inference_add_5_layer_call_and_return_conditional_losses_15066?
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
%__inference_add_5_layer_call_fn_15072?
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
H__inference_activation_11_layer_call_and_return_conditional_losses_15077?
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
-__inference_activation_11_layer_call_fn_15082?
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
#__inference_signature_wrapper_12796input_1"?
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
 ?
__inference__wrapped_model_9001??CJKLMbijklw~??????????????????????????????????????????????????????????????J?G
@?=
;?8
input_1+???????????????????????????
? "X?U
S
activation_11B??
activation_11,?????????????????????????????
H__inference_activation_10_layer_call_and_return_conditional_losses_14977?J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
-__inference_activation_10_layer_call_fn_14982?J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
H__inference_activation_11_layer_call_and_return_conditional_losses_15077?J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
-__inference_activation_11_layer_call_fn_15082?J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
G__inference_activation_1_layer_call_and_return_conditional_losses_13981?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
,__inference_activation_1_layer_call_fn_13986I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
G__inference_activation_2_layer_call_and_return_conditional_losses_14069?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
,__inference_activation_2_layer_call_fn_14074I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
G__inference_activation_3_layer_call_and_return_conditional_losses_14169?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
,__inference_activation_3_layer_call_fn_14174I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
G__inference_activation_4_layer_call_and_return_conditional_losses_14257?J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_activation_4_layer_call_fn_14262?J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
G__inference_activation_5_layer_call_and_return_conditional_losses_14435?J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_activation_5_layer_call_fn_14440?J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
G__inference_activation_6_layer_call_and_return_conditional_losses_14523?J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_activation_6_layer_call_fn_14528?J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
G__inference_activation_7_layer_call_and_return_conditional_losses_14623?J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_activation_7_layer_call_fn_14628?J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
G__inference_activation_8_layer_call_and_return_conditional_losses_14711?J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_activation_8_layer_call_fn_14716?J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
G__inference_activation_9_layer_call_and_return_conditional_losses_14889?J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_activation_9_layer_call_fn_14894?J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
E__inference_activation_layer_call_and_return_conditional_losses_13881?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
*__inference_activation_layer_call_fn_13886I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
@__inference_add_1_layer_call_and_return_conditional_losses_14158????
???
?|
<?9
inputs/0+???????????????????????????@
<?9
inputs/1+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
%__inference_add_1_layer_call_fn_14164????
???
?|
<?9
inputs/0+???????????????????????????@
<?9
inputs/1+???????????????????????????@
? "2?/+???????????????????????????@?
@__inference_add_2_layer_call_and_return_conditional_losses_14424????
???
??~
=?:
inputs/0,????????????????????????????
=?:
inputs/1,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
%__inference_add_2_layer_call_fn_14430????
???
??~
=?:
inputs/0,????????????????????????????
=?:
inputs/1,????????????????????????????
? "3?0,?????????????????????????????
@__inference_add_3_layer_call_and_return_conditional_losses_14612????
???
??~
=?:
inputs/0,????????????????????????????
=?:
inputs/1,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
%__inference_add_3_layer_call_fn_14618????
???
??~
=?:
inputs/0,????????????????????????????
=?:
inputs/1,????????????????????????????
? "3?0,?????????????????????????????
@__inference_add_4_layer_call_and_return_conditional_losses_14878????
???
??~
=?:
inputs/0,????????????????????????????
=?:
inputs/1,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
%__inference_add_4_layer_call_fn_14884????
???
??~
=?:
inputs/0,????????????????????????????
=?:
inputs/1,????????????????????????????
? "3?0,?????????????????????????????
@__inference_add_5_layer_call_and_return_conditional_losses_15066????
???
??~
=?:
inputs/0,????????????????????????????
=?:
inputs/1,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
%__inference_add_5_layer_call_fn_15072????
???
??~
=?:
inputs/0,????????????????????????????
=?:
inputs/1,????????????????????????????
? "3?0,?????????????????????????????
>__inference_add_layer_call_and_return_conditional_losses_13970????
???
?|
<?9
inputs/0+???????????????????????????@
<?9
inputs/1+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
#__inference_add_layer_call_fn_13976????
???
?|
<?9
inputs/0+???????????????????????????@
<?9
inputs/1+???????????????????????????@
? "2?/+???????????????????????????@?
>__inference_bn1_layer_call_and_return_conditional_losses_13744?JKLMM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
>__inference_bn1_layer_call_and_return_conditional_losses_13762?JKLMM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
#__inference_bn1_layer_call_fn_13775?JKLMM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
#__inference_bn1_layer_call_fn_13788?JKLMM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
@__inference_conv1_layer_call_and_return_conditional_losses_13717?CI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
%__inference_conv1_layer_call_fn_13724?CI?F
??<
:?7
inputs+???????????????????????????
? "2?/+???????????????????????????@?
G__inference_layer1.0.bn1_layer_call_and_return_conditional_losses_13832?ijklM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
G__inference_layer1.0.bn1_layer_call_and_return_conditional_losses_13850?ijklM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
,__inference_layer1.0.bn1_layer_call_fn_13863?ijklM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
,__inference_layer1.0.bn1_layer_call_fn_13876?ijklM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
G__inference_layer1.0.bn2_layer_call_and_return_conditional_losses_13920?~??M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
G__inference_layer1.0.bn2_layer_call_and_return_conditional_losses_13938?~??M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
,__inference_layer1.0.bn2_layer_call_fn_13951?~??M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
,__inference_layer1.0.bn2_layer_call_fn_13964?~??M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
I__inference_layer1.0.conv1_layer_call_and_return_conditional_losses_13805?bI?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
.__inference_layer1.0.conv1_layer_call_fn_13812?bI?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
I__inference_layer1.0.conv2_layer_call_and_return_conditional_losses_13893?wI?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
.__inference_layer1.0.conv2_layer_call_fn_13900?wI?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
G__inference_layer1.1.bn1_layer_call_and_return_conditional_losses_14020?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
G__inference_layer1.1.bn1_layer_call_and_return_conditional_losses_14038?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
,__inference_layer1.1.bn1_layer_call_fn_14051?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
,__inference_layer1.1.bn1_layer_call_fn_14064?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
G__inference_layer1.1.bn2_layer_call_and_return_conditional_losses_14108?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
G__inference_layer1.1.bn2_layer_call_and_return_conditional_losses_14126?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
,__inference_layer1.1.bn2_layer_call_fn_14139?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
,__inference_layer1.1.bn2_layer_call_fn_14152?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
I__inference_layer1.1.conv1_layer_call_and_return_conditional_losses_13993??I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
.__inference_layer1.1.conv1_layer_call_fn_14000??I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
I__inference_layer1.1.conv2_layer_call_and_return_conditional_losses_14081??I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
.__inference_layer1.1.conv2_layer_call_fn_14088??I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
G__inference_layer2.0.bn1_layer_call_and_return_conditional_losses_14208?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
G__inference_layer2.0.bn1_layer_call_and_return_conditional_losses_14226?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_layer2.0.bn1_layer_call_fn_14239?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
,__inference_layer2.0.bn1_layer_call_fn_14252?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
G__inference_layer2.0.bn2_layer_call_and_return_conditional_losses_14374?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
G__inference_layer2.0.bn2_layer_call_and_return_conditional_losses_14392?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_layer2.0.bn2_layer_call_fn_14405?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
,__inference_layer2.0.bn2_layer_call_fn_14418?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
I__inference_layer2.0.conv1_layer_call_and_return_conditional_losses_14181??I?F
??<
:?7
inputs+???????????????????????????@
? "@?=
6?3
0,????????????????????????????
? ?
.__inference_layer2.0.conv1_layer_call_fn_14188??I?F
??<
:?7
inputs+???????????????????????????@
? "3?0,?????????????????????????????
I__inference_layer2.0.conv2_layer_call_and_return_conditional_losses_14283??J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
.__inference_layer2.0.conv2_layer_call_fn_14290??J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
P__inference_layer2.0.downsample.0_layer_call_and_return_conditional_losses_14269??I?F
??<
:?7
inputs+???????????????????????????@
? "@?=
6?3
0,????????????????????????????
? ?
5__inference_layer2.0.downsample.0_layer_call_fn_14276??I?F
??<
:?7
inputs+???????????????????????????@
? "3?0,?????????????????????????????
P__inference_layer2.0.downsample.1_layer_call_and_return_conditional_losses_14310?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
P__inference_layer2.0.downsample.1_layer_call_and_return_conditional_losses_14328?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
5__inference_layer2.0.downsample.1_layer_call_fn_14341?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
5__inference_layer2.0.downsample.1_layer_call_fn_14354?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
F__inference_layer2.0.pad_layer_call_and_return_conditional_losses_9566?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_layer2.0.pad_layer_call_fn_9572?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_layer2.1.bn1_layer_call_and_return_conditional_losses_14474?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
G__inference_layer2.1.bn1_layer_call_and_return_conditional_losses_14492?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_layer2.1.bn1_layer_call_fn_14505?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
,__inference_layer2.1.bn1_layer_call_fn_14518?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
G__inference_layer2.1.bn2_layer_call_and_return_conditional_losses_14562?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
G__inference_layer2.1.bn2_layer_call_and_return_conditional_losses_14580?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_layer2.1.bn2_layer_call_fn_14593?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
,__inference_layer2.1.bn2_layer_call_fn_14606?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
I__inference_layer2.1.conv1_layer_call_and_return_conditional_losses_14447??J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
.__inference_layer2.1.conv1_layer_call_fn_14454??J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
I__inference_layer2.1.conv2_layer_call_and_return_conditional_losses_14535??J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
.__inference_layer2.1.conv2_layer_call_fn_14542??J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
G__inference_layer3.0.bn1_layer_call_and_return_conditional_losses_14662?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
G__inference_layer3.0.bn1_layer_call_and_return_conditional_losses_14680?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_layer3.0.bn1_layer_call_fn_14693?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
,__inference_layer3.0.bn1_layer_call_fn_14706?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
G__inference_layer3.0.bn2_layer_call_and_return_conditional_losses_14828?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
G__inference_layer3.0.bn2_layer_call_and_return_conditional_losses_14846?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_layer3.0.bn2_layer_call_fn_14859?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
,__inference_layer3.0.bn2_layer_call_fn_14872?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
I__inference_layer3.0.conv1_layer_call_and_return_conditional_losses_14635??J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
.__inference_layer3.0.conv1_layer_call_fn_14642??J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
I__inference_layer3.0.conv2_layer_call_and_return_conditional_losses_14737??J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
.__inference_layer3.0.conv2_layer_call_fn_14744??J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
P__inference_layer3.0.downsample.0_layer_call_and_return_conditional_losses_14723??J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
5__inference_layer3.0.downsample.0_layer_call_fn_14730??J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
P__inference_layer3.0.downsample.1_layer_call_and_return_conditional_losses_14764?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
P__inference_layer3.0.downsample.1_layer_call_and_return_conditional_losses_14782?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
5__inference_layer3.0.downsample.1_layer_call_fn_14795?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
5__inference_layer3.0.downsample.1_layer_call_fn_14808?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
G__inference_layer3.0.pad_layer_call_and_return_conditional_losses_10099?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_layer3.0.pad_layer_call_fn_10105?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_layer3.1.bn1_layer_call_and_return_conditional_losses_14928?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
G__inference_layer3.1.bn1_layer_call_and_return_conditional_losses_14946?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_layer3.1.bn1_layer_call_fn_14959?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
,__inference_layer3.1.bn1_layer_call_fn_14972?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
G__inference_layer3.1.bn2_layer_call_and_return_conditional_losses_15016?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
G__inference_layer3.1.bn2_layer_call_and_return_conditional_losses_15034?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_layer3.1.bn2_layer_call_fn_15047?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
,__inference_layer3.1.bn2_layer_call_fn_15060?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
I__inference_layer3.1.conv1_layer_call_and_return_conditional_losses_14901??J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
.__inference_layer3.1.conv1_layer_call_fn_14908??J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
I__inference_layer3.1.conv2_layer_call_and_return_conditional_losses_14989??J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
.__inference_layer3.1.conv2_layer_call_fn_14996??J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
A__inference_maxpool_layer_call_and_return_conditional_losses_9137?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
&__inference_maxpool_layer_call_fn_9143?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
@__inference_model_layer_call_and_return_conditional_losses_11704??CJKLMbijklw~??????????????????????????????????????????????????????????????R?O
H?E
;?8
input_1+???????????????????????????
p

 
? "@?=
6?3
0,????????????????????????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_11912??CJKLMbijklw~??????????????????????????????????????????????????????????????R?O
H?E
;?8
input_1+???????????????????????????
p 

 
? "@?=
6?3
0,????????????????????????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_13113??CJKLMbijklw~??????????????????????????????????????????????????????????????Q?N
G?D
:?7
inputs+???????????????????????????
p

 
? "@?=
6?3
0,????????????????????????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_13400??CJKLMbijklw~??????????????????????????????????????????????????????????????Q?N
G?D
:?7
inputs+???????????????????????????
p 

 
? "@?=
6?3
0,????????????????????????????
? ?
%__inference_model_layer_call_fn_12276??CJKLMbijklw~??????????????????????????????????????????????????????????????R?O
H?E
;?8
input_1+???????????????????????????
p

 
? "3?0,?????????????????????????????
%__inference_model_layer_call_fn_12639??CJKLMbijklw~??????????????????????????????????????????????????????????????R?O
H?E
;?8
input_1+???????????????????????????
p 

 
? "3?0,?????????????????????????????
%__inference_model_layer_call_fn_13555??CJKLMbijklw~??????????????????????????????????????????????????????????????Q?N
G?D
:?7
inputs+???????????????????????????
p

 
? "3?0,?????????????????????????????
%__inference_model_layer_call_fn_13710??CJKLMbijklw~??????????????????????????????????????????????????????????????Q?N
G?D
:?7
inputs+???????????????????????????
p 

 
? "3?0,?????????????????????????????
>__inference_pad1_layer_call_and_return_conditional_losses_9125?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
#__inference_pad1_layer_call_fn_9131?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
=__inference_pad_layer_call_and_return_conditional_losses_9008?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
"__inference_pad_layer_call_fn_9014?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
?__inference_relu_layer_call_and_return_conditional_losses_13793?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
$__inference_relu_layer_call_fn_13798I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
#__inference_signature_wrapper_12796??CJKLMbijklw~??????????????????????????????????????????????????????????????U?R
? 
K?H
F
input_1;?8
input_1+???????????????????????????"X?U
S
activation_11B??
activation_11,????????????????????????????