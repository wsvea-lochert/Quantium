Ø-
¾
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

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

DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ú
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
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype

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
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
0
Sigmoid
x"T
y"T"
Ttype:

2
Á
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ô'

first_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*"
shared_namefirst_conv/kernel

%first_conv/kernel/Read/ReadVariableOpReadVariableOpfirst_conv/kernel*'
_output_shapes
:à*
dtype0
w
first_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:à* 
shared_namefirst_conv/bias
p
#first_conv/bias/Read/ReadVariableOpReadVariableOpfirst_conv/bias*
_output_shapes	
:à*
dtype0

batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*-
shared_namebatch_normalization_11/gamma

0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes	
:à*
dtype0

batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*,
shared_namebatch_normalization_11/beta

/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes	
:à*
dtype0

"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*3
shared_name$"batch_normalization_11/moving_mean

6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes	
:à*
dtype0
¥
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*7
shared_name(&batch_normalization_11/moving_variance

:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes	
:à*
dtype0

conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:à* 
shared_nameconv2d_8/kernel
}
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*(
_output_shapes
:à*
dtype0
s
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_8/bias
l
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes	
:*
dtype0

batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_12/gamma

0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes	
:*
dtype0

batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_12/beta

/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes	
:*
dtype0

"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_12/moving_mean

6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes	
:*
dtype0
¥
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_12/moving_variance

:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes	
:*
dtype0

conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:à* 
shared_nameconv2d_9/kernel
}
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*(
_output_shapes
:à*
dtype0
s
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*
shared_nameconv2d_9/bias
l
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes	
:à*
dtype0

conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:à *!
shared_nameconv2d_10/kernel

$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*(
_output_shapes
:à *
dtype0
u
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_10/bias
n
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes	
: *
dtype0

batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_13/gamma

0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes	
: *
dtype0

batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_13/beta

/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes	
: *
dtype0

"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_13/moving_mean

6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes	
: *
dtype0
¥
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_13/moving_variance

:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes	
: *
dtype0

conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: `*!
shared_nameconv2d_11/kernel
~
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*'
_output_shapes
: `*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:`*
dtype0

batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*-
shared_namebatch_normalization_14/gamma

0batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_14/gamma*
_output_shapes
:`*
dtype0

batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*,
shared_namebatch_normalization_14/beta

/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_14/beta*
_output_shapes
:`*
dtype0

"batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"batch_normalization_14/moving_mean

6batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_14/moving_mean*
_output_shapes
:`*
dtype0
¤
&batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*7
shared_name(&batch_normalization_14/moving_variance

:batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_14/moving_variance*
_output_shapes
:`*
dtype0

conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*!
shared_nameconv2d_12/kernel
~
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*'
_output_shapes
:`*
dtype0
u
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_12/bias
n
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes	
:*
dtype0

batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_15/gamma

0batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_15/gamma*
_output_shapes	
:*
dtype0

batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_15/beta

/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_15/beta*
_output_shapes	
:*
dtype0

"batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_15/moving_mean

6batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
_output_shapes	
:*
dtype0
¥
&batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_15/moving_variance

:batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
_output_shapes	
:*
dtype0
­
$separable_conv2d_18/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$separable_conv2d_18/depthwise_kernel
¦
8separable_conv2d_18/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_18/depthwise_kernel*'
_output_shapes
:*
dtype0
®
$separable_conv2d_18/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*5
shared_name&$separable_conv2d_18/pointwise_kernel
§
8separable_conv2d_18/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_18/pointwise_kernel*(
_output_shapes
:À*
dtype0

separable_conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*)
shared_nameseparable_conv2d_18/bias

,separable_conv2d_18/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_18/bias*
_output_shapes	
:À*
dtype0

batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*-
shared_namebatch_normalization_16/gamma

0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes	
:À*
dtype0

batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*,
shared_namebatch_normalization_16/beta

/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes	
:À*
dtype0

"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*3
shared_name$"batch_normalization_16/moving_mean

6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes	
:À*
dtype0
¥
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*7
shared_name(&batch_normalization_16/moving_variance

:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes	
:À*
dtype0
­
$separable_conv2d_19/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*5
shared_name&$separable_conv2d_19/depthwise_kernel
¦
8separable_conv2d_19/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_19/depthwise_kernel*'
_output_shapes
:À*
dtype0
®
$separable_conv2d_19/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Àà*5
shared_name&$separable_conv2d_19/pointwise_kernel
§
8separable_conv2d_19/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_19/pointwise_kernel*(
_output_shapes
:Àà*
dtype0

separable_conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*)
shared_nameseparable_conv2d_19/bias

,separable_conv2d_19/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_19/bias*
_output_shapes	
:à*
dtype0
­
$separable_conv2d_20/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*5
shared_name&$separable_conv2d_20/depthwise_kernel
¦
8separable_conv2d_20/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_20/depthwise_kernel*'
_output_shapes
:à*
dtype0
®
$separable_conv2d_20/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*5
shared_name&$separable_conv2d_20/pointwise_kernel
§
8separable_conv2d_20/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_20/pointwise_kernel*(
_output_shapes
:à*
dtype0

separable_conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameseparable_conv2d_20/bias

,separable_conv2d_20/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_20/bias*
_output_shapes	
:*
dtype0
­
$separable_conv2d_21/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$separable_conv2d_21/depthwise_kernel
¦
8separable_conv2d_21/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_21/depthwise_kernel*'
_output_shapes
:*
dtype0
®
$separable_conv2d_21/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$separable_conv2d_21/pointwise_kernel
§
8separable_conv2d_21/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_21/pointwise_kernel*(
_output_shapes
:*
dtype0

separable_conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameseparable_conv2d_21/bias

,separable_conv2d_21/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_21/bias*
_output_shapes	
:*
dtype0
­
$separable_conv2d_22/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$separable_conv2d_22/depthwise_kernel
¦
8separable_conv2d_22/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_22/depthwise_kernel*'
_output_shapes
:*
dtype0
®
$separable_conv2d_22/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*5
shared_name&$separable_conv2d_22/pointwise_kernel
§
8separable_conv2d_22/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_22/pointwise_kernel*(
_output_shapes
:à*
dtype0

separable_conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*)
shared_nameseparable_conv2d_22/bias

,separable_conv2d_22/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_22/bias*
_output_shapes	
:à*
dtype0

batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*-
shared_namebatch_normalization_17/gamma

0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes	
:à*
dtype0

batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*,
shared_namebatch_normalization_17/beta

/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes	
:à*
dtype0

"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*3
shared_name$"batch_normalization_17/moving_mean

6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes	
:à*
dtype0
¥
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*7
shared_name(&batch_normalization_17/moving_variance

:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes	
:à*
dtype0
­
$separable_conv2d_23/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*5
shared_name&$separable_conv2d_23/depthwise_kernel
¦
8separable_conv2d_23/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_23/depthwise_kernel*'
_output_shapes
:à*
dtype0
®
$separable_conv2d_23/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:à *5
shared_name&$separable_conv2d_23/pointwise_kernel
§
8separable_conv2d_23/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_23/pointwise_kernel*(
_output_shapes
:à *
dtype0

separable_conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameseparable_conv2d_23/bias

,separable_conv2d_23/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_23/bias*
_output_shapes	
: *
dtype0

sep_out_0/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namesep_out_0/depthwise_kernel

.sep_out_0/depthwise_kernel/Read/ReadVariableOpReadVariableOpsep_out_0/depthwise_kernel*'
_output_shapes
: *
dtype0

sep_out_0/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: À*+
shared_namesep_out_0/pointwise_kernel

.sep_out_0/pointwise_kernel/Read/ReadVariableOpReadVariableOpsep_out_0/pointwise_kernel*(
_output_shapes
: À*
dtype0
u
sep_out_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*
shared_namesep_out_0/bias
n
"sep_out_0/bias/Read/ReadVariableOpReadVariableOpsep_out_0/bias*
_output_shapes	
:À*
dtype0

sep_out_1/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*+
shared_namesep_out_1/depthwise_kernel

.sep_out_1/depthwise_kernel/Read/ReadVariableOpReadVariableOpsep_out_1/depthwise_kernel*'
_output_shapes
:À*
dtype0

sep_out_1/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Àà*+
shared_namesep_out_1/pointwise_kernel

.sep_out_1/pointwise_kernel/Read/ReadVariableOpReadVariableOpsep_out_1/pointwise_kernel*(
_output_shapes
:Àà*
dtype0
u
sep_out_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*
shared_namesep_out_1/bias
n
"sep_out_1/bias/Read/ReadVariableOpReadVariableOpsep_out_1/bias*
_output_shapes	
:à*
dtype0

sep_out_2/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*+
shared_namesep_out_2/depthwise_kernel

.sep_out_2/depthwise_kernel/Read/ReadVariableOpReadVariableOpsep_out_2/depthwise_kernel*'
_output_shapes
:à*
dtype0

sep_out_2/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:àà*+
shared_namesep_out_2/pointwise_kernel

.sep_out_2/pointwise_kernel/Read/ReadVariableOpReadVariableOpsep_out_2/pointwise_kernel*(
_output_shapes
:àà*
dtype0
u
sep_out_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*
shared_namesep_out_2/bias
n
"sep_out_2/bias/Read/ReadVariableOpReadVariableOpsep_out_2/bias*
_output_shapes	
:à*
dtype0

output/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*(
shared_nameoutput/depthwise_kernel

+output/depthwise_kernel/Read/ReadVariableOpReadVariableOpoutput/depthwise_kernel*'
_output_shapes
:à*
dtype0

output/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*(
shared_nameoutput/pointwise_kernel

+output/pointwise_kernel/Read/ReadVariableOpReadVariableOpoutput/pointwise_kernel*'
_output_shapes
:à*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
î
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¨
valueB B
©
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
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
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer-17
layer_with_weights-9
layer-18
layer_with_weights-10
layer-19
layer-20
layer-21
layer-22
layer_with_weights-11
layer-23
layer_with_weights-12
layer-24
layer-25
layer_with_weights-13
layer-26
layer-27
layer_with_weights-14
layer-28
layer-29
layer-30
 layer_with_weights-15
 layer-31
!layer-32
"layer_with_weights-16
"layer-33
#layer_with_weights-17
#layer-34
$layer-35
%layer_with_weights-18
%layer-36
&layer-37
'layer-38
(layer_with_weights-19
(layer-39
)layer-40
*layer_with_weights-20
*layer-41
+layer-42
,layer_with_weights-21
,layer-43
-layer-44
.layer_with_weights-22
.layer-45
/	optimizer
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_default_save_signature
7
signatures*
* 
¦

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses*
Õ
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses*

K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses* 

Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses* 
¦

Wkernel
Xbias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses*
Õ
_axis
	`gamma
abeta
bmoving_mean
cmoving_variance
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses*

j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses* 
¥
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t_random_generator
u__call__
*v&call_and_return_all_conditional_losses* 
¦

wkernel
xbias
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬
	variables
trainable_variables
 regularization_losses
¡	keras_api
¢_random_generator
£__call__
+¤&call_and_return_all_conditional_losses* 
®
¥kernel
	¦bias
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses*
à
	­axis

®gamma
	¯beta
°moving_mean
±moving_variance
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses*

¸	variables
¹trainable_variables
ºregularization_losses
»	keras_api
¼__call__
+½&call_and_return_all_conditional_losses* 
®
¾kernel
	¿bias
À	variables
Átrainable_variables
Âregularization_losses
Ã	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses*
à
	Æaxis

Çgamma
	Èbeta
Émoving_mean
Êmoving_variance
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses*

Ñ	variables
Òtrainable_variables
Óregularization_losses
Ô	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses* 
¬
×	variables
Øtrainable_variables
Ùregularization_losses
Ú	keras_api
Û_random_generator
Ü__call__
+Ý&call_and_return_all_conditional_losses* 

Þ	variables
ßtrainable_variables
àregularization_losses
á	keras_api
â__call__
+ã&call_and_return_all_conditional_losses* 
Ï
ädepthwise_kernel
åpointwise_kernel
	æbias
ç	variables
ètrainable_variables
éregularization_losses
ê	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses*
à
	íaxis

îgamma
	ïbeta
ðmoving_mean
ñmoving_variance
ò	variables
ótrainable_variables
ôregularization_losses
õ	keras_api
ö__call__
+÷&call_and_return_all_conditional_losses*

ø	variables
ùtrainable_variables
úregularization_losses
û	keras_api
ü__call__
+ý&call_and_return_all_conditional_losses* 
Ï
þdepthwise_kernel
ÿpointwise_kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
Ï
depthwise_kernel
pointwise_kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
 _random_generator
¡__call__
+¢&call_and_return_all_conditional_losses* 
Ï
£depthwise_kernel
¤pointwise_kernel
	¥bias
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ª__call__
+«&call_and_return_all_conditional_losses*

¬	variables
­trainable_variables
®regularization_losses
¯	keras_api
°__call__
+±&call_and_return_all_conditional_losses* 
Ï
²depthwise_kernel
³pointwise_kernel
	´bias
µ	variables
¶trainable_variables
·regularization_losses
¸	keras_api
¹__call__
+º&call_and_return_all_conditional_losses*
à
	»axis

¼gamma
	½beta
¾moving_mean
¿moving_variance
À	variables
Átrainable_variables
Âregularization_losses
Ã	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses*

Æ	variables
Çtrainable_variables
Èregularization_losses
É	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses* 
Ï
Ìdepthwise_kernel
Ípointwise_kernel
	Îbias
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ò	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses*

Õ	variables
Ötrainable_variables
×regularization_losses
Ø	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses* 

Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses* 
Ï
ádepthwise_kernel
âpointwise_kernel
	ãbias
ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
è__call__
+é&call_and_return_all_conditional_losses*

ê	variables
ëtrainable_variables
ìregularization_losses
í	keras_api
î__call__
+ï&call_and_return_all_conditional_losses* 
Ï
ðdepthwise_kernel
ñpointwise_kernel
	òbias
ó	variables
ôtrainable_variables
õregularization_losses
ö	keras_api
÷__call__
+ø&call_and_return_all_conditional_losses*

ù	variables
útrainable_variables
ûregularization_losses
ü	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses* 
Ï
ÿdepthwise_kernel
pointwise_kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
Ï
depthwise_kernel
pointwise_kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
* 
â
80
91
A2
B3
C4
D5
W6
X7
`8
a9
b10
c11
w12
x13
14
15
16
17
18
19
¥20
¦21
®22
¯23
°24
±25
¾26
¿27
Ç28
È29
É30
Ê31
ä32
å33
æ34
î35
ï36
ð37
ñ38
þ39
ÿ40
41
42
43
44
£45
¤46
¥47
²48
³49
´50
¼51
½52
¾53
¿54
Ì55
Í56
Î57
á58
â59
ã60
ð61
ñ62
ò63
ÿ64
65
66
67
68
69*
è
80
91
A2
B3
W4
X5
`6
a7
w8
x9
10
11
12
13
¥14
¦15
®16
¯17
¾18
¿19
Ç20
È21
ä22
å23
æ24
î25
ï26
þ27
ÿ28
29
30
31
32
£33
¤34
¥35
²36
³37
´38
¼39
½40
Ì41
Í42
Î43
á44
â45
ã46
ð47
ñ48
ò49
ÿ50
51
52
53
54
55*
* 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
6_default_save_signature
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*
* 
* 
* 

serving_default* 
a[
VARIABLE_VALUEfirst_conv/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEfirst_conv/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

80
91*

80
91*
* 

non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_11/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_11/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_11/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_11/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
A0
B1
C2
D3*

A0
B1*
* 

¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

W0
X1*

W0
X1*
* 

±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_12/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_12/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_12/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_12/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
`0
a1
b2
c3*

`0
a1*
* 

¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
p	variables
qtrainable_variables
rregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses* 
* 
* 
* 
_Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

w0
x1*

w0
x1*
* 

Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ênon_trainable_variables
Ëlayers
Ìmetrics
 Ílayer_regularization_losses
Îlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_13/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_13/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_13/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_13/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
	variables
trainable_variables
 regularization_losses
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses* 
* 
* 
* 
`Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

¥0
¦1*

¥0
¦1*
* 

ãnon_trainable_variables
älayers
åmetrics
 ælayer_regularization_losses
çlayer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_14/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_14/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_14/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_14/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
®0
¯1
°2
±3*

®0
¯1*
* 

ènon_trainable_variables
élayers
êmetrics
 ëlayer_regularization_losses
ìlayer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ínon_trainable_variables
îlayers
ïmetrics
 ðlayer_regularization_losses
ñlayer_metrics
¸	variables
¹trainable_variables
ºregularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

¾0
¿1*

¾0
¿1*
* 

ònon_trainable_variables
ólayers
ômetrics
 õlayer_regularization_losses
ölayer_metrics
À	variables
Átrainable_variables
Âregularization_losses
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_15/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_15/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_15/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_15/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
Ç0
È1
É2
Ê3*

Ç0
È1*
* 

÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
Ë	variables
Ìtrainable_variables
Íregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
Ñ	variables
Òtrainable_variables
Óregularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
×	variables
Øtrainable_variables
Ùregularization_losses
Ü__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Þ	variables
ßtrainable_variables
àregularization_losses
â__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses* 
* 
* 
y
VARIABLE_VALUE$separable_conv2d_18/depthwise_kernelAlayer_with_weights-11/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE$separable_conv2d_18/pointwise_kernelAlayer_with_weights-11/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEseparable_conv2d_18/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

ä0
å1
æ2*

ä0
å1
æ2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ç	variables
ètrainable_variables
éregularization_losses
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_16/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_16/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_16/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_16/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
î0
ï1
ð2
ñ3*

î0
ï1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ò	variables
ótrainable_variables
ôregularization_losses
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ø	variables
ùtrainable_variables
úregularization_losses
ü__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses* 
* 
* 
y
VARIABLE_VALUE$separable_conv2d_19/depthwise_kernelAlayer_with_weights-13/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE$separable_conv2d_19/pointwise_kernelAlayer_with_weights-13/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEseparable_conv2d_19/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

þ0
ÿ1
2*

þ0
ÿ1
2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
y
VARIABLE_VALUE$separable_conv2d_20/depthwise_kernelAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE$separable_conv2d_20/pointwise_kernelAlayer_with_weights-14/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEseparable_conv2d_20/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1
2*

0
1
2*
* 

¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
	variables
trainable_variables
regularization_losses
¡__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses* 
* 
* 
* 
y
VARIABLE_VALUE$separable_conv2d_21/depthwise_kernelAlayer_with_weights-15/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE$separable_conv2d_21/pointwise_kernelAlayer_with_weights-15/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEseparable_conv2d_21/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

£0
¤1
¥2*

£0
¤1
¥2*
* 

³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
¦	variables
§trainable_variables
¨regularization_losses
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
¬	variables
­trainable_variables
®regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses* 
* 
* 
y
VARIABLE_VALUE$separable_conv2d_22/depthwise_kernelAlayer_with_weights-16/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE$separable_conv2d_22/pointwise_kernelAlayer_with_weights-16/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEseparable_conv2d_22/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*

²0
³1
´2*

²0
³1
´2*
* 

½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
µ	variables
¶trainable_variables
·regularization_losses
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_17/gamma6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_17/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_17/moving_mean<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_17/moving_variance@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
¼0
½1
¾2
¿3*

¼0
½1*
* 

Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
À	variables
Átrainable_variables
Âregularization_losses
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
Æ	variables
Çtrainable_variables
Èregularization_losses
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses* 
* 
* 
y
VARIABLE_VALUE$separable_conv2d_23/depthwise_kernelAlayer_with_weights-18/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE$separable_conv2d_23/pointwise_kernelAlayer_with_weights-18/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEseparable_conv2d_23/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ì0
Í1
Î2*

Ì0
Í1
Î2*
* 

Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
Õ	variables
Ötrainable_variables
×regularization_losses
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
Û	variables
Ütrainable_variables
Ýregularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses* 
* 
* 
uo
VARIABLE_VALUEsep_out_0/depthwise_kernelAlayer_with_weights-19/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEsep_out_0/pointwise_kernelAlayer_with_weights-19/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEsep_out_0/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*

á0
â1
ã2*

á0
â1
ã2*
* 

Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
ä	variables
åtrainable_variables
æregularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ànon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
ê	variables
ëtrainable_variables
ìregularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses* 
* 
* 
uo
VARIABLE_VALUEsep_out_1/depthwise_kernelAlayer_with_weights-20/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEsep_out_1/pointwise_kernelAlayer_with_weights-20/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEsep_out_1/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE*

ð0
ñ1
ò2*

ð0
ñ1
ò2*
* 

ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
ó	variables
ôtrainable_variables
õregularization_losses
÷__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
ù	variables
útrainable_variables
ûregularization_losses
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses* 
* 
* 
uo
VARIABLE_VALUEsep_out_2/depthwise_kernelAlayer_with_weights-21/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEsep_out_2/pointwise_kernelAlayer_with_weights-21/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEsep_out_2/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE*

ÿ0
1
2*

ÿ0
1
2*
* 

ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
rl
VARIABLE_VALUEoutput/depthwise_kernelAlayer_with_weights-22/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEoutput/pointwise_kernelAlayer_with_weights-22/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEoutput/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1
2*

0
1
2*
* 

ùnon_trainable_variables
úlayers
ûmetrics
 ülayer_regularization_losses
ýlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
t
C0
D1
b2
c3
4
5
°6
±7
É8
Ê9
ð10
ñ11
¾12
¿13*
ê
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
.45*

þ0
ÿ1*
* 
* 
* 
* 
* 
* 
* 
* 

C0
D1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

b0
c1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

°0
±1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

É0
Ê1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

ð0
ñ1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

¾0
¿1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

total

count
	variables
	keras_api*
M

total

count

_fn_kwargs
	variables
	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

	variables*

serving_default_input_6Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿàà
¾
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_6first_conv/kernelfirst_conv/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_varianceconv2d_12/kernelconv2d_12/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_variance$separable_conv2d_18/depthwise_kernel$separable_conv2d_18/pointwise_kernelseparable_conv2d_18/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_variance$separable_conv2d_19/depthwise_kernel$separable_conv2d_19/pointwise_kernelseparable_conv2d_19/bias$separable_conv2d_20/depthwise_kernel$separable_conv2d_20/pointwise_kernelseparable_conv2d_20/bias$separable_conv2d_21/depthwise_kernel$separable_conv2d_21/pointwise_kernelseparable_conv2d_21/bias$separable_conv2d_22/depthwise_kernel$separable_conv2d_22/pointwise_kernelseparable_conv2d_22/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_variance$separable_conv2d_23/depthwise_kernel$separable_conv2d_23/pointwise_kernelseparable_conv2d_23/biassep_out_0/depthwise_kernelsep_out_0/pointwise_kernelsep_out_0/biassep_out_1/depthwise_kernelsep_out_1/pointwise_kernelsep_out_1/biassep_out_2/depthwise_kernelsep_out_2/pointwise_kernelsep_out_2/biasoutput/depthwise_kerneloutput/pointwise_kerneloutput/bias*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*h
_read_only_resource_inputsJ
HF	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEF*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_79974
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Å
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%first_conv/kernel/Read/ReadVariableOp#first_conv/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp0batch_normalization_14/gamma/Read/ReadVariableOp/batch_normalization_14/beta/Read/ReadVariableOp6batch_normalization_14/moving_mean/Read/ReadVariableOp:batch_normalization_14/moving_variance/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp0batch_normalization_15/gamma/Read/ReadVariableOp/batch_normalization_15/beta/Read/ReadVariableOp6batch_normalization_15/moving_mean/Read/ReadVariableOp:batch_normalization_15/moving_variance/Read/ReadVariableOp8separable_conv2d_18/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_18/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_18/bias/Read/ReadVariableOp0batch_normalization_16/gamma/Read/ReadVariableOp/batch_normalization_16/beta/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOp8separable_conv2d_19/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_19/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_19/bias/Read/ReadVariableOp8separable_conv2d_20/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_20/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_20/bias/Read/ReadVariableOp8separable_conv2d_21/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_21/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_21/bias/Read/ReadVariableOp8separable_conv2d_22/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_22/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_22/bias/Read/ReadVariableOp0batch_normalization_17/gamma/Read/ReadVariableOp/batch_normalization_17/beta/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOp8separable_conv2d_23/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_23/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_23/bias/Read/ReadVariableOp.sep_out_0/depthwise_kernel/Read/ReadVariableOp.sep_out_0/pointwise_kernel/Read/ReadVariableOp"sep_out_0/bias/Read/ReadVariableOp.sep_out_1/depthwise_kernel/Read/ReadVariableOp.sep_out_1/pointwise_kernel/Read/ReadVariableOp"sep_out_1/bias/Read/ReadVariableOp.sep_out_2/depthwise_kernel/Read/ReadVariableOp.sep_out_2/pointwise_kernel/Read/ReadVariableOp"sep_out_2/bias/Read/ReadVariableOp+output/depthwise_kernel/Read/ReadVariableOp+output/pointwise_kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*W
TinP
N2L*
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
GPU2*0J 8 *'
f"R 
__inference__traced_save_81319
ø
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamefirst_conv/kernelfirst_conv/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_varianceconv2d_12/kernelconv2d_12/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_variance$separable_conv2d_18/depthwise_kernel$separable_conv2d_18/pointwise_kernelseparable_conv2d_18/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_variance$separable_conv2d_19/depthwise_kernel$separable_conv2d_19/pointwise_kernelseparable_conv2d_19/bias$separable_conv2d_20/depthwise_kernel$separable_conv2d_20/pointwise_kernelseparable_conv2d_20/bias$separable_conv2d_21/depthwise_kernel$separable_conv2d_21/pointwise_kernelseparable_conv2d_21/bias$separable_conv2d_22/depthwise_kernel$separable_conv2d_22/pointwise_kernelseparable_conv2d_22/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_variance$separable_conv2d_23/depthwise_kernel$separable_conv2d_23/pointwise_kernelseparable_conv2d_23/biassep_out_0/depthwise_kernelsep_out_0/pointwise_kernelsep_out_0/biassep_out_1/depthwise_kernelsep_out_1/pointwise_kernelsep_out_1/biassep_out_2/depthwise_kernelsep_out_2/pointwise_kernelsep_out_2/biasoutput/depthwise_kerneloutput/pointwise_kerneloutput/biastotalcounttotal_1count_1*V
TinO
M2K*
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
GPU2*0J 8 **
f%R#
!__inference__traced_restore_81551#
Ú
À
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_80421

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
û
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_80328

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
 
_user_specified_nameinputs
Ì

N__inference_separable_conv2d_19_layer_call_and_return_conditional_losses_80683

inputsC
(separable_conv2d_readvariableop_resource:ÀF
*separable_conv2d_readvariableop_1_resource:Àà.
biasadd_readvariableop_resource:	à
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:À*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:Àà*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @     o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ù
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ*
paddingSAME*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
	
Ñ
3__inference_separable_conv2d_23_layer_call_fn_80901

inputs"
unknown:à%
	unknown_0:à 
	unknown_1:	 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_23_layer_call_and_return_conditional_losses_77064
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
ð
d
H__inference_activation_29_layer_call_and_return_conditional_losses_77499

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
ø
d
H__inference_activation_21_layer_call_and_return_conditional_losses_80065

inputs
identityQ
ReluReluinputs*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿàààe
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿààà:Z V
2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà
 
_user_specified_nameinputs
ð
d
H__inference_activation_30_layer_call_and_return_conditional_losses_80792

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
ð
d
H__inference_activation_30_layer_call_and_return_conditional_losses_77520

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
±

ÿ
C__inference_conv2d_8_layer_call_and_return_conditional_losses_80094

inputs:
conv2d_readvariableop_resource:à.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:à*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpph
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿppà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà
 
_user_specified_nameinputs
ð
d
H__inference_activation_28_layer_call_and_return_conditional_losses_77485

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88àc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88à:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à
 
_user_specified_nameinputs
Ì

Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_80403

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`:`:`:`:`:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
ø
d
H__inference_activation_21_layer_call_and_return_conditional_losses_77274

inputs
identityQ
ReluReluinputs*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿàààe
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿààà:Z V
2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà
 
_user_specified_nameinputs
û
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_80537

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
º

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_80193

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¯
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppx
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppb
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
Ì

N__inference_separable_conv2d_18_layer_call_and_return_conditional_losses_80585

inputsC
(separable_conv2d_readvariableop_resource:F
*separable_conv2d_readvariableop_1_resource:À.
biasadd_readvariableop_resource:	À
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:À*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ù
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:À*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
I
-__inference_activation_28_layer_call_fn_80688

inputs
identity¿
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_28_layer_call_and_return_conditional_losses_77485i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88à:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_17_layer_call_fn_80844

inputs
unknown:	à
	unknown_0:	à
	unknown_1:	à
	unknown_2:	à
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_77034
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
º


E__inference_first_conv_layer_call_and_return_conditional_losses_77254

inputs9
conv2d_readvariableop_resource:à.
biasadd_readvariableop_resource:	à
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿàààj
IdentityIdentityBiasAdd:output:0^NoOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿàààw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_80512

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
d
H__inference_activation_25_layer_call_and_return_conditional_losses_77408

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp`:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
 
_user_specified_nameinputs
Ú
À
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_76690

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
û
¡
*__inference_first_conv_layer_call_fn_79983

inputs"
unknown:à
	unknown_0:	à
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_first_conv_layer_call_and_return_conditional_losses_77254z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
Ì

N__inference_separable_conv2d_23_layer_call_and_return_conditional_losses_80916

inputsC
(separable_conv2d_readvariableop_resource:àF
*separable_conv2d_readvariableop_1_resource:à .
biasadd_readvariableop_resource:	 
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:à *
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      `     o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ù
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
paddingSAME*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
: *
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
²
ÿ
D__inference_sep_out_1_layer_call_and_return_conditional_losses_81000

inputsC
(separable_conv2d_readvariableop_resource:ÀF
*separable_conv2d_readvariableop_1_resource:Àà.
biasadd_readvariableop_resource:	à
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:À*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:Àà*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      À     o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ú
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ*
paddingVALID*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
²
ÿ
D__inference_sep_out_0_layer_call_and_return_conditional_losses_80963

inputsC
(separable_conv2d_readvariableop_resource: F
*separable_conv2d_readvariableop_1_resource: À.
biasadd_readvariableop_resource:	À
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
: À*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ú
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:À*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_12_layer_call_fn_80120

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_76562
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±

ÿ
C__inference_conv2d_9_layer_call_and_return_conditional_losses_80212

inputs:
conv2d_readvariableop_resource:à.
biasadd_readvariableop_resource:	à
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:à*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppàh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppàw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿpp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
ð
d
H__inference_activation_22_layer_call_and_return_conditional_losses_77307

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
Ä
E
)__inference_dropout_3_layer_call_fn_80318

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_77376i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
 
_user_specified_nameinputs
Õè
ÒV
 __inference__wrapped_model_76433
input_6W
<mcfly_cnn_50epochs_first_conv_conv2d_readvariableop_resource:àL
=mcfly_cnn_50epochs_first_conv_biasadd_readvariableop_resource:	àP
Amcfly_cnn_50epochs_batch_normalization_11_readvariableop_resource:	àR
Cmcfly_cnn_50epochs_batch_normalization_11_readvariableop_1_resource:	àa
Rmcfly_cnn_50epochs_batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	àc
Tmcfly_cnn_50epochs_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	àV
:mcfly_cnn_50epochs_conv2d_8_conv2d_readvariableop_resource:àJ
;mcfly_cnn_50epochs_conv2d_8_biasadd_readvariableop_resource:	P
Amcfly_cnn_50epochs_batch_normalization_12_readvariableop_resource:	R
Cmcfly_cnn_50epochs_batch_normalization_12_readvariableop_1_resource:	a
Rmcfly_cnn_50epochs_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	c
Tmcfly_cnn_50epochs_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	V
:mcfly_cnn_50epochs_conv2d_9_conv2d_readvariableop_resource:àJ
;mcfly_cnn_50epochs_conv2d_9_biasadd_readvariableop_resource:	àW
;mcfly_cnn_50epochs_conv2d_10_conv2d_readvariableop_resource:à K
<mcfly_cnn_50epochs_conv2d_10_biasadd_readvariableop_resource:	 P
Amcfly_cnn_50epochs_batch_normalization_13_readvariableop_resource:	 R
Cmcfly_cnn_50epochs_batch_normalization_13_readvariableop_1_resource:	 a
Rmcfly_cnn_50epochs_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	 c
Tmcfly_cnn_50epochs_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	 V
;mcfly_cnn_50epochs_conv2d_11_conv2d_readvariableop_resource: `J
<mcfly_cnn_50epochs_conv2d_11_biasadd_readvariableop_resource:`O
Amcfly_cnn_50epochs_batch_normalization_14_readvariableop_resource:`Q
Cmcfly_cnn_50epochs_batch_normalization_14_readvariableop_1_resource:``
Rmcfly_cnn_50epochs_batch_normalization_14_fusedbatchnormv3_readvariableop_resource:`b
Tmcfly_cnn_50epochs_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:`V
;mcfly_cnn_50epochs_conv2d_12_conv2d_readvariableop_resource:`K
<mcfly_cnn_50epochs_conv2d_12_biasadd_readvariableop_resource:	P
Amcfly_cnn_50epochs_batch_normalization_15_readvariableop_resource:	R
Cmcfly_cnn_50epochs_batch_normalization_15_readvariableop_1_resource:	a
Rmcfly_cnn_50epochs_batch_normalization_15_fusedbatchnormv3_readvariableop_resource:	c
Tmcfly_cnn_50epochs_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource:	j
Omcfly_cnn_50epochs_separable_conv2d_18_separable_conv2d_readvariableop_resource:m
Qmcfly_cnn_50epochs_separable_conv2d_18_separable_conv2d_readvariableop_1_resource:ÀU
Fmcfly_cnn_50epochs_separable_conv2d_18_biasadd_readvariableop_resource:	ÀP
Amcfly_cnn_50epochs_batch_normalization_16_readvariableop_resource:	ÀR
Cmcfly_cnn_50epochs_batch_normalization_16_readvariableop_1_resource:	Àa
Rmcfly_cnn_50epochs_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:	Àc
Tmcfly_cnn_50epochs_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:	Àj
Omcfly_cnn_50epochs_separable_conv2d_19_separable_conv2d_readvariableop_resource:Àm
Qmcfly_cnn_50epochs_separable_conv2d_19_separable_conv2d_readvariableop_1_resource:ÀàU
Fmcfly_cnn_50epochs_separable_conv2d_19_biasadd_readvariableop_resource:	àj
Omcfly_cnn_50epochs_separable_conv2d_20_separable_conv2d_readvariableop_resource:àm
Qmcfly_cnn_50epochs_separable_conv2d_20_separable_conv2d_readvariableop_1_resource:àU
Fmcfly_cnn_50epochs_separable_conv2d_20_biasadd_readvariableop_resource:	j
Omcfly_cnn_50epochs_separable_conv2d_21_separable_conv2d_readvariableop_resource:m
Qmcfly_cnn_50epochs_separable_conv2d_21_separable_conv2d_readvariableop_1_resource:U
Fmcfly_cnn_50epochs_separable_conv2d_21_biasadd_readvariableop_resource:	j
Omcfly_cnn_50epochs_separable_conv2d_22_separable_conv2d_readvariableop_resource:m
Qmcfly_cnn_50epochs_separable_conv2d_22_separable_conv2d_readvariableop_1_resource:àU
Fmcfly_cnn_50epochs_separable_conv2d_22_biasadd_readvariableop_resource:	àP
Amcfly_cnn_50epochs_batch_normalization_17_readvariableop_resource:	àR
Cmcfly_cnn_50epochs_batch_normalization_17_readvariableop_1_resource:	àa
Rmcfly_cnn_50epochs_batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	àc
Tmcfly_cnn_50epochs_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	àj
Omcfly_cnn_50epochs_separable_conv2d_23_separable_conv2d_readvariableop_resource:àm
Qmcfly_cnn_50epochs_separable_conv2d_23_separable_conv2d_readvariableop_1_resource:à U
Fmcfly_cnn_50epochs_separable_conv2d_23_biasadd_readvariableop_resource:	 `
Emcfly_cnn_50epochs_sep_out_0_separable_conv2d_readvariableop_resource: c
Gmcfly_cnn_50epochs_sep_out_0_separable_conv2d_readvariableop_1_resource: ÀK
<mcfly_cnn_50epochs_sep_out_0_biasadd_readvariableop_resource:	À`
Emcfly_cnn_50epochs_sep_out_1_separable_conv2d_readvariableop_resource:Àc
Gmcfly_cnn_50epochs_sep_out_1_separable_conv2d_readvariableop_1_resource:ÀàK
<mcfly_cnn_50epochs_sep_out_1_biasadd_readvariableop_resource:	à`
Emcfly_cnn_50epochs_sep_out_2_separable_conv2d_readvariableop_resource:àc
Gmcfly_cnn_50epochs_sep_out_2_separable_conv2d_readvariableop_1_resource:ààK
<mcfly_cnn_50epochs_sep_out_2_biasadd_readvariableop_resource:	à]
Bmcfly_cnn_50epochs_output_separable_conv2d_readvariableop_resource:à_
Dmcfly_cnn_50epochs_output_separable_conv2d_readvariableop_1_resource:àG
9mcfly_cnn_50epochs_output_biasadd_readvariableop_resource:
identity¢IMcFly_cnn_50epochs/batch_normalization_11/FusedBatchNormV3/ReadVariableOp¢KMcFly_cnn_50epochs/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1¢8McFly_cnn_50epochs/batch_normalization_11/ReadVariableOp¢:McFly_cnn_50epochs/batch_normalization_11/ReadVariableOp_1¢IMcFly_cnn_50epochs/batch_normalization_12/FusedBatchNormV3/ReadVariableOp¢KMcFly_cnn_50epochs/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1¢8McFly_cnn_50epochs/batch_normalization_12/ReadVariableOp¢:McFly_cnn_50epochs/batch_normalization_12/ReadVariableOp_1¢IMcFly_cnn_50epochs/batch_normalization_13/FusedBatchNormV3/ReadVariableOp¢KMcFly_cnn_50epochs/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1¢8McFly_cnn_50epochs/batch_normalization_13/ReadVariableOp¢:McFly_cnn_50epochs/batch_normalization_13/ReadVariableOp_1¢IMcFly_cnn_50epochs/batch_normalization_14/FusedBatchNormV3/ReadVariableOp¢KMcFly_cnn_50epochs/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1¢8McFly_cnn_50epochs/batch_normalization_14/ReadVariableOp¢:McFly_cnn_50epochs/batch_normalization_14/ReadVariableOp_1¢IMcFly_cnn_50epochs/batch_normalization_15/FusedBatchNormV3/ReadVariableOp¢KMcFly_cnn_50epochs/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1¢8McFly_cnn_50epochs/batch_normalization_15/ReadVariableOp¢:McFly_cnn_50epochs/batch_normalization_15/ReadVariableOp_1¢IMcFly_cnn_50epochs/batch_normalization_16/FusedBatchNormV3/ReadVariableOp¢KMcFly_cnn_50epochs/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1¢8McFly_cnn_50epochs/batch_normalization_16/ReadVariableOp¢:McFly_cnn_50epochs/batch_normalization_16/ReadVariableOp_1¢IMcFly_cnn_50epochs/batch_normalization_17/FusedBatchNormV3/ReadVariableOp¢KMcFly_cnn_50epochs/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1¢8McFly_cnn_50epochs/batch_normalization_17/ReadVariableOp¢:McFly_cnn_50epochs/batch_normalization_17/ReadVariableOp_1¢3McFly_cnn_50epochs/conv2d_10/BiasAdd/ReadVariableOp¢2McFly_cnn_50epochs/conv2d_10/Conv2D/ReadVariableOp¢3McFly_cnn_50epochs/conv2d_11/BiasAdd/ReadVariableOp¢2McFly_cnn_50epochs/conv2d_11/Conv2D/ReadVariableOp¢3McFly_cnn_50epochs/conv2d_12/BiasAdd/ReadVariableOp¢2McFly_cnn_50epochs/conv2d_12/Conv2D/ReadVariableOp¢2McFly_cnn_50epochs/conv2d_8/BiasAdd/ReadVariableOp¢1McFly_cnn_50epochs/conv2d_8/Conv2D/ReadVariableOp¢2McFly_cnn_50epochs/conv2d_9/BiasAdd/ReadVariableOp¢1McFly_cnn_50epochs/conv2d_9/Conv2D/ReadVariableOp¢4McFly_cnn_50epochs/first_conv/BiasAdd/ReadVariableOp¢3McFly_cnn_50epochs/first_conv/Conv2D/ReadVariableOp¢0McFly_cnn_50epochs/output/BiasAdd/ReadVariableOp¢9McFly_cnn_50epochs/output/separable_conv2d/ReadVariableOp¢;McFly_cnn_50epochs/output/separable_conv2d/ReadVariableOp_1¢3McFly_cnn_50epochs/sep_out_0/BiasAdd/ReadVariableOp¢<McFly_cnn_50epochs/sep_out_0/separable_conv2d/ReadVariableOp¢>McFly_cnn_50epochs/sep_out_0/separable_conv2d/ReadVariableOp_1¢3McFly_cnn_50epochs/sep_out_1/BiasAdd/ReadVariableOp¢<McFly_cnn_50epochs/sep_out_1/separable_conv2d/ReadVariableOp¢>McFly_cnn_50epochs/sep_out_1/separable_conv2d/ReadVariableOp_1¢3McFly_cnn_50epochs/sep_out_2/BiasAdd/ReadVariableOp¢<McFly_cnn_50epochs/sep_out_2/separable_conv2d/ReadVariableOp¢>McFly_cnn_50epochs/sep_out_2/separable_conv2d/ReadVariableOp_1¢=McFly_cnn_50epochs/separable_conv2d_18/BiasAdd/ReadVariableOp¢FMcFly_cnn_50epochs/separable_conv2d_18/separable_conv2d/ReadVariableOp¢HMcFly_cnn_50epochs/separable_conv2d_18/separable_conv2d/ReadVariableOp_1¢=McFly_cnn_50epochs/separable_conv2d_19/BiasAdd/ReadVariableOp¢FMcFly_cnn_50epochs/separable_conv2d_19/separable_conv2d/ReadVariableOp¢HMcFly_cnn_50epochs/separable_conv2d_19/separable_conv2d/ReadVariableOp_1¢=McFly_cnn_50epochs/separable_conv2d_20/BiasAdd/ReadVariableOp¢FMcFly_cnn_50epochs/separable_conv2d_20/separable_conv2d/ReadVariableOp¢HMcFly_cnn_50epochs/separable_conv2d_20/separable_conv2d/ReadVariableOp_1¢=McFly_cnn_50epochs/separable_conv2d_21/BiasAdd/ReadVariableOp¢FMcFly_cnn_50epochs/separable_conv2d_21/separable_conv2d/ReadVariableOp¢HMcFly_cnn_50epochs/separable_conv2d_21/separable_conv2d/ReadVariableOp_1¢=McFly_cnn_50epochs/separable_conv2d_22/BiasAdd/ReadVariableOp¢FMcFly_cnn_50epochs/separable_conv2d_22/separable_conv2d/ReadVariableOp¢HMcFly_cnn_50epochs/separable_conv2d_22/separable_conv2d/ReadVariableOp_1¢=McFly_cnn_50epochs/separable_conv2d_23/BiasAdd/ReadVariableOp¢FMcFly_cnn_50epochs/separable_conv2d_23/separable_conv2d/ReadVariableOp¢HMcFly_cnn_50epochs/separable_conv2d_23/separable_conv2d/ReadVariableOp_1¹
3McFly_cnn_50epochs/first_conv/Conv2D/ReadVariableOpReadVariableOp<mcfly_cnn_50epochs_first_conv_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0Ù
$McFly_cnn_50epochs/first_conv/Conv2DConv2Dinput_6;McFly_cnn_50epochs/first_conv/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà*
paddingSAME*
strides
¯
4McFly_cnn_50epochs/first_conv/BiasAdd/ReadVariableOpReadVariableOp=mcfly_cnn_50epochs_first_conv_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0Ú
%McFly_cnn_50epochs/first_conv/BiasAddBiasAdd-McFly_cnn_50epochs/first_conv/Conv2D:output:0<McFly_cnn_50epochs/first_conv/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà·
8McFly_cnn_50epochs/batch_normalization_11/ReadVariableOpReadVariableOpAmcfly_cnn_50epochs_batch_normalization_11_readvariableop_resource*
_output_shapes	
:à*
dtype0»
:McFly_cnn_50epochs/batch_normalization_11/ReadVariableOp_1ReadVariableOpCmcfly_cnn_50epochs_batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:à*
dtype0Ù
IMcFly_cnn_50epochs/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpRmcfly_cnn_50epochs_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:à*
dtype0Ý
KMcFly_cnn_50epochs/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTmcfly_cnn_50epochs_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:à*
dtype0·
:McFly_cnn_50epochs/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3.McFly_cnn_50epochs/first_conv/BiasAdd:output:0@McFly_cnn_50epochs/batch_normalization_11/ReadVariableOp:value:0BMcFly_cnn_50epochs/batch_normalization_11/ReadVariableOp_1:value:0QMcFly_cnn_50epochs/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0SMcFly_cnn_50epochs/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:ÿÿÿÿÿÿÿÿÿààà:à:à:à:à:*
epsilon%o:*
is_training( ª
%McFly_cnn_50epochs/activation_21/ReluRelu>McFly_cnn_50epochs/batch_normalization_11/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿàààØ
+McFly_cnn_50epochs/max_pooling2d_10/MaxPoolMaxPool3McFly_cnn_50epochs/activation_21/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà*
ksize
*
paddingSAME*
strides
¶
1McFly_cnn_50epochs/conv2d_8/Conv2D/ReadVariableOpReadVariableOp:mcfly_cnn_50epochs_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:à*
dtype0
"McFly_cnn_50epochs/conv2d_8/Conv2DConv2D4McFly_cnn_50epochs/max_pooling2d_10/MaxPool:output:09McFly_cnn_50epochs/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
«
2McFly_cnn_50epochs/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp;mcfly_cnn_50epochs_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#McFly_cnn_50epochs/conv2d_8/BiasAddBiasAdd+McFly_cnn_50epochs/conv2d_8/Conv2D:output:0:McFly_cnn_50epochs/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp·
8McFly_cnn_50epochs/batch_normalization_12/ReadVariableOpReadVariableOpAmcfly_cnn_50epochs_batch_normalization_12_readvariableop_resource*
_output_shapes	
:*
dtype0»
:McFly_cnn_50epochs/batch_normalization_12/ReadVariableOp_1ReadVariableOpCmcfly_cnn_50epochs_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ù
IMcFly_cnn_50epochs/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpRmcfly_cnn_50epochs_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Ý
KMcFly_cnn_50epochs/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTmcfly_cnn_50epochs_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0³
:McFly_cnn_50epochs/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3,McFly_cnn_50epochs/conv2d_8/BiasAdd:output:0@McFly_cnn_50epochs/batch_normalization_12/ReadVariableOp:value:0BMcFly_cnn_50epochs/batch_normalization_12/ReadVariableOp_1:value:0QMcFly_cnn_50epochs/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0SMcFly_cnn_50epochs/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿpp:::::*
epsilon%o:*
is_training( ¨
%McFly_cnn_50epochs/activation_22/ReluRelu>McFly_cnn_50epochs/batch_normalization_12/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp¡
%McFly_cnn_50epochs/dropout_2/IdentityIdentity3McFly_cnn_50epochs/activation_22/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp¶
1McFly_cnn_50epochs/conv2d_9/Conv2D/ReadVariableOpReadVariableOp:mcfly_cnn_50epochs_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:à*
dtype0ú
"McFly_cnn_50epochs/conv2d_9/Conv2DConv2D.McFly_cnn_50epochs/dropout_2/Identity:output:09McFly_cnn_50epochs/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà*
paddingSAME*
strides
«
2McFly_cnn_50epochs/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp;mcfly_cnn_50epochs_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0Ò
#McFly_cnn_50epochs/conv2d_9/BiasAddBiasAdd+McFly_cnn_50epochs/conv2d_9/Conv2D:output:0:McFly_cnn_50epochs/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà
%McFly_cnn_50epochs/activation_23/ReluRelu,McFly_cnn_50epochs/conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà¸
2McFly_cnn_50epochs/conv2d_10/Conv2D/ReadVariableOpReadVariableOp;mcfly_cnn_50epochs_conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:à *
dtype0
#McFly_cnn_50epochs/conv2d_10/Conv2DConv2D3McFly_cnn_50epochs/activation_23/Relu:activations:0:McFly_cnn_50epochs/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp *
paddingSAME*
strides
­
3McFly_cnn_50epochs/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp<mcfly_cnn_50epochs_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0Õ
$McFly_cnn_50epochs/conv2d_10/BiasAddBiasAdd,McFly_cnn_50epochs/conv2d_10/Conv2D:output:0;McFly_cnn_50epochs/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp ·
8McFly_cnn_50epochs/batch_normalization_13/ReadVariableOpReadVariableOpAmcfly_cnn_50epochs_batch_normalization_13_readvariableop_resource*
_output_shapes	
: *
dtype0»
:McFly_cnn_50epochs/batch_normalization_13/ReadVariableOp_1ReadVariableOpCmcfly_cnn_50epochs_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
: *
dtype0Ù
IMcFly_cnn_50epochs/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpRmcfly_cnn_50epochs_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
: *
dtype0Ý
KMcFly_cnn_50epochs/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTmcfly_cnn_50epochs_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
: *
dtype0´
:McFly_cnn_50epochs/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3-McFly_cnn_50epochs/conv2d_10/BiasAdd:output:0@McFly_cnn_50epochs/batch_normalization_13/ReadVariableOp:value:0BMcFly_cnn_50epochs/batch_normalization_13/ReadVariableOp_1:value:0QMcFly_cnn_50epochs/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0SMcFly_cnn_50epochs/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿpp : : : : :*
epsilon%o:*
is_training( ¨
%McFly_cnn_50epochs/activation_24/ReluRelu>McFly_cnn_50epochs/batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp ¡
%McFly_cnn_50epochs/dropout_3/IdentityIdentity3McFly_cnn_50epochs/activation_24/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp ·
2McFly_cnn_50epochs/conv2d_11/Conv2D/ReadVariableOpReadVariableOp;mcfly_cnn_50epochs_conv2d_11_conv2d_readvariableop_resource*'
_output_shapes
: `*
dtype0û
#McFly_cnn_50epochs/conv2d_11/Conv2DConv2D.McFly_cnn_50epochs/dropout_3/Identity:output:0:McFly_cnn_50epochs/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`*
paddingSAME*
strides
¬
3McFly_cnn_50epochs/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp<mcfly_cnn_50epochs_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Ô
$McFly_cnn_50epochs/conv2d_11/BiasAddBiasAdd,McFly_cnn_50epochs/conv2d_11/Conv2D:output:0;McFly_cnn_50epochs/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`¶
8McFly_cnn_50epochs/batch_normalization_14/ReadVariableOpReadVariableOpAmcfly_cnn_50epochs_batch_normalization_14_readvariableop_resource*
_output_shapes
:`*
dtype0º
:McFly_cnn_50epochs/batch_normalization_14/ReadVariableOp_1ReadVariableOpCmcfly_cnn_50epochs_batch_normalization_14_readvariableop_1_resource*
_output_shapes
:`*
dtype0Ø
IMcFly_cnn_50epochs/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpRmcfly_cnn_50epochs_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0Ü
KMcFly_cnn_50epochs/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTmcfly_cnn_50epochs_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0¯
:McFly_cnn_50epochs/batch_normalization_14/FusedBatchNormV3FusedBatchNormV3-McFly_cnn_50epochs/conv2d_11/BiasAdd:output:0@McFly_cnn_50epochs/batch_normalization_14/ReadVariableOp:value:0BMcFly_cnn_50epochs/batch_normalization_14/ReadVariableOp_1:value:0QMcFly_cnn_50epochs/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0SMcFly_cnn_50epochs/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿpp`:`:`:`:`:*
epsilon%o:*
is_training( §
%McFly_cnn_50epochs/activation_25/ReluRelu>McFly_cnn_50epochs/batch_normalization_14/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`·
2McFly_cnn_50epochs/conv2d_12/Conv2D/ReadVariableOpReadVariableOp;mcfly_cnn_50epochs_conv2d_12_conv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0
#McFly_cnn_50epochs/conv2d_12/Conv2DConv2D3McFly_cnn_50epochs/activation_25/Relu:activations:0:McFly_cnn_50epochs/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
­
3McFly_cnn_50epochs/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp<mcfly_cnn_50epochs_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$McFly_cnn_50epochs/conv2d_12/BiasAddBiasAdd,McFly_cnn_50epochs/conv2d_12/Conv2D:output:0;McFly_cnn_50epochs/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp·
8McFly_cnn_50epochs/batch_normalization_15/ReadVariableOpReadVariableOpAmcfly_cnn_50epochs_batch_normalization_15_readvariableop_resource*
_output_shapes	
:*
dtype0»
:McFly_cnn_50epochs/batch_normalization_15/ReadVariableOp_1ReadVariableOpCmcfly_cnn_50epochs_batch_normalization_15_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ù
IMcFly_cnn_50epochs/batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOpRmcfly_cnn_50epochs_batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Ý
KMcFly_cnn_50epochs/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTmcfly_cnn_50epochs_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0´
:McFly_cnn_50epochs/batch_normalization_15/FusedBatchNormV3FusedBatchNormV3-McFly_cnn_50epochs/conv2d_12/BiasAdd:output:0@McFly_cnn_50epochs/batch_normalization_15/ReadVariableOp:value:0BMcFly_cnn_50epochs/batch_normalization_15/ReadVariableOp_1:value:0QMcFly_cnn_50epochs/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0SMcFly_cnn_50epochs/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿpp:::::*
epsilon%o:*
is_training( ¨
%McFly_cnn_50epochs/activation_26/ReluRelu>McFly_cnn_50epochs/batch_normalization_15/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp¡
%McFly_cnn_50epochs/dropout_4/IdentityIdentity3McFly_cnn_50epochs/activation_26/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppÓ
+McFly_cnn_50epochs/max_pooling2d_11/MaxPoolMaxPool.McFly_cnn_50epochs/dropout_4/Identity:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingSAME*
strides
ß
FMcFly_cnn_50epochs/separable_conv2d_18/separable_conv2d/ReadVariableOpReadVariableOpOmcfly_cnn_50epochs_separable_conv2d_18_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0ä
HMcFly_cnn_50epochs/separable_conv2d_18/separable_conv2d/ReadVariableOp_1ReadVariableOpQmcfly_cnn_50epochs_separable_conv2d_18_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:À*
dtype0
=McFly_cnn_50epochs/separable_conv2d_18/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
EMcFly_cnn_50epochs/separable_conv2d_18/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ã
AMcFly_cnn_50epochs/separable_conv2d_18/separable_conv2d/depthwiseDepthwiseConv2dNative4McFly_cnn_50epochs/max_pooling2d_11/MaxPool:output:0NMcFly_cnn_50epochs/separable_conv2d_18/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
Ã
7McFly_cnn_50epochs/separable_conv2d_18/separable_conv2dConv2DJMcFly_cnn_50epochs/separable_conv2d_18/separable_conv2d/depthwise:output:0PMcFly_cnn_50epochs/separable_conv2d_18/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À*
paddingVALID*
strides
Á
=McFly_cnn_50epochs/separable_conv2d_18/BiasAdd/ReadVariableOpReadVariableOpFmcfly_cnn_50epochs_separable_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:À*
dtype0ý
.McFly_cnn_50epochs/separable_conv2d_18/BiasAddBiasAdd@McFly_cnn_50epochs/separable_conv2d_18/separable_conv2d:output:0EMcFly_cnn_50epochs/separable_conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À·
8McFly_cnn_50epochs/batch_normalization_16/ReadVariableOpReadVariableOpAmcfly_cnn_50epochs_batch_normalization_16_readvariableop_resource*
_output_shapes	
:À*
dtype0»
:McFly_cnn_50epochs/batch_normalization_16/ReadVariableOp_1ReadVariableOpCmcfly_cnn_50epochs_batch_normalization_16_readvariableop_1_resource*
_output_shapes	
:À*
dtype0Ù
IMcFly_cnn_50epochs/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpRmcfly_cnn_50epochs_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:À*
dtype0Ý
KMcFly_cnn_50epochs/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTmcfly_cnn_50epochs_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:À*
dtype0¾
:McFly_cnn_50epochs/batch_normalization_16/FusedBatchNormV3FusedBatchNormV37McFly_cnn_50epochs/separable_conv2d_18/BiasAdd:output:0@McFly_cnn_50epochs/batch_normalization_16/ReadVariableOp:value:0BMcFly_cnn_50epochs/batch_normalization_16/ReadVariableOp_1:value:0QMcFly_cnn_50epochs/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0SMcFly_cnn_50epochs/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ88À:À:À:À:À:*
epsilon%o:*
is_training( ¨
%McFly_cnn_50epochs/activation_27/ReluRelu>McFly_cnn_50epochs/batch_normalization_16/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Àß
FMcFly_cnn_50epochs/separable_conv2d_19/separable_conv2d/ReadVariableOpReadVariableOpOmcfly_cnn_50epochs_separable_conv2d_19_separable_conv2d_readvariableop_resource*'
_output_shapes
:À*
dtype0ä
HMcFly_cnn_50epochs/separable_conv2d_19/separable_conv2d/ReadVariableOp_1ReadVariableOpQmcfly_cnn_50epochs_separable_conv2d_19_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:Àà*
dtype0
=McFly_cnn_50epochs/separable_conv2d_19/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @     
EMcFly_cnn_50epochs/separable_conv2d_19/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Â
AMcFly_cnn_50epochs/separable_conv2d_19/separable_conv2d/depthwiseDepthwiseConv2dNative3McFly_cnn_50epochs/activation_27/Relu:activations:0NMcFly_cnn_50epochs/separable_conv2d_19/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À*
paddingSAME*
strides
Ã
7McFly_cnn_50epochs/separable_conv2d_19/separable_conv2dConv2DJMcFly_cnn_50epochs/separable_conv2d_19/separable_conv2d/depthwise:output:0PMcFly_cnn_50epochs/separable_conv2d_19/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*
paddingVALID*
strides
Á
=McFly_cnn_50epochs/separable_conv2d_19/BiasAdd/ReadVariableOpReadVariableOpFmcfly_cnn_50epochs_separable_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0ý
.McFly_cnn_50epochs/separable_conv2d_19/BiasAddBiasAdd@McFly_cnn_50epochs/separable_conv2d_19/separable_conv2d:output:0EMcFly_cnn_50epochs/separable_conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à¡
%McFly_cnn_50epochs/activation_28/ReluRelu7McFly_cnn_50epochs/separable_conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88àß
FMcFly_cnn_50epochs/separable_conv2d_20/separable_conv2d/ReadVariableOpReadVariableOpOmcfly_cnn_50epochs_separable_conv2d_20_separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0ä
HMcFly_cnn_50epochs/separable_conv2d_20/separable_conv2d/ReadVariableOp_1ReadVariableOpQmcfly_cnn_50epochs_separable_conv2d_20_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:à*
dtype0
=McFly_cnn_50epochs/separable_conv2d_20/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      `     
EMcFly_cnn_50epochs/separable_conv2d_20/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Â
AMcFly_cnn_50epochs/separable_conv2d_20/separable_conv2d/depthwiseDepthwiseConv2dNative3McFly_cnn_50epochs/activation_28/Relu:activations:0NMcFly_cnn_50epochs/separable_conv2d_20/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*
paddingSAME*
strides
Ã
7McFly_cnn_50epochs/separable_conv2d_20/separable_conv2dConv2DJMcFly_cnn_50epochs/separable_conv2d_20/separable_conv2d/depthwise:output:0PMcFly_cnn_50epochs/separable_conv2d_20/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingVALID*
strides
Á
=McFly_cnn_50epochs/separable_conv2d_20/BiasAdd/ReadVariableOpReadVariableOpFmcfly_cnn_50epochs_separable_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ý
.McFly_cnn_50epochs/separable_conv2d_20/BiasAddBiasAdd@McFly_cnn_50epochs/separable_conv2d_20/separable_conv2d:output:0EMcFly_cnn_50epochs/separable_conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¡
%McFly_cnn_50epochs/activation_29/ReluRelu7McFly_cnn_50epochs/separable_conv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¡
%McFly_cnn_50epochs/dropout_5/IdentityIdentity3McFly_cnn_50epochs/activation_29/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88ß
FMcFly_cnn_50epochs/separable_conv2d_21/separable_conv2d/ReadVariableOpReadVariableOpOmcfly_cnn_50epochs_separable_conv2d_21_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0ä
HMcFly_cnn_50epochs/separable_conv2d_21/separable_conv2d/ReadVariableOp_1ReadVariableOpQmcfly_cnn_50epochs_separable_conv2d_21_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:*
dtype0
=McFly_cnn_50epochs/separable_conv2d_21/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
EMcFly_cnn_50epochs/separable_conv2d_21/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ½
AMcFly_cnn_50epochs/separable_conv2d_21/separable_conv2d/depthwiseDepthwiseConv2dNative.McFly_cnn_50epochs/dropout_5/Identity:output:0NMcFly_cnn_50epochs/separable_conv2d_21/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
Ã
7McFly_cnn_50epochs/separable_conv2d_21/separable_conv2dConv2DJMcFly_cnn_50epochs/separable_conv2d_21/separable_conv2d/depthwise:output:0PMcFly_cnn_50epochs/separable_conv2d_21/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingVALID*
strides
Á
=McFly_cnn_50epochs/separable_conv2d_21/BiasAdd/ReadVariableOpReadVariableOpFmcfly_cnn_50epochs_separable_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ý
.McFly_cnn_50epochs/separable_conv2d_21/BiasAddBiasAdd@McFly_cnn_50epochs/separable_conv2d_21/separable_conv2d:output:0EMcFly_cnn_50epochs/separable_conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¡
%McFly_cnn_50epochs/activation_30/ReluRelu7McFly_cnn_50epochs/separable_conv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88ß
FMcFly_cnn_50epochs/separable_conv2d_22/separable_conv2d/ReadVariableOpReadVariableOpOmcfly_cnn_50epochs_separable_conv2d_22_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0ä
HMcFly_cnn_50epochs/separable_conv2d_22/separable_conv2d/ReadVariableOp_1ReadVariableOpQmcfly_cnn_50epochs_separable_conv2d_22_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:à*
dtype0
=McFly_cnn_50epochs/separable_conv2d_22/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
EMcFly_cnn_50epochs/separable_conv2d_22/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Â
AMcFly_cnn_50epochs/separable_conv2d_22/separable_conv2d/depthwiseDepthwiseConv2dNative3McFly_cnn_50epochs/activation_30/Relu:activations:0NMcFly_cnn_50epochs/separable_conv2d_22/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
Ã
7McFly_cnn_50epochs/separable_conv2d_22/separable_conv2dConv2DJMcFly_cnn_50epochs/separable_conv2d_22/separable_conv2d/depthwise:output:0PMcFly_cnn_50epochs/separable_conv2d_22/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*
paddingVALID*
strides
Á
=McFly_cnn_50epochs/separable_conv2d_22/BiasAdd/ReadVariableOpReadVariableOpFmcfly_cnn_50epochs_separable_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0ý
.McFly_cnn_50epochs/separable_conv2d_22/BiasAddBiasAdd@McFly_cnn_50epochs/separable_conv2d_22/separable_conv2d:output:0EMcFly_cnn_50epochs/separable_conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à·
8McFly_cnn_50epochs/batch_normalization_17/ReadVariableOpReadVariableOpAmcfly_cnn_50epochs_batch_normalization_17_readvariableop_resource*
_output_shapes	
:à*
dtype0»
:McFly_cnn_50epochs/batch_normalization_17/ReadVariableOp_1ReadVariableOpCmcfly_cnn_50epochs_batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:à*
dtype0Ù
IMcFly_cnn_50epochs/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpRmcfly_cnn_50epochs_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:à*
dtype0Ý
KMcFly_cnn_50epochs/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTmcfly_cnn_50epochs_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:à*
dtype0¾
:McFly_cnn_50epochs/batch_normalization_17/FusedBatchNormV3FusedBatchNormV37McFly_cnn_50epochs/separable_conv2d_22/BiasAdd:output:0@McFly_cnn_50epochs/batch_normalization_17/ReadVariableOp:value:0BMcFly_cnn_50epochs/batch_normalization_17/ReadVariableOp_1:value:0QMcFly_cnn_50epochs/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0SMcFly_cnn_50epochs/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ88à:à:à:à:à:*
epsilon%o:*
is_training( ¨
%McFly_cnn_50epochs/activation_31/ReluRelu>McFly_cnn_50epochs/batch_normalization_17/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88àß
FMcFly_cnn_50epochs/separable_conv2d_23/separable_conv2d/ReadVariableOpReadVariableOpOmcfly_cnn_50epochs_separable_conv2d_23_separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0ä
HMcFly_cnn_50epochs/separable_conv2d_23/separable_conv2d/ReadVariableOp_1ReadVariableOpQmcfly_cnn_50epochs_separable_conv2d_23_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:à *
dtype0
=McFly_cnn_50epochs/separable_conv2d_23/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      `     
EMcFly_cnn_50epochs/separable_conv2d_23/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Â
AMcFly_cnn_50epochs/separable_conv2d_23/separable_conv2d/depthwiseDepthwiseConv2dNative3McFly_cnn_50epochs/activation_31/Relu:activations:0NMcFly_cnn_50epochs/separable_conv2d_23/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*
paddingSAME*
strides
Ã
7McFly_cnn_50epochs/separable_conv2d_23/separable_conv2dConv2DJMcFly_cnn_50epochs/separable_conv2d_23/separable_conv2d/depthwise:output:0PMcFly_cnn_50epochs/separable_conv2d_23/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 *
paddingVALID*
strides
Á
=McFly_cnn_50epochs/separable_conv2d_23/BiasAdd/ReadVariableOpReadVariableOpFmcfly_cnn_50epochs_separable_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0ý
.McFly_cnn_50epochs/separable_conv2d_23/BiasAddBiasAdd@McFly_cnn_50epochs/separable_conv2d_23/separable_conv2d:output:0EMcFly_cnn_50epochs/separable_conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 ¡
%McFly_cnn_50epochs/activation_32/ReluRelu7McFly_cnn_50epochs/separable_conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 Ø
+McFly_cnn_50epochs/max_pooling2d_12/MaxPoolMaxPool3McFly_cnn_50epochs/activation_32/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingSAME*
strides
Ë
<McFly_cnn_50epochs/sep_out_0/separable_conv2d/ReadVariableOpReadVariableOpEmcfly_cnn_50epochs_sep_out_0_separable_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype0Ð
>McFly_cnn_50epochs/sep_out_0/separable_conv2d/ReadVariableOp_1ReadVariableOpGmcfly_cnn_50epochs_sep_out_0_separable_conv2d_readvariableop_1_resource*(
_output_shapes
: À*
dtype0
3McFly_cnn_50epochs/sep_out_0/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
;McFly_cnn_50epochs/sep_out_0/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      °
7McFly_cnn_50epochs/sep_out_0/separable_conv2d/depthwiseDepthwiseConv2dNative4McFly_cnn_50epochs/max_pooling2d_12/MaxPool:output:0DMcFly_cnn_50epochs/sep_out_0/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
¥
-McFly_cnn_50epochs/sep_out_0/separable_conv2dConv2D@McFly_cnn_50epochs/sep_out_0/separable_conv2d/depthwise:output:0FMcFly_cnn_50epochs/sep_out_0/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
paddingVALID*
strides
­
3McFly_cnn_50epochs/sep_out_0/BiasAdd/ReadVariableOpReadVariableOp<mcfly_cnn_50epochs_sep_out_0_biasadd_readvariableop_resource*
_output_shapes	
:À*
dtype0ß
$McFly_cnn_50epochs/sep_out_0/BiasAddBiasAdd6McFly_cnn_50epochs/sep_out_0/separable_conv2d:output:0;McFly_cnn_50epochs/sep_out_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
!McFly_cnn_50epochs/sep_out_0/ReluRelu-McFly_cnn_50epochs/sep_out_0/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀÔ
+McFly_cnn_50epochs/max_pooling2d_13/MaxPoolMaxPool/McFly_cnn_50epochs/sep_out_0/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
ksize
*
paddingSAME*
strides
Ë
<McFly_cnn_50epochs/sep_out_1/separable_conv2d/ReadVariableOpReadVariableOpEmcfly_cnn_50epochs_sep_out_1_separable_conv2d_readvariableop_resource*'
_output_shapes
:À*
dtype0Ð
>McFly_cnn_50epochs/sep_out_1/separable_conv2d/ReadVariableOp_1ReadVariableOpGmcfly_cnn_50epochs_sep_out_1_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:Àà*
dtype0
3McFly_cnn_50epochs/sep_out_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      À     
;McFly_cnn_50epochs/sep_out_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      °
7McFly_cnn_50epochs/sep_out_1/separable_conv2d/depthwiseDepthwiseConv2dNative4McFly_cnn_50epochs/max_pooling2d_13/MaxPool:output:0DMcFly_cnn_50epochs/sep_out_1/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
paddingVALID*
strides
¥
-McFly_cnn_50epochs/sep_out_1/separable_conv2dConv2D@McFly_cnn_50epochs/sep_out_1/separable_conv2d/depthwise:output:0FMcFly_cnn_50epochs/sep_out_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
­
3McFly_cnn_50epochs/sep_out_1/BiasAdd/ReadVariableOpReadVariableOp<mcfly_cnn_50epochs_sep_out_1_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0ß
$McFly_cnn_50epochs/sep_out_1/BiasAddBiasAdd6McFly_cnn_50epochs/sep_out_1/separable_conv2d:output:0;McFly_cnn_50epochs/sep_out_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
!McFly_cnn_50epochs/sep_out_1/ReluRelu-McFly_cnn_50epochs/sep_out_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÔ
+McFly_cnn_50epochs/max_pooling2d_14/MaxPoolMaxPool/McFly_cnn_50epochs/sep_out_1/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
ksize
*
paddingSAME*
strides
Ë
<McFly_cnn_50epochs/sep_out_2/separable_conv2d/ReadVariableOpReadVariableOpEmcfly_cnn_50epochs_sep_out_2_separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0Ð
>McFly_cnn_50epochs/sep_out_2/separable_conv2d/ReadVariableOp_1ReadVariableOpGmcfly_cnn_50epochs_sep_out_2_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:àà*
dtype0
3McFly_cnn_50epochs/sep_out_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      à      
;McFly_cnn_50epochs/sep_out_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      °
7McFly_cnn_50epochs/sep_out_2/separable_conv2d/depthwiseDepthwiseConv2dNative4McFly_cnn_50epochs/max_pooling2d_14/MaxPool:output:0DMcFly_cnn_50epochs/sep_out_2/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
¥
-McFly_cnn_50epochs/sep_out_2/separable_conv2dConv2D@McFly_cnn_50epochs/sep_out_2/separable_conv2d/depthwise:output:0FMcFly_cnn_50epochs/sep_out_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
­
3McFly_cnn_50epochs/sep_out_2/BiasAdd/ReadVariableOpReadVariableOp<mcfly_cnn_50epochs_sep_out_2_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0ß
$McFly_cnn_50epochs/sep_out_2/BiasAddBiasAdd6McFly_cnn_50epochs/sep_out_2/separable_conv2d:output:0;McFly_cnn_50epochs/sep_out_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
!McFly_cnn_50epochs/sep_out_2/ReluRelu-McFly_cnn_50epochs/sep_out_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÔ
+McFly_cnn_50epochs/max_pooling2d_15/MaxPoolMaxPool/McFly_cnn_50epochs/sep_out_2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
ksize
*
paddingSAME*
strides
Å
9McFly_cnn_50epochs/output/separable_conv2d/ReadVariableOpReadVariableOpBmcfly_cnn_50epochs_output_separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0É
;McFly_cnn_50epochs/output/separable_conv2d/ReadVariableOp_1ReadVariableOpDmcfly_cnn_50epochs_output_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:à*
dtype0
0McFly_cnn_50epochs/output/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      à     
8McFly_cnn_50epochs/output/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ª
4McFly_cnn_50epochs/output/separable_conv2d/depthwiseDepthwiseConv2dNative4McFly_cnn_50epochs/max_pooling2d_15/MaxPool:output:0AMcFly_cnn_50epochs/output/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides

*McFly_cnn_50epochs/output/separable_conv2dConv2D=McFly_cnn_50epochs/output/separable_conv2d/depthwise:output:0CMcFly_cnn_50epochs/output/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
¦
0McFly_cnn_50epochs/output/BiasAdd/ReadVariableOpReadVariableOp9mcfly_cnn_50epochs_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Õ
!McFly_cnn_50epochs/output/BiasAddBiasAdd3McFly_cnn_50epochs/output/separable_conv2d:output:08McFly_cnn_50epochs/output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!McFly_cnn_50epochs/output/SigmoidSigmoid*McFly_cnn_50epochs/output/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
IdentityIdentity%McFly_cnn_50epochs/output/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$
NoOpNoOpJ^McFly_cnn_50epochs/batch_normalization_11/FusedBatchNormV3/ReadVariableOpL^McFly_cnn_50epochs/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_19^McFly_cnn_50epochs/batch_normalization_11/ReadVariableOp;^McFly_cnn_50epochs/batch_normalization_11/ReadVariableOp_1J^McFly_cnn_50epochs/batch_normalization_12/FusedBatchNormV3/ReadVariableOpL^McFly_cnn_50epochs/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_19^McFly_cnn_50epochs/batch_normalization_12/ReadVariableOp;^McFly_cnn_50epochs/batch_normalization_12/ReadVariableOp_1J^McFly_cnn_50epochs/batch_normalization_13/FusedBatchNormV3/ReadVariableOpL^McFly_cnn_50epochs/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_19^McFly_cnn_50epochs/batch_normalization_13/ReadVariableOp;^McFly_cnn_50epochs/batch_normalization_13/ReadVariableOp_1J^McFly_cnn_50epochs/batch_normalization_14/FusedBatchNormV3/ReadVariableOpL^McFly_cnn_50epochs/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_19^McFly_cnn_50epochs/batch_normalization_14/ReadVariableOp;^McFly_cnn_50epochs/batch_normalization_14/ReadVariableOp_1J^McFly_cnn_50epochs/batch_normalization_15/FusedBatchNormV3/ReadVariableOpL^McFly_cnn_50epochs/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_19^McFly_cnn_50epochs/batch_normalization_15/ReadVariableOp;^McFly_cnn_50epochs/batch_normalization_15/ReadVariableOp_1J^McFly_cnn_50epochs/batch_normalization_16/FusedBatchNormV3/ReadVariableOpL^McFly_cnn_50epochs/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_19^McFly_cnn_50epochs/batch_normalization_16/ReadVariableOp;^McFly_cnn_50epochs/batch_normalization_16/ReadVariableOp_1J^McFly_cnn_50epochs/batch_normalization_17/FusedBatchNormV3/ReadVariableOpL^McFly_cnn_50epochs/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_19^McFly_cnn_50epochs/batch_normalization_17/ReadVariableOp;^McFly_cnn_50epochs/batch_normalization_17/ReadVariableOp_14^McFly_cnn_50epochs/conv2d_10/BiasAdd/ReadVariableOp3^McFly_cnn_50epochs/conv2d_10/Conv2D/ReadVariableOp4^McFly_cnn_50epochs/conv2d_11/BiasAdd/ReadVariableOp3^McFly_cnn_50epochs/conv2d_11/Conv2D/ReadVariableOp4^McFly_cnn_50epochs/conv2d_12/BiasAdd/ReadVariableOp3^McFly_cnn_50epochs/conv2d_12/Conv2D/ReadVariableOp3^McFly_cnn_50epochs/conv2d_8/BiasAdd/ReadVariableOp2^McFly_cnn_50epochs/conv2d_8/Conv2D/ReadVariableOp3^McFly_cnn_50epochs/conv2d_9/BiasAdd/ReadVariableOp2^McFly_cnn_50epochs/conv2d_9/Conv2D/ReadVariableOp5^McFly_cnn_50epochs/first_conv/BiasAdd/ReadVariableOp4^McFly_cnn_50epochs/first_conv/Conv2D/ReadVariableOp1^McFly_cnn_50epochs/output/BiasAdd/ReadVariableOp:^McFly_cnn_50epochs/output/separable_conv2d/ReadVariableOp<^McFly_cnn_50epochs/output/separable_conv2d/ReadVariableOp_14^McFly_cnn_50epochs/sep_out_0/BiasAdd/ReadVariableOp=^McFly_cnn_50epochs/sep_out_0/separable_conv2d/ReadVariableOp?^McFly_cnn_50epochs/sep_out_0/separable_conv2d/ReadVariableOp_14^McFly_cnn_50epochs/sep_out_1/BiasAdd/ReadVariableOp=^McFly_cnn_50epochs/sep_out_1/separable_conv2d/ReadVariableOp?^McFly_cnn_50epochs/sep_out_1/separable_conv2d/ReadVariableOp_14^McFly_cnn_50epochs/sep_out_2/BiasAdd/ReadVariableOp=^McFly_cnn_50epochs/sep_out_2/separable_conv2d/ReadVariableOp?^McFly_cnn_50epochs/sep_out_2/separable_conv2d/ReadVariableOp_1>^McFly_cnn_50epochs/separable_conv2d_18/BiasAdd/ReadVariableOpG^McFly_cnn_50epochs/separable_conv2d_18/separable_conv2d/ReadVariableOpI^McFly_cnn_50epochs/separable_conv2d_18/separable_conv2d/ReadVariableOp_1>^McFly_cnn_50epochs/separable_conv2d_19/BiasAdd/ReadVariableOpG^McFly_cnn_50epochs/separable_conv2d_19/separable_conv2d/ReadVariableOpI^McFly_cnn_50epochs/separable_conv2d_19/separable_conv2d/ReadVariableOp_1>^McFly_cnn_50epochs/separable_conv2d_20/BiasAdd/ReadVariableOpG^McFly_cnn_50epochs/separable_conv2d_20/separable_conv2d/ReadVariableOpI^McFly_cnn_50epochs/separable_conv2d_20/separable_conv2d/ReadVariableOp_1>^McFly_cnn_50epochs/separable_conv2d_21/BiasAdd/ReadVariableOpG^McFly_cnn_50epochs/separable_conv2d_21/separable_conv2d/ReadVariableOpI^McFly_cnn_50epochs/separable_conv2d_21/separable_conv2d/ReadVariableOp_1>^McFly_cnn_50epochs/separable_conv2d_22/BiasAdd/ReadVariableOpG^McFly_cnn_50epochs/separable_conv2d_22/separable_conv2d/ReadVariableOpI^McFly_cnn_50epochs/separable_conv2d_22/separable_conv2d/ReadVariableOp_1>^McFly_cnn_50epochs/separable_conv2d_23/BiasAdd/ReadVariableOpG^McFly_cnn_50epochs/separable_conv2d_23/separable_conv2d/ReadVariableOpI^McFly_cnn_50epochs/separable_conv2d_23/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¾
_input_shapes¬
©:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
IMcFly_cnn_50epochs/batch_normalization_11/FusedBatchNormV3/ReadVariableOpIMcFly_cnn_50epochs/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2
KMcFly_cnn_50epochs/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1KMcFly_cnn_50epochs/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12t
8McFly_cnn_50epochs/batch_normalization_11/ReadVariableOp8McFly_cnn_50epochs/batch_normalization_11/ReadVariableOp2x
:McFly_cnn_50epochs/batch_normalization_11/ReadVariableOp_1:McFly_cnn_50epochs/batch_normalization_11/ReadVariableOp_12
IMcFly_cnn_50epochs/batch_normalization_12/FusedBatchNormV3/ReadVariableOpIMcFly_cnn_50epochs/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2
KMcFly_cnn_50epochs/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1KMcFly_cnn_50epochs/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12t
8McFly_cnn_50epochs/batch_normalization_12/ReadVariableOp8McFly_cnn_50epochs/batch_normalization_12/ReadVariableOp2x
:McFly_cnn_50epochs/batch_normalization_12/ReadVariableOp_1:McFly_cnn_50epochs/batch_normalization_12/ReadVariableOp_12
IMcFly_cnn_50epochs/batch_normalization_13/FusedBatchNormV3/ReadVariableOpIMcFly_cnn_50epochs/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2
KMcFly_cnn_50epochs/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1KMcFly_cnn_50epochs/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12t
8McFly_cnn_50epochs/batch_normalization_13/ReadVariableOp8McFly_cnn_50epochs/batch_normalization_13/ReadVariableOp2x
:McFly_cnn_50epochs/batch_normalization_13/ReadVariableOp_1:McFly_cnn_50epochs/batch_normalization_13/ReadVariableOp_12
IMcFly_cnn_50epochs/batch_normalization_14/FusedBatchNormV3/ReadVariableOpIMcFly_cnn_50epochs/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2
KMcFly_cnn_50epochs/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1KMcFly_cnn_50epochs/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12t
8McFly_cnn_50epochs/batch_normalization_14/ReadVariableOp8McFly_cnn_50epochs/batch_normalization_14/ReadVariableOp2x
:McFly_cnn_50epochs/batch_normalization_14/ReadVariableOp_1:McFly_cnn_50epochs/batch_normalization_14/ReadVariableOp_12
IMcFly_cnn_50epochs/batch_normalization_15/FusedBatchNormV3/ReadVariableOpIMcFly_cnn_50epochs/batch_normalization_15/FusedBatchNormV3/ReadVariableOp2
KMcFly_cnn_50epochs/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1KMcFly_cnn_50epochs/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12t
8McFly_cnn_50epochs/batch_normalization_15/ReadVariableOp8McFly_cnn_50epochs/batch_normalization_15/ReadVariableOp2x
:McFly_cnn_50epochs/batch_normalization_15/ReadVariableOp_1:McFly_cnn_50epochs/batch_normalization_15/ReadVariableOp_12
IMcFly_cnn_50epochs/batch_normalization_16/FusedBatchNormV3/ReadVariableOpIMcFly_cnn_50epochs/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2
KMcFly_cnn_50epochs/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1KMcFly_cnn_50epochs/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12t
8McFly_cnn_50epochs/batch_normalization_16/ReadVariableOp8McFly_cnn_50epochs/batch_normalization_16/ReadVariableOp2x
:McFly_cnn_50epochs/batch_normalization_16/ReadVariableOp_1:McFly_cnn_50epochs/batch_normalization_16/ReadVariableOp_12
IMcFly_cnn_50epochs/batch_normalization_17/FusedBatchNormV3/ReadVariableOpIMcFly_cnn_50epochs/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2
KMcFly_cnn_50epochs/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1KMcFly_cnn_50epochs/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12t
8McFly_cnn_50epochs/batch_normalization_17/ReadVariableOp8McFly_cnn_50epochs/batch_normalization_17/ReadVariableOp2x
:McFly_cnn_50epochs/batch_normalization_17/ReadVariableOp_1:McFly_cnn_50epochs/batch_normalization_17/ReadVariableOp_12j
3McFly_cnn_50epochs/conv2d_10/BiasAdd/ReadVariableOp3McFly_cnn_50epochs/conv2d_10/BiasAdd/ReadVariableOp2h
2McFly_cnn_50epochs/conv2d_10/Conv2D/ReadVariableOp2McFly_cnn_50epochs/conv2d_10/Conv2D/ReadVariableOp2j
3McFly_cnn_50epochs/conv2d_11/BiasAdd/ReadVariableOp3McFly_cnn_50epochs/conv2d_11/BiasAdd/ReadVariableOp2h
2McFly_cnn_50epochs/conv2d_11/Conv2D/ReadVariableOp2McFly_cnn_50epochs/conv2d_11/Conv2D/ReadVariableOp2j
3McFly_cnn_50epochs/conv2d_12/BiasAdd/ReadVariableOp3McFly_cnn_50epochs/conv2d_12/BiasAdd/ReadVariableOp2h
2McFly_cnn_50epochs/conv2d_12/Conv2D/ReadVariableOp2McFly_cnn_50epochs/conv2d_12/Conv2D/ReadVariableOp2h
2McFly_cnn_50epochs/conv2d_8/BiasAdd/ReadVariableOp2McFly_cnn_50epochs/conv2d_8/BiasAdd/ReadVariableOp2f
1McFly_cnn_50epochs/conv2d_8/Conv2D/ReadVariableOp1McFly_cnn_50epochs/conv2d_8/Conv2D/ReadVariableOp2h
2McFly_cnn_50epochs/conv2d_9/BiasAdd/ReadVariableOp2McFly_cnn_50epochs/conv2d_9/BiasAdd/ReadVariableOp2f
1McFly_cnn_50epochs/conv2d_9/Conv2D/ReadVariableOp1McFly_cnn_50epochs/conv2d_9/Conv2D/ReadVariableOp2l
4McFly_cnn_50epochs/first_conv/BiasAdd/ReadVariableOp4McFly_cnn_50epochs/first_conv/BiasAdd/ReadVariableOp2j
3McFly_cnn_50epochs/first_conv/Conv2D/ReadVariableOp3McFly_cnn_50epochs/first_conv/Conv2D/ReadVariableOp2d
0McFly_cnn_50epochs/output/BiasAdd/ReadVariableOp0McFly_cnn_50epochs/output/BiasAdd/ReadVariableOp2v
9McFly_cnn_50epochs/output/separable_conv2d/ReadVariableOp9McFly_cnn_50epochs/output/separable_conv2d/ReadVariableOp2z
;McFly_cnn_50epochs/output/separable_conv2d/ReadVariableOp_1;McFly_cnn_50epochs/output/separable_conv2d/ReadVariableOp_12j
3McFly_cnn_50epochs/sep_out_0/BiasAdd/ReadVariableOp3McFly_cnn_50epochs/sep_out_0/BiasAdd/ReadVariableOp2|
<McFly_cnn_50epochs/sep_out_0/separable_conv2d/ReadVariableOp<McFly_cnn_50epochs/sep_out_0/separable_conv2d/ReadVariableOp2
>McFly_cnn_50epochs/sep_out_0/separable_conv2d/ReadVariableOp_1>McFly_cnn_50epochs/sep_out_0/separable_conv2d/ReadVariableOp_12j
3McFly_cnn_50epochs/sep_out_1/BiasAdd/ReadVariableOp3McFly_cnn_50epochs/sep_out_1/BiasAdd/ReadVariableOp2|
<McFly_cnn_50epochs/sep_out_1/separable_conv2d/ReadVariableOp<McFly_cnn_50epochs/sep_out_1/separable_conv2d/ReadVariableOp2
>McFly_cnn_50epochs/sep_out_1/separable_conv2d/ReadVariableOp_1>McFly_cnn_50epochs/sep_out_1/separable_conv2d/ReadVariableOp_12j
3McFly_cnn_50epochs/sep_out_2/BiasAdd/ReadVariableOp3McFly_cnn_50epochs/sep_out_2/BiasAdd/ReadVariableOp2|
<McFly_cnn_50epochs/sep_out_2/separable_conv2d/ReadVariableOp<McFly_cnn_50epochs/sep_out_2/separable_conv2d/ReadVariableOp2
>McFly_cnn_50epochs/sep_out_2/separable_conv2d/ReadVariableOp_1>McFly_cnn_50epochs/sep_out_2/separable_conv2d/ReadVariableOp_12~
=McFly_cnn_50epochs/separable_conv2d_18/BiasAdd/ReadVariableOp=McFly_cnn_50epochs/separable_conv2d_18/BiasAdd/ReadVariableOp2
FMcFly_cnn_50epochs/separable_conv2d_18/separable_conv2d/ReadVariableOpFMcFly_cnn_50epochs/separable_conv2d_18/separable_conv2d/ReadVariableOp2
HMcFly_cnn_50epochs/separable_conv2d_18/separable_conv2d/ReadVariableOp_1HMcFly_cnn_50epochs/separable_conv2d_18/separable_conv2d/ReadVariableOp_12~
=McFly_cnn_50epochs/separable_conv2d_19/BiasAdd/ReadVariableOp=McFly_cnn_50epochs/separable_conv2d_19/BiasAdd/ReadVariableOp2
FMcFly_cnn_50epochs/separable_conv2d_19/separable_conv2d/ReadVariableOpFMcFly_cnn_50epochs/separable_conv2d_19/separable_conv2d/ReadVariableOp2
HMcFly_cnn_50epochs/separable_conv2d_19/separable_conv2d/ReadVariableOp_1HMcFly_cnn_50epochs/separable_conv2d_19/separable_conv2d/ReadVariableOp_12~
=McFly_cnn_50epochs/separable_conv2d_20/BiasAdd/ReadVariableOp=McFly_cnn_50epochs/separable_conv2d_20/BiasAdd/ReadVariableOp2
FMcFly_cnn_50epochs/separable_conv2d_20/separable_conv2d/ReadVariableOpFMcFly_cnn_50epochs/separable_conv2d_20/separable_conv2d/ReadVariableOp2
HMcFly_cnn_50epochs/separable_conv2d_20/separable_conv2d/ReadVariableOp_1HMcFly_cnn_50epochs/separable_conv2d_20/separable_conv2d/ReadVariableOp_12~
=McFly_cnn_50epochs/separable_conv2d_21/BiasAdd/ReadVariableOp=McFly_cnn_50epochs/separable_conv2d_21/BiasAdd/ReadVariableOp2
FMcFly_cnn_50epochs/separable_conv2d_21/separable_conv2d/ReadVariableOpFMcFly_cnn_50epochs/separable_conv2d_21/separable_conv2d/ReadVariableOp2
HMcFly_cnn_50epochs/separable_conv2d_21/separable_conv2d/ReadVariableOp_1HMcFly_cnn_50epochs/separable_conv2d_21/separable_conv2d/ReadVariableOp_12~
=McFly_cnn_50epochs/separable_conv2d_22/BiasAdd/ReadVariableOp=McFly_cnn_50epochs/separable_conv2d_22/BiasAdd/ReadVariableOp2
FMcFly_cnn_50epochs/separable_conv2d_22/separable_conv2d/ReadVariableOpFMcFly_cnn_50epochs/separable_conv2d_22/separable_conv2d/ReadVariableOp2
HMcFly_cnn_50epochs/separable_conv2d_22/separable_conv2d/ReadVariableOp_1HMcFly_cnn_50epochs/separable_conv2d_22/separable_conv2d/ReadVariableOp_12~
=McFly_cnn_50epochs/separable_conv2d_23/BiasAdd/ReadVariableOp=McFly_cnn_50epochs/separable_conv2d_23/BiasAdd/ReadVariableOp2
FMcFly_cnn_50epochs/separable_conv2d_23/separable_conv2d/ReadVariableOpFMcFly_cnn_50epochs/separable_conv2d_23/separable_conv2d/ReadVariableOp2
HMcFly_cnn_50epochs/separable_conv2d_23/separable_conv2d/ReadVariableOp_1HMcFly_cnn_50epochs/separable_conv2d_23/separable_conv2d/ReadVariableOp_1:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_6
®

ÿ
D__inference_conv2d_12_layer_call_and_return_conditional_losses_77420

inputs9
conv2d_readvariableop_resource:`.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpph
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿpp`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_13_layer_call_fn_80267

inputs
unknown:	 
	unknown_0:	 
	unknown_1:	 
	unknown_2:	 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_76626
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
!
þ
2__inference_McFly_cnn_50epochs_layer_call_fn_79251

inputs"
unknown:à
	unknown_0:	à
	unknown_1:	à
	unknown_2:	à
	unknown_3:	à
	unknown_4:	à%
	unknown_5:à
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:à

unknown_12:	à&

unknown_13:à 

unknown_14:	 

unknown_15:	 

unknown_16:	 

unknown_17:	 

unknown_18:	 %

unknown_19: `

unknown_20:`

unknown_21:`

unknown_22:`

unknown_23:`

unknown_24:`%

unknown_25:`

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	%

unknown_31:&

unknown_32:À

unknown_33:	À

unknown_34:	À

unknown_35:	À

unknown_36:	À

unknown_37:	À%

unknown_38:À&

unknown_39:Àà

unknown_40:	à%

unknown_41:à&

unknown_42:à

unknown_43:	%

unknown_44:&

unknown_45:

unknown_46:	%

unknown_47:&

unknown_48:à

unknown_49:	à

unknown_50:	à

unknown_51:	à

unknown_52:	à

unknown_53:	à%

unknown_54:à&

unknown_55:à 

unknown_56:	 %

unknown_57: &

unknown_58: À

unknown_59:	À%

unknown_60:À&

unknown_61:Àà

unknown_62:	à%

unknown_63:à&

unknown_64:àà

unknown_65:	à%

unknown_66:à%

unknown_67:à

unknown_68:
identity¢StatefulPartitionedCall

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
unknown_68*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*Z
_read_only_resource_inputs<
:8	
!"#$%()*+,-./01234589:;<=>?@ABCDEF*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_78295w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¾
_input_shapes¬
©:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_80559

inputs
identity¡
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

b
)__inference_dropout_4_layer_call_fn_80532

inputs
identity¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_77814x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_16_layer_call_fn_80598

inputs
unknown:	À
	unknown_0:	À
	unknown_1:	À
	unknown_2:	À
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_76827
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
	
Ñ
3__inference_separable_conv2d_18_layer_call_fn_80570

inputs"
unknown:%
	unknown_0:À
	unknown_1:	À
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_18_layer_call_and_return_conditional_losses_76796
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_80055

inputs&
readvariableop_resource:	à(
readvariableop_1_resource:	à7
(fusedbatchnormv3_readvariableop_resource:	à9
*fusedbatchnormv3_readvariableop_1_resource:	à
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:à*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:à*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:à*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:à*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà:à:à:à:à:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_80138

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_80285

inputs&
readvariableop_resource:	 (
readvariableop_1_resource:	 7
(fusedbatchnormv3_readvariableop_resource:	 9
*fusedbatchnormv3_readvariableop_1_resource:	 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
: *
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
: *
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
!
ÿ
2__inference_McFly_cnn_50epochs_layer_call_fn_77735
input_6"
unknown:à
	unknown_0:	à
	unknown_1:	à
	unknown_2:	à
	unknown_3:	à
	unknown_4:	à%
	unknown_5:à
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:à

unknown_12:	à&

unknown_13:à 

unknown_14:	 

unknown_15:	 

unknown_16:	 

unknown_17:	 

unknown_18:	 %

unknown_19: `

unknown_20:`

unknown_21:`

unknown_22:`

unknown_23:`

unknown_24:`%

unknown_25:`

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	%

unknown_31:&

unknown_32:À

unknown_33:	À

unknown_34:	À

unknown_35:	À

unknown_36:	À

unknown_37:	À%

unknown_38:À&

unknown_39:Àà

unknown_40:	à%

unknown_41:à&

unknown_42:à

unknown_43:	%

unknown_44:&

unknown_45:

unknown_46:	%

unknown_47:&

unknown_48:à

unknown_49:	à

unknown_50:	à

unknown_51:	à

unknown_52:	à

unknown_53:	à%

unknown_54:à&

unknown_55:à 

unknown_56:	 %

unknown_57: &

unknown_58: À

unknown_59:	À%

unknown_60:À&

unknown_61:Àà

unknown_62:	à%

unknown_63:à&

unknown_64:àà

unknown_65:	à%

unknown_66:à%

unknown_67:à

unknown_68:
identity¢StatefulPartitionedCall

StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_68*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*h
_read_only_resource_inputsJ
HF	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEF*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_77592w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¾
_input_shapes¬
©:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_6
ð
d
H__inference_activation_22_layer_call_and_return_conditional_losses_80166

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
ò
Ç
)__inference_sep_out_1_layer_call_fn_80984

inputs"
unknown:À%
	unknown_0:Àà
	unknown_1:	à
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sep_out_1_layer_call_and_return_conditional_losses_77146
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_16_layer_call_fn_80611

inputs
unknown:	À
	unknown_0:	À
	unknown_1:	À
	unknown_2:	À
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_76858
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
ò
Ç
)__inference_sep_out_2_layer_call_fn_81021

inputs"
unknown:à%
	unknown_0:àà
	unknown_1:	à
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sep_out_2_layer_call_and_return_conditional_losses_77187
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_14_layer_call_fn_80372

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_76659
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_76486

inputs&
readvariableop_resource:	à(
readvariableop_1_resource:	à7
(fusedbatchnormv3_readvariableop_resource:	à9
*fusedbatchnormv3_readvariableop_1_resource:	à
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:à*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:à*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:à*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:à*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà:à:à:à:à:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_11_layer_call_fn_80554

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_76774
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
E
)__inference_dropout_2_layer_call_fn_80171

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_77314i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
ð
d
H__inference_activation_31_layer_call_and_return_conditional_losses_77543

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88àc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88à:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_81010

inputs
identity¡
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì

N__inference_separable_conv2d_19_layer_call_and_return_conditional_losses_76888

inputsC
(separable_conv2d_readvariableop_resource:ÀF
*separable_conv2d_readvariableop_1_resource:Àà.
biasadd_readvariableop_resource:	à
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:À*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:Àà*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @     o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ù
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ*
paddingSAME*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
è
Â
&__inference_output_layer_call_fn_81058

inputs"
unknown:à$
	unknown_0:à
	unknown_1:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_77228
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
û
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_77314

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_76455

inputs&
readvariableop_resource:	à(
readvariableop_1_resource:	à7
(fusedbatchnormv3_readvariableop_resource:	à9
*fusedbatchnormv3_readvariableop_1_resource:	à
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:à*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:à*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:à*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:à*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà:à:à:à:à:*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_76562

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
!
þ
2__inference_McFly_cnn_50epochs_layer_call_fn_79106

inputs"
unknown:à
	unknown_0:	à
	unknown_1:	à
	unknown_2:	à
	unknown_3:	à
	unknown_4:	à%
	unknown_5:à
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:à

unknown_12:	à&

unknown_13:à 

unknown_14:	 

unknown_15:	 

unknown_16:	 

unknown_17:	 

unknown_18:	 %

unknown_19: `

unknown_20:`

unknown_21:`

unknown_22:`

unknown_23:`

unknown_24:`%

unknown_25:`

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	%

unknown_31:&

unknown_32:À

unknown_33:	À

unknown_34:	À

unknown_35:	À

unknown_36:	À

unknown_37:	À%

unknown_38:À&

unknown_39:Àà

unknown_40:	à%

unknown_41:à&

unknown_42:à

unknown_43:	%

unknown_44:&

unknown_45:

unknown_46:	%

unknown_47:&

unknown_48:à

unknown_49:	à

unknown_50:	à

unknown_51:	à

unknown_52:	à

unknown_53:	à%

unknown_54:à&

unknown_55:à 

unknown_56:	 %

unknown_57: &

unknown_58: À

unknown_59:	À%

unknown_60:À&

unknown_61:Àà

unknown_62:	à%

unknown_63:à&

unknown_64:àà

unknown_65:	à%

unknown_66:à%

unknown_67:à

unknown_68:
identity¢StatefulPartitionedCall

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
unknown_68*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*h
_read_only_resource_inputsJ
HF	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEF*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_77592w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¾
_input_shapes¬
©:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_12_layer_call_fn_80107

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_76531
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_77376

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
 
_user_specified_nameinputs
Ì
I
-__inference_activation_23_layer_call_fn_80217

inputs
identity¿
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_23_layer_call_and_return_conditional_losses_77337i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿppà:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà
 
_user_specified_nameinputs
ð
d
H__inference_activation_32_layer_call_and_return_conditional_losses_77557

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88 :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_76595

inputs&
readvariableop_resource:	 (
readvariableop_1_resource:	 7
(fusedbatchnormv3_readvariableop_resource:	 9
*fusedbatchnormv3_readvariableop_1_resource:	 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
: *
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
: *
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
º

c
D__inference_dropout_3_layer_call_and_return_conditional_losses_77869

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¯
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
 
_user_specified_nameinputs
Ì

N__inference_separable_conv2d_21_layer_call_and_return_conditional_losses_76944

inputsC
(separable_conv2d_readvariableop_resource:F
*separable_conv2d_readvariableop_1_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ù
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_80494

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì

N__inference_separable_conv2d_22_layer_call_and_return_conditional_losses_76972

inputsC
(separable_conv2d_readvariableop_resource:F
*separable_conv2d_readvariableop_1_resource:à.
biasadd_readvariableop_resource:	à
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:à*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ù
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÈÑ
Ï 
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_78772
input_6+
first_conv_78586:à
first_conv_78588:	à+
batch_normalization_11_78591:	à+
batch_normalization_11_78593:	à+
batch_normalization_11_78595:	à+
batch_normalization_11_78597:	à*
conv2d_8_78602:à
conv2d_8_78604:	+
batch_normalization_12_78607:	+
batch_normalization_12_78609:	+
batch_normalization_12_78611:	+
batch_normalization_12_78613:	*
conv2d_9_78618:à
conv2d_9_78620:	à+
conv2d_10_78624:à 
conv2d_10_78626:	 +
batch_normalization_13_78629:	 +
batch_normalization_13_78631:	 +
batch_normalization_13_78633:	 +
batch_normalization_13_78635:	 *
conv2d_11_78640: `
conv2d_11_78642:`*
batch_normalization_14_78645:`*
batch_normalization_14_78647:`*
batch_normalization_14_78649:`*
batch_normalization_14_78651:`*
conv2d_12_78655:`
conv2d_12_78657:	+
batch_normalization_15_78660:	+
batch_normalization_15_78662:	+
batch_normalization_15_78664:	+
batch_normalization_15_78666:	4
separable_conv2d_18_78672:5
separable_conv2d_18_78674:À(
separable_conv2d_18_78676:	À+
batch_normalization_16_78679:	À+
batch_normalization_16_78681:	À+
batch_normalization_16_78683:	À+
batch_normalization_16_78685:	À4
separable_conv2d_19_78689:À5
separable_conv2d_19_78691:Àà(
separable_conv2d_19_78693:	à4
separable_conv2d_20_78697:à5
separable_conv2d_20_78699:à(
separable_conv2d_20_78701:	4
separable_conv2d_21_78706:5
separable_conv2d_21_78708:(
separable_conv2d_21_78710:	4
separable_conv2d_22_78714:5
separable_conv2d_22_78716:à(
separable_conv2d_22_78718:	à+
batch_normalization_17_78721:	à+
batch_normalization_17_78723:	à+
batch_normalization_17_78725:	à+
batch_normalization_17_78727:	à4
separable_conv2d_23_78731:à5
separable_conv2d_23_78733:à (
separable_conv2d_23_78735:	 *
sep_out_0_78740: +
sep_out_0_78742: À
sep_out_0_78744:	À*
sep_out_1_78748:À+
sep_out_1_78750:Àà
sep_out_1_78752:	à*
sep_out_2_78756:à+
sep_out_2_78758:àà
sep_out_2_78760:	à'
output_78764:à'
output_78766:à
output_78768:
identity¢.batch_normalization_11/StatefulPartitionedCall¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢.batch_normalization_14/StatefulPartitionedCall¢.batch_normalization_15/StatefulPartitionedCall¢.batch_normalization_16/StatefulPartitionedCall¢.batch_normalization_17/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCall¢"first_conv/StatefulPartitionedCall¢output/StatefulPartitionedCall¢!sep_out_0/StatefulPartitionedCall¢!sep_out_1/StatefulPartitionedCall¢!sep_out_2/StatefulPartitionedCall¢+separable_conv2d_18/StatefulPartitionedCall¢+separable_conv2d_19/StatefulPartitionedCall¢+separable_conv2d_20/StatefulPartitionedCall¢+separable_conv2d_21/StatefulPartitionedCall¢+separable_conv2d_22/StatefulPartitionedCall¢+separable_conv2d_23/StatefulPartitionedCall
"first_conv/StatefulPartitionedCallStatefulPartitionedCallinput_6first_conv_78586first_conv_78588*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_first_conv_layer_call_and_return_conditional_losses_77254
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall+first_conv/StatefulPartitionedCall:output:0batch_normalization_11_78591batch_normalization_11_78593batch_normalization_11_78595batch_normalization_11_78597*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_76455
activation_21/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_77274ó
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_76506
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_8_78602conv2d_8_78604*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_77287
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_12_78607batch_normalization_12_78609batch_normalization_12_78611batch_normalization_12_78613*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_76531þ
activation_22/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_22_layer_call_and_return_conditional_losses_77307å
dropout_2/PartitionedCallPartitionedCall&activation_22/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_77314
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv2d_9_78618conv2d_9_78620*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_77326ð
activation_23/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_23_layer_call_and_return_conditional_losses_77337
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0conv2d_10_78624conv2d_10_78626*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_77349
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_13_78629batch_normalization_13_78631batch_normalization_13_78633batch_normalization_13_78635*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_76595þ
activation_24/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_24_layer_call_and_return_conditional_losses_77369å
dropout_3/PartitionedCallPartitionedCall&activation_24/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_77376
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0conv2d_11_78640conv2d_11_78642*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_77388
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_14_78645batch_normalization_14_78647batch_normalization_14_78649batch_normalization_14_78651*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_76659ý
activation_25/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_25_layer_call_and_return_conditional_losses_77408
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall&activation_25/PartitionedCall:output:0conv2d_12_78655conv2d_12_78657*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_77420
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_15_78660batch_normalization_15_78662batch_normalization_15_78664batch_normalization_15_78666*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_76723þ
activation_26/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_26_layer_call_and_return_conditional_losses_77440å
dropout_4/PartitionedCallPartitionedCall&activation_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_77447ï
 max_pooling2d_11/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_76774å
+separable_conv2d_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0separable_conv2d_18_78672separable_conv2d_18_78674separable_conv2d_18_78676*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_18_layer_call_and_return_conditional_losses_76796
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_18/StatefulPartitionedCall:output:0batch_normalization_16_78679batch_normalization_16_78681batch_normalization_16_78683batch_normalization_16_78685*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_76827þ
activation_27/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_27_layer_call_and_return_conditional_losses_77471â
+separable_conv2d_19/StatefulPartitionedCallStatefulPartitionedCall&activation_27/PartitionedCall:output:0separable_conv2d_19_78689separable_conv2d_19_78691separable_conv2d_19_78693*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_19_layer_call_and_return_conditional_losses_76888û
activation_28/PartitionedCallPartitionedCall4separable_conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_28_layer_call_and_return_conditional_losses_77485â
+separable_conv2d_20/StatefulPartitionedCallStatefulPartitionedCall&activation_28/PartitionedCall:output:0separable_conv2d_20_78697separable_conv2d_20_78699separable_conv2d_20_78701*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_20_layer_call_and_return_conditional_losses_76916û
activation_29/PartitionedCallPartitionedCall4separable_conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_29_layer_call_and_return_conditional_losses_77499å
dropout_5/PartitionedCallPartitionedCall&activation_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_77506Þ
+separable_conv2d_21/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0separable_conv2d_21_78706separable_conv2d_21_78708separable_conv2d_21_78710*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_21_layer_call_and_return_conditional_losses_76944û
activation_30/PartitionedCallPartitionedCall4separable_conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_30_layer_call_and_return_conditional_losses_77520â
+separable_conv2d_22/StatefulPartitionedCallStatefulPartitionedCall&activation_30/PartitionedCall:output:0separable_conv2d_22_78714separable_conv2d_22_78716separable_conv2d_22_78718*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_22_layer_call_and_return_conditional_losses_76972
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_22/StatefulPartitionedCall:output:0batch_normalization_17_78721batch_normalization_17_78723batch_normalization_17_78725batch_normalization_17_78727*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_77003þ
activation_31/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_31_layer_call_and_return_conditional_losses_77543â
+separable_conv2d_23/StatefulPartitionedCallStatefulPartitionedCall&activation_31/PartitionedCall:output:0separable_conv2d_23_78731separable_conv2d_23_78733separable_conv2d_23_78735*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_23_layer_call_and_return_conditional_losses_77064û
activation_32/PartitionedCallPartitionedCall4separable_conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_32_layer_call_and_return_conditional_losses_77557ó
 max_pooling2d_12/PartitionedCallPartitionedCall&activation_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_77082³
!sep_out_0/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0sep_out_0_78740sep_out_0_78742sep_out_0_78744*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sep_out_0_layer_call_and_return_conditional_losses_77105÷
 max_pooling2d_13/PartitionedCallPartitionedCall*sep_out_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_77123³
!sep_out_1/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0sep_out_1_78748sep_out_1_78750sep_out_1_78752*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sep_out_1_layer_call_and_return_conditional_losses_77146÷
 max_pooling2d_14/PartitionedCallPartitionedCall*sep_out_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_77164³
!sep_out_2/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_14/PartitionedCall:output:0sep_out_2_78756sep_out_2_78758sep_out_2_78760*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sep_out_2_layer_call_and_return_conditional_losses_77187÷
 max_pooling2d_15/PartitionedCallPartitionedCall*sep_out_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_77205£
output/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0output_78764output_78766output_78768*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_77228~
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall#^first_conv/StatefulPartitionedCall^output/StatefulPartitionedCall"^sep_out_0/StatefulPartitionedCall"^sep_out_1/StatefulPartitionedCall"^sep_out_2/StatefulPartitionedCall,^separable_conv2d_18/StatefulPartitionedCall,^separable_conv2d_19/StatefulPartitionedCall,^separable_conv2d_20/StatefulPartitionedCall,^separable_conv2d_21/StatefulPartitionedCall,^separable_conv2d_22/StatefulPartitionedCall,^separable_conv2d_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¾
_input_shapes¬
©:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2H
"first_conv/StatefulPartitionedCall"first_conv/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!sep_out_0/StatefulPartitionedCall!sep_out_0/StatefulPartitionedCall2F
!sep_out_1/StatefulPartitionedCall!sep_out_1/StatefulPartitionedCall2F
!sep_out_2/StatefulPartitionedCall!sep_out_2/StatefulPartitionedCall2Z
+separable_conv2d_18/StatefulPartitionedCall+separable_conv2d_18/StatefulPartitionedCall2Z
+separable_conv2d_19/StatefulPartitionedCall+separable_conv2d_19/StatefulPartitionedCall2Z
+separable_conv2d_20/StatefulPartitionedCall+separable_conv2d_20/StatefulPartitionedCall2Z
+separable_conv2d_21/StatefulPartitionedCall+separable_conv2d_21/StatefulPartitionedCall2Z
+separable_conv2d_22/StatefulPartitionedCall+separable_conv2d_22/StatefulPartitionedCall2Z
+separable_conv2d_23/StatefulPartitionedCall+separable_conv2d_23/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_6
	
Ñ
3__inference_separable_conv2d_22_layer_call_fn_80803

inputs"
unknown:%
	unknown_0:à
	unknown_1:	à
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_22_layer_call_and_return_conditional_losses_76972
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_10_layer_call_fn_80070

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_76506
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
ê3
!__inference__traced_restore_81551
file_prefix=
"assignvariableop_first_conv_kernel:à1
"assignvariableop_1_first_conv_bias:	à>
/assignvariableop_2_batch_normalization_11_gamma:	à=
.assignvariableop_3_batch_normalization_11_beta:	àD
5assignvariableop_4_batch_normalization_11_moving_mean:	àH
9assignvariableop_5_batch_normalization_11_moving_variance:	à>
"assignvariableop_6_conv2d_8_kernel:à/
 assignvariableop_7_conv2d_8_bias:	>
/assignvariableop_8_batch_normalization_12_gamma:	=
.assignvariableop_9_batch_normalization_12_beta:	E
6assignvariableop_10_batch_normalization_12_moving_mean:	I
:assignvariableop_11_batch_normalization_12_moving_variance:	?
#assignvariableop_12_conv2d_9_kernel:à0
!assignvariableop_13_conv2d_9_bias:	à@
$assignvariableop_14_conv2d_10_kernel:à 1
"assignvariableop_15_conv2d_10_bias:	 ?
0assignvariableop_16_batch_normalization_13_gamma:	 >
/assignvariableop_17_batch_normalization_13_beta:	 E
6assignvariableop_18_batch_normalization_13_moving_mean:	 I
:assignvariableop_19_batch_normalization_13_moving_variance:	 ?
$assignvariableop_20_conv2d_11_kernel: `0
"assignvariableop_21_conv2d_11_bias:`>
0assignvariableop_22_batch_normalization_14_gamma:`=
/assignvariableop_23_batch_normalization_14_beta:`D
6assignvariableop_24_batch_normalization_14_moving_mean:`H
:assignvariableop_25_batch_normalization_14_moving_variance:`?
$assignvariableop_26_conv2d_12_kernel:`1
"assignvariableop_27_conv2d_12_bias:	?
0assignvariableop_28_batch_normalization_15_gamma:	>
/assignvariableop_29_batch_normalization_15_beta:	E
6assignvariableop_30_batch_normalization_15_moving_mean:	I
:assignvariableop_31_batch_normalization_15_moving_variance:	S
8assignvariableop_32_separable_conv2d_18_depthwise_kernel:T
8assignvariableop_33_separable_conv2d_18_pointwise_kernel:À;
,assignvariableop_34_separable_conv2d_18_bias:	À?
0assignvariableop_35_batch_normalization_16_gamma:	À>
/assignvariableop_36_batch_normalization_16_beta:	ÀE
6assignvariableop_37_batch_normalization_16_moving_mean:	ÀI
:assignvariableop_38_batch_normalization_16_moving_variance:	ÀS
8assignvariableop_39_separable_conv2d_19_depthwise_kernel:ÀT
8assignvariableop_40_separable_conv2d_19_pointwise_kernel:Àà;
,assignvariableop_41_separable_conv2d_19_bias:	àS
8assignvariableop_42_separable_conv2d_20_depthwise_kernel:àT
8assignvariableop_43_separable_conv2d_20_pointwise_kernel:à;
,assignvariableop_44_separable_conv2d_20_bias:	S
8assignvariableop_45_separable_conv2d_21_depthwise_kernel:T
8assignvariableop_46_separable_conv2d_21_pointwise_kernel:;
,assignvariableop_47_separable_conv2d_21_bias:	S
8assignvariableop_48_separable_conv2d_22_depthwise_kernel:T
8assignvariableop_49_separable_conv2d_22_pointwise_kernel:à;
,assignvariableop_50_separable_conv2d_22_bias:	à?
0assignvariableop_51_batch_normalization_17_gamma:	à>
/assignvariableop_52_batch_normalization_17_beta:	àE
6assignvariableop_53_batch_normalization_17_moving_mean:	àI
:assignvariableop_54_batch_normalization_17_moving_variance:	àS
8assignvariableop_55_separable_conv2d_23_depthwise_kernel:àT
8assignvariableop_56_separable_conv2d_23_pointwise_kernel:à ;
,assignvariableop_57_separable_conv2d_23_bias:	 I
.assignvariableop_58_sep_out_0_depthwise_kernel: J
.assignvariableop_59_sep_out_0_pointwise_kernel: À1
"assignvariableop_60_sep_out_0_bias:	ÀI
.assignvariableop_61_sep_out_1_depthwise_kernel:ÀJ
.assignvariableop_62_sep_out_1_pointwise_kernel:Àà1
"assignvariableop_63_sep_out_1_bias:	àI
.assignvariableop_64_sep_out_2_depthwise_kernel:àJ
.assignvariableop_65_sep_out_2_pointwise_kernel:àà1
"assignvariableop_66_sep_out_2_bias:	àF
+assignvariableop_67_output_depthwise_kernel:àF
+assignvariableop_68_output_pointwise_kernel:à-
assignvariableop_69_output_bias:#
assignvariableop_70_total: #
assignvariableop_71_count: %
assignvariableop_72_total_1: %
assignvariableop_73_count_1: 
identity_75¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_8¢AssignVariableOp_9Ô#
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*ú"
valueð"Bí"KB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-11/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-11/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-13/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-13/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-15/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-15/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-16/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-16/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-18/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-18/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-19/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-19/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-20/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-20/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-21/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-21/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-22/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-22/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*«
value¡BKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Â
_output_shapes¯
¬:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Y
dtypesO
M2K[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp"assignvariableop_first_conv_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_first_conv_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_11_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_11_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_11_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_11_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_8_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_8_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_12_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_12_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_12_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_12_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_9_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_9_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_10_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_10_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_normalization_13_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_13_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_18AssignVariableOp6assignvariableop_18_batch_normalization_13_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_19AssignVariableOp:assignvariableop_19_batch_normalization_13_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp$assignvariableop_20_conv2d_11_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv2d_11_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_22AssignVariableOp0assignvariableop_22_batch_normalization_14_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_23AssignVariableOp/assignvariableop_23_batch_normalization_14_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_24AssignVariableOp6assignvariableop_24_batch_normalization_14_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_25AssignVariableOp:assignvariableop_25_batch_normalization_14_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_12_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv2d_12_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_28AssignVariableOp0assignvariableop_28_batch_normalization_15_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_29AssignVariableOp/assignvariableop_29_batch_normalization_15_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_30AssignVariableOp6assignvariableop_30_batch_normalization_15_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_31AssignVariableOp:assignvariableop_31_batch_normalization_15_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_32AssignVariableOp8assignvariableop_32_separable_conv2d_18_depthwise_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_33AssignVariableOp8assignvariableop_33_separable_conv2d_18_pointwise_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp,assignvariableop_34_separable_conv2d_18_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_35AssignVariableOp0assignvariableop_35_batch_normalization_16_gammaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_36AssignVariableOp/assignvariableop_36_batch_normalization_16_betaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_37AssignVariableOp6assignvariableop_37_batch_normalization_16_moving_meanIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_38AssignVariableOp:assignvariableop_38_batch_normalization_16_moving_varianceIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_39AssignVariableOp8assignvariableop_39_separable_conv2d_19_depthwise_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_40AssignVariableOp8assignvariableop_40_separable_conv2d_19_pointwise_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp,assignvariableop_41_separable_conv2d_19_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_42AssignVariableOp8assignvariableop_42_separable_conv2d_20_depthwise_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_43AssignVariableOp8assignvariableop_43_separable_conv2d_20_pointwise_kernelIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp,assignvariableop_44_separable_conv2d_20_biasIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_45AssignVariableOp8assignvariableop_45_separable_conv2d_21_depthwise_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_46AssignVariableOp8assignvariableop_46_separable_conv2d_21_pointwise_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp,assignvariableop_47_separable_conv2d_21_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_48AssignVariableOp8assignvariableop_48_separable_conv2d_22_depthwise_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_49AssignVariableOp8assignvariableop_49_separable_conv2d_22_pointwise_kernelIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp,assignvariableop_50_separable_conv2d_22_biasIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_51AssignVariableOp0assignvariableop_51_batch_normalization_17_gammaIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_52AssignVariableOp/assignvariableop_52_batch_normalization_17_betaIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_53AssignVariableOp6assignvariableop_53_batch_normalization_17_moving_meanIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_54AssignVariableOp:assignvariableop_54_batch_normalization_17_moving_varianceIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_55AssignVariableOp8assignvariableop_55_separable_conv2d_23_depthwise_kernelIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_56AssignVariableOp8assignvariableop_56_separable_conv2d_23_pointwise_kernelIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp,assignvariableop_57_separable_conv2d_23_biasIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp.assignvariableop_58_sep_out_0_depthwise_kernelIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_59AssignVariableOp.assignvariableop_59_sep_out_0_pointwise_kernelIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_60AssignVariableOp"assignvariableop_60_sep_out_0_biasIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOp.assignvariableop_61_sep_out_1_depthwise_kernelIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOp.assignvariableop_62_sep_out_1_pointwise_kernelIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOp"assignvariableop_63_sep_out_1_biasIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOp.assignvariableop_64_sep_out_2_depthwise_kernelIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_65AssignVariableOp.assignvariableop_65_sep_out_2_pointwise_kernelIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_66AssignVariableOp"assignvariableop_66_sep_out_2_biasIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_67AssignVariableOp+assignvariableop_67_output_depthwise_kernelIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_68AssignVariableOp+assignvariableop_68_output_pointwise_kernelIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOpassignvariableop_69_output_biasIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOpassignvariableop_70_totalIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOpassignvariableop_71_countIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_72AssignVariableOpassignvariableop_72_total_1Identity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_73AssignVariableOpassignvariableop_73_count_1Identity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 «
Identity_74Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_75IdentityIdentity_74:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_75Identity_75:output:0*«
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_73AssignVariableOp_732(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
û
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_77506

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
º

c
D__inference_dropout_3_layer_call_and_return_conditional_losses_80340

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¯
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
 
_user_specified_nameinputs
ò
Ç
)__inference_sep_out_0_layer_call_fn_80947

inputs"
unknown: %
	unknown_0: À
	unknown_1:	À
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sep_out_0_layer_call_and_return_conditional_losses_77105
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_80862

inputs&
readvariableop_resource:	à(
readvariableop_1_resource:	à7
(fusedbatchnormv3_readvariableop_resource:	à9
*fusedbatchnormv3_readvariableop_1_resource:	à
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:à*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:à*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:à*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:à*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà:à:à:à:à:*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
Ì
I
-__inference_activation_32_layer_call_fn_80921

inputs
identity¿
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_32_layer_call_and_return_conditional_losses_77557i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88 :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 
 
_user_specified_nameinputs
Ì

Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_76659

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`:`:`:`:`:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
Ì
I
-__inference_activation_30_layer_call_fn_80787

inputs
identity¿
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_30_layer_call_and_return_conditional_losses_77520i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
º


E__inference_first_conv_layer_call_and_return_conditional_losses_79993

inputs9
conv2d_readvariableop_resource:à.
biasadd_readvariableop_resource:	à
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿàààj
IdentityIdentityBiasAdd:output:0^NoOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿàààw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
±

ÿ
C__inference_conv2d_9_layer_call_and_return_conditional_losses_77326

inputs:
conv2d_readvariableop_resource:à.
biasadd_readvariableop_resource:	à
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:à*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppàh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppàw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿpp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
¦
ú
A__inference_output_layer_call_and_return_conditional_losses_81074

inputsC
(separable_conv2d_readvariableop_resource:àE
*separable_conv2d_readvariableop_1_resource:à-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*'
_output_shapes
:à*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      à     o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ú
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
ß
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿt
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_77205

inputs
identity¡
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

ÿ
D__inference_conv2d_12_layer_call_and_return_conditional_losses_80450

inputs9
conv2d_readvariableop_resource:`.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpph
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿpp`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
 
_user_specified_nameinputs
Ì

N__inference_separable_conv2d_20_layer_call_and_return_conditional_losses_76916

inputsC
(separable_conv2d_readvariableop_resource:àF
*separable_conv2d_readvariableop_1_resource:à.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:à*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      `     o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ù
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
paddingSAME*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
ÅÑ
Î 
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_77592

inputs+
first_conv_77255:à
first_conv_77257:	à+
batch_normalization_11_77260:	à+
batch_normalization_11_77262:	à+
batch_normalization_11_77264:	à+
batch_normalization_11_77266:	à*
conv2d_8_77288:à
conv2d_8_77290:	+
batch_normalization_12_77293:	+
batch_normalization_12_77295:	+
batch_normalization_12_77297:	+
batch_normalization_12_77299:	*
conv2d_9_77327:à
conv2d_9_77329:	à+
conv2d_10_77350:à 
conv2d_10_77352:	 +
batch_normalization_13_77355:	 +
batch_normalization_13_77357:	 +
batch_normalization_13_77359:	 +
batch_normalization_13_77361:	 *
conv2d_11_77389: `
conv2d_11_77391:`*
batch_normalization_14_77394:`*
batch_normalization_14_77396:`*
batch_normalization_14_77398:`*
batch_normalization_14_77400:`*
conv2d_12_77421:`
conv2d_12_77423:	+
batch_normalization_15_77426:	+
batch_normalization_15_77428:	+
batch_normalization_15_77430:	+
batch_normalization_15_77432:	4
separable_conv2d_18_77450:5
separable_conv2d_18_77452:À(
separable_conv2d_18_77454:	À+
batch_normalization_16_77457:	À+
batch_normalization_16_77459:	À+
batch_normalization_16_77461:	À+
batch_normalization_16_77463:	À4
separable_conv2d_19_77473:À5
separable_conv2d_19_77475:Àà(
separable_conv2d_19_77477:	à4
separable_conv2d_20_77487:à5
separable_conv2d_20_77489:à(
separable_conv2d_20_77491:	4
separable_conv2d_21_77508:5
separable_conv2d_21_77510:(
separable_conv2d_21_77512:	4
separable_conv2d_22_77522:5
separable_conv2d_22_77524:à(
separable_conv2d_22_77526:	à+
batch_normalization_17_77529:	à+
batch_normalization_17_77531:	à+
batch_normalization_17_77533:	à+
batch_normalization_17_77535:	à4
separable_conv2d_23_77545:à5
separable_conv2d_23_77547:à (
separable_conv2d_23_77549:	 *
sep_out_0_77560: +
sep_out_0_77562: À
sep_out_0_77564:	À*
sep_out_1_77568:À+
sep_out_1_77570:Àà
sep_out_1_77572:	à*
sep_out_2_77576:à+
sep_out_2_77578:àà
sep_out_2_77580:	à'
output_77584:à'
output_77586:à
output_77588:
identity¢.batch_normalization_11/StatefulPartitionedCall¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢.batch_normalization_14/StatefulPartitionedCall¢.batch_normalization_15/StatefulPartitionedCall¢.batch_normalization_16/StatefulPartitionedCall¢.batch_normalization_17/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCall¢"first_conv/StatefulPartitionedCall¢output/StatefulPartitionedCall¢!sep_out_0/StatefulPartitionedCall¢!sep_out_1/StatefulPartitionedCall¢!sep_out_2/StatefulPartitionedCall¢+separable_conv2d_18/StatefulPartitionedCall¢+separable_conv2d_19/StatefulPartitionedCall¢+separable_conv2d_20/StatefulPartitionedCall¢+separable_conv2d_21/StatefulPartitionedCall¢+separable_conv2d_22/StatefulPartitionedCall¢+separable_conv2d_23/StatefulPartitionedCall
"first_conv/StatefulPartitionedCallStatefulPartitionedCallinputsfirst_conv_77255first_conv_77257*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_first_conv_layer_call_and_return_conditional_losses_77254
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall+first_conv/StatefulPartitionedCall:output:0batch_normalization_11_77260batch_normalization_11_77262batch_normalization_11_77264batch_normalization_11_77266*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_76455
activation_21/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_77274ó
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_76506
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_8_77288conv2d_8_77290*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_77287
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_12_77293batch_normalization_12_77295batch_normalization_12_77297batch_normalization_12_77299*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_76531þ
activation_22/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_22_layer_call_and_return_conditional_losses_77307å
dropout_2/PartitionedCallPartitionedCall&activation_22/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_77314
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv2d_9_77327conv2d_9_77329*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_77326ð
activation_23/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_23_layer_call_and_return_conditional_losses_77337
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0conv2d_10_77350conv2d_10_77352*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_77349
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_13_77355batch_normalization_13_77357batch_normalization_13_77359batch_normalization_13_77361*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_76595þ
activation_24/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_24_layer_call_and_return_conditional_losses_77369å
dropout_3/PartitionedCallPartitionedCall&activation_24/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_77376
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0conv2d_11_77389conv2d_11_77391*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_77388
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_14_77394batch_normalization_14_77396batch_normalization_14_77398batch_normalization_14_77400*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_76659ý
activation_25/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_25_layer_call_and_return_conditional_losses_77408
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall&activation_25/PartitionedCall:output:0conv2d_12_77421conv2d_12_77423*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_77420
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_15_77426batch_normalization_15_77428batch_normalization_15_77430batch_normalization_15_77432*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_76723þ
activation_26/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_26_layer_call_and_return_conditional_losses_77440å
dropout_4/PartitionedCallPartitionedCall&activation_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_77447ï
 max_pooling2d_11/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_76774å
+separable_conv2d_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0separable_conv2d_18_77450separable_conv2d_18_77452separable_conv2d_18_77454*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_18_layer_call_and_return_conditional_losses_76796
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_18/StatefulPartitionedCall:output:0batch_normalization_16_77457batch_normalization_16_77459batch_normalization_16_77461batch_normalization_16_77463*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_76827þ
activation_27/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_27_layer_call_and_return_conditional_losses_77471â
+separable_conv2d_19/StatefulPartitionedCallStatefulPartitionedCall&activation_27/PartitionedCall:output:0separable_conv2d_19_77473separable_conv2d_19_77475separable_conv2d_19_77477*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_19_layer_call_and_return_conditional_losses_76888û
activation_28/PartitionedCallPartitionedCall4separable_conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_28_layer_call_and_return_conditional_losses_77485â
+separable_conv2d_20/StatefulPartitionedCallStatefulPartitionedCall&activation_28/PartitionedCall:output:0separable_conv2d_20_77487separable_conv2d_20_77489separable_conv2d_20_77491*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_20_layer_call_and_return_conditional_losses_76916û
activation_29/PartitionedCallPartitionedCall4separable_conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_29_layer_call_and_return_conditional_losses_77499å
dropout_5/PartitionedCallPartitionedCall&activation_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_77506Þ
+separable_conv2d_21/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0separable_conv2d_21_77508separable_conv2d_21_77510separable_conv2d_21_77512*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_21_layer_call_and_return_conditional_losses_76944û
activation_30/PartitionedCallPartitionedCall4separable_conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_30_layer_call_and_return_conditional_losses_77520â
+separable_conv2d_22/StatefulPartitionedCallStatefulPartitionedCall&activation_30/PartitionedCall:output:0separable_conv2d_22_77522separable_conv2d_22_77524separable_conv2d_22_77526*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_22_layer_call_and_return_conditional_losses_76972
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_22/StatefulPartitionedCall:output:0batch_normalization_17_77529batch_normalization_17_77531batch_normalization_17_77533batch_normalization_17_77535*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_77003þ
activation_31/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_31_layer_call_and_return_conditional_losses_77543â
+separable_conv2d_23/StatefulPartitionedCallStatefulPartitionedCall&activation_31/PartitionedCall:output:0separable_conv2d_23_77545separable_conv2d_23_77547separable_conv2d_23_77549*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_23_layer_call_and_return_conditional_losses_77064û
activation_32/PartitionedCallPartitionedCall4separable_conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_32_layer_call_and_return_conditional_losses_77557ó
 max_pooling2d_12/PartitionedCallPartitionedCall&activation_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_77082³
!sep_out_0/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0sep_out_0_77560sep_out_0_77562sep_out_0_77564*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sep_out_0_layer_call_and_return_conditional_losses_77105÷
 max_pooling2d_13/PartitionedCallPartitionedCall*sep_out_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_77123³
!sep_out_1/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0sep_out_1_77568sep_out_1_77570sep_out_1_77572*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sep_out_1_layer_call_and_return_conditional_losses_77146÷
 max_pooling2d_14/PartitionedCallPartitionedCall*sep_out_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_77164³
!sep_out_2/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_14/PartitionedCall:output:0sep_out_2_77576sep_out_2_77578sep_out_2_77580*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sep_out_2_layer_call_and_return_conditional_losses_77187÷
 max_pooling2d_15/PartitionedCallPartitionedCall*sep_out_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_77205£
output/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0output_77584output_77586output_77588*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_77228~
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall#^first_conv/StatefulPartitionedCall^output/StatefulPartitionedCall"^sep_out_0/StatefulPartitionedCall"^sep_out_1/StatefulPartitionedCall"^sep_out_2/StatefulPartitionedCall,^separable_conv2d_18/StatefulPartitionedCall,^separable_conv2d_19/StatefulPartitionedCall,^separable_conv2d_20/StatefulPartitionedCall,^separable_conv2d_21/StatefulPartitionedCall,^separable_conv2d_22/StatefulPartitionedCall,^separable_conv2d_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¾
_input_shapes¬
©:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2H
"first_conv/StatefulPartitionedCall"first_conv/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!sep_out_0/StatefulPartitionedCall!sep_out_0/StatefulPartitionedCall2F
!sep_out_1/StatefulPartitionedCall!sep_out_1/StatefulPartitionedCall2F
!sep_out_2/StatefulPartitionedCall!sep_out_2/StatefulPartitionedCall2Z
+separable_conv2d_18/StatefulPartitionedCall+separable_conv2d_18/StatefulPartitionedCall2Z
+separable_conv2d_19/StatefulPartitionedCall+separable_conv2d_19/StatefulPartitionedCall2Z
+separable_conv2d_20/StatefulPartitionedCall+separable_conv2d_20/StatefulPartitionedCall2Z
+separable_conv2d_21/StatefulPartitionedCall+separable_conv2d_21/StatefulPartitionedCall2Z
+separable_conv2d_22/StatefulPartitionedCall+separable_conv2d_22/StatefulPartitionedCall2Z
+separable_conv2d_23/StatefulPartitionedCall+separable_conv2d_23/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_80156

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
I
-__inference_activation_22_layer_call_fn_80161

inputs
identity¿
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_22_layer_call_and_return_conditional_losses_77307i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
Æ×
ß!
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_78961
input_6+
first_conv_78775:à
first_conv_78777:	à+
batch_normalization_11_78780:	à+
batch_normalization_11_78782:	à+
batch_normalization_11_78784:	à+
batch_normalization_11_78786:	à*
conv2d_8_78791:à
conv2d_8_78793:	+
batch_normalization_12_78796:	+
batch_normalization_12_78798:	+
batch_normalization_12_78800:	+
batch_normalization_12_78802:	*
conv2d_9_78807:à
conv2d_9_78809:	à+
conv2d_10_78813:à 
conv2d_10_78815:	 +
batch_normalization_13_78818:	 +
batch_normalization_13_78820:	 +
batch_normalization_13_78822:	 +
batch_normalization_13_78824:	 *
conv2d_11_78829: `
conv2d_11_78831:`*
batch_normalization_14_78834:`*
batch_normalization_14_78836:`*
batch_normalization_14_78838:`*
batch_normalization_14_78840:`*
conv2d_12_78844:`
conv2d_12_78846:	+
batch_normalization_15_78849:	+
batch_normalization_15_78851:	+
batch_normalization_15_78853:	+
batch_normalization_15_78855:	4
separable_conv2d_18_78861:5
separable_conv2d_18_78863:À(
separable_conv2d_18_78865:	À+
batch_normalization_16_78868:	À+
batch_normalization_16_78870:	À+
batch_normalization_16_78872:	À+
batch_normalization_16_78874:	À4
separable_conv2d_19_78878:À5
separable_conv2d_19_78880:Àà(
separable_conv2d_19_78882:	à4
separable_conv2d_20_78886:à5
separable_conv2d_20_78888:à(
separable_conv2d_20_78890:	4
separable_conv2d_21_78895:5
separable_conv2d_21_78897:(
separable_conv2d_21_78899:	4
separable_conv2d_22_78903:5
separable_conv2d_22_78905:à(
separable_conv2d_22_78907:	à+
batch_normalization_17_78910:	à+
batch_normalization_17_78912:	à+
batch_normalization_17_78914:	à+
batch_normalization_17_78916:	à4
separable_conv2d_23_78920:à5
separable_conv2d_23_78922:à (
separable_conv2d_23_78924:	 *
sep_out_0_78929: +
sep_out_0_78931: À
sep_out_0_78933:	À*
sep_out_1_78937:À+
sep_out_1_78939:Àà
sep_out_1_78941:	à*
sep_out_2_78945:à+
sep_out_2_78947:àà
sep_out_2_78949:	à'
output_78953:à'
output_78955:à
output_78957:
identity¢.batch_normalization_11/StatefulPartitionedCall¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢.batch_normalization_14/StatefulPartitionedCall¢.batch_normalization_15/StatefulPartitionedCall¢.batch_normalization_16/StatefulPartitionedCall¢.batch_normalization_17/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall¢"first_conv/StatefulPartitionedCall¢output/StatefulPartitionedCall¢!sep_out_0/StatefulPartitionedCall¢!sep_out_1/StatefulPartitionedCall¢!sep_out_2/StatefulPartitionedCall¢+separable_conv2d_18/StatefulPartitionedCall¢+separable_conv2d_19/StatefulPartitionedCall¢+separable_conv2d_20/StatefulPartitionedCall¢+separable_conv2d_21/StatefulPartitionedCall¢+separable_conv2d_22/StatefulPartitionedCall¢+separable_conv2d_23/StatefulPartitionedCall
"first_conv/StatefulPartitionedCallStatefulPartitionedCallinput_6first_conv_78775first_conv_78777*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_first_conv_layer_call_and_return_conditional_losses_77254
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall+first_conv/StatefulPartitionedCall:output:0batch_normalization_11_78780batch_normalization_11_78782batch_normalization_11_78784batch_normalization_11_78786*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_76486
activation_21/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_77274ó
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_76506
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_8_78791conv2d_8_78793*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_77287
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_12_78796batch_normalization_12_78798batch_normalization_12_78800batch_normalization_12_78802*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_76562þ
activation_22/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_22_layer_call_and_return_conditional_losses_77307õ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_77924
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv2d_9_78807conv2d_9_78809*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_77326ð
activation_23/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_23_layer_call_and_return_conditional_losses_77337
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0conv2d_10_78813conv2d_10_78815*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_77349
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_13_78818batch_normalization_13_78820batch_normalization_13_78822batch_normalization_13_78824*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_76626þ
activation_24/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_24_layer_call_and_return_conditional_losses_77369
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall&activation_24/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_77869 
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0conv2d_11_78829conv2d_11_78831*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_77388
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_14_78834batch_normalization_14_78836batch_normalization_14_78838batch_normalization_14_78840*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_76690ý
activation_25/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_25_layer_call_and_return_conditional_losses_77408
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall&activation_25/PartitionedCall:output:0conv2d_12_78844conv2d_12_78846*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_77420
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_15_78849batch_normalization_15_78851batch_normalization_15_78853batch_normalization_15_78855*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_76754þ
activation_26/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_26_layer_call_and_return_conditional_losses_77440
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall&activation_26/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_77814÷
 max_pooling2d_11/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_76774å
+separable_conv2d_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0separable_conv2d_18_78861separable_conv2d_18_78863separable_conv2d_18_78865*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_18_layer_call_and_return_conditional_losses_76796
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_18/StatefulPartitionedCall:output:0batch_normalization_16_78868batch_normalization_16_78870batch_normalization_16_78872batch_normalization_16_78874*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_76858þ
activation_27/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_27_layer_call_and_return_conditional_losses_77471â
+separable_conv2d_19/StatefulPartitionedCallStatefulPartitionedCall&activation_27/PartitionedCall:output:0separable_conv2d_19_78878separable_conv2d_19_78880separable_conv2d_19_78882*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_19_layer_call_and_return_conditional_losses_76888û
activation_28/PartitionedCallPartitionedCall4separable_conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_28_layer_call_and_return_conditional_losses_77485â
+separable_conv2d_20/StatefulPartitionedCallStatefulPartitionedCall&activation_28/PartitionedCall:output:0separable_conv2d_20_78886separable_conv2d_20_78888separable_conv2d_20_78890*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_20_layer_call_and_return_conditional_losses_76916û
activation_29/PartitionedCallPartitionedCall4separable_conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_29_layer_call_and_return_conditional_losses_77499
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall&activation_29/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_77773æ
+separable_conv2d_21/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0separable_conv2d_21_78895separable_conv2d_21_78897separable_conv2d_21_78899*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_21_layer_call_and_return_conditional_losses_76944û
activation_30/PartitionedCallPartitionedCall4separable_conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_30_layer_call_and_return_conditional_losses_77520â
+separable_conv2d_22/StatefulPartitionedCallStatefulPartitionedCall&activation_30/PartitionedCall:output:0separable_conv2d_22_78903separable_conv2d_22_78905separable_conv2d_22_78907*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_22_layer_call_and_return_conditional_losses_76972
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_22/StatefulPartitionedCall:output:0batch_normalization_17_78910batch_normalization_17_78912batch_normalization_17_78914batch_normalization_17_78916*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_77034þ
activation_31/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_31_layer_call_and_return_conditional_losses_77543â
+separable_conv2d_23/StatefulPartitionedCallStatefulPartitionedCall&activation_31/PartitionedCall:output:0separable_conv2d_23_78920separable_conv2d_23_78922separable_conv2d_23_78924*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_23_layer_call_and_return_conditional_losses_77064û
activation_32/PartitionedCallPartitionedCall4separable_conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_32_layer_call_and_return_conditional_losses_77557ó
 max_pooling2d_12/PartitionedCallPartitionedCall&activation_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_77082³
!sep_out_0/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0sep_out_0_78929sep_out_0_78931sep_out_0_78933*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sep_out_0_layer_call_and_return_conditional_losses_77105÷
 max_pooling2d_13/PartitionedCallPartitionedCall*sep_out_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_77123³
!sep_out_1/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0sep_out_1_78937sep_out_1_78939sep_out_1_78941*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sep_out_1_layer_call_and_return_conditional_losses_77146÷
 max_pooling2d_14/PartitionedCallPartitionedCall*sep_out_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_77164³
!sep_out_2/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_14/PartitionedCall:output:0sep_out_2_78945sep_out_2_78947sep_out_2_78949*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sep_out_2_layer_call_and_return_conditional_losses_77187÷
 max_pooling2d_15/PartitionedCallPartitionedCall*sep_out_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_77205£
output/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0output_78953output_78955output_78957*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_77228~
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥	
NoOpNoOp/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall#^first_conv/StatefulPartitionedCall^output/StatefulPartitionedCall"^sep_out_0/StatefulPartitionedCall"^sep_out_1/StatefulPartitionedCall"^sep_out_2/StatefulPartitionedCall,^separable_conv2d_18/StatefulPartitionedCall,^separable_conv2d_19/StatefulPartitionedCall,^separable_conv2d_20/StatefulPartitionedCall,^separable_conv2d_21/StatefulPartitionedCall,^separable_conv2d_22/StatefulPartitionedCall,^separable_conv2d_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¾
_input_shapes¬
©:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2H
"first_conv/StatefulPartitionedCall"first_conv/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!sep_out_0/StatefulPartitionedCall!sep_out_0/StatefulPartitionedCall2F
!sep_out_1/StatefulPartitionedCall!sep_out_1/StatefulPartitionedCall2F
!sep_out_2/StatefulPartitionedCall!sep_out_2/StatefulPartitionedCall2Z
+separable_conv2d_18/StatefulPartitionedCall+separable_conv2d_18/StatefulPartitionedCall2Z
+separable_conv2d_19/StatefulPartitionedCall+separable_conv2d_19/StatefulPartitionedCall2Z
+separable_conv2d_20/StatefulPartitionedCall+separable_conv2d_20/StatefulPartitionedCall2Z
+separable_conv2d_21/StatefulPartitionedCall+separable_conv2d_21/StatefulPartitionedCall2Z
+separable_conv2d_22/StatefulPartitionedCall+separable_conv2d_22/StatefulPartitionedCall2Z
+separable_conv2d_23/StatefulPartitionedCall+separable_conv2d_23/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_6
Ä
E
)__inference_dropout_5_layer_call_fn_80734

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_77506i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
²
ÿ
D__inference_sep_out_1_layer_call_and_return_conditional_losses_77146

inputsC
(separable_conv2d_readvariableop_resource:ÀF
*separable_conv2d_readvariableop_1_resource:Àà.
biasadd_readvariableop_resource:	à
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:À*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:Àà*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      À     o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ú
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ*
paddingVALID*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs

b
)__inference_dropout_3_layer_call_fn_80323

inputs
identity¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_77869x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
 
_user_specified_nameinputs
û
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_80744

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_77164

inputs
identity¡
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
I
-__inference_activation_26_layer_call_fn_80517

inputs
identity¿
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_26_layer_call_and_return_conditional_losses_77440i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_76774

inputs
identity¡
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð

)__inference_conv2d_11_layer_call_fn_80349

inputs"
unknown: `
	unknown_0:`
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_77388w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿpp : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_80647

inputs&
readvariableop_resource:	À(
readvariableop_1_resource:	À7
(fusedbatchnormv3_readvariableop_resource:	À9
*fusedbatchnormv3_readvariableop_1_resource:	À
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:À*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:À*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:À*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:À*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ:À:À:À:À:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
«

þ
D__inference_conv2d_11_layer_call_and_return_conditional_losses_77388

inputs9
conv2d_readvariableop_resource: `-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: `*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿpp : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_80936

inputs
identity¡
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_14_layer_call_fn_80385

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_76690
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
ð
d
H__inference_activation_27_layer_call_and_return_conditional_losses_77471

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Àc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88À:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_76858

inputs&
readvariableop_resource:	À(
readvariableop_1_resource:	À7
(fusedbatchnormv3_readvariableop_resource:	À9
*fusedbatchnormv3_readvariableop_1_resource:	À
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:À*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:À*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:À*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:À*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ:À:À:À:À:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
«

þ
D__inference_conv2d_11_layer_call_and_return_conditional_losses_80359

inputs9
conv2d_readvariableop_resource: `-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: `*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿpp : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
 
_user_specified_nameinputs
º

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_77924

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¯
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppx
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppb
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_76506

inputs
identity¡
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
d
H__inference_activation_24_layer_call_and_return_conditional_losses_80313

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
 
_user_specified_nameinputs
Ì
I
-__inference_activation_24_layer_call_fn_80308

inputs
identity¿
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_24_layer_call_and_return_conditional_losses_77369i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
 
_user_specified_nameinputs
û
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_77447

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_11_layer_call_fn_80019

inputs
unknown:	à
	unknown_0:	à
	unknown_1:	à
	unknown_2:	à
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_76486
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
Ä
E
)__inference_dropout_4_layer_call_fn_80527

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_77447i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
ð
d
H__inference_activation_24_layer_call_and_return_conditional_losses_77369

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_76626

inputs&
readvariableop_resource:	 (
readvariableop_1_resource:	 7
(fusedbatchnormv3_readvariableop_resource:	 9
*fusedbatchnormv3_readvariableop_1_resource:	 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
: *
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
: *
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ã×
Þ!
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_78295

inputs+
first_conv_78109:à
first_conv_78111:	à+
batch_normalization_11_78114:	à+
batch_normalization_11_78116:	à+
batch_normalization_11_78118:	à+
batch_normalization_11_78120:	à*
conv2d_8_78125:à
conv2d_8_78127:	+
batch_normalization_12_78130:	+
batch_normalization_12_78132:	+
batch_normalization_12_78134:	+
batch_normalization_12_78136:	*
conv2d_9_78141:à
conv2d_9_78143:	à+
conv2d_10_78147:à 
conv2d_10_78149:	 +
batch_normalization_13_78152:	 +
batch_normalization_13_78154:	 +
batch_normalization_13_78156:	 +
batch_normalization_13_78158:	 *
conv2d_11_78163: `
conv2d_11_78165:`*
batch_normalization_14_78168:`*
batch_normalization_14_78170:`*
batch_normalization_14_78172:`*
batch_normalization_14_78174:`*
conv2d_12_78178:`
conv2d_12_78180:	+
batch_normalization_15_78183:	+
batch_normalization_15_78185:	+
batch_normalization_15_78187:	+
batch_normalization_15_78189:	4
separable_conv2d_18_78195:5
separable_conv2d_18_78197:À(
separable_conv2d_18_78199:	À+
batch_normalization_16_78202:	À+
batch_normalization_16_78204:	À+
batch_normalization_16_78206:	À+
batch_normalization_16_78208:	À4
separable_conv2d_19_78212:À5
separable_conv2d_19_78214:Àà(
separable_conv2d_19_78216:	à4
separable_conv2d_20_78220:à5
separable_conv2d_20_78222:à(
separable_conv2d_20_78224:	4
separable_conv2d_21_78229:5
separable_conv2d_21_78231:(
separable_conv2d_21_78233:	4
separable_conv2d_22_78237:5
separable_conv2d_22_78239:à(
separable_conv2d_22_78241:	à+
batch_normalization_17_78244:	à+
batch_normalization_17_78246:	à+
batch_normalization_17_78248:	à+
batch_normalization_17_78250:	à4
separable_conv2d_23_78254:à5
separable_conv2d_23_78256:à (
separable_conv2d_23_78258:	 *
sep_out_0_78263: +
sep_out_0_78265: À
sep_out_0_78267:	À*
sep_out_1_78271:À+
sep_out_1_78273:Àà
sep_out_1_78275:	à*
sep_out_2_78279:à+
sep_out_2_78281:àà
sep_out_2_78283:	à'
output_78287:à'
output_78289:à
output_78291:
identity¢.batch_normalization_11/StatefulPartitionedCall¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢.batch_normalization_14/StatefulPartitionedCall¢.batch_normalization_15/StatefulPartitionedCall¢.batch_normalization_16/StatefulPartitionedCall¢.batch_normalization_17/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall¢"first_conv/StatefulPartitionedCall¢output/StatefulPartitionedCall¢!sep_out_0/StatefulPartitionedCall¢!sep_out_1/StatefulPartitionedCall¢!sep_out_2/StatefulPartitionedCall¢+separable_conv2d_18/StatefulPartitionedCall¢+separable_conv2d_19/StatefulPartitionedCall¢+separable_conv2d_20/StatefulPartitionedCall¢+separable_conv2d_21/StatefulPartitionedCall¢+separable_conv2d_22/StatefulPartitionedCall¢+separable_conv2d_23/StatefulPartitionedCall
"first_conv/StatefulPartitionedCallStatefulPartitionedCallinputsfirst_conv_78109first_conv_78111*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_first_conv_layer_call_and_return_conditional_losses_77254
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall+first_conv/StatefulPartitionedCall:output:0batch_normalization_11_78114batch_normalization_11_78116batch_normalization_11_78118batch_normalization_11_78120*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_76486
activation_21/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_77274ó
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_76506
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_8_78125conv2d_8_78127*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_77287
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_12_78130batch_normalization_12_78132batch_normalization_12_78134batch_normalization_12_78136*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_76562þ
activation_22/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_22_layer_call_and_return_conditional_losses_77307õ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_77924
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv2d_9_78141conv2d_9_78143*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_77326ð
activation_23/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_23_layer_call_and_return_conditional_losses_77337
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0conv2d_10_78147conv2d_10_78149*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_77349
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_13_78152batch_normalization_13_78154batch_normalization_13_78156batch_normalization_13_78158*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_76626þ
activation_24/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_24_layer_call_and_return_conditional_losses_77369
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall&activation_24/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_77869 
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0conv2d_11_78163conv2d_11_78165*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_77388
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_14_78168batch_normalization_14_78170batch_normalization_14_78172batch_normalization_14_78174*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_76690ý
activation_25/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_25_layer_call_and_return_conditional_losses_77408
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall&activation_25/PartitionedCall:output:0conv2d_12_78178conv2d_12_78180*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_77420
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_15_78183batch_normalization_15_78185batch_normalization_15_78187batch_normalization_15_78189*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_76754þ
activation_26/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_26_layer_call_and_return_conditional_losses_77440
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall&activation_26/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_77814÷
 max_pooling2d_11/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_76774å
+separable_conv2d_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0separable_conv2d_18_78195separable_conv2d_18_78197separable_conv2d_18_78199*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_18_layer_call_and_return_conditional_losses_76796
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_18/StatefulPartitionedCall:output:0batch_normalization_16_78202batch_normalization_16_78204batch_normalization_16_78206batch_normalization_16_78208*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_76858þ
activation_27/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_27_layer_call_and_return_conditional_losses_77471â
+separable_conv2d_19/StatefulPartitionedCallStatefulPartitionedCall&activation_27/PartitionedCall:output:0separable_conv2d_19_78212separable_conv2d_19_78214separable_conv2d_19_78216*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_19_layer_call_and_return_conditional_losses_76888û
activation_28/PartitionedCallPartitionedCall4separable_conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_28_layer_call_and_return_conditional_losses_77485â
+separable_conv2d_20/StatefulPartitionedCallStatefulPartitionedCall&activation_28/PartitionedCall:output:0separable_conv2d_20_78220separable_conv2d_20_78222separable_conv2d_20_78224*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_20_layer_call_and_return_conditional_losses_76916û
activation_29/PartitionedCallPartitionedCall4separable_conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_29_layer_call_and_return_conditional_losses_77499
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall&activation_29/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_77773æ
+separable_conv2d_21/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0separable_conv2d_21_78229separable_conv2d_21_78231separable_conv2d_21_78233*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_21_layer_call_and_return_conditional_losses_76944û
activation_30/PartitionedCallPartitionedCall4separable_conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_30_layer_call_and_return_conditional_losses_77520â
+separable_conv2d_22/StatefulPartitionedCallStatefulPartitionedCall&activation_30/PartitionedCall:output:0separable_conv2d_22_78237separable_conv2d_22_78239separable_conv2d_22_78241*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_22_layer_call_and_return_conditional_losses_76972
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_22/StatefulPartitionedCall:output:0batch_normalization_17_78244batch_normalization_17_78246batch_normalization_17_78248batch_normalization_17_78250*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_77034þ
activation_31/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_31_layer_call_and_return_conditional_losses_77543â
+separable_conv2d_23/StatefulPartitionedCallStatefulPartitionedCall&activation_31/PartitionedCall:output:0separable_conv2d_23_78254separable_conv2d_23_78256separable_conv2d_23_78258*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_23_layer_call_and_return_conditional_losses_77064û
activation_32/PartitionedCallPartitionedCall4separable_conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_32_layer_call_and_return_conditional_losses_77557ó
 max_pooling2d_12/PartitionedCallPartitionedCall&activation_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_77082³
!sep_out_0/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0sep_out_0_78263sep_out_0_78265sep_out_0_78267*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sep_out_0_layer_call_and_return_conditional_losses_77105÷
 max_pooling2d_13/PartitionedCallPartitionedCall*sep_out_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_77123³
!sep_out_1/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0sep_out_1_78271sep_out_1_78273sep_out_1_78275*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sep_out_1_layer_call_and_return_conditional_losses_77146÷
 max_pooling2d_14/PartitionedCallPartitionedCall*sep_out_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_77164³
!sep_out_2/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_14/PartitionedCall:output:0sep_out_2_78279sep_out_2_78281sep_out_2_78283*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sep_out_2_layer_call_and_return_conditional_losses_77187÷
 max_pooling2d_15/PartitionedCallPartitionedCall*sep_out_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_77205£
output/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0output_78287output_78289output_78291*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_77228~
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥	
NoOpNoOp/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall#^first_conv/StatefulPartitionedCall^output/StatefulPartitionedCall"^sep_out_0/StatefulPartitionedCall"^sep_out_1/StatefulPartitionedCall"^sep_out_2/StatefulPartitionedCall,^separable_conv2d_18/StatefulPartitionedCall,^separable_conv2d_19/StatefulPartitionedCall,^separable_conv2d_20/StatefulPartitionedCall,^separable_conv2d_21/StatefulPartitionedCall,^separable_conv2d_22/StatefulPartitionedCall,^separable_conv2d_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¾
_input_shapes¬
©:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2H
"first_conv/StatefulPartitionedCall"first_conv/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!sep_out_0/StatefulPartitionedCall!sep_out_0/StatefulPartitionedCall2F
!sep_out_1/StatefulPartitionedCall!sep_out_1/StatefulPartitionedCall2F
!sep_out_2/StatefulPartitionedCall!sep_out_2/StatefulPartitionedCall2Z
+separable_conv2d_18/StatefulPartitionedCall+separable_conv2d_18/StatefulPartitionedCall2Z
+separable_conv2d_19/StatefulPartitionedCall+separable_conv2d_19/StatefulPartitionedCall2Z
+separable_conv2d_20/StatefulPartitionedCall+separable_conv2d_20/StatefulPartitionedCall2Z
+separable_conv2d_21/StatefulPartitionedCall+separable_conv2d_21/StatefulPartitionedCall2Z
+separable_conv2d_22/StatefulPartitionedCall+separable_conv2d_22/StatefulPartitionedCall2Z
+separable_conv2d_23/StatefulPartitionedCall+separable_conv2d_23/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
Ì
I
-__inference_activation_27_layer_call_fn_80652

inputs
identity¿
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_27_layer_call_and_return_conditional_losses_77471i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88À:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À
 
_user_specified_nameinputs
ì
d
H__inference_activation_25_layer_call_and_return_conditional_losses_80431

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp`:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
 
_user_specified_nameinputs
ð
d
H__inference_activation_26_layer_call_and_return_conditional_losses_80522

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_77082

inputs
identity¡
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
d
H__inference_activation_27_layer_call_and_return_conditional_losses_80657

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Àc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88À:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À
 
_user_specified_nameinputs

b
)__inference_dropout_5_layer_call_fn_80739

inputs
identity¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_77773x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ8822
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
Û 
ð
#__inference_signature_wrapper_79974
input_6"
unknown:à
	unknown_0:	à
	unknown_1:	à
	unknown_2:	à
	unknown_3:	à
	unknown_4:	à%
	unknown_5:à
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:à

unknown_12:	à&

unknown_13:à 

unknown_14:	 

unknown_15:	 

unknown_16:	 

unknown_17:	 

unknown_18:	 %

unknown_19: `

unknown_20:`

unknown_21:`

unknown_22:`

unknown_23:`

unknown_24:`%

unknown_25:`

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	%

unknown_31:&

unknown_32:À

unknown_33:	À

unknown_34:	À

unknown_35:	À

unknown_36:	À

unknown_37:	À%

unknown_38:À&

unknown_39:Àà

unknown_40:	à%

unknown_41:à&

unknown_42:à

unknown_43:	%

unknown_44:&

unknown_45:

unknown_46:	%

unknown_47:&

unknown_48:à

unknown_49:	à

unknown_50:	à

unknown_51:	à

unknown_52:	à

unknown_53:	à%

unknown_54:à&

unknown_55:à 

unknown_56:	 %

unknown_57: &

unknown_58: À

unknown_59:	À%

unknown_60:À&

unknown_61:Àà

unknown_62:	à%

unknown_63:à&

unknown_64:àà

unknown_65:	à%

unknown_66:à%

unknown_67:à

unknown_68:
identity¢StatefulPartitionedCallð	
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_68*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*h
_read_only_resource_inputsJ
HF	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEF*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_76433w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¾
_input_shapes¬
©:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_6

g
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_80973

inputs
identity¡
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì

N__inference_separable_conv2d_22_layer_call_and_return_conditional_losses_80818

inputsC
(separable_conv2d_readvariableop_resource:F
*separable_conv2d_readvariableop_1_resource:à.
biasadd_readvariableop_resource:	à
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:à*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ù
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ñ
3__inference_separable_conv2d_19_layer_call_fn_80668

inputs"
unknown:À%
	unknown_0:Àà
	unknown_1:	à
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_19_layer_call_and_return_conditional_losses_76888
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
º

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_80549

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¯
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppx
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppb
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
ð
d
H__inference_activation_32_layer_call_and_return_conditional_losses_80926

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88 :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 
 
_user_specified_nameinputs
Ì
I
-__inference_activation_29_layer_call_fn_80724

inputs
identity¿
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_29_layer_call_and_return_conditional_losses_77499i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
¦
ú
A__inference_output_layer_call_and_return_conditional_losses_77228

inputsC
(separable_conv2d_readvariableop_resource:àE
*separable_conv2d_readvariableop_1_resource:à-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*'
_output_shapes
:à*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      à     o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ú
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
ß
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿt
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
ð
d
H__inference_activation_28_layer_call_and_return_conditional_losses_80693

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88àc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88à:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à
 
_user_specified_nameinputs
º

c
D__inference_dropout_5_layer_call_and_return_conditional_losses_77773

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¯
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_80075

inputs
identity¡
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_77003

inputs&
readvariableop_resource:	à(
readvariableop_1_resource:	à7
(fusedbatchnormv3_readvariableop_resource:	à9
*fusedbatchnormv3_readvariableop_1_resource:	à
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:à*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:à*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:à*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:à*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà:à:à:à:à:*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
²


D__inference_conv2d_10_layer_call_and_return_conditional_losses_80241

inputs:
conv2d_readvariableop_resource:à .
biasadd_readvariableop_resource:	 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:à *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp *
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿppà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_76754

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì

N__inference_separable_conv2d_20_layer_call_and_return_conditional_losses_80719

inputsC
(separable_conv2d_readvariableop_resource:àF
*separable_conv2d_readvariableop_1_resource:à.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:à*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      `     o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ù
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
paddingSAME*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
Ì
I
-__inference_activation_31_layer_call_fn_80885

inputs
identity¿
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_31_layer_call_and_return_conditional_losses_77543i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88à:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à
 
_user_specified_nameinputs
ñ
 
)__inference_conv2d_12_layer_call_fn_80440

inputs"
unknown:`
	unknown_0:	
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_77420x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿpp`: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
 
_user_specified_nameinputs
ð
d
H__inference_activation_23_layer_call_and_return_conditional_losses_77337

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppàc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿppà:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà
 
_user_specified_nameinputs
ò
 
(__inference_conv2d_9_layer_call_fn_80202

inputs#
unknown:à
	unknown_0:	à
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_77326x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿpp: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_76723

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

b
)__inference_dropout_2_layer_call_fn_80176

inputs
identity¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_77924x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
û
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_80181

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_14_layer_call_fn_81005

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_77164
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
ÿ
D__inference_sep_out_0_layer_call_and_return_conditional_losses_77105

inputsC
(separable_conv2d_readvariableop_resource: F
*separable_conv2d_readvariableop_1_resource: À.
biasadd_readvariableop_resource:	À
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
: À*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ú
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:À*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
È
I
-__inference_activation_25_layer_call_fn_80426

inputs
identity¾
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_25_layer_call_and_return_conditional_losses_77408h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp`:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_80880

inputs&
readvariableop_resource:	à(
readvariableop_1_resource:	à7
(fusedbatchnormv3_readvariableop_resource:	à9
*fusedbatchnormv3_readvariableop_1_resource:	à
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:à*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:à*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:à*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:à*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà:à:à:à:à:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
ô
¡
)__inference_conv2d_10_layer_call_fn_80231

inputs#
unknown:à 
	unknown_0:	 
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_77349x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿppà: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà
 
_user_specified_nameinputs
¥
÷"
__inference__traced_save_81319
file_prefix0
,savev2_first_conv_kernel_read_readvariableop.
*savev2_first_conv_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop;
7savev2_batch_normalization_14_gamma_read_readvariableop:
6savev2_batch_normalization_14_beta_read_readvariableopA
=savev2_batch_normalization_14_moving_mean_read_readvariableopE
Asavev2_batch_normalization_14_moving_variance_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop;
7savev2_batch_normalization_15_gamma_read_readvariableop:
6savev2_batch_normalization_15_beta_read_readvariableopA
=savev2_batch_normalization_15_moving_mean_read_readvariableopE
Asavev2_batch_normalization_15_moving_variance_read_readvariableopC
?savev2_separable_conv2d_18_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_18_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_18_bias_read_readvariableop;
7savev2_batch_normalization_16_gamma_read_readvariableop:
6savev2_batch_normalization_16_beta_read_readvariableopA
=savev2_batch_normalization_16_moving_mean_read_readvariableopE
Asavev2_batch_normalization_16_moving_variance_read_readvariableopC
?savev2_separable_conv2d_19_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_19_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_19_bias_read_readvariableopC
?savev2_separable_conv2d_20_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_20_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_20_bias_read_readvariableopC
?savev2_separable_conv2d_21_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_21_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_21_bias_read_readvariableopC
?savev2_separable_conv2d_22_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_22_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_22_bias_read_readvariableop;
7savev2_batch_normalization_17_gamma_read_readvariableop:
6savev2_batch_normalization_17_beta_read_readvariableopA
=savev2_batch_normalization_17_moving_mean_read_readvariableopE
Asavev2_batch_normalization_17_moving_variance_read_readvariableopC
?savev2_separable_conv2d_23_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_23_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_23_bias_read_readvariableop9
5savev2_sep_out_0_depthwise_kernel_read_readvariableop9
5savev2_sep_out_0_pointwise_kernel_read_readvariableop-
)savev2_sep_out_0_bias_read_readvariableop9
5savev2_sep_out_1_depthwise_kernel_read_readvariableop9
5savev2_sep_out_1_pointwise_kernel_read_readvariableop-
)savev2_sep_out_1_bias_read_readvariableop9
5savev2_sep_out_2_depthwise_kernel_read_readvariableop9
5savev2_sep_out_2_pointwise_kernel_read_readvariableop-
)savev2_sep_out_2_bias_read_readvariableop6
2savev2_output_depthwise_kernel_read_readvariableop6
2savev2_output_pointwise_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ñ#
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*ú"
valueð"Bí"KB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-11/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-11/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-13/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-13/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-15/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-15/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-16/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-16/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-18/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-18/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-19/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-19/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-20/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-20/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-21/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-21/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-22/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-22/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*«
value¡BKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Þ!
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_first_conv_kernel_read_readvariableop*savev2_first_conv_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop7savev2_batch_normalization_14_gamma_read_readvariableop6savev2_batch_normalization_14_beta_read_readvariableop=savev2_batch_normalization_14_moving_mean_read_readvariableopAsavev2_batch_normalization_14_moving_variance_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop7savev2_batch_normalization_15_gamma_read_readvariableop6savev2_batch_normalization_15_beta_read_readvariableop=savev2_batch_normalization_15_moving_mean_read_readvariableopAsavev2_batch_normalization_15_moving_variance_read_readvariableop?savev2_separable_conv2d_18_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_18_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_18_bias_read_readvariableop7savev2_batch_normalization_16_gamma_read_readvariableop6savev2_batch_normalization_16_beta_read_readvariableop=savev2_batch_normalization_16_moving_mean_read_readvariableopAsavev2_batch_normalization_16_moving_variance_read_readvariableop?savev2_separable_conv2d_19_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_19_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_19_bias_read_readvariableop?savev2_separable_conv2d_20_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_20_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_20_bias_read_readvariableop?savev2_separable_conv2d_21_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_21_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_21_bias_read_readvariableop?savev2_separable_conv2d_22_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_22_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_22_bias_read_readvariableop7savev2_batch_normalization_17_gamma_read_readvariableop6savev2_batch_normalization_17_beta_read_readvariableop=savev2_batch_normalization_17_moving_mean_read_readvariableopAsavev2_batch_normalization_17_moving_variance_read_readvariableop?savev2_separable_conv2d_23_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_23_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_23_bias_read_readvariableop5savev2_sep_out_0_depthwise_kernel_read_readvariableop5savev2_sep_out_0_pointwise_kernel_read_readvariableop)savev2_sep_out_0_bias_read_readvariableop5savev2_sep_out_1_depthwise_kernel_read_readvariableop5savev2_sep_out_1_pointwise_kernel_read_readvariableop)savev2_sep_out_1_bias_read_readvariableop5savev2_sep_out_2_depthwise_kernel_read_readvariableop5savev2_sep_out_2_pointwise_kernel_read_readvariableop)savev2_sep_out_2_bias_read_readvariableop2savev2_output_depthwise_kernel_read_readvariableop2savev2_output_pointwise_kernel_read_readvariableop&savev2_output_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Y
dtypesO
M2K
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*É
_input_shapes·
´: :à:à:à:à:à:à:à::::::à:à:à : : : : : : `:`:`:`:`:`:`:::::::À:À:À:À:À:À:À:Àà:à:à:à::::::à:à:à:à:à:à:à:à : : : À:À:À:Àà:à:à:àà:à:à:à:: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:à:!

_output_shapes	
:à:!

_output_shapes	
:à:!

_output_shapes	
:à:!

_output_shapes	
:à:!

_output_shapes	
:à:.*
(
_output_shapes
:à:!

_output_shapes	
::!	

_output_shapes	
::!


_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
:à:!

_output_shapes	
:à:.*
(
_output_shapes
:à :!

_output_shapes	
: :!

_output_shapes	
: :!

_output_shapes	
: :!

_output_shapes	
: :!

_output_shapes	
: :-)
'
_output_shapes
: `: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`:-)
'
_output_shapes
:`:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::! 

_output_shapes	
::-!)
'
_output_shapes
::."*
(
_output_shapes
:À:!#

_output_shapes	
:À:!$

_output_shapes	
:À:!%

_output_shapes	
:À:!&

_output_shapes	
:À:!'

_output_shapes	
:À:-()
'
_output_shapes
:À:.)*
(
_output_shapes
:Àà:!*

_output_shapes	
:à:-+)
'
_output_shapes
:à:.,*
(
_output_shapes
:à:!-

_output_shapes	
::-.)
'
_output_shapes
::./*
(
_output_shapes
::!0

_output_shapes	
::-1)
'
_output_shapes
::.2*
(
_output_shapes
:à:!3

_output_shapes	
:à:!4

_output_shapes	
:à:!5

_output_shapes	
:à:!6

_output_shapes	
:à:!7

_output_shapes	
:à:-8)
'
_output_shapes
:à:.9*
(
_output_shapes
:à :!:

_output_shapes	
: :-;)
'
_output_shapes
: :.<*
(
_output_shapes
: À:!=

_output_shapes	
:À:->)
'
_output_shapes
:À:.?*
(
_output_shapes
:Àà:!@

_output_shapes	
:à:-A)
'
_output_shapes
:à:.B*
(
_output_shapes
:àà:!C

_output_shapes	
:à:-D)
'
_output_shapes
:à:-E)
'
_output_shapes
:à: F

_output_shapes
::G

_output_shapes
: :H

_output_shapes
: :I

_output_shapes
: :J

_output_shapes
: :K

_output_shapes
: 
	
Ñ
3__inference_separable_conv2d_21_layer_call_fn_80767

inputs"
unknown:%
	unknown_0:
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_21_layer_call_and_return_conditional_losses_76944
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ñ
3__inference_separable_conv2d_20_layer_call_fn_80704

inputs"
unknown:à%
	unknown_0:à
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_separable_conv2d_20_layer_call_and_return_conditional_losses_76916
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_15_layer_call_fn_81042

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_77205
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_80037

inputs&
readvariableop_resource:	à(
readvariableop_1_resource:	à7
(fusedbatchnormv3_readvariableop_resource:	à9
*fusedbatchnormv3_readvariableop_1_resource:	à
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:à*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:à*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:à*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:à*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà:à:à:à:à:*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
Ì

N__inference_separable_conv2d_18_layer_call_and_return_conditional_losses_76796

inputsC
(separable_conv2d_readvariableop_resource:F
*separable_conv2d_readvariableop_1_resource:À.
biasadd_readvariableop_resource:	À
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:À*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ù
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:À*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
I
-__inference_activation_21_layer_call_fn_80060

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_77274k
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿààà:Z V
2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà
 
_user_specified_nameinputs
²


D__inference_conv2d_10_layer_call_and_return_conditional_losses_77349

inputs:
conv2d_readvariableop_resource:à .
biasadd_readvariableop_resource:	 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:à *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp *
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿppà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà
 
_user_specified_nameinputs
ð
d
H__inference_activation_23_layer_call_and_return_conditional_losses_80222

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppàc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿppà:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà
 
_user_specified_nameinputs
ò
 
(__inference_conv2d_8_layer_call_fn_80084

inputs#
unknown:à
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_77287x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿppà: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_76531

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
ÿ
D__inference_sep_out_2_layer_call_and_return_conditional_losses_77187

inputsC
(separable_conv2d_readvariableop_resource:àF
*separable_conv2d_readvariableop_1_resource:àà.
biasadd_readvariableop_resource:	à
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:àà*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      à      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ú
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
Ì

N__inference_separable_conv2d_23_layer_call_and_return_conditional_losses_77064

inputsC
(separable_conv2d_readvariableop_resource:àF
*separable_conv2d_readvariableop_1_resource:à .
biasadd_readvariableop_resource:	 
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:à *
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      `     o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ù
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
paddingSAME*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
: *
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_17_layer_call_fn_80831

inputs
unknown:	à
	unknown_0:	à
	unknown_1:	à
	unknown_2:	à
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_77003
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_11_layer_call_fn_80006

inputs
unknown:	à
	unknown_0:	à
	unknown_1:	à
	unknown_2:	à
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_76455
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
ü
B
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_79525

inputsD
)first_conv_conv2d_readvariableop_resource:à9
*first_conv_biasadd_readvariableop_resource:	à=
.batch_normalization_11_readvariableop_resource:	à?
0batch_normalization_11_readvariableop_1_resource:	àN
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	àP
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	àC
'conv2d_8_conv2d_readvariableop_resource:à7
(conv2d_8_biasadd_readvariableop_resource:	=
.batch_normalization_12_readvariableop_resource:	?
0batch_normalization_12_readvariableop_1_resource:	N
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_9_conv2d_readvariableop_resource:à7
(conv2d_9_biasadd_readvariableop_resource:	àD
(conv2d_10_conv2d_readvariableop_resource:à 8
)conv2d_10_biasadd_readvariableop_resource:	 =
.batch_normalization_13_readvariableop_resource:	 ?
0batch_normalization_13_readvariableop_1_resource:	 N
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	 P
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	 C
(conv2d_11_conv2d_readvariableop_resource: `7
)conv2d_11_biasadd_readvariableop_resource:`<
.batch_normalization_14_readvariableop_resource:`>
0batch_normalization_14_readvariableop_1_resource:`M
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:`O
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:`C
(conv2d_12_conv2d_readvariableop_resource:`8
)conv2d_12_biasadd_readvariableop_resource:	=
.batch_normalization_15_readvariableop_resource:	?
0batch_normalization_15_readvariableop_1_resource:	N
?batch_normalization_15_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource:	W
<separable_conv2d_18_separable_conv2d_readvariableop_resource:Z
>separable_conv2d_18_separable_conv2d_readvariableop_1_resource:ÀB
3separable_conv2d_18_biasadd_readvariableop_resource:	À=
.batch_normalization_16_readvariableop_resource:	À?
0batch_normalization_16_readvariableop_1_resource:	ÀN
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:	ÀP
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:	ÀW
<separable_conv2d_19_separable_conv2d_readvariableop_resource:ÀZ
>separable_conv2d_19_separable_conv2d_readvariableop_1_resource:ÀàB
3separable_conv2d_19_biasadd_readvariableop_resource:	àW
<separable_conv2d_20_separable_conv2d_readvariableop_resource:àZ
>separable_conv2d_20_separable_conv2d_readvariableop_1_resource:àB
3separable_conv2d_20_biasadd_readvariableop_resource:	W
<separable_conv2d_21_separable_conv2d_readvariableop_resource:Z
>separable_conv2d_21_separable_conv2d_readvariableop_1_resource:B
3separable_conv2d_21_biasadd_readvariableop_resource:	W
<separable_conv2d_22_separable_conv2d_readvariableop_resource:Z
>separable_conv2d_22_separable_conv2d_readvariableop_1_resource:àB
3separable_conv2d_22_biasadd_readvariableop_resource:	à=
.batch_normalization_17_readvariableop_resource:	à?
0batch_normalization_17_readvariableop_1_resource:	àN
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	àP
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	àW
<separable_conv2d_23_separable_conv2d_readvariableop_resource:àZ
>separable_conv2d_23_separable_conv2d_readvariableop_1_resource:à B
3separable_conv2d_23_biasadd_readvariableop_resource:	 M
2sep_out_0_separable_conv2d_readvariableop_resource: P
4sep_out_0_separable_conv2d_readvariableop_1_resource: À8
)sep_out_0_biasadd_readvariableop_resource:	ÀM
2sep_out_1_separable_conv2d_readvariableop_resource:ÀP
4sep_out_1_separable_conv2d_readvariableop_1_resource:Àà8
)sep_out_1_biasadd_readvariableop_resource:	àM
2sep_out_2_separable_conv2d_readvariableop_resource:àP
4sep_out_2_separable_conv2d_readvariableop_1_resource:àà8
)sep_out_2_biasadd_readvariableop_resource:	àJ
/output_separable_conv2d_readvariableop_resource:àL
1output_separable_conv2d_readvariableop_1_resource:à4
&output_biasadd_readvariableop_resource:
identity¢6batch_normalization_11/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_11/ReadVariableOp¢'batch_normalization_11/ReadVariableOp_1¢6batch_normalization_12/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_12/ReadVariableOp¢'batch_normalization_12/ReadVariableOp_1¢6batch_normalization_13/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_13/ReadVariableOp¢'batch_normalization_13/ReadVariableOp_1¢6batch_normalization_14/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_14/ReadVariableOp¢'batch_normalization_14/ReadVariableOp_1¢6batch_normalization_15/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_15/ReadVariableOp¢'batch_normalization_15/ReadVariableOp_1¢6batch_normalization_16/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_16/ReadVariableOp¢'batch_normalization_16/ReadVariableOp_1¢6batch_normalization_17/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_17/ReadVariableOp¢'batch_normalization_17/ReadVariableOp_1¢ conv2d_10/BiasAdd/ReadVariableOp¢conv2d_10/Conv2D/ReadVariableOp¢ conv2d_11/BiasAdd/ReadVariableOp¢conv2d_11/Conv2D/ReadVariableOp¢ conv2d_12/BiasAdd/ReadVariableOp¢conv2d_12/Conv2D/ReadVariableOp¢conv2d_8/BiasAdd/ReadVariableOp¢conv2d_8/Conv2D/ReadVariableOp¢conv2d_9/BiasAdd/ReadVariableOp¢conv2d_9/Conv2D/ReadVariableOp¢!first_conv/BiasAdd/ReadVariableOp¢ first_conv/Conv2D/ReadVariableOp¢output/BiasAdd/ReadVariableOp¢&output/separable_conv2d/ReadVariableOp¢(output/separable_conv2d/ReadVariableOp_1¢ sep_out_0/BiasAdd/ReadVariableOp¢)sep_out_0/separable_conv2d/ReadVariableOp¢+sep_out_0/separable_conv2d/ReadVariableOp_1¢ sep_out_1/BiasAdd/ReadVariableOp¢)sep_out_1/separable_conv2d/ReadVariableOp¢+sep_out_1/separable_conv2d/ReadVariableOp_1¢ sep_out_2/BiasAdd/ReadVariableOp¢)sep_out_2/separable_conv2d/ReadVariableOp¢+sep_out_2/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_18/BiasAdd/ReadVariableOp¢3separable_conv2d_18/separable_conv2d/ReadVariableOp¢5separable_conv2d_18/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_19/BiasAdd/ReadVariableOp¢3separable_conv2d_19/separable_conv2d/ReadVariableOp¢5separable_conv2d_19/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_20/BiasAdd/ReadVariableOp¢3separable_conv2d_20/separable_conv2d/ReadVariableOp¢5separable_conv2d_20/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_21/BiasAdd/ReadVariableOp¢3separable_conv2d_21/separable_conv2d/ReadVariableOp¢5separable_conv2d_21/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_22/BiasAdd/ReadVariableOp¢3separable_conv2d_22/separable_conv2d/ReadVariableOp¢5separable_conv2d_22/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_23/BiasAdd/ReadVariableOp¢3separable_conv2d_23/separable_conv2d/ReadVariableOp¢5separable_conv2d_23/separable_conv2d/ReadVariableOp_1
 first_conv/Conv2D/ReadVariableOpReadVariableOp)first_conv_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0²
first_conv/Conv2DConv2Dinputs(first_conv/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà*
paddingSAME*
strides

!first_conv/BiasAdd/ReadVariableOpReadVariableOp*first_conv_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0¡
first_conv/BiasAddBiasAddfirst_conv/Conv2D:output:0)first_conv/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:à*
dtype0
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:à*
dtype0³
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:à*
dtype0·
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:à*
dtype0Å
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3first_conv/BiasAdd:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:ÿÿÿÿÿÿÿÿÿààà:à:à:à:à:*
epsilon%o:*
is_training( 
activation_21/ReluRelu+batch_normalization_11/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà²
max_pooling2d_10/MaxPoolMaxPool activation_21/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà*
ksize
*
paddingSAME*
strides

conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:à*
dtype0Ç
conv2d_8/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:*
dtype0³
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0·
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Á
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿpp:::::*
epsilon%o:*
is_training( 
activation_22/ReluRelu+batch_normalization_12/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp{
dropout_2/IdentityIdentity activation_22/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:à*
dtype0Á
conv2d_9/Conv2DConv2Ddropout_2/Identity:output:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà*
paddingSAME*
strides

conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppàp
activation_23/ReluReluconv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:à *
dtype0È
conv2d_10/Conv2DConv2D activation_23/Relu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp *
paddingSAME*
strides

 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
: *
dtype0
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
: *
dtype0³
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
: *
dtype0·
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
: *
dtype0Â
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_10/BiasAdd:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿpp : : : : :*
epsilon%o:*
is_training( 
activation_24/ReluRelu+batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp {
dropout_3/IdentityIdentity activation_24/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*'
_output_shapes
: `*
dtype0Â
conv2d_11/Conv2DConv2Ddropout_3/Identity:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`*
paddingSAME*
strides

 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes
:`*
dtype0
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes
:`*
dtype0²
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0¶
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0½
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3conv2d_11/BiasAdd:output:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿpp`:`:`:`:`:*
epsilon%o:*
is_training( 
activation_25/ReluRelu+batch_normalization_14/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0È
conv2d_12/Conv2DConv2D activation_25/Relu:activations:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes	
:*
dtype0³
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0·
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Â
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3conv2d_12/BiasAdd:output:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿpp:::::*
epsilon%o:*
is_training( 
activation_26/ReluRelu+batch_normalization_15/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp{
dropout_4/IdentityIdentity activation_26/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp­
max_pooling2d_11/MaxPoolMaxPooldropout_4/Identity:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingSAME*
strides
¹
3separable_conv2d_18/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_18_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0¾
5separable_conv2d_18/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_18_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:À*
dtype0
*separable_conv2d_18/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
2separable_conv2d_18/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
.separable_conv2d_18/separable_conv2d/depthwiseDepthwiseConv2dNative!max_pooling2d_11/MaxPool:output:0;separable_conv2d_18/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

$separable_conv2d_18/separable_conv2dConv2D7separable_conv2d_18/separable_conv2d/depthwise:output:0=separable_conv2d_18/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À*
paddingVALID*
strides

*separable_conv2d_18/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:À*
dtype0Ä
separable_conv2d_18/BiasAddBiasAdd-separable_conv2d_18/separable_conv2d:output:02separable_conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes	
:À*
dtype0
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes	
:À*
dtype0³
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:À*
dtype0·
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:À*
dtype0Ì
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_18/BiasAdd:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ88À:À:À:À:À:*
epsilon%o:*
is_training( 
activation_27/ReluRelu+batch_normalization_16/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À¹
3separable_conv2d_19/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_19_separable_conv2d_readvariableop_resource*'
_output_shapes
:À*
dtype0¾
5separable_conv2d_19/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_19_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:Àà*
dtype0
*separable_conv2d_19/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @     
2separable_conv2d_19/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
.separable_conv2d_19/separable_conv2d/depthwiseDepthwiseConv2dNative activation_27/Relu:activations:0;separable_conv2d_19/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À*
paddingSAME*
strides

$separable_conv2d_19/separable_conv2dConv2D7separable_conv2d_19/separable_conv2d/depthwise:output:0=separable_conv2d_19/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*
paddingVALID*
strides

*separable_conv2d_19/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0Ä
separable_conv2d_19/BiasAddBiasAdd-separable_conv2d_19/separable_conv2d:output:02separable_conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à{
activation_28/ReluRelu$separable_conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à¹
3separable_conv2d_20/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_20_separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0¾
5separable_conv2d_20/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_20_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:à*
dtype0
*separable_conv2d_20/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      `     
2separable_conv2d_20/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
.separable_conv2d_20/separable_conv2d/depthwiseDepthwiseConv2dNative activation_28/Relu:activations:0;separable_conv2d_20/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*
paddingSAME*
strides

$separable_conv2d_20/separable_conv2dConv2D7separable_conv2d_20/separable_conv2d/depthwise:output:0=separable_conv2d_20/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingVALID*
strides

*separable_conv2d_20/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ä
separable_conv2d_20/BiasAddBiasAdd-separable_conv2d_20/separable_conv2d:output:02separable_conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88{
activation_29/ReluRelu$separable_conv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88{
dropout_5/IdentityIdentity activation_29/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¹
3separable_conv2d_21/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_21_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0¾
5separable_conv2d_21/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_21_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:*
dtype0
*separable_conv2d_21/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
2separable_conv2d_21/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
.separable_conv2d_21/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_5/Identity:output:0;separable_conv2d_21/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

$separable_conv2d_21/separable_conv2dConv2D7separable_conv2d_21/separable_conv2d/depthwise:output:0=separable_conv2d_21/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingVALID*
strides

*separable_conv2d_21/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ä
separable_conv2d_21/BiasAddBiasAdd-separable_conv2d_21/separable_conv2d:output:02separable_conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88{
activation_30/ReluRelu$separable_conv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¹
3separable_conv2d_22/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_22_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0¾
5separable_conv2d_22/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_22_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:à*
dtype0
*separable_conv2d_22/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
2separable_conv2d_22/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
.separable_conv2d_22/separable_conv2d/depthwiseDepthwiseConv2dNative activation_30/Relu:activations:0;separable_conv2d_22/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

$separable_conv2d_22/separable_conv2dConv2D7separable_conv2d_22/separable_conv2d/depthwise:output:0=separable_conv2d_22/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*
paddingVALID*
strides

*separable_conv2d_22/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0Ä
separable_conv2d_22/BiasAddBiasAdd-separable_conv2d_22/separable_conv2d:output:02separable_conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes	
:à*
dtype0
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:à*
dtype0³
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:à*
dtype0·
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:à*
dtype0Ì
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_22/BiasAdd:output:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ88à:à:à:à:à:*
epsilon%o:*
is_training( 
activation_31/ReluRelu+batch_normalization_17/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à¹
3separable_conv2d_23/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_23_separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0¾
5separable_conv2d_23/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_23_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:à *
dtype0
*separable_conv2d_23/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      `     
2separable_conv2d_23/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
.separable_conv2d_23/separable_conv2d/depthwiseDepthwiseConv2dNative activation_31/Relu:activations:0;separable_conv2d_23/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*
paddingSAME*
strides

$separable_conv2d_23/separable_conv2dConv2D7separable_conv2d_23/separable_conv2d/depthwise:output:0=separable_conv2d_23/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 *
paddingVALID*
strides

*separable_conv2d_23/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0Ä
separable_conv2d_23/BiasAddBiasAdd-separable_conv2d_23/separable_conv2d:output:02separable_conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 {
activation_32/ReluRelu$separable_conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 ²
max_pooling2d_12/MaxPoolMaxPool activation_32/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingSAME*
strides
¥
)sep_out_0/separable_conv2d/ReadVariableOpReadVariableOp2sep_out_0_separable_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype0ª
+sep_out_0/separable_conv2d/ReadVariableOp_1ReadVariableOp4sep_out_0_separable_conv2d_readvariableop_1_resource*(
_output_shapes
: À*
dtype0y
 sep_out_0/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            y
(sep_out_0/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ÷
$sep_out_0/separable_conv2d/depthwiseDepthwiseConv2dNative!max_pooling2d_12/MaxPool:output:01sep_out_0/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
ì
sep_out_0/separable_conv2dConv2D-sep_out_0/separable_conv2d/depthwise:output:03sep_out_0/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
paddingVALID*
strides

 sep_out_0/BiasAdd/ReadVariableOpReadVariableOp)sep_out_0_biasadd_readvariableop_resource*
_output_shapes	
:À*
dtype0¦
sep_out_0/BiasAddBiasAdd#sep_out_0/separable_conv2d:output:0(sep_out_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀm
sep_out_0/ReluRelusep_out_0/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ®
max_pooling2d_13/MaxPoolMaxPoolsep_out_0/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
ksize
*
paddingSAME*
strides
¥
)sep_out_1/separable_conv2d/ReadVariableOpReadVariableOp2sep_out_1_separable_conv2d_readvariableop_resource*'
_output_shapes
:À*
dtype0ª
+sep_out_1/separable_conv2d/ReadVariableOp_1ReadVariableOp4sep_out_1_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:Àà*
dtype0y
 sep_out_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      À     y
(sep_out_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ÷
$sep_out_1/separable_conv2d/depthwiseDepthwiseConv2dNative!max_pooling2d_13/MaxPool:output:01sep_out_1/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
paddingVALID*
strides
ì
sep_out_1/separable_conv2dConv2D-sep_out_1/separable_conv2d/depthwise:output:03sep_out_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides

 sep_out_1/BiasAdd/ReadVariableOpReadVariableOp)sep_out_1_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0¦
sep_out_1/BiasAddBiasAdd#sep_out_1/separable_conv2d:output:0(sep_out_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿàm
sep_out_1/ReluRelusep_out_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà®
max_pooling2d_14/MaxPoolMaxPoolsep_out_1/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
ksize
*
paddingSAME*
strides
¥
)sep_out_2/separable_conv2d/ReadVariableOpReadVariableOp2sep_out_2_separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0ª
+sep_out_2/separable_conv2d/ReadVariableOp_1ReadVariableOp4sep_out_2_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:àà*
dtype0y
 sep_out_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      à      y
(sep_out_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ÷
$sep_out_2/separable_conv2d/depthwiseDepthwiseConv2dNative!max_pooling2d_14/MaxPool:output:01sep_out_2/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
ì
sep_out_2/separable_conv2dConv2D-sep_out_2/separable_conv2d/depthwise:output:03sep_out_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides

 sep_out_2/BiasAdd/ReadVariableOpReadVariableOp)sep_out_2_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0¦
sep_out_2/BiasAddBiasAdd#sep_out_2/separable_conv2d:output:0(sep_out_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿàm
sep_out_2/ReluRelusep_out_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà®
max_pooling2d_15/MaxPoolMaxPoolsep_out_2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
ksize
*
paddingSAME*
strides

&output/separable_conv2d/ReadVariableOpReadVariableOp/output_separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0£
(output/separable_conv2d/ReadVariableOp_1ReadVariableOp1output_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:à*
dtype0v
output/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      à     v
%output/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ñ
!output/separable_conv2d/depthwiseDepthwiseConv2dNative!max_pooling2d_15/MaxPool:output:0.output/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
â
output/separable_conv2dConv2D*output/separable_conv2d/depthwise:output:00output/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output/BiasAddBiasAdd output/separable_conv2d:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentityoutput/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
NoOpNoOp7^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_17^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_17^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_17^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp"^first_conv/BiasAdd/ReadVariableOp!^first_conv/Conv2D/ReadVariableOp^output/BiasAdd/ReadVariableOp'^output/separable_conv2d/ReadVariableOp)^output/separable_conv2d/ReadVariableOp_1!^sep_out_0/BiasAdd/ReadVariableOp*^sep_out_0/separable_conv2d/ReadVariableOp,^sep_out_0/separable_conv2d/ReadVariableOp_1!^sep_out_1/BiasAdd/ReadVariableOp*^sep_out_1/separable_conv2d/ReadVariableOp,^sep_out_1/separable_conv2d/ReadVariableOp_1!^sep_out_2/BiasAdd/ReadVariableOp*^sep_out_2/separable_conv2d/ReadVariableOp,^sep_out_2/separable_conv2d/ReadVariableOp_1+^separable_conv2d_18/BiasAdd/ReadVariableOp4^separable_conv2d_18/separable_conv2d/ReadVariableOp6^separable_conv2d_18/separable_conv2d/ReadVariableOp_1+^separable_conv2d_19/BiasAdd/ReadVariableOp4^separable_conv2d_19/separable_conv2d/ReadVariableOp6^separable_conv2d_19/separable_conv2d/ReadVariableOp_1+^separable_conv2d_20/BiasAdd/ReadVariableOp4^separable_conv2d_20/separable_conv2d/ReadVariableOp6^separable_conv2d_20/separable_conv2d/ReadVariableOp_1+^separable_conv2d_21/BiasAdd/ReadVariableOp4^separable_conv2d_21/separable_conv2d/ReadVariableOp6^separable_conv2d_21/separable_conv2d/ReadVariableOp_1+^separable_conv2d_22/BiasAdd/ReadVariableOp4^separable_conv2d_22/separable_conv2d/ReadVariableOp6^separable_conv2d_22/separable_conv2d/ReadVariableOp_1+^separable_conv2d_23/BiasAdd/ReadVariableOp4^separable_conv2d_23/separable_conv2d/ReadVariableOp6^separable_conv2d_23/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¾
_input_shapes¬
©:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2F
!first_conv/BiasAdd/ReadVariableOp!first_conv/BiasAdd/ReadVariableOp2D
 first_conv/Conv2D/ReadVariableOp first_conv/Conv2D/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2P
&output/separable_conv2d/ReadVariableOp&output/separable_conv2d/ReadVariableOp2T
(output/separable_conv2d/ReadVariableOp_1(output/separable_conv2d/ReadVariableOp_12D
 sep_out_0/BiasAdd/ReadVariableOp sep_out_0/BiasAdd/ReadVariableOp2V
)sep_out_0/separable_conv2d/ReadVariableOp)sep_out_0/separable_conv2d/ReadVariableOp2Z
+sep_out_0/separable_conv2d/ReadVariableOp_1+sep_out_0/separable_conv2d/ReadVariableOp_12D
 sep_out_1/BiasAdd/ReadVariableOp sep_out_1/BiasAdd/ReadVariableOp2V
)sep_out_1/separable_conv2d/ReadVariableOp)sep_out_1/separable_conv2d/ReadVariableOp2Z
+sep_out_1/separable_conv2d/ReadVariableOp_1+sep_out_1/separable_conv2d/ReadVariableOp_12D
 sep_out_2/BiasAdd/ReadVariableOp sep_out_2/BiasAdd/ReadVariableOp2V
)sep_out_2/separable_conv2d/ReadVariableOp)sep_out_2/separable_conv2d/ReadVariableOp2Z
+sep_out_2/separable_conv2d/ReadVariableOp_1+sep_out_2/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_18/BiasAdd/ReadVariableOp*separable_conv2d_18/BiasAdd/ReadVariableOp2j
3separable_conv2d_18/separable_conv2d/ReadVariableOp3separable_conv2d_18/separable_conv2d/ReadVariableOp2n
5separable_conv2d_18/separable_conv2d/ReadVariableOp_15separable_conv2d_18/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_19/BiasAdd/ReadVariableOp*separable_conv2d_19/BiasAdd/ReadVariableOp2j
3separable_conv2d_19/separable_conv2d/ReadVariableOp3separable_conv2d_19/separable_conv2d/ReadVariableOp2n
5separable_conv2d_19/separable_conv2d/ReadVariableOp_15separable_conv2d_19/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_20/BiasAdd/ReadVariableOp*separable_conv2d_20/BiasAdd/ReadVariableOp2j
3separable_conv2d_20/separable_conv2d/ReadVariableOp3separable_conv2d_20/separable_conv2d/ReadVariableOp2n
5separable_conv2d_20/separable_conv2d/ReadVariableOp_15separable_conv2d_20/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_21/BiasAdd/ReadVariableOp*separable_conv2d_21/BiasAdd/ReadVariableOp2j
3separable_conv2d_21/separable_conv2d/ReadVariableOp3separable_conv2d_21/separable_conv2d/ReadVariableOp2n
5separable_conv2d_21/separable_conv2d/ReadVariableOp_15separable_conv2d_21/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_22/BiasAdd/ReadVariableOp*separable_conv2d_22/BiasAdd/ReadVariableOp2j
3separable_conv2d_22/separable_conv2d/ReadVariableOp3separable_conv2d_22/separable_conv2d/ReadVariableOp2n
5separable_conv2d_22/separable_conv2d/ReadVariableOp_15separable_conv2d_22/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_23/BiasAdd/ReadVariableOp*separable_conv2d_23/BiasAdd/ReadVariableOp2j
3separable_conv2d_23/separable_conv2d/ReadVariableOp3separable_conv2d_23/separable_conv2d/ReadVariableOp2n
5separable_conv2d_23/separable_conv2d/ReadVariableOp_15separable_conv2d_23/separable_conv2d/ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
º

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_77814

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¯
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppx
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppb
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_15_layer_call_fn_80463

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_76723
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
d
H__inference_activation_31_layer_call_and_return_conditional_losses_80890

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88àc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88à:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à
 
_user_specified_nameinputs
ð
d
H__inference_activation_29_layer_call_and_return_conditional_losses_80729

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_77123

inputs
identity¡
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_77034

inputs&
readvariableop_resource:	à(
readvariableop_1_resource:	à7
(fusedbatchnormv3_readvariableop_resource:	à9
*fusedbatchnormv3_readvariableop_1_resource:	à
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:à*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:à*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:à*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:à*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà:à:à:à:à:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_13_layer_call_fn_80254

inputs
unknown:	 
	unknown_0:	 
	unknown_1:	 
	unknown_2:	 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_76595
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_12_layer_call_fn_80931

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_77082
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì

N__inference_separable_conv2d_21_layer_call_and_return_conditional_losses_80782

inputsC
(separable_conv2d_readvariableop_resource:F
*separable_conv2d_readvariableop_1_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ù
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
Ä
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_80303

inputs&
readvariableop_resource:	 (
readvariableop_1_resource:	 7
(fusedbatchnormv3_readvariableop_resource:	 9
*fusedbatchnormv3_readvariableop_1_resource:	 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
: *
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
: *
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
±

ÿ
C__inference_conv2d_8_layer_call_and_return_conditional_losses_77287

inputs:
conv2d_readvariableop_resource:à.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:à*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpph
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿppà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà
 
_user_specified_nameinputs
º

c
D__inference_dropout_5_layer_call_and_return_conditional_losses_80756

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¯
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ88:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
ÅÌ
ØF
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_79827

inputsD
)first_conv_conv2d_readvariableop_resource:à9
*first_conv_biasadd_readvariableop_resource:	à=
.batch_normalization_11_readvariableop_resource:	à?
0batch_normalization_11_readvariableop_1_resource:	àN
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	àP
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	àC
'conv2d_8_conv2d_readvariableop_resource:à7
(conv2d_8_biasadd_readvariableop_resource:	=
.batch_normalization_12_readvariableop_resource:	?
0batch_normalization_12_readvariableop_1_resource:	N
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_9_conv2d_readvariableop_resource:à7
(conv2d_9_biasadd_readvariableop_resource:	àD
(conv2d_10_conv2d_readvariableop_resource:à 8
)conv2d_10_biasadd_readvariableop_resource:	 =
.batch_normalization_13_readvariableop_resource:	 ?
0batch_normalization_13_readvariableop_1_resource:	 N
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	 P
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	 C
(conv2d_11_conv2d_readvariableop_resource: `7
)conv2d_11_biasadd_readvariableop_resource:`<
.batch_normalization_14_readvariableop_resource:`>
0batch_normalization_14_readvariableop_1_resource:`M
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:`O
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:`C
(conv2d_12_conv2d_readvariableop_resource:`8
)conv2d_12_biasadd_readvariableop_resource:	=
.batch_normalization_15_readvariableop_resource:	?
0batch_normalization_15_readvariableop_1_resource:	N
?batch_normalization_15_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource:	W
<separable_conv2d_18_separable_conv2d_readvariableop_resource:Z
>separable_conv2d_18_separable_conv2d_readvariableop_1_resource:ÀB
3separable_conv2d_18_biasadd_readvariableop_resource:	À=
.batch_normalization_16_readvariableop_resource:	À?
0batch_normalization_16_readvariableop_1_resource:	ÀN
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:	ÀP
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:	ÀW
<separable_conv2d_19_separable_conv2d_readvariableop_resource:ÀZ
>separable_conv2d_19_separable_conv2d_readvariableop_1_resource:ÀàB
3separable_conv2d_19_biasadd_readvariableop_resource:	àW
<separable_conv2d_20_separable_conv2d_readvariableop_resource:àZ
>separable_conv2d_20_separable_conv2d_readvariableop_1_resource:àB
3separable_conv2d_20_biasadd_readvariableop_resource:	W
<separable_conv2d_21_separable_conv2d_readvariableop_resource:Z
>separable_conv2d_21_separable_conv2d_readvariableop_1_resource:B
3separable_conv2d_21_biasadd_readvariableop_resource:	W
<separable_conv2d_22_separable_conv2d_readvariableop_resource:Z
>separable_conv2d_22_separable_conv2d_readvariableop_1_resource:àB
3separable_conv2d_22_biasadd_readvariableop_resource:	à=
.batch_normalization_17_readvariableop_resource:	à?
0batch_normalization_17_readvariableop_1_resource:	àN
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	àP
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	àW
<separable_conv2d_23_separable_conv2d_readvariableop_resource:àZ
>separable_conv2d_23_separable_conv2d_readvariableop_1_resource:à B
3separable_conv2d_23_biasadd_readvariableop_resource:	 M
2sep_out_0_separable_conv2d_readvariableop_resource: P
4sep_out_0_separable_conv2d_readvariableop_1_resource: À8
)sep_out_0_biasadd_readvariableop_resource:	ÀM
2sep_out_1_separable_conv2d_readvariableop_resource:ÀP
4sep_out_1_separable_conv2d_readvariableop_1_resource:Àà8
)sep_out_1_biasadd_readvariableop_resource:	àM
2sep_out_2_separable_conv2d_readvariableop_resource:àP
4sep_out_2_separable_conv2d_readvariableop_1_resource:àà8
)sep_out_2_biasadd_readvariableop_resource:	àJ
/output_separable_conv2d_readvariableop_resource:àL
1output_separable_conv2d_readvariableop_1_resource:à4
&output_biasadd_readvariableop_resource:
identity¢%batch_normalization_11/AssignNewValue¢'batch_normalization_11/AssignNewValue_1¢6batch_normalization_11/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_11/ReadVariableOp¢'batch_normalization_11/ReadVariableOp_1¢%batch_normalization_12/AssignNewValue¢'batch_normalization_12/AssignNewValue_1¢6batch_normalization_12/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_12/ReadVariableOp¢'batch_normalization_12/ReadVariableOp_1¢%batch_normalization_13/AssignNewValue¢'batch_normalization_13/AssignNewValue_1¢6batch_normalization_13/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_13/ReadVariableOp¢'batch_normalization_13/ReadVariableOp_1¢%batch_normalization_14/AssignNewValue¢'batch_normalization_14/AssignNewValue_1¢6batch_normalization_14/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_14/ReadVariableOp¢'batch_normalization_14/ReadVariableOp_1¢%batch_normalization_15/AssignNewValue¢'batch_normalization_15/AssignNewValue_1¢6batch_normalization_15/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_15/ReadVariableOp¢'batch_normalization_15/ReadVariableOp_1¢%batch_normalization_16/AssignNewValue¢'batch_normalization_16/AssignNewValue_1¢6batch_normalization_16/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_16/ReadVariableOp¢'batch_normalization_16/ReadVariableOp_1¢%batch_normalization_17/AssignNewValue¢'batch_normalization_17/AssignNewValue_1¢6batch_normalization_17/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_17/ReadVariableOp¢'batch_normalization_17/ReadVariableOp_1¢ conv2d_10/BiasAdd/ReadVariableOp¢conv2d_10/Conv2D/ReadVariableOp¢ conv2d_11/BiasAdd/ReadVariableOp¢conv2d_11/Conv2D/ReadVariableOp¢ conv2d_12/BiasAdd/ReadVariableOp¢conv2d_12/Conv2D/ReadVariableOp¢conv2d_8/BiasAdd/ReadVariableOp¢conv2d_8/Conv2D/ReadVariableOp¢conv2d_9/BiasAdd/ReadVariableOp¢conv2d_9/Conv2D/ReadVariableOp¢!first_conv/BiasAdd/ReadVariableOp¢ first_conv/Conv2D/ReadVariableOp¢output/BiasAdd/ReadVariableOp¢&output/separable_conv2d/ReadVariableOp¢(output/separable_conv2d/ReadVariableOp_1¢ sep_out_0/BiasAdd/ReadVariableOp¢)sep_out_0/separable_conv2d/ReadVariableOp¢+sep_out_0/separable_conv2d/ReadVariableOp_1¢ sep_out_1/BiasAdd/ReadVariableOp¢)sep_out_1/separable_conv2d/ReadVariableOp¢+sep_out_1/separable_conv2d/ReadVariableOp_1¢ sep_out_2/BiasAdd/ReadVariableOp¢)sep_out_2/separable_conv2d/ReadVariableOp¢+sep_out_2/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_18/BiasAdd/ReadVariableOp¢3separable_conv2d_18/separable_conv2d/ReadVariableOp¢5separable_conv2d_18/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_19/BiasAdd/ReadVariableOp¢3separable_conv2d_19/separable_conv2d/ReadVariableOp¢5separable_conv2d_19/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_20/BiasAdd/ReadVariableOp¢3separable_conv2d_20/separable_conv2d/ReadVariableOp¢5separable_conv2d_20/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_21/BiasAdd/ReadVariableOp¢3separable_conv2d_21/separable_conv2d/ReadVariableOp¢5separable_conv2d_21/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_22/BiasAdd/ReadVariableOp¢3separable_conv2d_22/separable_conv2d/ReadVariableOp¢5separable_conv2d_22/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_23/BiasAdd/ReadVariableOp¢3separable_conv2d_23/separable_conv2d/ReadVariableOp¢5separable_conv2d_23/separable_conv2d/ReadVariableOp_1
 first_conv/Conv2D/ReadVariableOpReadVariableOp)first_conv_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0²
first_conv/Conv2DConv2Dinputs(first_conv/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà*
paddingSAME*
strides

!first_conv/BiasAdd/ReadVariableOpReadVariableOp*first_conv_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0¡
first_conv/BiasAddBiasAddfirst_conv/Conv2D:output:0)first_conv/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:à*
dtype0
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:à*
dtype0³
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:à*
dtype0·
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:à*
dtype0Ó
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3first_conv/BiasAdd:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:ÿÿÿÿÿÿÿÿÿààà:à:à:à:à:*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_11/AssignNewValueAssignVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_11/AssignNewValue_1AssignVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_21/ReluRelu+batch_normalization_11/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿààà²
max_pooling2d_10/MaxPoolMaxPool activation_21/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà*
ksize
*
paddingSAME*
strides

conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:à*
dtype0Ç
conv2d_8/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:*
dtype0³
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0·
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ï
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿpp:::::*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_12/AssignNewValueAssignVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource4batch_normalization_12/FusedBatchNormV3:batch_mean:07^batch_normalization_12/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_12/AssignNewValue_1AssignVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_12/FusedBatchNormV3:batch_variance:09^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_22/ReluRelu+batch_normalization_12/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_2/dropout/MulMul activation_22/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppg
dropout_2/dropout/ShapeShape activation_22/Relu:activations:0*
T0*
_output_shapes
:©
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Í
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:à*
dtype0Á
conv2d_9/Conv2DConv2Ddropout_2/dropout/Mul_1:z:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà*
paddingSAME*
strides

conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppàp
activation_23/ReluReluconv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppà
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:à *
dtype0È
conv2d_10/Conv2DConv2D activation_23/Relu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp *
paddingSAME*
strides

 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
: *
dtype0
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
: *
dtype0³
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
: *
dtype0·
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
: *
dtype0Ð
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_10/BiasAdd:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿpp : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_13/AssignNewValueAssignVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource4batch_normalization_13/FusedBatchNormV3:batch_mean:07^batch_normalization_13/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_13/AssignNewValue_1AssignVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_13/FusedBatchNormV3:batch_variance:09^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_24/ReluRelu+batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp \
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_3/dropout/MulMul activation_24/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp g
dropout_3/dropout/ShapeShape activation_24/Relu:activations:0*
T0*
_output_shapes
:©
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp *
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Í
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp 
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*'
_output_shapes
: `*
dtype0Â
conv2d_11/Conv2DConv2Ddropout_3/dropout/Mul_1:z:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`*
paddingSAME*
strides

 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes
:`*
dtype0
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes
:`*
dtype0²
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0¶
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0Ë
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3conv2d_11/BiasAdd:output:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿpp`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_14/AssignNewValueAssignVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource4batch_normalization_14/FusedBatchNormV3:batch_mean:07^batch_normalization_14/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_14/AssignNewValue_1AssignVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_14/FusedBatchNormV3:batch_variance:09^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_25/ReluRelu+batch_normalization_14/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0È
conv2d_12/Conv2DConv2D activation_25/Relu:activations:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes	
:*
dtype0³
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0·
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ð
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3conv2d_12/BiasAdd:output:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿpp:::::*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_15/AssignNewValueAssignVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource4batch_normalization_15/FusedBatchNormV3:batch_mean:07^batch_normalization_15/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_15/AssignNewValue_1AssignVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_15/FusedBatchNormV3:batch_variance:09^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_26/ReluRelu+batch_normalization_15/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?
dropout_4/dropout/MulMul activation_26/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppg
dropout_4/dropout/ShapeShape activation_26/Relu:activations:0*
T0*
_output_shapes
:©
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Í
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp­
max_pooling2d_11/MaxPoolMaxPooldropout_4/dropout/Mul_1:z:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingSAME*
strides
¹
3separable_conv2d_18/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_18_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0¾
5separable_conv2d_18/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_18_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:À*
dtype0
*separable_conv2d_18/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
2separable_conv2d_18/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
.separable_conv2d_18/separable_conv2d/depthwiseDepthwiseConv2dNative!max_pooling2d_11/MaxPool:output:0;separable_conv2d_18/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

$separable_conv2d_18/separable_conv2dConv2D7separable_conv2d_18/separable_conv2d/depthwise:output:0=separable_conv2d_18/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À*
paddingVALID*
strides

*separable_conv2d_18/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:À*
dtype0Ä
separable_conv2d_18/BiasAddBiasAdd-separable_conv2d_18/separable_conv2d:output:02separable_conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes	
:À*
dtype0
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes	
:À*
dtype0³
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:À*
dtype0·
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:À*
dtype0Ú
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_18/BiasAdd:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ88À:À:À:À:À:*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_16/AssignNewValueAssignVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource4batch_normalization_16/FusedBatchNormV3:batch_mean:07^batch_normalization_16/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_16/AssignNewValue_1AssignVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_16/FusedBatchNormV3:batch_variance:09^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_27/ReluRelu+batch_normalization_16/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À¹
3separable_conv2d_19/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_19_separable_conv2d_readvariableop_resource*'
_output_shapes
:À*
dtype0¾
5separable_conv2d_19/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_19_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:Àà*
dtype0
*separable_conv2d_19/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @     
2separable_conv2d_19/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
.separable_conv2d_19/separable_conv2d/depthwiseDepthwiseConv2dNative activation_27/Relu:activations:0;separable_conv2d_19/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88À*
paddingSAME*
strides

$separable_conv2d_19/separable_conv2dConv2D7separable_conv2d_19/separable_conv2d/depthwise:output:0=separable_conv2d_19/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*
paddingVALID*
strides

*separable_conv2d_19/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0Ä
separable_conv2d_19/BiasAddBiasAdd-separable_conv2d_19/separable_conv2d:output:02separable_conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à{
activation_28/ReluRelu$separable_conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à¹
3separable_conv2d_20/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_20_separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0¾
5separable_conv2d_20/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_20_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:à*
dtype0
*separable_conv2d_20/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      `     
2separable_conv2d_20/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
.separable_conv2d_20/separable_conv2d/depthwiseDepthwiseConv2dNative activation_28/Relu:activations:0;separable_conv2d_20/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*
paddingSAME*
strides

$separable_conv2d_20/separable_conv2dConv2D7separable_conv2d_20/separable_conv2d/depthwise:output:0=separable_conv2d_20/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingVALID*
strides

*separable_conv2d_20/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ä
separable_conv2d_20/BiasAddBiasAdd-separable_conv2d_20/separable_conv2d:output:02separable_conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88{
activation_29/ReluRelu$separable_conv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_5/dropout/MulMul activation_29/Relu:activations:0 dropout_5/dropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88g
dropout_5/dropout/ShapeShape activation_29/Relu:activations:0*
T0*
_output_shapes
:©
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
dtype0e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Í
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¹
3separable_conv2d_21/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_21_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0¾
5separable_conv2d_21/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_21_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:*
dtype0
*separable_conv2d_21/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
2separable_conv2d_21/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
.separable_conv2d_21/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_5/dropout/Mul_1:z:0;separable_conv2d_21/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

$separable_conv2d_21/separable_conv2dConv2D7separable_conv2d_21/separable_conv2d/depthwise:output:0=separable_conv2d_21/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingVALID*
strides

*separable_conv2d_21/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ä
separable_conv2d_21/BiasAddBiasAdd-separable_conv2d_21/separable_conv2d:output:02separable_conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88{
activation_30/ReluRelu$separable_conv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¹
3separable_conv2d_22/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_22_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0¾
5separable_conv2d_22/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_22_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:à*
dtype0
*separable_conv2d_22/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
2separable_conv2d_22/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
.separable_conv2d_22/separable_conv2d/depthwiseDepthwiseConv2dNative activation_30/Relu:activations:0;separable_conv2d_22/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

$separable_conv2d_22/separable_conv2dConv2D7separable_conv2d_22/separable_conv2d/depthwise:output:0=separable_conv2d_22/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*
paddingVALID*
strides

*separable_conv2d_22/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0Ä
separable_conv2d_22/BiasAddBiasAdd-separable_conv2d_22/separable_conv2d:output:02separable_conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes	
:à*
dtype0
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:à*
dtype0³
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:à*
dtype0·
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:à*
dtype0Ú
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_22/BiasAdd:output:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ88à:à:à:à:à:*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_17/AssignNewValueAssignVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource4batch_normalization_17/FusedBatchNormV3:batch_mean:07^batch_normalization_17/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_17/AssignNewValue_1AssignVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_17/FusedBatchNormV3:batch_variance:09^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_31/ReluRelu+batch_normalization_17/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à¹
3separable_conv2d_23/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_23_separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0¾
5separable_conv2d_23/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_23_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:à *
dtype0
*separable_conv2d_23/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      `     
2separable_conv2d_23/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
.separable_conv2d_23/separable_conv2d/depthwiseDepthwiseConv2dNative activation_31/Relu:activations:0;separable_conv2d_23/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88à*
paddingSAME*
strides

$separable_conv2d_23/separable_conv2dConv2D7separable_conv2d_23/separable_conv2d/depthwise:output:0=separable_conv2d_23/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 *
paddingVALID*
strides

*separable_conv2d_23/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0Ä
separable_conv2d_23/BiasAddBiasAdd-separable_conv2d_23/separable_conv2d:output:02separable_conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 {
activation_32/ReluRelu$separable_conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88 ²
max_pooling2d_12/MaxPoolMaxPool activation_32/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingSAME*
strides
¥
)sep_out_0/separable_conv2d/ReadVariableOpReadVariableOp2sep_out_0_separable_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype0ª
+sep_out_0/separable_conv2d/ReadVariableOp_1ReadVariableOp4sep_out_0_separable_conv2d_readvariableop_1_resource*(
_output_shapes
: À*
dtype0y
 sep_out_0/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            y
(sep_out_0/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ÷
$sep_out_0/separable_conv2d/depthwiseDepthwiseConv2dNative!max_pooling2d_12/MaxPool:output:01sep_out_0/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
ì
sep_out_0/separable_conv2dConv2D-sep_out_0/separable_conv2d/depthwise:output:03sep_out_0/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
paddingVALID*
strides

 sep_out_0/BiasAdd/ReadVariableOpReadVariableOp)sep_out_0_biasadd_readvariableop_resource*
_output_shapes	
:À*
dtype0¦
sep_out_0/BiasAddBiasAdd#sep_out_0/separable_conv2d:output:0(sep_out_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀm
sep_out_0/ReluRelusep_out_0/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ®
max_pooling2d_13/MaxPoolMaxPoolsep_out_0/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
ksize
*
paddingSAME*
strides
¥
)sep_out_1/separable_conv2d/ReadVariableOpReadVariableOp2sep_out_1_separable_conv2d_readvariableop_resource*'
_output_shapes
:À*
dtype0ª
+sep_out_1/separable_conv2d/ReadVariableOp_1ReadVariableOp4sep_out_1_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:Àà*
dtype0y
 sep_out_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      À     y
(sep_out_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ÷
$sep_out_1/separable_conv2d/depthwiseDepthwiseConv2dNative!max_pooling2d_13/MaxPool:output:01sep_out_1/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
paddingVALID*
strides
ì
sep_out_1/separable_conv2dConv2D-sep_out_1/separable_conv2d/depthwise:output:03sep_out_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides

 sep_out_1/BiasAdd/ReadVariableOpReadVariableOp)sep_out_1_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0¦
sep_out_1/BiasAddBiasAdd#sep_out_1/separable_conv2d:output:0(sep_out_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿàm
sep_out_1/ReluRelusep_out_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà®
max_pooling2d_14/MaxPoolMaxPoolsep_out_1/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
ksize
*
paddingSAME*
strides
¥
)sep_out_2/separable_conv2d/ReadVariableOpReadVariableOp2sep_out_2_separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0ª
+sep_out_2/separable_conv2d/ReadVariableOp_1ReadVariableOp4sep_out_2_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:àà*
dtype0y
 sep_out_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      à      y
(sep_out_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ÷
$sep_out_2/separable_conv2d/depthwiseDepthwiseConv2dNative!max_pooling2d_14/MaxPool:output:01sep_out_2/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
ì
sep_out_2/separable_conv2dConv2D-sep_out_2/separable_conv2d/depthwise:output:03sep_out_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides

 sep_out_2/BiasAdd/ReadVariableOpReadVariableOp)sep_out_2_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0¦
sep_out_2/BiasAddBiasAdd#sep_out_2/separable_conv2d:output:0(sep_out_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿàm
sep_out_2/ReluRelusep_out_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà®
max_pooling2d_15/MaxPoolMaxPoolsep_out_2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
ksize
*
paddingSAME*
strides

&output/separable_conv2d/ReadVariableOpReadVariableOp/output_separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0£
(output/separable_conv2d/ReadVariableOp_1ReadVariableOp1output_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:à*
dtype0v
output/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      à     v
%output/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ñ
!output/separable_conv2d/depthwiseDepthwiseConv2dNative!max_pooling2d_15/MaxPool:output:0.output/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
â
output/separable_conv2dConv2D*output/separable_conv2d/depthwise:output:00output/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output/BiasAddBiasAdd output/separable_conv2d:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentityoutput/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1&^batch_normalization_12/AssignNewValue(^batch_normalization_12/AssignNewValue_17^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_1&^batch_normalization_13/AssignNewValue(^batch_normalization_13/AssignNewValue_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_1&^batch_normalization_14/AssignNewValue(^batch_normalization_14/AssignNewValue_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_1&^batch_normalization_15/AssignNewValue(^batch_normalization_15/AssignNewValue_17^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_1&^batch_normalization_16/AssignNewValue(^batch_normalization_16/AssignNewValue_17^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_1&^batch_normalization_17/AssignNewValue(^batch_normalization_17/AssignNewValue_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp"^first_conv/BiasAdd/ReadVariableOp!^first_conv/Conv2D/ReadVariableOp^output/BiasAdd/ReadVariableOp'^output/separable_conv2d/ReadVariableOp)^output/separable_conv2d/ReadVariableOp_1!^sep_out_0/BiasAdd/ReadVariableOp*^sep_out_0/separable_conv2d/ReadVariableOp,^sep_out_0/separable_conv2d/ReadVariableOp_1!^sep_out_1/BiasAdd/ReadVariableOp*^sep_out_1/separable_conv2d/ReadVariableOp,^sep_out_1/separable_conv2d/ReadVariableOp_1!^sep_out_2/BiasAdd/ReadVariableOp*^sep_out_2/separable_conv2d/ReadVariableOp,^sep_out_2/separable_conv2d/ReadVariableOp_1+^separable_conv2d_18/BiasAdd/ReadVariableOp4^separable_conv2d_18/separable_conv2d/ReadVariableOp6^separable_conv2d_18/separable_conv2d/ReadVariableOp_1+^separable_conv2d_19/BiasAdd/ReadVariableOp4^separable_conv2d_19/separable_conv2d/ReadVariableOp6^separable_conv2d_19/separable_conv2d/ReadVariableOp_1+^separable_conv2d_20/BiasAdd/ReadVariableOp4^separable_conv2d_20/separable_conv2d/ReadVariableOp6^separable_conv2d_20/separable_conv2d/ReadVariableOp_1+^separable_conv2d_21/BiasAdd/ReadVariableOp4^separable_conv2d_21/separable_conv2d/ReadVariableOp6^separable_conv2d_21/separable_conv2d/ReadVariableOp_1+^separable_conv2d_22/BiasAdd/ReadVariableOp4^separable_conv2d_22/separable_conv2d/ReadVariableOp6^separable_conv2d_22/separable_conv2d/ReadVariableOp_1+^separable_conv2d_23/BiasAdd/ReadVariableOp4^separable_conv2d_23/separable_conv2d/ReadVariableOp6^separable_conv2d_23/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¾
_input_shapes¬
©:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12N
%batch_normalization_12/AssignNewValue%batch_normalization_12/AssignNewValue2R
'batch_normalization_12/AssignNewValue_1'batch_normalization_12/AssignNewValue_12p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12N
%batch_normalization_13/AssignNewValue%batch_normalization_13/AssignNewValue2R
'batch_normalization_13/AssignNewValue_1'batch_normalization_13/AssignNewValue_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12N
%batch_normalization_14/AssignNewValue%batch_normalization_14/AssignNewValue2R
'batch_normalization_14/AssignNewValue_1'batch_normalization_14/AssignNewValue_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12N
%batch_normalization_15/AssignNewValue%batch_normalization_15/AssignNewValue2R
'batch_normalization_15/AssignNewValue_1'batch_normalization_15/AssignNewValue_12p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12N
%batch_normalization_16/AssignNewValue%batch_normalization_16/AssignNewValue2R
'batch_normalization_16/AssignNewValue_1'batch_normalization_16/AssignNewValue_12p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12N
%batch_normalization_17/AssignNewValue%batch_normalization_17/AssignNewValue2R
'batch_normalization_17/AssignNewValue_1'batch_normalization_17/AssignNewValue_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2F
!first_conv/BiasAdd/ReadVariableOp!first_conv/BiasAdd/ReadVariableOp2D
 first_conv/Conv2D/ReadVariableOp first_conv/Conv2D/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2P
&output/separable_conv2d/ReadVariableOp&output/separable_conv2d/ReadVariableOp2T
(output/separable_conv2d/ReadVariableOp_1(output/separable_conv2d/ReadVariableOp_12D
 sep_out_0/BiasAdd/ReadVariableOp sep_out_0/BiasAdd/ReadVariableOp2V
)sep_out_0/separable_conv2d/ReadVariableOp)sep_out_0/separable_conv2d/ReadVariableOp2Z
+sep_out_0/separable_conv2d/ReadVariableOp_1+sep_out_0/separable_conv2d/ReadVariableOp_12D
 sep_out_1/BiasAdd/ReadVariableOp sep_out_1/BiasAdd/ReadVariableOp2V
)sep_out_1/separable_conv2d/ReadVariableOp)sep_out_1/separable_conv2d/ReadVariableOp2Z
+sep_out_1/separable_conv2d/ReadVariableOp_1+sep_out_1/separable_conv2d/ReadVariableOp_12D
 sep_out_2/BiasAdd/ReadVariableOp sep_out_2/BiasAdd/ReadVariableOp2V
)sep_out_2/separable_conv2d/ReadVariableOp)sep_out_2/separable_conv2d/ReadVariableOp2Z
+sep_out_2/separable_conv2d/ReadVariableOp_1+sep_out_2/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_18/BiasAdd/ReadVariableOp*separable_conv2d_18/BiasAdd/ReadVariableOp2j
3separable_conv2d_18/separable_conv2d/ReadVariableOp3separable_conv2d_18/separable_conv2d/ReadVariableOp2n
5separable_conv2d_18/separable_conv2d/ReadVariableOp_15separable_conv2d_18/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_19/BiasAdd/ReadVariableOp*separable_conv2d_19/BiasAdd/ReadVariableOp2j
3separable_conv2d_19/separable_conv2d/ReadVariableOp3separable_conv2d_19/separable_conv2d/ReadVariableOp2n
5separable_conv2d_19/separable_conv2d/ReadVariableOp_15separable_conv2d_19/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_20/BiasAdd/ReadVariableOp*separable_conv2d_20/BiasAdd/ReadVariableOp2j
3separable_conv2d_20/separable_conv2d/ReadVariableOp3separable_conv2d_20/separable_conv2d/ReadVariableOp2n
5separable_conv2d_20/separable_conv2d/ReadVariableOp_15separable_conv2d_20/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_21/BiasAdd/ReadVariableOp*separable_conv2d_21/BiasAdd/ReadVariableOp2j
3separable_conv2d_21/separable_conv2d/ReadVariableOp3separable_conv2d_21/separable_conv2d/ReadVariableOp2n
5separable_conv2d_21/separable_conv2d/ReadVariableOp_15separable_conv2d_21/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_22/BiasAdd/ReadVariableOp*separable_conv2d_22/BiasAdd/ReadVariableOp2j
3separable_conv2d_22/separable_conv2d/ReadVariableOp3separable_conv2d_22/separable_conv2d/ReadVariableOp2n
5separable_conv2d_22/separable_conv2d/ReadVariableOp_15separable_conv2d_22/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_23/BiasAdd/ReadVariableOp*separable_conv2d_23/BiasAdd/ReadVariableOp2j
3separable_conv2d_23/separable_conv2d/ReadVariableOp3separable_conv2d_23/separable_conv2d/ReadVariableOp2n
5separable_conv2d_23/separable_conv2d/ReadVariableOp_15separable_conv2d_23/separable_conv2d/ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
²
ÿ
D__inference_sep_out_2_layer_call_and_return_conditional_losses_81037

inputsC
(separable_conv2d_readvariableop_resource:àF
*separable_conv2d_readvariableop_1_resource:àà.
biasadd_readvariableop_resource:	à
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:à*
dtype0
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:àà*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      à      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ú
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
à
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà¥
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
!
ÿ
2__inference_McFly_cnn_50epochs_layer_call_fn_78583
input_6"
unknown:à
	unknown_0:	à
	unknown_1:	à
	unknown_2:	à
	unknown_3:	à
	unknown_4:	à%
	unknown_5:à
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:à

unknown_12:	à&

unknown_13:à 

unknown_14:	 

unknown_15:	 

unknown_16:	 

unknown_17:	 

unknown_18:	 %

unknown_19: `

unknown_20:`

unknown_21:`

unknown_22:`

unknown_23:`

unknown_24:`%

unknown_25:`

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	%

unknown_31:&

unknown_32:À

unknown_33:	À

unknown_34:	À

unknown_35:	À

unknown_36:	À

unknown_37:	À%

unknown_38:À&

unknown_39:Àà

unknown_40:	à%

unknown_41:à&

unknown_42:à

unknown_43:	%

unknown_44:&

unknown_45:

unknown_46:	%

unknown_47:&

unknown_48:à

unknown_49:	à

unknown_50:	à

unknown_51:	à

unknown_52:	à

unknown_53:	à%

unknown_54:à&

unknown_55:à 

unknown_56:	 %

unknown_57: &

unknown_58: À

unknown_59:	À%

unknown_60:À&

unknown_61:Àà

unknown_62:	à%

unknown_63:à&

unknown_64:àà

unknown_65:	à%

unknown_66:à%

unknown_67:à

unknown_68:
identity¢StatefulPartitionedCall

StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_68*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*Z
_read_only_resource_inputs<
:8	
!"#$%()*+,-./01234589:;<=>?@ABCDEF*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_78295w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¾
_input_shapes¬
©:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_6
	
Õ
6__inference_batch_normalization_15_layer_call_fn_80476

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_76754
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_76827

inputs&
readvariableop_resource:	À(
readvariableop_1_resource:	À7
(fusedbatchnormv3_readvariableop_resource:	À9
*fusedbatchnormv3_readvariableop_1_resource:	À
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:À*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:À*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:À*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:À*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ:À:À:À:À:*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_13_layer_call_fn_80968

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_77123
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
d
H__inference_activation_26_layer_call_and_return_conditional_losses_77440

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿpp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_81047

inputs
identity¡
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_80629

inputs&
readvariableop_resource:	À(
readvariableop_1_resource:	À7
(fusedbatchnormv3_readvariableop_resource:	À9
*fusedbatchnormv3_readvariableop_1_resource:	À
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:À*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:À*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:À*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:À*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ:À:À:À:À:*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*»
serving_default§
E
input_6:
serving_default_input_6:0ÿÿÿÿÿÿÿÿÿààB
output8
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:®Ú
À
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
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
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer-17
layer_with_weights-9
layer-18
layer_with_weights-10
layer-19
layer-20
layer-21
layer-22
layer_with_weights-11
layer-23
layer_with_weights-12
layer-24
layer-25
layer_with_weights-13
layer-26
layer-27
layer_with_weights-14
layer-28
layer-29
layer-30
 layer_with_weights-15
 layer-31
!layer-32
"layer_with_weights-16
"layer-33
#layer_with_weights-17
#layer-34
$layer-35
%layer_with_weights-18
%layer-36
&layer-37
'layer-38
(layer_with_weights-19
(layer-39
)layer-40
*layer_with_weights-20
*layer-41
+layer-42
,layer_with_weights-21
,layer-43
-layer-44
.layer_with_weights-22
.layer-45
/	optimizer
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_default_save_signature
7
signatures"
_tf_keras_network
"
_tf_keras_input_layer
»

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Wkernel
Xbias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
_axis
	`gamma
abeta
bmoving_mean
cmoving_variance
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t_random_generator
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer
»

wkernel
xbias
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
ª
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
trainable_variables
 regularization_losses
¡	keras_api
¢_random_generator
£__call__
+¤&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
¥kernel
	¦bias
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	­axis

®gamma
	¯beta
°moving_mean
±moving_variance
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"
_tf_keras_layer
«
¸	variables
¹trainable_variables
ºregularization_losses
»	keras_api
¼__call__
+½&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
¾kernel
	¿bias
À	variables
Átrainable_variables
Âregularization_losses
Ã	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	Æaxis

Çgamma
	Èbeta
Émoving_mean
Êmoving_variance
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ñ	variables
Òtrainable_variables
Óregularization_losses
Ô	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
×	variables
Øtrainable_variables
Ùregularization_losses
Ú	keras_api
Û_random_generator
Ü__call__
+Ý&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Þ	variables
ßtrainable_variables
àregularization_losses
á	keras_api
â__call__
+ã&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
ädepthwise_kernel
åpointwise_kernel
	æbias
ç	variables
ètrainable_variables
éregularization_losses
ê	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	íaxis

îgamma
	ïbeta
ðmoving_mean
ñmoving_variance
ò	variables
ótrainable_variables
ôregularization_losses
õ	keras_api
ö__call__
+÷&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ø	variables
ùtrainable_variables
úregularization_losses
û	keras_api
ü__call__
+ý&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
þdepthwise_kernel
ÿpointwise_kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
depthwise_kernel
pointwise_kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
 _random_generator
¡__call__
+¢&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
£depthwise_kernel
¤pointwise_kernel
	¥bias
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ª__call__
+«&call_and_return_all_conditional_losses"
_tf_keras_layer
«
¬	variables
­trainable_variables
®regularization_losses
¯	keras_api
°__call__
+±&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
²depthwise_kernel
³pointwise_kernel
	´bias
µ	variables
¶trainable_variables
·regularization_losses
¸	keras_api
¹__call__
+º&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	»axis

¼gamma
	½beta
¾moving_mean
¿moving_variance
À	variables
Átrainable_variables
Âregularization_losses
Ã	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Æ	variables
Çtrainable_variables
Èregularization_losses
É	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
Ìdepthwise_kernel
Ípointwise_kernel
	Îbias
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ò	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Õ	variables
Ötrainable_variables
×regularization_losses
Ø	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
ádepthwise_kernel
âpointwise_kernel
	ãbias
ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
è__call__
+é&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ê	variables
ëtrainable_variables
ìregularization_losses
í	keras_api
î__call__
+ï&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
ðdepthwise_kernel
ñpointwise_kernel
	òbias
ó	variables
ôtrainable_variables
õregularization_losses
ö	keras_api
÷__call__
+ø&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ù	variables
útrainable_variables
ûregularization_losses
ü	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
ÿdepthwise_kernel
pointwise_kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
depthwise_kernel
pointwise_kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
"
	optimizer
þ
80
91
A2
B3
C4
D5
W6
X7
`8
a9
b10
c11
w12
x13
14
15
16
17
18
19
¥20
¦21
®22
¯23
°24
±25
¾26
¿27
Ç28
È29
É30
Ê31
ä32
å33
æ34
î35
ï36
ð37
ñ38
þ39
ÿ40
41
42
43
44
£45
¤46
¥47
²48
³49
´50
¼51
½52
¾53
¿54
Ì55
Í56
Î57
á58
â59
ã60
ð61
ñ62
ò63
ÿ64
65
66
67
68
69"
trackable_list_wrapper

80
91
A2
B3
W4
X5
`6
a7
w8
x9
10
11
12
13
¥14
¦15
®16
¯17
¾18
¿19
Ç20
È21
ä22
å23
æ24
î25
ï26
þ27
ÿ28
29
30
31
32
£33
¤34
¥35
²36
³37
´38
¼39
½40
Ì41
Í42
Î43
á44
â45
ã46
ð47
ñ48
ò49
ÿ50
51
52
53
54
55"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
6_default_save_signature
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
2
2__inference_McFly_cnn_50epochs_layer_call_fn_77735
2__inference_McFly_cnn_50epochs_layer_call_fn_79106
2__inference_McFly_cnn_50epochs_layer_call_fn_79251
2__inference_McFly_cnn_50epochs_layer_call_fn_78583À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_79525
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_79827
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_78772
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_78961À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ËBÈ
 __inference__wrapped_model_76433input_6"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
-
serving_default"
signature_map
,:*à2first_conv/kernel
:à2first_conv/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_first_conv_layer_call_fn_79983¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_first_conv_layer_call_and_return_conditional_losses_79993¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:)à2batch_normalization_11/gamma
*:(à2batch_normalization_11/beta
3:1à (2"batch_normalization_11/moving_mean
7:5à (2&batch_normalization_11/moving_variance
<
A0
B1
C2
D3"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
ª2§
6__inference_batch_normalization_11_layer_call_fn_80006
6__inference_batch_normalization_11_layer_call_fn_80019´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_80037
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_80055´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_activation_21_layer_call_fn_80060¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_21_layer_call_and_return_conditional_losses_80065¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_max_pooling2d_10_layer_call_fn_80070¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_80075¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
+:)à2conv2d_8/kernel
:2conv2d_8/bias
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_conv2d_8_layer_call_fn_80084¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_conv2d_8_layer_call_and_return_conditional_losses_80094¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:)2batch_normalization_12/gamma
*:(2batch_normalization_12/beta
3:1 (2"batch_normalization_12/moving_mean
7:5 (2&batch_normalization_12/moving_variance
<
`0
a1
b2
c3"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
ª2§
6__inference_batch_normalization_12_layer_call_fn_80107
6__inference_batch_normalization_12_layer_call_fn_80120´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_80138
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_80156´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_activation_22_layer_call_fn_80161¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_22_layer_call_and_return_conditional_losses_80166¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
p	variables
qtrainable_variables
rregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout_2_layer_call_fn_80171
)__inference_dropout_2_layer_call_fn_80176´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_2_layer_call_and_return_conditional_losses_80181
D__inference_dropout_2_layer_call_and_return_conditional_losses_80193´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
+:)à2conv2d_9/kernel
:à2conv2d_9/bias
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_conv2d_9_layer_call_fn_80202¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_conv2d_9_layer_call_and_return_conditional_losses_80212¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
·
Ênon_trainable_variables
Ëlayers
Ìmetrics
 Ílayer_regularization_losses
Îlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_activation_23_layer_call_fn_80217¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_23_layer_call_and_return_conditional_losses_80222¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,:*à 2conv2d_10/kernel
: 2conv2d_10/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv2d_10_layer_call_fn_80231¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_10_layer_call_and_return_conditional_losses_80241¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:) 2batch_normalization_13/gamma
*:( 2batch_normalization_13/beta
3:1  (2"batch_normalization_13/moving_mean
7:5  (2&batch_normalization_13/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ª2§
6__inference_batch_normalization_13_layer_call_fn_80254
6__inference_batch_normalization_13_layer_call_fn_80267´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_80285
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_80303´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_activation_24_layer_call_fn_80308¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_24_layer_call_and_return_conditional_losses_80313¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
	variables
trainable_variables
 regularization_losses
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout_3_layer_call_fn_80318
)__inference_dropout_3_layer_call_fn_80323´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_3_layer_call_and_return_conditional_losses_80328
D__inference_dropout_3_layer_call_and_return_conditional_losses_80340´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
+:) `2conv2d_11/kernel
:`2conv2d_11/bias
0
¥0
¦1"
trackable_list_wrapper
0
¥0
¦1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ãnon_trainable_variables
älayers
åmetrics
 ælayer_regularization_losses
çlayer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv2d_11_layer_call_fn_80349¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_11_layer_call_and_return_conditional_losses_80359¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
*:(`2batch_normalization_14/gamma
):'`2batch_normalization_14/beta
2:0` (2"batch_normalization_14/moving_mean
6:4` (2&batch_normalization_14/moving_variance
@
®0
¯1
°2
±3"
trackable_list_wrapper
0
®0
¯1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ènon_trainable_variables
élayers
êmetrics
 ëlayer_regularization_losses
ìlayer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
ª2§
6__inference_batch_normalization_14_layer_call_fn_80372
6__inference_batch_normalization_14_layer_call_fn_80385´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_80403
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_80421´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ínon_trainable_variables
îlayers
ïmetrics
 ðlayer_regularization_losses
ñlayer_metrics
¸	variables
¹trainable_variables
ºregularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_activation_25_layer_call_fn_80426¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_25_layer_call_and_return_conditional_losses_80431¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
+:)`2conv2d_12/kernel
:2conv2d_12/bias
0
¾0
¿1"
trackable_list_wrapper
0
¾0
¿1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ònon_trainable_variables
ólayers
ômetrics
 õlayer_regularization_losses
ölayer_metrics
À	variables
Átrainable_variables
Âregularization_losses
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv2d_12_layer_call_fn_80440¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_12_layer_call_and_return_conditional_losses_80450¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:)2batch_normalization_15/gamma
*:(2batch_normalization_15/beta
3:1 (2"batch_normalization_15/moving_mean
7:5 (2&batch_normalization_15/moving_variance
@
Ç0
È1
É2
Ê3"
trackable_list_wrapper
0
Ç0
È1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
Ë	variables
Ìtrainable_variables
Íregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
ª2§
6__inference_batch_normalization_15_layer_call_fn_80463
6__inference_batch_normalization_15_layer_call_fn_80476´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_80494
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_80512´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
Ñ	variables
Òtrainable_variables
Óregularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_activation_26_layer_call_fn_80517¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_26_layer_call_and_return_conditional_losses_80522¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
×	variables
Øtrainable_variables
Ùregularization_losses
Ü__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout_4_layer_call_fn_80527
)__inference_dropout_4_layer_call_fn_80532´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_4_layer_call_and_return_conditional_losses_80537
D__inference_dropout_4_layer_call_and_return_conditional_losses_80549´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Þ	variables
ßtrainable_variables
àregularization_losses
â__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_max_pooling2d_11_layer_call_fn_80554¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_80559¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
?:=2$separable_conv2d_18/depthwise_kernel
@:>À2$separable_conv2d_18/pointwise_kernel
':%À2separable_conv2d_18/bias
8
ä0
å1
æ2"
trackable_list_wrapper
8
ä0
å1
æ2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ç	variables
ètrainable_variables
éregularization_losses
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses"
_generic_user_object
Ý2Ú
3__inference_separable_conv2d_18_layer_call_fn_80570¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_separable_conv2d_18_layer_call_and_return_conditional_losses_80585¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:)À2batch_normalization_16/gamma
*:(À2batch_normalization_16/beta
3:1À (2"batch_normalization_16/moving_mean
7:5À (2&batch_normalization_16/moving_variance
@
î0
ï1
ð2
ñ3"
trackable_list_wrapper
0
î0
ï1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ò	variables
ótrainable_variables
ôregularization_losses
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
ª2§
6__inference_batch_normalization_16_layer_call_fn_80598
6__inference_batch_normalization_16_layer_call_fn_80611´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_80629
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_80647´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ø	variables
ùtrainable_variables
úregularization_losses
ü__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_activation_27_layer_call_fn_80652¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_27_layer_call_and_return_conditional_losses_80657¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
?:=À2$separable_conv2d_19/depthwise_kernel
@:>Àà2$separable_conv2d_19/pointwise_kernel
':%à2separable_conv2d_19/bias
8
þ0
ÿ1
2"
trackable_list_wrapper
8
þ0
ÿ1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ý2Ú
3__inference_separable_conv2d_19_layer_call_fn_80668¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_separable_conv2d_19_layer_call_and_return_conditional_losses_80683¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_activation_28_layer_call_fn_80688¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_28_layer_call_and_return_conditional_losses_80693¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
?:=à2$separable_conv2d_20/depthwise_kernel
@:>à2$separable_conv2d_20/pointwise_kernel
':%2separable_conv2d_20/bias
8
0
1
2"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ý2Ú
3__inference_separable_conv2d_20_layer_call_fn_80704¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_separable_conv2d_20_layer_call_and_return_conditional_losses_80719¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_activation_29_layer_call_fn_80724¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_29_layer_call_and_return_conditional_losses_80729¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
	variables
trainable_variables
regularization_losses
¡__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout_5_layer_call_fn_80734
)__inference_dropout_5_layer_call_fn_80739´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_5_layer_call_and_return_conditional_losses_80744
D__inference_dropout_5_layer_call_and_return_conditional_losses_80756´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
?:=2$separable_conv2d_21/depthwise_kernel
@:>2$separable_conv2d_21/pointwise_kernel
':%2separable_conv2d_21/bias
8
£0
¤1
¥2"
trackable_list_wrapper
8
£0
¤1
¥2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
¦	variables
§trainable_variables
¨regularization_losses
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
Ý2Ú
3__inference_separable_conv2d_21_layer_call_fn_80767¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_separable_conv2d_21_layer_call_and_return_conditional_losses_80782¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
¬	variables
­trainable_variables
®regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_activation_30_layer_call_fn_80787¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_30_layer_call_and_return_conditional_losses_80792¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
?:=2$separable_conv2d_22/depthwise_kernel
@:>à2$separable_conv2d_22/pointwise_kernel
':%à2separable_conv2d_22/bias
8
²0
³1
´2"
trackable_list_wrapper
8
²0
³1
´2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
µ	variables
¶trainable_variables
·regularization_losses
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
Ý2Ú
3__inference_separable_conv2d_22_layer_call_fn_80803¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_separable_conv2d_22_layer_call_and_return_conditional_losses_80818¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:)à2batch_normalization_17/gamma
*:(à2batch_normalization_17/beta
3:1à (2"batch_normalization_17/moving_mean
7:5à (2&batch_normalization_17/moving_variance
@
¼0
½1
¾2
¿3"
trackable_list_wrapper
0
¼0
½1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
À	variables
Átrainable_variables
Âregularization_losses
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
ª2§
6__inference_batch_normalization_17_layer_call_fn_80831
6__inference_batch_normalization_17_layer_call_fn_80844´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_80862
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_80880´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
Æ	variables
Çtrainable_variables
Èregularization_losses
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_activation_31_layer_call_fn_80885¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_31_layer_call_and_return_conditional_losses_80890¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
?:=à2$separable_conv2d_23/depthwise_kernel
@:>à 2$separable_conv2d_23/pointwise_kernel
':% 2separable_conv2d_23/bias
8
Ì0
Í1
Î2"
trackable_list_wrapper
8
Ì0
Í1
Î2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
Ý2Ú
3__inference_separable_conv2d_23_layer_call_fn_80901¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_separable_conv2d_23_layer_call_and_return_conditional_losses_80916¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
Õ	variables
Ötrainable_variables
×regularization_losses
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_activation_32_layer_call_fn_80921¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_32_layer_call_and_return_conditional_losses_80926¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
Û	variables
Ütrainable_variables
Ýregularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_max_pooling2d_12_layer_call_fn_80931¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_80936¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5:3 2sep_out_0/depthwise_kernel
6:4 À2sep_out_0/pointwise_kernel
:À2sep_out_0/bias
8
á0
â1
ã2"
trackable_list_wrapper
8
á0
â1
ã2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
ä	variables
åtrainable_variables
æregularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_sep_out_0_layer_call_fn_80947¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_sep_out_0_layer_call_and_return_conditional_losses_80963¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ànon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
ê	variables
ëtrainable_variables
ìregularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_max_pooling2d_13_layer_call_fn_80968¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_80973¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5:3À2sep_out_1/depthwise_kernel
6:4Àà2sep_out_1/pointwise_kernel
:à2sep_out_1/bias
8
ð0
ñ1
ò2"
trackable_list_wrapper
8
ð0
ñ1
ò2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
ó	variables
ôtrainable_variables
õregularization_losses
÷__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_sep_out_1_layer_call_fn_80984¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_sep_out_1_layer_call_and_return_conditional_losses_81000¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
ù	variables
útrainable_variables
ûregularization_losses
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_max_pooling2d_14_layer_call_fn_81005¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_81010¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5:3à2sep_out_2/depthwise_kernel
6:4àà2sep_out_2/pointwise_kernel
:à2sep_out_2/bias
8
ÿ0
1
2"
trackable_list_wrapper
8
ÿ0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_sep_out_2_layer_call_fn_81021¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_sep_out_2_layer_call_and_return_conditional_losses_81037¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_max_pooling2d_15_layer_call_fn_81042¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_81047¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2:0à2output/depthwise_kernel
2:0à2output/pointwise_kernel
:2output/bias
8
0
1
2"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ùnon_trainable_variables
úlayers
ûmetrics
 ülayer_regularization_losses
ýlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_output_layer_call_fn_81058¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_output_layer_call_and_return_conditional_losses_81074¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 

C0
D1
b2
c3
4
5
°6
±7
É8
Ê9
ð10
ñ11
¾12
¿13"
trackable_list_wrapper

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
.45"
trackable_list_wrapper
0
þ0
ÿ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÊBÇ
#__inference_signature_wrapper_79974input_6"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
C0
D1"
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
b0
c1"
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
0
0
1"
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
°0
±1"
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
É0
Ê1"
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
ð0
ñ1"
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
0
¾0
¿1"
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

total

count
	variables
	keras_api"
_tf_keras_metric
c

total

count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_objectÅ
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_78772ó~89ABCDWX`abcwx¥¦®¯°±¾¿ÇÈÉÊäåæîïðñþÿ£¤¥²³´¼½¾¿ÌÍÎáâãðñòÿB¢?
8¢5
+(
input_6ÿÿÿÿÿÿÿÿÿàà
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Å
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_78961ó~89ABCDWX`abcwx¥¦®¯°±¾¿ÇÈÉÊäåæîïðñþÿ£¤¥²³´¼½¾¿ÌÍÎáâãðñòÿB¢?
8¢5
+(
input_6ÿÿÿÿÿÿÿÿÿàà
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ä
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_79525ò~89ABCDWX`abcwx¥¦®¯°±¾¿ÇÈÉÊäåæîïðñþÿ£¤¥²³´¼½¾¿ÌÍÎáâãðñòÿA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ä
M__inference_McFly_cnn_50epochs_layer_call_and_return_conditional_losses_79827ò~89ABCDWX`abcwx¥¦®¯°±¾¿ÇÈÉÊäåæîïðñþÿ£¤¥²³´¼½¾¿ÌÍÎáâãðñòÿA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
2__inference_McFly_cnn_50epochs_layer_call_fn_77735æ~89ABCDWX`abcwx¥¦®¯°±¾¿ÇÈÉÊäåæîïðñþÿ£¤¥²³´¼½¾¿ÌÍÎáâãðñòÿB¢?
8¢5
+(
input_6ÿÿÿÿÿÿÿÿÿàà
p 

 
ª " ÿÿÿÿÿÿÿÿÿ
2__inference_McFly_cnn_50epochs_layer_call_fn_78583æ~89ABCDWX`abcwx¥¦®¯°±¾¿ÇÈÉÊäåæîïðñþÿ£¤¥²³´¼½¾¿ÌÍÎáâãðñòÿB¢?
8¢5
+(
input_6ÿÿÿÿÿÿÿÿÿàà
p

 
ª " ÿÿÿÿÿÿÿÿÿ
2__inference_McFly_cnn_50epochs_layer_call_fn_79106å~89ABCDWX`abcwx¥¦®¯°±¾¿ÇÈÉÊäåæîïðñþÿ£¤¥²³´¼½¾¿ÌÍÎáâãðñòÿA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 

 
ª " ÿÿÿÿÿÿÿÿÿ
2__inference_McFly_cnn_50epochs_layer_call_fn_79251å~89ABCDWX`abcwx¥¦®¯°±¾¿ÇÈÉÊäåæîïðñþÿ£¤¥²³´¼½¾¿ÌÍÎáâãðñòÿA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p

 
ª " ÿÿÿÿÿÿÿÿÿ
 __inference__wrapped_model_76433õ~89ABCDWX`abcwx¥¦®¯°±¾¿ÇÈÉÊäåæîïðñþÿ£¤¥²³´¼½¾¿ÌÍÎáâãðñòÿ:¢7
0¢-
+(
input_6ÿÿÿÿÿÿÿÿÿàà
ª "7ª4
2
output(%
outputÿÿÿÿÿÿÿÿÿº
H__inference_activation_21_layer_call_and_return_conditional_losses_80065n:¢7
0¢-
+(
inputsÿÿÿÿÿÿÿÿÿààà
ª "0¢-
&#
0ÿÿÿÿÿÿÿÿÿààà
 
-__inference_activation_21_layer_call_fn_80060a:¢7
0¢-
+(
inputsÿÿÿÿÿÿÿÿÿààà
ª "# ÿÿÿÿÿÿÿÿÿààà¶
H__inference_activation_22_layer_call_and_return_conditional_losses_80166j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿpp
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿpp
 
-__inference_activation_22_layer_call_fn_80161]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿpp
ª "!ÿÿÿÿÿÿÿÿÿpp¶
H__inference_activation_23_layer_call_and_return_conditional_losses_80222j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿppà
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿppà
 
-__inference_activation_23_layer_call_fn_80217]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿppà
ª "!ÿÿÿÿÿÿÿÿÿppà¶
H__inference_activation_24_layer_call_and_return_conditional_losses_80313j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿpp 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿpp 
 
-__inference_activation_24_layer_call_fn_80308]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿpp 
ª "!ÿÿÿÿÿÿÿÿÿpp ´
H__inference_activation_25_layer_call_and_return_conditional_losses_80431h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿpp`
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿpp`
 
-__inference_activation_25_layer_call_fn_80426[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿpp`
ª " ÿÿÿÿÿÿÿÿÿpp`¶
H__inference_activation_26_layer_call_and_return_conditional_losses_80522j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿpp
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿpp
 
-__inference_activation_26_layer_call_fn_80517]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿpp
ª "!ÿÿÿÿÿÿÿÿÿpp¶
H__inference_activation_27_layer_call_and_return_conditional_losses_80657j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88À
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ88À
 
-__inference_activation_27_layer_call_fn_80652]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88À
ª "!ÿÿÿÿÿÿÿÿÿ88À¶
H__inference_activation_28_layer_call_and_return_conditional_losses_80693j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88à
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ88à
 
-__inference_activation_28_layer_call_fn_80688]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88à
ª "!ÿÿÿÿÿÿÿÿÿ88à¶
H__inference_activation_29_layer_call_and_return_conditional_losses_80729j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ88
 
-__inference_activation_29_layer_call_fn_80724]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88
ª "!ÿÿÿÿÿÿÿÿÿ88¶
H__inference_activation_30_layer_call_and_return_conditional_losses_80792j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ88
 
-__inference_activation_30_layer_call_fn_80787]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88
ª "!ÿÿÿÿÿÿÿÿÿ88¶
H__inference_activation_31_layer_call_and_return_conditional_losses_80890j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88à
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ88à
 
-__inference_activation_31_layer_call_fn_80885]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88à
ª "!ÿÿÿÿÿÿÿÿÿ88à¶
H__inference_activation_32_layer_call_and_return_conditional_losses_80926j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ88 
 
-__inference_activation_32_layer_call_fn_80921]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88 
ª "!ÿÿÿÿÿÿÿÿÿ88 î
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_80037ABCDN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 î
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_80055ABCDN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 Æ
6__inference_batch_normalization_11_layer_call_fn_80006ABCDN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàÆ
6__inference_batch_normalization_11_layer_call_fn_80019ABCDN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàî
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_80138`abcN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 î
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_80156`abcN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
6__inference_batch_normalization_12_layer_call_fn_80107`abcN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
6__inference_batch_normalization_12_layer_call_fn_80120`abcN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿò
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_80285N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ò
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_80303N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ê
6__inference_batch_normalization_13_layer_call_fn_80254N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ê
6__inference_batch_normalization_13_layer_call_fn_80267N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ð
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_80403®¯°±M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 ð
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_80421®¯°±M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 È
6__inference_batch_normalization_14_layer_call_fn_80372®¯°±M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`È
6__inference_batch_normalization_14_layer_call_fn_80385®¯°±M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`ò
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_80494ÇÈÉÊN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ò
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_80512ÇÈÉÊN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ê
6__inference_batch_normalization_15_layer_call_fn_80463ÇÈÉÊN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÊ
6__inference_batch_normalization_15_layer_call_fn_80476ÇÈÉÊN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿò
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_80629îïðñN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 ò
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_80647îïðñN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 Ê
6__inference_batch_normalization_16_layer_call_fn_80598îïðñN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀÊ
6__inference_batch_normalization_16_layer_call_fn_80611îïðñN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀò
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_80862¼½¾¿N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 ò
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_80880¼½¾¿N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 Ê
6__inference_batch_normalization_17_layer_call_fn_80831¼½¾¿N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàÊ
6__inference_batch_normalization_17_layer_call_fn_80844¼½¾¿N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà¸
D__inference_conv2d_10_layer_call_and_return_conditional_losses_80241p8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿppà
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿpp 
 
)__inference_conv2d_10_layer_call_fn_80231c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿppà
ª "!ÿÿÿÿÿÿÿÿÿpp ·
D__inference_conv2d_11_layer_call_and_return_conditional_losses_80359o¥¦8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿpp 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿpp`
 
)__inference_conv2d_11_layer_call_fn_80349b¥¦8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿpp 
ª " ÿÿÿÿÿÿÿÿÿpp`·
D__inference_conv2d_12_layer_call_and_return_conditional_losses_80450o¾¿7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿpp`
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿpp
 
)__inference_conv2d_12_layer_call_fn_80440b¾¿7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿpp`
ª "!ÿÿÿÿÿÿÿÿÿppµ
C__inference_conv2d_8_layer_call_and_return_conditional_losses_80094nWX8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿppà
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿpp
 
(__inference_conv2d_8_layer_call_fn_80084aWX8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿppà
ª "!ÿÿÿÿÿÿÿÿÿppµ
C__inference_conv2d_9_layer_call_and_return_conditional_losses_80212nwx8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿpp
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿppà
 
(__inference_conv2d_9_layer_call_fn_80202awx8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿpp
ª "!ÿÿÿÿÿÿÿÿÿppà¶
D__inference_dropout_2_layer_call_and_return_conditional_losses_80181n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿpp
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿpp
 ¶
D__inference_dropout_2_layer_call_and_return_conditional_losses_80193n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿpp
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿpp
 
)__inference_dropout_2_layer_call_fn_80171a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿpp
p 
ª "!ÿÿÿÿÿÿÿÿÿpp
)__inference_dropout_2_layer_call_fn_80176a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿpp
p
ª "!ÿÿÿÿÿÿÿÿÿpp¶
D__inference_dropout_3_layer_call_and_return_conditional_losses_80328n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿpp 
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿpp 
 ¶
D__inference_dropout_3_layer_call_and_return_conditional_losses_80340n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿpp 
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿpp 
 
)__inference_dropout_3_layer_call_fn_80318a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿpp 
p 
ª "!ÿÿÿÿÿÿÿÿÿpp 
)__inference_dropout_3_layer_call_fn_80323a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿpp 
p
ª "!ÿÿÿÿÿÿÿÿÿpp ¶
D__inference_dropout_4_layer_call_and_return_conditional_losses_80537n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿpp
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿpp
 ¶
D__inference_dropout_4_layer_call_and_return_conditional_losses_80549n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿpp
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿpp
 
)__inference_dropout_4_layer_call_fn_80527a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿpp
p 
ª "!ÿÿÿÿÿÿÿÿÿpp
)__inference_dropout_4_layer_call_fn_80532a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿpp
p
ª "!ÿÿÿÿÿÿÿÿÿpp¶
D__inference_dropout_5_layer_call_and_return_conditional_losses_80744n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ88
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ88
 ¶
D__inference_dropout_5_layer_call_and_return_conditional_losses_80756n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ88
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ88
 
)__inference_dropout_5_layer_call_fn_80734a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ88
p 
ª "!ÿÿÿÿÿÿÿÿÿ88
)__inference_dropout_5_layer_call_fn_80739a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ88
p
ª "!ÿÿÿÿÿÿÿÿÿ88º
E__inference_first_conv_layer_call_and_return_conditional_losses_79993q899¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿàà
ª "0¢-
&#
0ÿÿÿÿÿÿÿÿÿààà
 
*__inference_first_conv_layer_call_fn_79983d899¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿàà
ª "# ÿÿÿÿÿÿÿÿÿàààî
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_80075R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_10_layer_call_fn_80070R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_80559R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_11_layer_call_fn_80554R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_80936R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_12_layer_call_fn_80931R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_80973R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_13_layer_call_fn_80968R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_81010R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_14_layer_call_fn_81005R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_81047R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_15_layer_call_fn_81042R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÛ
A__inference_output_layer_call_and_return_conditional_losses_81074J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ³
&__inference_output_layer_call_fn_81058J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿß
D__inference_sep_out_0_layer_call_and_return_conditional_losses_80963áâãJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 ·
)__inference_sep_out_0_layer_call_fn_80947áâãJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀß
D__inference_sep_out_1_layer_call_and_return_conditional_losses_81000ðñòJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 ·
)__inference_sep_out_1_layer_call_fn_80984ðñòJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàß
D__inference_sep_out_2_layer_call_and_return_conditional_losses_81037ÿJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 ·
)__inference_sep_out_2_layer_call_fn_81021ÿJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàé
N__inference_separable_conv2d_18_layer_call_and_return_conditional_losses_80585äåæJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 Á
3__inference_separable_conv2d_18_layer_call_fn_80570äåæJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀé
N__inference_separable_conv2d_19_layer_call_and_return_conditional_losses_80683þÿJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 Á
3__inference_separable_conv2d_19_layer_call_fn_80668þÿJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàé
N__inference_separable_conv2d_20_layer_call_and_return_conditional_losses_80719J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Á
3__inference_separable_conv2d_20_layer_call_fn_80704J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿé
N__inference_separable_conv2d_21_layer_call_and_return_conditional_losses_80782£¤¥J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Á
3__inference_separable_conv2d_21_layer_call_fn_80767£¤¥J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿé
N__inference_separable_conv2d_22_layer_call_and_return_conditional_losses_80818²³´J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
 Á
3__inference_separable_conv2d_22_layer_call_fn_80803²³´J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàé
N__inference_separable_conv2d_23_layer_call_and_return_conditional_losses_80916ÌÍÎJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Á
3__inference_separable_conv2d_23_layer_call_fn_80901ÌÍÎJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¨
#__inference_signature_wrapper_79974~89ABCDWX`abcwx¥¦®¯°±¾¿ÇÈÉÊäåæîïðñþÿ£¤¥²³´¼½¾¿ÌÍÎáâãðñòÿE¢B
¢ 
;ª8
6
input_6+(
input_6ÿÿÿÿÿÿÿÿÿàà"7ª4
2
output(%
outputÿÿÿÿÿÿÿÿÿ