       £K"	  АҐK6„Abrain.Event:2ЛФ§ШБЭ     |IЩЖ	ЌДҐK6„A"фЇ
П
)Adam/iterations/Initializer/initial_valueConst*
dtype0	*
_output_shapes
: *"
_class
loc:@Adam/iterations*
value	B	 R 
І
Adam/iterationsVarHandleOp* 
shared_nameAdam/iterations*"
_class
loc:@Adam/iterations*
	container *
shape: *
dtype0	*
_output_shapes
: 
o
0Adam/iterations/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/iterations*
_output_shapes
: 
Ч
Adam/iterations/AssignAssignVariableOpAdam/iterations)Adam/iterations/Initializer/initial_value*"
_class
loc:@Adam/iterations*
dtype0	
П
#Adam/iterations/Read/ReadVariableOpReadVariableOpAdam/iterations*
dtype0	*
_output_shapes
: *"
_class
loc:@Adam/iterations
В
!Adam/lr/Initializer/initial_valueConst*
_class
loc:@Adam/lr*
valueB
 *љ7Ж5*
dtype0*
_output_shapes
: 
П
Adam/lrVarHandleOp*
dtype0*
_output_shapes
: *
shared_name	Adam/lr*
_class
loc:@Adam/lr*
	container *
shape: 
_
(Adam/lr/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/lr*
_output_shapes
: 
w
Adam/lr/AssignAssignVariableOpAdam/lr!Adam/lr/Initializer/initial_value*
_class
loc:@Adam/lr*
dtype0
w
Adam/lr/Read/ReadVariableOpReadVariableOpAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 
К
%Adam/beta_1/Initializer/initial_valueConst*
_class
loc:@Adam/beta_1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Ы
Adam/beta_1VarHandleOp*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_nameAdam/beta_1*
_class
loc:@Adam/beta_1
g
,Adam/beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/beta_1*
_output_shapes
: 
З
Adam/beta_1/AssignAssignVariableOpAdam/beta_1%Adam/beta_1/Initializer/initial_value*
_class
loc:@Adam/beta_1*
dtype0
Г
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 
К
%Adam/beta_2/Initializer/initial_valueConst*
_class
loc:@Adam/beta_2*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
Ы
Adam/beta_2VarHandleOp*
shape: *
dtype0*
_output_shapes
: *
shared_nameAdam/beta_2*
_class
loc:@Adam/beta_2*
	container 
g
,Adam/beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/beta_2*
_output_shapes
: 
З
Adam/beta_2/AssignAssignVariableOpAdam/beta_2%Adam/beta_2/Initializer/initial_value*
_class
loc:@Adam/beta_2*
dtype0
Г
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 
И
$Adam/decay/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@Adam/decay*
valueB
 *    
Ш

Adam/decayVarHandleOp*
dtype0*
_output_shapes
: *
shared_name
Adam/decay*
_class
loc:@Adam/decay*
	container *
shape: 
e
+Adam/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Adam/decay*
_output_shapes
: 
Г
Adam/decay/AssignAssignVariableOp
Adam/decay$Adam/decay/Initializer/initial_value*
dtype0*
_class
loc:@Adam/decay
А
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: *
_class
loc:@Adam/decay
•
0dense_12/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@dense_12/kernel*
valueB"   d   
Ч
.dense_12/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_12/kernel*
valueB
 *чzXљ*
dtype0*
_output_shapes
: 
Ч
.dense_12/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@dense_12/kernel*
valueB
 *чzX=*
dtype0*
_output_shapes
: 
п
8dense_12/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_12/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes
:	Аd*

seed *
T0*"
_class
loc:@dense_12/kernel
Џ
.dense_12/kernel/Initializer/random_uniform/subSub.dense_12/kernel/Initializer/random_uniform/max.dense_12/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@dense_12/kernel
н
.dense_12/kernel/Initializer/random_uniform/mulMul8dense_12/kernel/Initializer/random_uniform/RandomUniform.dense_12/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_12/kernel*
_output_shapes
:	Аd
я
*dense_12/kernel/Initializer/random_uniformAdd.dense_12/kernel/Initializer/random_uniform/mul.dense_12/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_12/kernel*
_output_shapes
:	Аd
∞
dense_12/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_namedense_12/kernel*"
_class
loc:@dense_12/kernel*
	container *
shape:	Аd
o
0dense_12/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_12/kernel*
_output_shapes
: 
Ш
dense_12/kernel/AssignAssignVariableOpdense_12/kernel*dense_12/kernel/Initializer/random_uniform*
dtype0*"
_class
loc:@dense_12/kernel
Ш
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*"
_class
loc:@dense_12/kernel*
dtype0*
_output_shapes
:	Аd
О
dense_12/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:d* 
_class
loc:@dense_12/bias*
valueBd*    
•
dense_12/biasVarHandleOp*
	container *
shape:d*
dtype0*
_output_shapes
: *
shared_namedense_12/bias* 
_class
loc:@dense_12/bias
k
.dense_12/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_12/bias*
_output_shapes
: 
З
dense_12/bias/AssignAssignVariableOpdense_12/biasdense_12/bias/Initializer/zeros* 
_class
loc:@dense_12/bias*
dtype0
Н
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
dtype0*
_output_shapes
:d* 
_class
loc:@dense_12/bias
•
0dense_13/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@dense_13/kernel*
valueB"d   d   *
dtype0*
_output_shapes
:
Ч
.dense_13/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_13/kernel*
valueB
 *ђ\1Њ*
dtype0*
_output_shapes
: 
Ч
.dense_13/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@dense_13/kernel*
valueB
 *ђ\1>*
dtype0*
_output_shapes
: 
о
8dense_13/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_13/kernel/Initializer/random_uniform/shape*

seed *
T0*"
_class
loc:@dense_13/kernel*
seed2 *
dtype0*
_output_shapes

:dd
Џ
.dense_13/kernel/Initializer/random_uniform/subSub.dense_13/kernel/Initializer/random_uniform/max.dense_13/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes
: 
м
.dense_13/kernel/Initializer/random_uniform/mulMul8dense_13/kernel/Initializer/random_uniform/RandomUniform.dense_13/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes

:dd
ё
*dense_13/kernel/Initializer/random_uniformAdd.dense_13/kernel/Initializer/random_uniform/mul.dense_13/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes

:dd
ѓ
dense_13/kernelVarHandleOp* 
shared_namedense_13/kernel*"
_class
loc:@dense_13/kernel*
	container *
shape
:dd*
dtype0*
_output_shapes
: 
o
0dense_13/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_13/kernel*
_output_shapes
: 
Ш
dense_13/kernel/AssignAssignVariableOpdense_13/kernel*dense_13/kernel/Initializer/random_uniform*"
_class
loc:@dense_13/kernel*
dtype0
Ч
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*"
_class
loc:@dense_13/kernel*
dtype0*
_output_shapes

:dd
О
dense_13/bias/Initializer/zerosConst* 
_class
loc:@dense_13/bias*
valueBd*    *
dtype0*
_output_shapes
:d
•
dense_13/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_13/bias* 
_class
loc:@dense_13/bias*
	container *
shape:d
k
.dense_13/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_13/bias*
_output_shapes
: 
З
dense_13/bias/AssignAssignVariableOpdense_13/biasdense_13/bias/Initializer/zeros* 
_class
loc:@dense_13/bias*
dtype0
Н
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias* 
_class
loc:@dense_13/bias*
dtype0*
_output_shapes
:d
•
0dense_14/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@dense_14/kernel*
valueB"d   d   *
dtype0*
_output_shapes
:
Ч
.dense_14/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_14/kernel*
valueB
 *ђ\1Њ*
dtype0*
_output_shapes
: 
Ч
.dense_14/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@dense_14/kernel*
valueB
 *ђ\1>*
dtype0*
_output_shapes
: 
о
8dense_14/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_14/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:dd*

seed *
T0*"
_class
loc:@dense_14/kernel*
seed2 
Џ
.dense_14/kernel/Initializer/random_uniform/subSub.dense_14/kernel/Initializer/random_uniform/max.dense_14/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_14/kernel*
_output_shapes
: 
м
.dense_14/kernel/Initializer/random_uniform/mulMul8dense_14/kernel/Initializer/random_uniform/RandomUniform.dense_14/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_14/kernel*
_output_shapes

:dd
ё
*dense_14/kernel/Initializer/random_uniformAdd.dense_14/kernel/Initializer/random_uniform/mul.dense_14/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_14/kernel*
_output_shapes

:dd
ѓ
dense_14/kernelVarHandleOp*
	container *
shape
:dd*
dtype0*
_output_shapes
: * 
shared_namedense_14/kernel*"
_class
loc:@dense_14/kernel
o
0dense_14/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_14/kernel*
_output_shapes
: 
Ш
dense_14/kernel/AssignAssignVariableOpdense_14/kernel*dense_14/kernel/Initializer/random_uniform*
dtype0*"
_class
loc:@dense_14/kernel
Ч
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*"
_class
loc:@dense_14/kernel*
dtype0*
_output_shapes

:dd
О
dense_14/bias/Initializer/zerosConst* 
_class
loc:@dense_14/bias*
valueBd*    *
dtype0*
_output_shapes
:d
•
dense_14/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_14/bias* 
_class
loc:@dense_14/bias*
	container *
shape:d
k
.dense_14/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_14/bias*
_output_shapes
: 
З
dense_14/bias/AssignAssignVariableOpdense_14/biasdense_14/bias/Initializer/zeros* 
_class
loc:@dense_14/bias*
dtype0
Н
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias* 
_class
loc:@dense_14/bias*
dtype0*
_output_shapes
:d
•
0dense_15/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@dense_15/kernel*
valueB"d   d   *
dtype0*
_output_shapes
:
Ч
.dense_15/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_15/kernel*
valueB
 *ђ\1Њ*
dtype0*
_output_shapes
: 
Ч
.dense_15/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@dense_15/kernel*
valueB
 *ђ\1>*
dtype0*
_output_shapes
: 
о
8dense_15/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_15/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes

:dd*

seed *
T0*"
_class
loc:@dense_15/kernel
Џ
.dense_15/kernel/Initializer/random_uniform/subSub.dense_15/kernel/Initializer/random_uniform/max.dense_15/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_15/kernel*
_output_shapes
: 
м
.dense_15/kernel/Initializer/random_uniform/mulMul8dense_15/kernel/Initializer/random_uniform/RandomUniform.dense_15/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_15/kernel*
_output_shapes

:dd
ё
*dense_15/kernel/Initializer/random_uniformAdd.dense_15/kernel/Initializer/random_uniform/mul.dense_15/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_15/kernel*
_output_shapes

:dd
ѓ
dense_15/kernelVarHandleOp* 
shared_namedense_15/kernel*"
_class
loc:@dense_15/kernel*
	container *
shape
:dd*
dtype0*
_output_shapes
: 
o
0dense_15/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_15/kernel*
_output_shapes
: 
Ш
dense_15/kernel/AssignAssignVariableOpdense_15/kernel*dense_15/kernel/Initializer/random_uniform*"
_class
loc:@dense_15/kernel*
dtype0
Ч
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*"
_class
loc:@dense_15/kernel*
dtype0*
_output_shapes

:dd
О
dense_15/bias/Initializer/zerosConst* 
_class
loc:@dense_15/bias*
valueBd*    *
dtype0*
_output_shapes
:d
•
dense_15/biasVarHandleOp*
shared_namedense_15/bias* 
_class
loc:@dense_15/bias*
	container *
shape:d*
dtype0*
_output_shapes
: 
k
.dense_15/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_15/bias*
_output_shapes
: 
З
dense_15/bias/AssignAssignVariableOpdense_15/biasdense_15/bias/Initializer/zeros* 
_class
loc:@dense_15/bias*
dtype0
Н
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
dtype0*
_output_shapes
:d* 
_class
loc:@dense_15/bias
•
0dense_16/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@dense_16/kernel*
valueB"d   d   *
dtype0*
_output_shapes
:
Ч
.dense_16/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_16/kernel*
valueB
 *ђ\1Њ*
dtype0*
_output_shapes
: 
Ч
.dense_16/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@dense_16/kernel*
valueB
 *ђ\1>*
dtype0*
_output_shapes
: 
о
8dense_16/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_16/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@dense_16/kernel*
seed2 *
dtype0*
_output_shapes

:dd*

seed 
Џ
.dense_16/kernel/Initializer/random_uniform/subSub.dense_16/kernel/Initializer/random_uniform/max.dense_16/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@dense_16/kernel
м
.dense_16/kernel/Initializer/random_uniform/mulMul8dense_16/kernel/Initializer/random_uniform/RandomUniform.dense_16/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_16/kernel*
_output_shapes

:dd
ё
*dense_16/kernel/Initializer/random_uniformAdd.dense_16/kernel/Initializer/random_uniform/mul.dense_16/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_16/kernel*
_output_shapes

:dd
ѓ
dense_16/kernelVarHandleOp*
shape
:dd*
dtype0*
_output_shapes
: * 
shared_namedense_16/kernel*"
_class
loc:@dense_16/kernel*
	container 
o
0dense_16/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_16/kernel*
_output_shapes
: 
Ш
dense_16/kernel/AssignAssignVariableOpdense_16/kernel*dense_16/kernel/Initializer/random_uniform*"
_class
loc:@dense_16/kernel*
dtype0
Ч
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
dtype0*
_output_shapes

:dd*"
_class
loc:@dense_16/kernel
О
dense_16/bias/Initializer/zerosConst* 
_class
loc:@dense_16/bias*
valueBd*    *
dtype0*
_output_shapes
:d
•
dense_16/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_16/bias* 
_class
loc:@dense_16/bias*
	container *
shape:d
k
.dense_16/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_16/bias*
_output_shapes
: 
З
dense_16/bias/AssignAssignVariableOpdense_16/biasdense_16/bias/Initializer/zeros* 
_class
loc:@dense_16/bias*
dtype0
Н
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias* 
_class
loc:@dense_16/bias*
dtype0*
_output_shapes
:d
•
0dense_17/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@dense_17/kernel*
valueB"d      *
dtype0*
_output_shapes
:
Ч
.dense_17/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_17/kernel*
valueB
 *B[xЊ*
dtype0*
_output_shapes
: 
Ч
.dense_17/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@dense_17/kernel*
valueB
 *B[x>*
dtype0*
_output_shapes
: 
о
8dense_17/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_17/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@dense_17/kernel*
seed2 *
dtype0*
_output_shapes

:d*

seed 
Џ
.dense_17/kernel/Initializer/random_uniform/subSub.dense_17/kernel/Initializer/random_uniform/max.dense_17/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_17/kernel*
_output_shapes
: 
м
.dense_17/kernel/Initializer/random_uniform/mulMul8dense_17/kernel/Initializer/random_uniform/RandomUniform.dense_17/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_17/kernel*
_output_shapes

:d
ё
*dense_17/kernel/Initializer/random_uniformAdd.dense_17/kernel/Initializer/random_uniform/mul.dense_17/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_17/kernel*
_output_shapes

:d
ѓ
dense_17/kernelVarHandleOp*
	container *
shape
:d*
dtype0*
_output_shapes
: * 
shared_namedense_17/kernel*"
_class
loc:@dense_17/kernel
o
0dense_17/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_17/kernel*
_output_shapes
: 
Ш
dense_17/kernel/AssignAssignVariableOpdense_17/kernel*dense_17/kernel/Initializer/random_uniform*"
_class
loc:@dense_17/kernel*
dtype0
Ч
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
dtype0*
_output_shapes

:d*"
_class
loc:@dense_17/kernel
О
dense_17/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:* 
_class
loc:@dense_17/bias*
valueB*    
•
dense_17/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_17/bias* 
_class
loc:@dense_17/bias*
	container *
shape:
k
.dense_17/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_17/bias*
_output_shapes
: 
З
dense_17/bias/AssignAssignVariableOpdense_17/biasdense_17/bias/Initializer/zeros*
dtype0* 
_class
loc:@dense_17/bias
Н
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias* 
_class
loc:@dense_17/bias*
dtype0*
_output_shapes
:
l
input_1Placeholder*
shape:€€€€€€€€€А*
dtype0*(
_output_shapes
:€€€€€€€€€А
f
MatMul/ReadVariableOpReadVariableOpdense_12/kernel*
dtype0*
_output_shapes
:	Аd
И
MatMulMatMulinput_1MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b( 
`
BiasAdd/ReadVariableOpReadVariableOpdense_12/bias*
dtype0*
_output_shapes
:d
{
BiasAddBiasAddMatMulBiasAdd/ReadVariableOp*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d*
T0
G
ReluReluBiasAdd*'
_output_shapes
:€€€€€€€€€d*
T0
\
keras_learning_phase/inputConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 
d
cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
I
cond/switch_tIdentitycond/Switch:1*
T0
*
_output_shapes
: 
G
cond/switch_fIdentitycond/Switch*
_output_shapes
: *
T0

O
cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
k
cond/dropout/keep_probConst^cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
m
cond/dropout/ShapeShapecond/dropout/Shape/Switch:1*
_output_shapes
:*
T0*
out_type0
Х
cond/dropout/Shape/SwitchSwitchRelucond/pred_id*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d*
T0*
_class
	loc:@Relu
t
cond/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
t
cond/dropout/random_uniform/maxConst^cond/switch_t*
valueB
 *  А?*
dtype0*
_output_shapes
: 
¶
)cond/dropout/random_uniform/RandomUniformRandomUniformcond/dropout/Shape*

seed *
T0*
dtype0*'
_output_shapes
:€€€€€€€€€d*
seed2 
Й
cond/dropout/random_uniform/subSubcond/dropout/random_uniform/maxcond/dropout/random_uniform/min*
T0*
_output_shapes
: 
§
cond/dropout/random_uniform/mulMul)cond/dropout/random_uniform/RandomUniformcond/dropout/random_uniform/sub*'
_output_shapes
:€€€€€€€€€d*
T0
Ц
cond/dropout/random_uniformAddcond/dropout/random_uniform/mulcond/dropout/random_uniform/min*
T0*'
_output_shapes
:€€€€€€€€€d
~
cond/dropout/addAddcond/dropout/keep_probcond/dropout/random_uniform*
T0*'
_output_shapes
:€€€€€€€€€d
_
cond/dropout/FloorFloorcond/dropout/add*
T0*'
_output_shapes
:€€€€€€€€€d
В
cond/dropout/divRealDivcond/dropout/Shape/Switch:1cond/dropout/keep_prob*
T0*'
_output_shapes
:€€€€€€€€€d
o
cond/dropout/mulMulcond/dropout/divcond/dropout/Floor*'
_output_shapes
:€€€€€€€€€d*
T0
a
cond/IdentityIdentitycond/Identity/Switch*'
_output_shapes
:€€€€€€€€€d*
T0
Р
cond/Identity/SwitchSwitchRelucond/pred_id*
T0*
_class
	loc:@Relu*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
q

cond/MergeMergecond/Identitycond/dropout/mul*
T0*
N*)
_output_shapes
:€€€€€€€€€d: 
g
MatMul_1/ReadVariableOpReadVariableOpdense_13/kernel*
dtype0*
_output_shapes

:dd
П
MatMul_1MatMul
cond/MergeMatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b( 
b
BiasAdd_1/ReadVariableOpReadVariableOpdense_13/bias*
dtype0*
_output_shapes
:d
Б
	BiasAdd_1BiasAddMatMul_1BiasAdd_1/ReadVariableOp*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d*
T0
K
Relu_1Relu	BiasAdd_1*
T0*'
_output_shapes
:€€€€€€€€€d
f
cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
M
cond_1/switch_tIdentitycond_1/Switch:1*
_output_shapes
: *
T0

K
cond_1/switch_fIdentitycond_1/Switch*
T0
*
_output_shapes
: 
Q
cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
o
cond_1/dropout/keep_probConst^cond_1/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
q
cond_1/dropout/ShapeShapecond_1/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
Э
cond_1/dropout/Shape/SwitchSwitchRelu_1cond_1/pred_id*
T0*
_class
loc:@Relu_1*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
x
!cond_1/dropout/random_uniform/minConst^cond_1/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
x
!cond_1/dropout/random_uniform/maxConst^cond_1/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  А?
™
+cond_1/dropout/random_uniform/RandomUniformRandomUniformcond_1/dropout/Shape*
T0*
dtype0*'
_output_shapes
:€€€€€€€€€d*
seed2 *

seed 
П
!cond_1/dropout/random_uniform/subSub!cond_1/dropout/random_uniform/max!cond_1/dropout/random_uniform/min*
T0*
_output_shapes
: 
™
!cond_1/dropout/random_uniform/mulMul+cond_1/dropout/random_uniform/RandomUniform!cond_1/dropout/random_uniform/sub*
T0*'
_output_shapes
:€€€€€€€€€d
Ь
cond_1/dropout/random_uniformAdd!cond_1/dropout/random_uniform/mul!cond_1/dropout/random_uniform/min*
T0*'
_output_shapes
:€€€€€€€€€d
Д
cond_1/dropout/addAddcond_1/dropout/keep_probcond_1/dropout/random_uniform*
T0*'
_output_shapes
:€€€€€€€€€d
c
cond_1/dropout/FloorFloorcond_1/dropout/add*
T0*'
_output_shapes
:€€€€€€€€€d
И
cond_1/dropout/divRealDivcond_1/dropout/Shape/Switch:1cond_1/dropout/keep_prob*
T0*'
_output_shapes
:€€€€€€€€€d
u
cond_1/dropout/mulMulcond_1/dropout/divcond_1/dropout/Floor*
T0*'
_output_shapes
:€€€€€€€€€d
e
cond_1/IdentityIdentitycond_1/Identity/Switch*'
_output_shapes
:€€€€€€€€€d*
T0
Ш
cond_1/Identity/SwitchSwitchRelu_1cond_1/pred_id*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d*
T0*
_class
loc:@Relu_1
w
cond_1/MergeMergecond_1/Identitycond_1/dropout/mul*
N*)
_output_shapes
:€€€€€€€€€d: *
T0
g
MatMul_2/ReadVariableOpReadVariableOpdense_14/kernel*
dtype0*
_output_shapes

:dd
С
MatMul_2MatMulcond_1/MergeMatMul_2/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b( 
b
BiasAdd_2/ReadVariableOpReadVariableOpdense_14/bias*
dtype0*
_output_shapes
:d
Б
	BiasAdd_2BiasAddMatMul_2BiasAdd_2/ReadVariableOp*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d*
T0
K
Relu_2Relu	BiasAdd_2*
T0*'
_output_shapes
:€€€€€€€€€d
f
cond_2/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
M
cond_2/switch_tIdentitycond_2/Switch:1*
T0
*
_output_shapes
: 
K
cond_2/switch_fIdentitycond_2/Switch*
T0
*
_output_shapes
: 
Q
cond_2/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
o
cond_2/dropout/keep_probConst^cond_2/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
q
cond_2/dropout/ShapeShapecond_2/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
Э
cond_2/dropout/Shape/SwitchSwitchRelu_2cond_2/pred_id*
T0*
_class
loc:@Relu_2*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
x
!cond_2/dropout/random_uniform/minConst^cond_2/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
x
!cond_2/dropout/random_uniform/maxConst^cond_2/switch_t*
valueB
 *  А?*
dtype0*
_output_shapes
: 
™
+cond_2/dropout/random_uniform/RandomUniformRandomUniformcond_2/dropout/Shape*
T0*
dtype0*'
_output_shapes
:€€€€€€€€€d*
seed2 *

seed 
П
!cond_2/dropout/random_uniform/subSub!cond_2/dropout/random_uniform/max!cond_2/dropout/random_uniform/min*
T0*
_output_shapes
: 
™
!cond_2/dropout/random_uniform/mulMul+cond_2/dropout/random_uniform/RandomUniform!cond_2/dropout/random_uniform/sub*
T0*'
_output_shapes
:€€€€€€€€€d
Ь
cond_2/dropout/random_uniformAdd!cond_2/dropout/random_uniform/mul!cond_2/dropout/random_uniform/min*
T0*'
_output_shapes
:€€€€€€€€€d
Д
cond_2/dropout/addAddcond_2/dropout/keep_probcond_2/dropout/random_uniform*'
_output_shapes
:€€€€€€€€€d*
T0
c
cond_2/dropout/FloorFloorcond_2/dropout/add*
T0*'
_output_shapes
:€€€€€€€€€d
И
cond_2/dropout/divRealDivcond_2/dropout/Shape/Switch:1cond_2/dropout/keep_prob*
T0*'
_output_shapes
:€€€€€€€€€d
u
cond_2/dropout/mulMulcond_2/dropout/divcond_2/dropout/Floor*
T0*'
_output_shapes
:€€€€€€€€€d
e
cond_2/IdentityIdentitycond_2/Identity/Switch*
T0*'
_output_shapes
:€€€€€€€€€d
Ш
cond_2/Identity/SwitchSwitchRelu_2cond_2/pred_id*
T0*
_class
loc:@Relu_2*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
w
cond_2/MergeMergecond_2/Identitycond_2/dropout/mul*
T0*
N*)
_output_shapes
:€€€€€€€€€d: 
g
MatMul_3/ReadVariableOpReadVariableOpdense_15/kernel*
dtype0*
_output_shapes

:dd
С
MatMul_3MatMulcond_2/MergeMatMul_3/ReadVariableOp*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b( *
T0
b
BiasAdd_3/ReadVariableOpReadVariableOpdense_15/bias*
dtype0*
_output_shapes
:d
Б
	BiasAdd_3BiasAddMatMul_3BiasAdd_3/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d
K
Relu_3Relu	BiasAdd_3*
T0*'
_output_shapes
:€€€€€€€€€d
f
cond_3/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
M
cond_3/switch_tIdentitycond_3/Switch:1*
_output_shapes
: *
T0

K
cond_3/switch_fIdentitycond_3/Switch*
_output_shapes
: *
T0

Q
cond_3/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

o
cond_3/dropout/keep_probConst^cond_3/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
q
cond_3/dropout/ShapeShapecond_3/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
Э
cond_3/dropout/Shape/SwitchSwitchRelu_3cond_3/pred_id*
T0*
_class
loc:@Relu_3*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
x
!cond_3/dropout/random_uniform/minConst^cond_3/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
x
!cond_3/dropout/random_uniform/maxConst^cond_3/switch_t*
valueB
 *  А?*
dtype0*
_output_shapes
: 
™
+cond_3/dropout/random_uniform/RandomUniformRandomUniformcond_3/dropout/Shape*
T0*
dtype0*'
_output_shapes
:€€€€€€€€€d*
seed2 *

seed 
П
!cond_3/dropout/random_uniform/subSub!cond_3/dropout/random_uniform/max!cond_3/dropout/random_uniform/min*
T0*
_output_shapes
: 
™
!cond_3/dropout/random_uniform/mulMul+cond_3/dropout/random_uniform/RandomUniform!cond_3/dropout/random_uniform/sub*
T0*'
_output_shapes
:€€€€€€€€€d
Ь
cond_3/dropout/random_uniformAdd!cond_3/dropout/random_uniform/mul!cond_3/dropout/random_uniform/min*
T0*'
_output_shapes
:€€€€€€€€€d
Д
cond_3/dropout/addAddcond_3/dropout/keep_probcond_3/dropout/random_uniform*
T0*'
_output_shapes
:€€€€€€€€€d
c
cond_3/dropout/FloorFloorcond_3/dropout/add*
T0*'
_output_shapes
:€€€€€€€€€d
И
cond_3/dropout/divRealDivcond_3/dropout/Shape/Switch:1cond_3/dropout/keep_prob*
T0*'
_output_shapes
:€€€€€€€€€d
u
cond_3/dropout/mulMulcond_3/dropout/divcond_3/dropout/Floor*
T0*'
_output_shapes
:€€€€€€€€€d
e
cond_3/IdentityIdentitycond_3/Identity/Switch*
T0*'
_output_shapes
:€€€€€€€€€d
Ш
cond_3/Identity/SwitchSwitchRelu_3cond_3/pred_id*
T0*
_class
loc:@Relu_3*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
w
cond_3/MergeMergecond_3/Identitycond_3/dropout/mul*
T0*
N*)
_output_shapes
:€€€€€€€€€d: 
g
MatMul_4/ReadVariableOpReadVariableOpdense_16/kernel*
dtype0*
_output_shapes

:dd
С
MatMul_4MatMulcond_3/MergeMatMul_4/ReadVariableOp*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b( *
T0
b
BiasAdd_4/ReadVariableOpReadVariableOpdense_16/bias*
dtype0*
_output_shapes
:d
Б
	BiasAdd_4BiasAddMatMul_4BiasAdd_4/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d
K
Relu_4Relu	BiasAdd_4*
T0*'
_output_shapes
:€€€€€€€€€d
f
cond_4/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

M
cond_4/switch_tIdentitycond_4/Switch:1*
_output_shapes
: *
T0

K
cond_4/switch_fIdentitycond_4/Switch*
T0
*
_output_shapes
: 
Q
cond_4/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

o
cond_4/dropout/keep_probConst^cond_4/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
q
cond_4/dropout/ShapeShapecond_4/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
Э
cond_4/dropout/Shape/SwitchSwitchRelu_4cond_4/pred_id*
T0*
_class
loc:@Relu_4*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
x
!cond_4/dropout/random_uniform/minConst^cond_4/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
x
!cond_4/dropout/random_uniform/maxConst^cond_4/switch_t*
valueB
 *  А?*
dtype0*
_output_shapes
: 
™
+cond_4/dropout/random_uniform/RandomUniformRandomUniformcond_4/dropout/Shape*
T0*
dtype0*'
_output_shapes
:€€€€€€€€€d*
seed2 *

seed 
П
!cond_4/dropout/random_uniform/subSub!cond_4/dropout/random_uniform/max!cond_4/dropout/random_uniform/min*
T0*
_output_shapes
: 
™
!cond_4/dropout/random_uniform/mulMul+cond_4/dropout/random_uniform/RandomUniform!cond_4/dropout/random_uniform/sub*
T0*'
_output_shapes
:€€€€€€€€€d
Ь
cond_4/dropout/random_uniformAdd!cond_4/dropout/random_uniform/mul!cond_4/dropout/random_uniform/min*
T0*'
_output_shapes
:€€€€€€€€€d
Д
cond_4/dropout/addAddcond_4/dropout/keep_probcond_4/dropout/random_uniform*
T0*'
_output_shapes
:€€€€€€€€€d
c
cond_4/dropout/FloorFloorcond_4/dropout/add*
T0*'
_output_shapes
:€€€€€€€€€d
И
cond_4/dropout/divRealDivcond_4/dropout/Shape/Switch:1cond_4/dropout/keep_prob*
T0*'
_output_shapes
:€€€€€€€€€d
u
cond_4/dropout/mulMulcond_4/dropout/divcond_4/dropout/Floor*
T0*'
_output_shapes
:€€€€€€€€€d
e
cond_4/IdentityIdentitycond_4/Identity/Switch*
T0*'
_output_shapes
:€€€€€€€€€d
Ш
cond_4/Identity/SwitchSwitchRelu_4cond_4/pred_id*
T0*
_class
loc:@Relu_4*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
w
cond_4/MergeMergecond_4/Identitycond_4/dropout/mul*
T0*
N*)
_output_shapes
:€€€€€€€€€d: 
g
MatMul_5/ReadVariableOpReadVariableOpdense_17/kernel*
dtype0*
_output_shapes

:d
С
MatMul_5MatMulcond_4/MergeMatMul_5/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
b
BiasAdd_5/ReadVariableOpReadVariableOpdense_17/bias*
dtype0*
_output_shapes
:
Б
	BiasAdd_5BiasAddMatMul_5BiasAdd_5/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
O
SoftmaxSoftmax	BiasAdd_5*
T0*'
_output_shapes
:€€€€€€€€€
Д
output_1_targetPlaceholder*
dtype0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*%
shape:€€€€€€€€€€€€€€€€€€
R
ConstConst*
valueB*  А?*
dtype0*
_output_shapes
:
Д
output_1_sample_weightsPlaceholderWithDefaultConst*
shape:€€€€€€€€€*
dtype0*#
_output_shapes
:€€€€€€€€€
j
(loss/output_1_loss/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Я
loss/output_1_loss/SumSumSoftmax(loss/output_1_loss/Sum/reduction_indices*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(*

Tidx0
x
loss/output_1_loss/truedivRealDivSoftmaxloss/output_1_loss/Sum*
T0*'
_output_shapes
:€€€€€€€€€
]
loss/output_1_loss/ConstConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
]
loss/output_1_loss/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
r
loss/output_1_loss/subSubloss/output_1_loss/sub/xloss/output_1_loss/Const*
T0*
_output_shapes
: 
Щ
(loss/output_1_loss/clip_by_value/MinimumMinimumloss/output_1_loss/truedivloss/output_1_loss/sub*'
_output_shapes
:€€€€€€€€€*
T0
°
 loss/output_1_loss/clip_by_valueMaximum(loss/output_1_loss/clip_by_value/Minimumloss/output_1_loss/Const*'
_output_shapes
:€€€€€€€€€*
T0
q
loss/output_1_loss/LogLog loss/output_1_loss/clip_by_value*'
_output_shapes
:€€€€€€€€€*
T0
x
loss/output_1_loss/mulMuloutput_1_targetloss/output_1_loss/Log*
T0*'
_output_shapes
:€€€€€€€€€
l
*loss/output_1_loss/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Ѓ
loss/output_1_loss/Sum_1Sumloss/output_1_loss/mul*loss/output_1_loss/Sum_1/reduction_indices*
T0*#
_output_shapes
:€€€€€€€€€*
	keep_dims( *

Tidx0
e
loss/output_1_loss/NegNegloss/output_1_loss/Sum_1*
T0*#
_output_shapes
:€€€€€€€€€
Ю
Gloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeShapeoutput_1_sample_weights*
_output_shapes
:*
T0*
out_type0
И
Floss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
Ь
Floss/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeShapeloss/output_1_loss/Neg*
_output_shapes
:*
T0*
out_type0
З
Eloss/output_1_loss/broadcast_weights/assert_broadcastable/values/rankConst*
dtype0*
_output_shapes
: *
value	B :
З
Eloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
ь
Closs/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarEqualEloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar/xFloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
Ж
Oloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/SwitchSwitchCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : *
T0

—
Qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_tIdentityQloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
ѕ
Qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentityOloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
¬
Ploss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: *
T0

н
Qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1SwitchCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarPloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
_output_shapes
: : *
T0
*V
_classL
JHloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar
Л
oloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualvloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switchxloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
Ц
vloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchEloss/output_1_loss/broadcast_weights/assert_broadcastable/values/rankPloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
_output_shapes
: : *
T0*X
_classN
LJloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/values/rank
Ъ
xloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchFloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rankPloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*Y
_classO
MKloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank*
_output_shapes
: : 
ш
iloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitcholoss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankoloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
Е
kloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentitykloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
Г
kloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityiloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
И
jloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityoloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
Љ
Вloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
“
~loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsЙloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Вloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
∞
Еloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchFloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shapePloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*Y
_classO
MKloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
М
Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchЕloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchjloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id* 
_output_shapes
::*
T0*Y
_classO
MKloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape
√
Гloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
і
Гloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
ћ
}loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillГloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeГloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
_output_shapes

:*
T0*

index_type0
ѓ
loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
ƒ
zloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2~loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims}loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
T0*
N*
_output_shapes

:*

Tidx0
Њ
Дloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
ў
Аloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsЛloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Дloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

:
і
Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchGloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapePloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*Z
_classP
NLloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::
С
Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchЗloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchjloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*Z
_classP
NLloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::
Я
Мloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationАloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1zloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
validate_indices(*
T0*<
_output_shapes*
(:€€€€€€€€€:€€€€€€€€€:*
set_operationa-b
ѕ
Дloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeОloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
•
uloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
value	B : 
Ы
sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualuloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xДloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
ъ
kloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switcholoss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankjloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
_output_shapes
: : *
T0
*В
_classx
vtloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank
€
hloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergekloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
¬
Nloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMergehloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeSloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1*
N*
_output_shapes
: : *
T0

І
?loss/output_1_loss/broadcast_weights/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Р
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
Ы
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_2Const**
value!B Boutput_1_sample_weights:0*
dtype0*
_output_shapes
: 
П
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
Ъ
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_4Const*)
value B Bloss/output_1_loss/Neg:0*
dtype0*
_output_shapes
: 
М
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
Щ
Lloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitchNloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeNloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
Ћ
Nloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentityNloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
…
Nloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityLloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
 
Mloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_idIdentityNloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
£
Jloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOpO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
Е
Xloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependencyIdentityNloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tK^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*a
_classW
USloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
М
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
у
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
ю
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f**
value!B Boutput_1_sample_weights:0*
dtype0*
_output_shapes
: 
т
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *
valueB Bvalues.shape=
э
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*)
value B Bloss/output_1_loss/Neg:0*
dtype0*
_output_shapes
: 
п
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
”
Lloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssertSloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
В
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSwitchNloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0
*a
_classW
USloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
ю
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchGloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeMloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0*Z
_classP
NLloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::
ь
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchFloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeMloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id* 
_output_shapes
::*
T0*Y
_classO
MKloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape
о
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarMloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0
*V
_classL
JHloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : 
Й
Zloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1IdentityNloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fM^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert*
T0
*a
_classW
USloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
ґ
Kloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/MergeMergeZloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1Xloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
Ў
4loss/output_1_loss/broadcast_weights/ones_like/ShapeShapeloss/output_1_loss/NegL^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
«
4loss/output_1_loss/broadcast_weights/ones_like/ConstConstL^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *
valueB
 *  А?
в
.loss/output_1_loss/broadcast_weights/ones_likeFill4loss/output_1_loss/broadcast_weights/ones_like/Shape4loss/output_1_loss/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:€€€€€€€€€
Ґ
$loss/output_1_loss/broadcast_weightsMuloutput_1_sample_weights.loss/output_1_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:€€€€€€€€€
Л
loss/output_1_loss/Mul_1Mulloss/output_1_loss/Neg$loss/output_1_loss/broadcast_weights*
T0*#
_output_shapes
:€€€€€€€€€
d
loss/output_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
У
loss/output_1_loss/Sum_2Sumloss/output_1_loss/Mul_1loss/output_1_loss/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
d
loss/output_1_loss/Const_2Const*
dtype0*
_output_shapes
:*
valueB: 
Я
loss/output_1_loss/Sum_3Sum$loss/output_1_loss/broadcast_weightsloss/output_1_loss/Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
|
loss/output_1_loss/truediv_1RealDivloss/output_1_loss/Sum_2loss/output_1_loss/Sum_3*
T0*
_output_shapes
: 
b
loss/output_1_loss/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss/output_1_loss/GreaterGreaterloss/output_1_loss/Sum_3loss/output_1_loss/zeros_like*
T0*
_output_shapes
: 
Э
loss/output_1_loss/SelectSelectloss/output_1_loss/Greaterloss/output_1_loss/truediv_1loss/output_1_loss/zeros_like*
_output_shapes
: *
T0
]
loss/output_1_loss/Const_3Const*
valueB *
dtype0*
_output_shapes
: 
Ф
loss/output_1_loss/MeanMeanloss/output_1_loss/Selectloss/output_1_loss/Const_3*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
U
loss/mulMul
loss/mul/xloss/output_1_loss/Mean*
T0*
_output_shapes
: 
g
metrics/acc/ArgMax/dimensionConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ш
metrics/acc/ArgMaxArgMaxoutput_1_targetmetrics/acc/ArgMax/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:€€€€€€€€€
i
metrics/acc/ArgMax_1/dimensionConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ф
metrics/acc/ArgMax_1ArgMaxSoftmaxmetrics/acc/ArgMax_1/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:€€€€€€€€€
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*#
_output_shapes
:€€€€€€€€€*
T0	
x
metrics/acc/CastCastmetrics/acc/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:€€€€€€€€€*

DstT0
[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
metrics/acc/MeanMeanmetrics/acc/Castmetrics/acc/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
g
metrics/f1_score/mulMuloutput_1_targetSoftmax*
T0*'
_output_shapes
:€€€€€€€€€
[
metrics/f1_score/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
]
metrics/f1_score/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  А?
У
&metrics/f1_score/clip_by_value/MinimumMinimummetrics/f1_score/mulmetrics/f1_score/Const_1*
T0*'
_output_shapes
:€€€€€€€€€
Ы
metrics/f1_score/clip_by_valueMaximum&metrics/f1_score/clip_by_value/Minimummetrics/f1_score/Const*
T0*'
_output_shapes
:€€€€€€€€€
q
metrics/f1_score/RoundRoundmetrics/f1_score/clip_by_value*
T0*'
_output_shapes
:€€€€€€€€€
i
metrics/f1_score/Const_2Const*
dtype0*
_output_shapes
:*
valueB"       
Л
metrics/f1_score/SumSummetrics/f1_score/Roundmetrics/f1_score/Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
]
metrics/f1_score/Const_3Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
metrics/f1_score/Const_4Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
И
(metrics/f1_score/clip_by_value_1/MinimumMinimumSoftmaxmetrics/f1_score/Const_4*
T0*'
_output_shapes
:€€€€€€€€€
°
 metrics/f1_score/clip_by_value_1Maximum(metrics/f1_score/clip_by_value_1/Minimummetrics/f1_score/Const_3*'
_output_shapes
:€€€€€€€€€*
T0
u
metrics/f1_score/Round_1Round metrics/f1_score/clip_by_value_1*
T0*'
_output_shapes
:€€€€€€€€€
i
metrics/f1_score/Const_5Const*
valueB"       *
dtype0*
_output_shapes
:
П
metrics/f1_score/Sum_1Summetrics/f1_score/Round_1metrics/f1_score/Const_5*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
[
metrics/f1_score/add/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
l
metrics/f1_score/addAddmetrics/f1_score/Sum_1metrics/f1_score/add/y*
_output_shapes
: *
T0
p
metrics/f1_score/truedivRealDivmetrics/f1_score/Summetrics/f1_score/add*
T0*
_output_shapes
: 
i
metrics/f1_score/mul_1Muloutput_1_targetSoftmax*
T0*'
_output_shapes
:€€€€€€€€€
]
metrics/f1_score/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
metrics/f1_score/Const_7Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ч
(metrics/f1_score/clip_by_value_2/MinimumMinimummetrics/f1_score/mul_1metrics/f1_score/Const_7*
T0*'
_output_shapes
:€€€€€€€€€
°
 metrics/f1_score/clip_by_value_2Maximum(metrics/f1_score/clip_by_value_2/Minimummetrics/f1_score/Const_6*
T0*'
_output_shapes
:€€€€€€€€€
u
metrics/f1_score/Round_2Round metrics/f1_score/clip_by_value_2*
T0*'
_output_shapes
:€€€€€€€€€
i
metrics/f1_score/Const_8Const*
valueB"       *
dtype0*
_output_shapes
:
П
metrics/f1_score/Sum_2Summetrics/f1_score/Round_2metrics/f1_score/Const_8*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
]
metrics/f1_score/Const_9Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
metrics/f1_score/Const_10Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ъ
(metrics/f1_score/clip_by_value_3/MinimumMinimumoutput_1_targetmetrics/f1_score/Const_10*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
™
 metrics/f1_score/clip_by_value_3Maximum(metrics/f1_score/clip_by_value_3/Minimummetrics/f1_score/Const_9*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
~
metrics/f1_score/Round_3Round metrics/f1_score/clip_by_value_3*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
j
metrics/f1_score/Const_11Const*
dtype0*
_output_shapes
:*
valueB"       
Р
metrics/f1_score/Sum_3Summetrics/f1_score/Round_3metrics/f1_score/Const_11*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
]
metrics/f1_score/add_1/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
p
metrics/f1_score/add_1Addmetrics/f1_score/Sum_3metrics/f1_score/add_1/y*
T0*
_output_shapes
: 
v
metrics/f1_score/truediv_1RealDivmetrics/f1_score/Sum_2metrics/f1_score/add_1*
T0*
_output_shapes
: 
]
metrics/f1_score/mul_2/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
r
metrics/f1_score/mul_2Mulmetrics/f1_score/mul_2/xmetrics/f1_score/truediv*
T0*
_output_shapes
: 
r
metrics/f1_score/mul_3Mulmetrics/f1_score/mul_2metrics/f1_score/truediv_1*
T0*
_output_shapes
: 
t
metrics/f1_score/add_2Addmetrics/f1_score/truedivmetrics/f1_score/truediv_1*
_output_shapes
: *
T0
]
metrics/f1_score/add_3/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
p
metrics/f1_score/add_3Addmetrics/f1_score/add_2metrics/f1_score/add_3/y*
_output_shapes
: *
T0
v
metrics/f1_score/truediv_2RealDivmetrics/f1_score/mul_3metrics/f1_score/add_3*
_output_shapes
: *
T0
\
metrics/f1_score/Const_12Const*
valueB *
dtype0*
_output_shapes
: 
Т
metrics/f1_score/MeanMeanmetrics/f1_score/truediv_2metrics/f1_score/Const_12*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
}
training/Adam/gradients/ShapeConst*
_class
loc:@loss/mul*
valueB *
dtype0*
_output_shapes
: 
Г
!training/Adam/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
_class
loc:@loss/mul*
valueB
 *  А?
ґ
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*
_class
loc:@loss/mul*

index_type0*
_output_shapes
: 
•
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/output_1_loss/Mean*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
Ъ
+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
_output_shapes
: *
T0*
_class
loc:@loss/mul
±
Btraining/Adam/gradients/loss/output_1_loss/Mean_grad/Reshape/shapeConst**
_class 
loc:@loss/output_1_loss/Mean*
valueB *
dtype0*
_output_shapes
: 
У
<training/Adam/gradients/loss/output_1_loss/Mean_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Btraining/Adam/gradients/loss/output_1_loss/Mean_grad/Reshape/shape*
T0**
_class 
loc:@loss/output_1_loss/Mean*
Tshape0*
_output_shapes
: 
©
:training/Adam/gradients/loss/output_1_loss/Mean_grad/ConstConst**
_class 
loc:@loss/output_1_loss/Mean*
valueB *
dtype0*
_output_shapes
: 
Ъ
9training/Adam/gradients/loss/output_1_loss/Mean_grad/TileTile<training/Adam/gradients/loss/output_1_loss/Mean_grad/Reshape:training/Adam/gradients/loss/output_1_loss/Mean_grad/Const*

Tmultiples0*
T0**
_class 
loc:@loss/output_1_loss/Mean*
_output_shapes
: 
≠
<training/Adam/gradients/loss/output_1_loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
: **
_class 
loc:@loss/output_1_loss/Mean*
valueB
 *  А?
Н
<training/Adam/gradients/loss/output_1_loss/Mean_grad/truedivRealDiv9training/Adam/gradients/loss/output_1_loss/Mean_grad/Tile<training/Adam/gradients/loss/output_1_loss/Mean_grad/Const_1*
T0**
_class 
loc:@loss/output_1_loss/Mean*
_output_shapes
: 
і
Atraining/Adam/gradients/loss/output_1_loss/Select_grad/zeros_likeConst*,
_class"
 loc:@loss/output_1_loss/Select*
valueB
 *    *
dtype0*
_output_shapes
: 
≥
=training/Adam/gradients/loss/output_1_loss/Select_grad/SelectSelectloss/output_1_loss/Greater<training/Adam/gradients/loss/output_1_loss/Mean_grad/truedivAtraining/Adam/gradients/loss/output_1_loss/Select_grad/zeros_like*
T0*,
_class"
 loc:@loss/output_1_loss/Select*
_output_shapes
: 
µ
?training/Adam/gradients/loss/output_1_loss/Select_grad/Select_1Selectloss/output_1_loss/GreaterAtraining/Adam/gradients/loss/output_1_loss/Select_grad/zeros_like<training/Adam/gradients/loss/output_1_loss/Mean_grad/truediv*
T0*,
_class"
 loc:@loss/output_1_loss/Select*
_output_shapes
: 
≥
?training/Adam/gradients/loss/output_1_loss/truediv_1_grad/ShapeConst*/
_class%
#!loc:@loss/output_1_loss/truediv_1*
valueB *
dtype0*
_output_shapes
: 
µ
Atraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/Shape_1Const*/
_class%
#!loc:@loss/output_1_loss/truediv_1*
valueB *
dtype0*
_output_shapes
: 
Џ
Otraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgs?training/Adam/gradients/loss/output_1_loss/truediv_1_grad/ShapeAtraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1
ч
Atraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/RealDivRealDiv=training/Adam/gradients/loss/output_1_loss/Select_grad/Selectloss/output_1_loss/Sum_3*
_output_shapes
: *
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1
«
=training/Adam/gradients/loss/output_1_loss/truediv_1_grad/SumSumAtraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/RealDivOtraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/BroadcastGradientArgs*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1
ђ
Atraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/ReshapeReshape=training/Adam/gradients/loss/output_1_loss/truediv_1_grad/Sum?training/Adam/gradients/loss/output_1_loss/truediv_1_grad/Shape*
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1*
Tshape0*
_output_shapes
: 
∞
=training/Adam/gradients/loss/output_1_loss/truediv_1_grad/NegNegloss/output_1_loss/Sum_2*
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1*
_output_shapes
: 
щ
Ctraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/RealDiv_1RealDiv=training/Adam/gradients/loss/output_1_loss/truediv_1_grad/Negloss/output_1_loss/Sum_3*
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1*
_output_shapes
: 
€
Ctraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/RealDiv_2RealDivCtraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/RealDiv_1loss/output_1_loss/Sum_3*
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1*
_output_shapes
: 
Ъ
=training/Adam/gradients/loss/output_1_loss/truediv_1_grad/mulMul=training/Adam/gradients/loss/output_1_loss/Select_grad/SelectCtraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/RealDiv_2*
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1*
_output_shapes
: 
«
?training/Adam/gradients/loss/output_1_loss/truediv_1_grad/Sum_1Sum=training/Adam/gradients/loss/output_1_loss/truediv_1_grad/mulQtraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1
≤
Ctraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/Reshape_1Reshape?training/Adam/gradients/loss/output_1_loss/truediv_1_grad/Sum_1Atraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/Shape_1*
_output_shapes
: *
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1*
Tshape0
Ї
Ctraining/Adam/gradients/loss/output_1_loss/Sum_2_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*+
_class!
loc:@loss/output_1_loss/Sum_2*
valueB:
∞
=training/Adam/gradients/loss/output_1_loss/Sum_2_grad/ReshapeReshapeAtraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/ReshapeCtraining/Adam/gradients/loss/output_1_loss/Sum_2_grad/Reshape/shape*
T0*+
_class!
loc:@loss/output_1_loss/Sum_2*
Tshape0*
_output_shapes
:
ј
;training/Adam/gradients/loss/output_1_loss/Sum_2_grad/ShapeShapeloss/output_1_loss/Mul_1*
T0*+
_class!
loc:@loss/output_1_loss/Sum_2*
out_type0*
_output_shapes
:
Ђ
:training/Adam/gradients/loss/output_1_loss/Sum_2_grad/TileTile=training/Adam/gradients/loss/output_1_loss/Sum_2_grad/Reshape;training/Adam/gradients/loss/output_1_loss/Sum_2_grad/Shape*

Tmultiples0*
T0*+
_class!
loc:@loss/output_1_loss/Sum_2*#
_output_shapes
:€€€€€€€€€
Њ
;training/Adam/gradients/loss/output_1_loss/Mul_1_grad/ShapeShapeloss/output_1_loss/Neg*
_output_shapes
:*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1*
out_type0
ќ
=training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Shape_1Shape$loss/output_1_loss/broadcast_weights*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1*
out_type0*
_output_shapes
:
 
Ktraining/Adam/gradients/loss/output_1_loss/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs;training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Shape=training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1
э
9training/Adam/gradients/loss/output_1_loss/Mul_1_grad/MulMul:training/Adam/gradients/loss/output_1_loss/Sum_2_grad/Tile$loss/output_1_loss/broadcast_weights*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1*#
_output_shapes
:€€€€€€€€€
µ
9training/Adam/gradients/loss/output_1_loss/Mul_1_grad/SumSum9training/Adam/gradients/loss/output_1_loss/Mul_1_grad/MulKtraining/Adam/gradients/loss/output_1_loss/Mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1*
_output_shapes
:
©
=training/Adam/gradients/loss/output_1_loss/Mul_1_grad/ReshapeReshape9training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Sum;training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Shape*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1*
Tshape0*#
_output_shapes
:€€€€€€€€€
с
;training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Mul_1Mulloss/output_1_loss/Neg:training/Adam/gradients/loss/output_1_loss/Sum_2_grad/Tile*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1*#
_output_shapes
:€€€€€€€€€
ї
;training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Sum_1Sum;training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Mul_1Mtraining/Adam/gradients/loss/output_1_loss/Mul_1_grad/BroadcastGradientArgs:1*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
ѓ
?training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Reshape_1Reshape;training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Sum_1=training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Shape_1*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1*
Tshape0*#
_output_shapes
:€€€€€€€€€
÷
7training/Adam/gradients/loss/output_1_loss/Neg_grad/NegNeg=training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Reshape*
T0*)
_class
loc:@loss/output_1_loss/Neg*#
_output_shapes
:€€€€€€€€€
Њ
;training/Adam/gradients/loss/output_1_loss/Sum_1_grad/ShapeShapeloss/output_1_loss/mul*
_output_shapes
:*
T0*+
_class!
loc:@loss/output_1_loss/Sum_1*
out_type0
©
:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/SizeConst*
dtype0*
_output_shapes
: *+
_class!
loc:@loss/output_1_loss/Sum_1*
value	B :
ц
9training/Adam/gradients/loss/output_1_loss/Sum_1_grad/addAdd*loss/output_1_loss/Sum_1/reduction_indices:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Size*
T0*+
_class!
loc:@loss/output_1_loss/Sum_1*
_output_shapes
: 
К
9training/Adam/gradients/loss/output_1_loss/Sum_1_grad/modFloorMod9training/Adam/gradients/loss/output_1_loss/Sum_1_grad/add:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Size*
T0*+
_class!
loc:@loss/output_1_loss/Sum_1*
_output_shapes
: 
≠
=training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Shape_1Const*+
_class!
loc:@loss/output_1_loss/Sum_1*
valueB *
dtype0*
_output_shapes
: 
∞
Atraining/Adam/gradients/loss/output_1_loss/Sum_1_grad/range/startConst*+
_class!
loc:@loss/output_1_loss/Sum_1*
value	B : *
dtype0*
_output_shapes
: 
∞
Atraining/Adam/gradients/loss/output_1_loss/Sum_1_grad/range/deltaConst*+
_class!
loc:@loss/output_1_loss/Sum_1*
value	B :*
dtype0*
_output_shapes
: 
џ
;training/Adam/gradients/loss/output_1_loss/Sum_1_grad/rangeRangeAtraining/Adam/gradients/loss/output_1_loss/Sum_1_grad/range/start:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/SizeAtraining/Adam/gradients/loss/output_1_loss/Sum_1_grad/range/delta*

Tidx0*+
_class!
loc:@loss/output_1_loss/Sum_1*
_output_shapes
:
ѓ
@training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Fill/valueConst*+
_class!
loc:@loss/output_1_loss/Sum_1*
value	B :*
dtype0*
_output_shapes
: 
£
:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/FillFill=training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Shape_1@training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Fill/value*
_output_shapes
: *
T0*+
_class!
loc:@loss/output_1_loss/Sum_1*

index_type0
†
Ctraining/Adam/gradients/loss/output_1_loss/Sum_1_grad/DynamicStitchDynamicStitch;training/Adam/gradients/loss/output_1_loss/Sum_1_grad/range9training/Adam/gradients/loss/output_1_loss/Sum_1_grad/mod;training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Shape:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Fill*
N*
_output_shapes
:*
T0*+
_class!
loc:@loss/output_1_loss/Sum_1
Ѓ
?training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Maximum/yConst*+
_class!
loc:@loss/output_1_loss/Sum_1*
value	B :*
dtype0*
_output_shapes
: 
†
=training/Adam/gradients/loss/output_1_loss/Sum_1_grad/MaximumMaximumCtraining/Adam/gradients/loss/output_1_loss/Sum_1_grad/DynamicStitch?training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Maximum/y*
T0*+
_class!
loc:@loss/output_1_loss/Sum_1*
_output_shapes
:
Ш
>training/Adam/gradients/loss/output_1_loss/Sum_1_grad/floordivFloorDiv;training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Shape=training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Maximum*
T0*+
_class!
loc:@loss/output_1_loss/Sum_1*
_output_shapes
:
Љ
=training/Adam/gradients/loss/output_1_loss/Sum_1_grad/ReshapeReshape7training/Adam/gradients/loss/output_1_loss/Neg_grad/NegCtraining/Adam/gradients/loss/output_1_loss/Sum_1_grad/DynamicStitch*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0*+
_class!
loc:@loss/output_1_loss/Sum_1*
Tshape0
≤
:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/TileTile=training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Reshape>training/Adam/gradients/loss/output_1_loss/Sum_1_grad/floordiv*

Tmultiples0*
T0*+
_class!
loc:@loss/output_1_loss/Sum_1*'
_output_shapes
:€€€€€€€€€
≥
9training/Adam/gradients/loss/output_1_loss/mul_grad/ShapeShapeoutput_1_target*
T0*)
_class
loc:@loss/output_1_loss/mul*
out_type0*
_output_shapes
:
Љ
;training/Adam/gradients/loss/output_1_loss/mul_grad/Shape_1Shapeloss/output_1_loss/Log*
T0*)
_class
loc:@loss/output_1_loss/mul*
out_type0*
_output_shapes
:
¬
Itraining/Adam/gradients/loss/output_1_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9training/Adam/gradients/loss/output_1_loss/mul_grad/Shape;training/Adam/gradients/loss/output_1_loss/mul_grad/Shape_1*
T0*)
_class
loc:@loss/output_1_loss/mul*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
п
7training/Adam/gradients/loss/output_1_loss/mul_grad/MulMul:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Tileloss/output_1_loss/Log*'
_output_shapes
:€€€€€€€€€*
T0*)
_class
loc:@loss/output_1_loss/mul
≠
7training/Adam/gradients/loss/output_1_loss/mul_grad/SumSum7training/Adam/gradients/loss/output_1_loss/mul_grad/MulItraining/Adam/gradients/loss/output_1_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/output_1_loss/mul*
_output_shapes
:
Ѓ
;training/Adam/gradients/loss/output_1_loss/mul_grad/ReshapeReshape7training/Adam/gradients/loss/output_1_loss/mul_grad/Sum9training/Adam/gradients/loss/output_1_loss/mul_grad/Shape*
T0*)
_class
loc:@loss/output_1_loss/mul*
Tshape0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
к
9training/Adam/gradients/loss/output_1_loss/mul_grad/Mul_1Muloutput_1_target:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Tile*
T0*)
_class
loc:@loss/output_1_loss/mul*'
_output_shapes
:€€€€€€€€€
≥
9training/Adam/gradients/loss/output_1_loss/mul_grad/Sum_1Sum9training/Adam/gradients/loss/output_1_loss/mul_grad/Mul_1Ktraining/Adam/gradients/loss/output_1_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/output_1_loss/mul*
_output_shapes
:
Ђ
=training/Adam/gradients/loss/output_1_loss/mul_grad/Reshape_1Reshape9training/Adam/gradients/loss/output_1_loss/mul_grad/Sum_1;training/Adam/gradients/loss/output_1_loss/mul_grad/Shape_1*
T0*)
_class
loc:@loss/output_1_loss/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€
Л
>training/Adam/gradients/loss/output_1_loss/Log_grad/Reciprocal
Reciprocal loss/output_1_loss/clip_by_value>^training/Adam/gradients/loss/output_1_loss/mul_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€*
T0*)
_class
loc:@loss/output_1_loss/Log
Ъ
7training/Adam/gradients/loss/output_1_loss/Log_grad/mulMul=training/Adam/gradients/loss/output_1_loss/mul_grad/Reshape_1>training/Adam/gradients/loss/output_1_loss/Log_grad/Reciprocal*'
_output_shapes
:€€€€€€€€€*
T0*)
_class
loc:@loss/output_1_loss/Log
а
Ctraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/ShapeShape(loss/output_1_loss/clip_by_value/Minimum*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*
out_type0*
_output_shapes
:
љ
Etraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Shape_1Const*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*
valueB *
dtype0*
_output_shapes
: 
с
Etraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Shape_2Shape7training/Adam/gradients/loss/output_1_loss/Log_grad/mul*
_output_shapes
:*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*
out_type0
√
Itraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *3
_class)
'%loc:@loss/output_1_loss/clip_by_value*
valueB
 *    
÷
Ctraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/zerosFillEtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Shape_2Itraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/zeros/Const*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*

index_type0*'
_output_shapes
:€€€€€€€€€
Е
Jtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/GreaterEqualGreaterEqual(loss/output_1_loss/clip_by_value/Minimumloss/output_1_loss/Const*'
_output_shapes
:€€€€€€€€€*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value
к
Straining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsCtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/ShapeEtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Shape_1*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
€
Dtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/SelectSelectJtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/GreaterEqual7training/Adam/gradients/loss/output_1_loss/Log_grad/mulCtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/zeros*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*'
_output_shapes
:€€€€€€€€€
Б
Ftraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Select_1SelectJtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/GreaterEqualCtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/zeros7training/Adam/gradients/loss/output_1_loss/Log_grad/mul*'
_output_shapes
:€€€€€€€€€*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value
Ў
Atraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/SumSumDtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/SelectStraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/BroadcastGradientArgs*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*
_output_shapes
:*
	keep_dims( *

Tidx0
Ќ
Etraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/ReshapeReshapeAtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/SumCtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*
Tshape0
ё
Ctraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Sum_1SumFtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Select_1Utraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/BroadcastGradientArgs:1*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*
_output_shapes
:*
	keep_dims( *

Tidx0
¬
Gtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Reshape_1ReshapeCtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Sum_1Etraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Shape_1*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*
Tshape0*
_output_shapes
: 
в
Ktraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/ShapeShapeloss/output_1_loss/truediv*
_output_shapes
:*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*
out_type0
Ќ
Mtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Shape_1Const*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*
valueB *
dtype0*
_output_shapes
: 
П
Mtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Shape_2ShapeEtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Reshape*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*
out_type0*
_output_shapes
:
”
Qtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/zeros/ConstConst*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*
valueB
 *    *
dtype0*
_output_shapes
: 
ц
Ktraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/zerosFillMtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Shape_2Qtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/zeros/Const*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*

index_type0*'
_output_shapes
:€€€€€€€€€
€
Otraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualloss/output_1_loss/truedivloss/output_1_loss/sub*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*'
_output_shapes
:€€€€€€€€€
К
[training/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/ShapeMtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum
™
Ltraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/SelectSelectOtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/LessEqualEtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/ReshapeKtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/zeros*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*'
_output_shapes
:€€€€€€€€€
ђ
Ntraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Select_1SelectOtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/LessEqualKtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/zerosEtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Reshape*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*'
_output_shapes
:€€€€€€€€€
ш
Itraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/SumSumLtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Select[training/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0
н
Mtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/ReshapeReshapeItraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/SumKtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Shape*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*
Tshape0*'
_output_shapes
:€€€€€€€€€
ю
Ktraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Sum_1SumNtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Select_1]training/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*
_output_shapes
:
в
Otraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeKtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Sum_1Mtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Shape_1*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*
Tshape0*
_output_shapes
: 
≥
=training/Adam/gradients/loss/output_1_loss/truediv_grad/ShapeShapeSoftmax*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*
out_type0*
_output_shapes
:
ƒ
?training/Adam/gradients/loss/output_1_loss/truediv_grad/Shape_1Shapeloss/output_1_loss/Sum*
_output_shapes
:*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*
out_type0
“
Mtraining/Adam/gradients/loss/output_1_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/loss/output_1_loss/truediv_grad/Shape?training/Adam/gradients/loss/output_1_loss/truediv_grad/Shape_1*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Т
?training/Adam/gradients/loss/output_1_loss/truediv_grad/RealDivRealDivMtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Reshapeloss/output_1_loss/Sum*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*'
_output_shapes
:€€€€€€€€€
Ѕ
;training/Adam/gradients/loss/output_1_loss/truediv_grad/SumSum?training/Adam/gradients/loss/output_1_loss/truediv_grad/RealDivMtraining/Adam/gradients/loss/output_1_loss/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@loss/output_1_loss/truediv
µ
?training/Adam/gradients/loss/output_1_loss/truediv_grad/ReshapeReshape;training/Adam/gradients/loss/output_1_loss/truediv_grad/Sum=training/Adam/gradients/loss/output_1_loss/truediv_grad/Shape*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*
Tshape0*'
_output_shapes
:€€€€€€€€€
ђ
;training/Adam/gradients/loss/output_1_loss/truediv_grad/NegNegSoftmax*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*'
_output_shapes
:€€€€€€€€€
В
Atraining/Adam/gradients/loss/output_1_loss/truediv_grad/RealDiv_1RealDiv;training/Adam/gradients/loss/output_1_loss/truediv_grad/Negloss/output_1_loss/Sum*'
_output_shapes
:€€€€€€€€€*
T0*-
_class#
!loc:@loss/output_1_loss/truediv
И
Atraining/Adam/gradients/loss/output_1_loss/truediv_grad/RealDiv_2RealDivAtraining/Adam/gradients/loss/output_1_loss/truediv_grad/RealDiv_1loss/output_1_loss/Sum*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*'
_output_shapes
:€€€€€€€€€
µ
;training/Adam/gradients/loss/output_1_loss/truediv_grad/mulMulMtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/ReshapeAtraining/Adam/gradients/loss/output_1_loss/truediv_grad/RealDiv_2*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*'
_output_shapes
:€€€€€€€€€
Ѕ
=training/Adam/gradients/loss/output_1_loss/truediv_grad/Sum_1Sum;training/Adam/gradients/loss/output_1_loss/truediv_grad/mulOtraining/Adam/gradients/loss/output_1_loss/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*
_output_shapes
:
ї
Atraining/Adam/gradients/loss/output_1_loss/truediv_grad/Reshape_1Reshape=training/Adam/gradients/loss/output_1_loss/truediv_grad/Sum_1?training/Adam/gradients/loss/output_1_loss/truediv_grad/Shape_1*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*
Tshape0*'
_output_shapes
:€€€€€€€€€
Ђ
9training/Adam/gradients/loss/output_1_loss/Sum_grad/ShapeShapeSoftmax*
T0*)
_class
loc:@loss/output_1_loss/Sum*
out_type0*
_output_shapes
:
•
8training/Adam/gradients/loss/output_1_loss/Sum_grad/SizeConst*)
_class
loc:@loss/output_1_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 
о
7training/Adam/gradients/loss/output_1_loss/Sum_grad/addAdd(loss/output_1_loss/Sum/reduction_indices8training/Adam/gradients/loss/output_1_loss/Sum_grad/Size*
_output_shapes
: *
T0*)
_class
loc:@loss/output_1_loss/Sum
В
7training/Adam/gradients/loss/output_1_loss/Sum_grad/modFloorMod7training/Adam/gradients/loss/output_1_loss/Sum_grad/add8training/Adam/gradients/loss/output_1_loss/Sum_grad/Size*
_output_shapes
: *
T0*)
_class
loc:@loss/output_1_loss/Sum
©
;training/Adam/gradients/loss/output_1_loss/Sum_grad/Shape_1Const*)
_class
loc:@loss/output_1_loss/Sum*
valueB *
dtype0*
_output_shapes
: 
ђ
?training/Adam/gradients/loss/output_1_loss/Sum_grad/range/startConst*)
_class
loc:@loss/output_1_loss/Sum*
value	B : *
dtype0*
_output_shapes
: 
ђ
?training/Adam/gradients/loss/output_1_loss/Sum_grad/range/deltaConst*)
_class
loc:@loss/output_1_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 
—
9training/Adam/gradients/loss/output_1_loss/Sum_grad/rangeRange?training/Adam/gradients/loss/output_1_loss/Sum_grad/range/start8training/Adam/gradients/loss/output_1_loss/Sum_grad/Size?training/Adam/gradients/loss/output_1_loss/Sum_grad/range/delta*
_output_shapes
:*

Tidx0*)
_class
loc:@loss/output_1_loss/Sum
Ђ
>training/Adam/gradients/loss/output_1_loss/Sum_grad/Fill/valueConst*
dtype0*
_output_shapes
: *)
_class
loc:@loss/output_1_loss/Sum*
value	B :
Ы
8training/Adam/gradients/loss/output_1_loss/Sum_grad/FillFill;training/Adam/gradients/loss/output_1_loss/Sum_grad/Shape_1>training/Adam/gradients/loss/output_1_loss/Sum_grad/Fill/value*
T0*)
_class
loc:@loss/output_1_loss/Sum*

index_type0*
_output_shapes
: 
Ф
Atraining/Adam/gradients/loss/output_1_loss/Sum_grad/DynamicStitchDynamicStitch9training/Adam/gradients/loss/output_1_loss/Sum_grad/range7training/Adam/gradients/loss/output_1_loss/Sum_grad/mod9training/Adam/gradients/loss/output_1_loss/Sum_grad/Shape8training/Adam/gradients/loss/output_1_loss/Sum_grad/Fill*
T0*)
_class
loc:@loss/output_1_loss/Sum*
N*
_output_shapes
:
™
=training/Adam/gradients/loss/output_1_loss/Sum_grad/Maximum/yConst*)
_class
loc:@loss/output_1_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 
Ш
;training/Adam/gradients/loss/output_1_loss/Sum_grad/MaximumMaximumAtraining/Adam/gradients/loss/output_1_loss/Sum_grad/DynamicStitch=training/Adam/gradients/loss/output_1_loss/Sum_grad/Maximum/y*
T0*)
_class
loc:@loss/output_1_loss/Sum*
_output_shapes
:
Р
<training/Adam/gradients/loss/output_1_loss/Sum_grad/floordivFloorDiv9training/Adam/gradients/loss/output_1_loss/Sum_grad/Shape;training/Adam/gradients/loss/output_1_loss/Sum_grad/Maximum*
_output_shapes
:*
T0*)
_class
loc:@loss/output_1_loss/Sum
ј
;training/Adam/gradients/loss/output_1_loss/Sum_grad/ReshapeReshapeAtraining/Adam/gradients/loss/output_1_loss/truediv_grad/Reshape_1Atraining/Adam/gradients/loss/output_1_loss/Sum_grad/DynamicStitch*
T0*)
_class
loc:@loss/output_1_loss/Sum*
Tshape0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
™
8training/Adam/gradients/loss/output_1_loss/Sum_grad/TileTile;training/Adam/gradients/loss/output_1_loss/Sum_grad/Reshape<training/Adam/gradients/loss/output_1_loss/Sum_grad/floordiv*
T0*)
_class
loc:@loss/output_1_loss/Sum*'
_output_shapes
:€€€€€€€€€*

Tmultiples0
Й
training/Adam/gradients/AddNAddN?training/Adam/gradients/loss/output_1_loss/truediv_grad/Reshape8training/Adam/gradients/loss/output_1_loss/Sum_grad/Tile*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*
N*'
_output_shapes
:€€€€€€€€€
§
(training/Adam/gradients/Softmax_grad/mulMultraining/Adam/gradients/AddNSoftmax*
T0*
_class
loc:@Softmax*'
_output_shapes
:€€€€€€€€€
°
:training/Adam/gradients/Softmax_grad/Sum/reduction_indicesConst*
_class
loc:@Softmax*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
А
(training/Adam/gradients/Softmax_grad/SumSum(training/Adam/gradients/Softmax_grad/mul:training/Adam/gradients/Softmax_grad/Sum/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_class
loc:@Softmax*'
_output_shapes
:€€€€€€€€€
≈
(training/Adam/gradients/Softmax_grad/subSubtraining/Adam/gradients/AddN(training/Adam/gradients/Softmax_grad/Sum*
T0*
_class
loc:@Softmax*'
_output_shapes
:€€€€€€€€€
≤
*training/Adam/gradients/Softmax_grad/mul_1Mul(training/Adam/gradients/Softmax_grad/subSoftmax*
T0*
_class
loc:@Softmax*'
_output_shapes
:€€€€€€€€€
«
2training/Adam/gradients/BiasAdd_5_grad/BiasAddGradBiasAddGrad*training/Adam/gradients/Softmax_grad/mul_1*
T0*
_class
loc:@BiasAdd_5*
data_formatNHWC*
_output_shapes
:
р
,training/Adam/gradients/MatMul_5_grad/MatMulMatMul*training/Adam/gradients/Softmax_grad/mul_1MatMul_5/ReadVariableOp*
transpose_b(*
T0*
_class
loc:@MatMul_5*'
_output_shapes
:€€€€€€€€€d*
transpose_a( 
ё
.training/Adam/gradients/MatMul_5_grad/MatMul_1MatMulcond_4/Merge*training/Adam/gradients/Softmax_grad/mul_1*
transpose_b( *
T0*
_class
loc:@MatMul_5*
_output_shapes

:d*
transpose_a(
Ё
3training/Adam/gradients/cond_4/Merge_grad/cond_gradSwitch,training/Adam/gradients/MatMul_5_grad/MatMulcond_4/pred_id*
T0*
_class
loc:@MatMul_5*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Ѓ
5training/Adam/gradients/cond_4/dropout/mul_grad/ShapeShapecond_4/dropout/div*
_output_shapes
:*
T0*%
_class
loc:@cond_4/dropout/mul*
out_type0
≤
7training/Adam/gradients/cond_4/dropout/mul_grad/Shape_1Shapecond_4/dropout/Floor*
T0*%
_class
loc:@cond_4/dropout/mul*
out_type0*
_output_shapes
:
≤
Etraining/Adam/gradients/cond_4/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/cond_4/dropout/mul_grad/Shape7training/Adam/gradients/cond_4/dropout/mul_grad/Shape_1*
T0*%
_class
loc:@cond_4/dropout/mul*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
а
3training/Adam/gradients/cond_4/dropout/mul_grad/MulMul5training/Adam/gradients/cond_4/Merge_grad/cond_grad:1cond_4/dropout/Floor*
T0*%
_class
loc:@cond_4/dropout/mul*'
_output_shapes
:€€€€€€€€€d
Э
3training/Adam/gradients/cond_4/dropout/mul_grad/SumSum3training/Adam/gradients/cond_4/dropout/mul_grad/MulEtraining/Adam/gradients/cond_4/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_4/dropout/mul
Х
7training/Adam/gradients/cond_4/dropout/mul_grad/ReshapeReshape3training/Adam/gradients/cond_4/dropout/mul_grad/Sum5training/Adam/gradients/cond_4/dropout/mul_grad/Shape*
T0*%
_class
loc:@cond_4/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
а
5training/Adam/gradients/cond_4/dropout/mul_grad/Mul_1Mulcond_4/dropout/div5training/Adam/gradients/cond_4/Merge_grad/cond_grad:1*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_4/dropout/mul
£
5training/Adam/gradients/cond_4/dropout/mul_grad/Sum_1Sum5training/Adam/gradients/cond_4/dropout/mul_grad/Mul_1Gtraining/Adam/gradients/cond_4/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_4/dropout/mul
Ы
9training/Adam/gradients/cond_4/dropout/mul_grad/Reshape_1Reshape5training/Adam/gradients/cond_4/dropout/mul_grad/Sum_17training/Adam/gradients/cond_4/dropout/mul_grad/Shape_1*
T0*%
_class
loc:@cond_4/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
†
training/Adam/gradients/SwitchSwitchRelu_4cond_4/pred_id*
T0*
_class
loc:@Relu_4*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Ы
 training/Adam/gradients/IdentityIdentity training/Adam/gradients/Switch:1*
T0*
_class
loc:@Relu_4*'
_output_shapes
:€€€€€€€€€d
Ъ
training/Adam/gradients/Shape_1Shape training/Adam/gradients/Switch:1*
_output_shapes
:*
T0*
_class
loc:@Relu_4*
out_type0
¶
#training/Adam/gradients/zeros/ConstConst!^training/Adam/gradients/Identity*
_class
loc:@Relu_4*
valueB
 *    *
dtype0*
_output_shapes
: 
 
training/Adam/gradients/zerosFilltraining/Adam/gradients/Shape_1#training/Adam/gradients/zeros/Const*
T0*
_class
loc:@Relu_4*

index_type0*'
_output_shapes
:€€€€€€€€€d
т
=training/Adam/gradients/cond_4/Identity/Switch_grad/cond_gradMerge3training/Adam/gradients/cond_4/Merge_grad/cond_gradtraining/Adam/gradients/zeros*
N*)
_output_shapes
:€€€€€€€€€d: *
T0*
_class
loc:@Relu_4
є
5training/Adam/gradients/cond_4/dropout/div_grad/ShapeShapecond_4/dropout/Shape/Switch:1*
_output_shapes
:*
T0*%
_class
loc:@cond_4/dropout/div*
out_type0
°
7training/Adam/gradients/cond_4/dropout/div_grad/Shape_1Const*%
_class
loc:@cond_4/dropout/div*
valueB *
dtype0*
_output_shapes
: 
≤
Etraining/Adam/gradients/cond_4/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/cond_4/dropout/div_grad/Shape7training/Adam/gradients/cond_4/dropout/div_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*%
_class
loc:@cond_4/dropout/div
о
7training/Adam/gradients/cond_4/dropout/div_grad/RealDivRealDiv7training/Adam/gradients/cond_4/dropout/mul_grad/Reshapecond_4/dropout/keep_prob*
T0*%
_class
loc:@cond_4/dropout/div*'
_output_shapes
:€€€€€€€€€d
°
3training/Adam/gradients/cond_4/dropout/div_grad/SumSum7training/Adam/gradients/cond_4/dropout/div_grad/RealDivEtraining/Adam/gradients/cond_4/dropout/div_grad/BroadcastGradientArgs*
T0*%
_class
loc:@cond_4/dropout/div*
_output_shapes
:*
	keep_dims( *

Tidx0
Х
7training/Adam/gradients/cond_4/dropout/div_grad/ReshapeReshape3training/Adam/gradients/cond_4/dropout/div_grad/Sum5training/Adam/gradients/cond_4/dropout/div_grad/Shape*
T0*%
_class
loc:@cond_4/dropout/div*
Tshape0*'
_output_shapes
:€€€€€€€€€d
≤
3training/Adam/gradients/cond_4/dropout/div_grad/NegNegcond_4/dropout/Shape/Switch:1*
T0*%
_class
loc:@cond_4/dropout/div*'
_output_shapes
:€€€€€€€€€d
м
9training/Adam/gradients/cond_4/dropout/div_grad/RealDiv_1RealDiv3training/Adam/gradients/cond_4/dropout/div_grad/Negcond_4/dropout/keep_prob*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_4/dropout/div
т
9training/Adam/gradients/cond_4/dropout/div_grad/RealDiv_2RealDiv9training/Adam/gradients/cond_4/dropout/div_grad/RealDiv_1cond_4/dropout/keep_prob*
T0*%
_class
loc:@cond_4/dropout/div*'
_output_shapes
:€€€€€€€€€d
З
3training/Adam/gradients/cond_4/dropout/div_grad/mulMul7training/Adam/gradients/cond_4/dropout/mul_grad/Reshape9training/Adam/gradients/cond_4/dropout/div_grad/RealDiv_2*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_4/dropout/div
°
5training/Adam/gradients/cond_4/dropout/div_grad/Sum_1Sum3training/Adam/gradients/cond_4/dropout/div_grad/mulGtraining/Adam/gradients/cond_4/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_4/dropout/div
К
9training/Adam/gradients/cond_4/dropout/div_grad/Reshape_1Reshape5training/Adam/gradients/cond_4/dropout/div_grad/Sum_17training/Adam/gradients/cond_4/dropout/div_grad/Shape_1*
T0*%
_class
loc:@cond_4/dropout/div*
Tshape0*
_output_shapes
: 
Ґ
 training/Adam/gradients/Switch_1SwitchRelu_4cond_4/pred_id*
T0*
_class
loc:@Relu_4*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Э
"training/Adam/gradients/Identity_1Identity training/Adam/gradients/Switch_1*'
_output_shapes
:€€€€€€€€€d*
T0*
_class
loc:@Relu_4
Ъ
training/Adam/gradients/Shape_2Shape training/Adam/gradients/Switch_1*
T0*
_class
loc:@Relu_4*
out_type0*
_output_shapes
:
™
%training/Adam/gradients/zeros_1/ConstConst#^training/Adam/gradients/Identity_1*
_class
loc:@Relu_4*
valueB
 *    *
dtype0*
_output_shapes
: 
ќ
training/Adam/gradients/zeros_1Filltraining/Adam/gradients/Shape_2%training/Adam/gradients/zeros_1/Const*
T0*
_class
loc:@Relu_4*

index_type0*'
_output_shapes
:€€€€€€€€€d
э
Btraining/Adam/gradients/cond_4/dropout/Shape/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_17training/Adam/gradients/cond_4/dropout/div_grad/Reshape*
T0*
_class
loc:@Relu_4*
N*)
_output_shapes
:€€€€€€€€€d: 
€
training/Adam/gradients/AddN_1AddN=training/Adam/gradients/cond_4/Identity/Switch_grad/cond_gradBtraining/Adam/gradients/cond_4/dropout/Shape/Switch_grad/cond_grad*
T0*
_class
loc:@Relu_4*
N*'
_output_shapes
:€€€€€€€€€d
≠
,training/Adam/gradients/Relu_4_grad/ReluGradReluGradtraining/Adam/gradients/AddN_1Relu_4*
T0*
_class
loc:@Relu_4*'
_output_shapes
:€€€€€€€€€d
…
2training/Adam/gradients/BiasAdd_4_grad/BiasAddGradBiasAddGrad,training/Adam/gradients/Relu_4_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:d*
T0*
_class
loc:@BiasAdd_4
т
,training/Adam/gradients/MatMul_4_grad/MatMulMatMul,training/Adam/gradients/Relu_4_grad/ReluGradMatMul_4/ReadVariableOp*
T0*
_class
loc:@MatMul_4*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b(
а
.training/Adam/gradients/MatMul_4_grad/MatMul_1MatMulcond_3/Merge,training/Adam/gradients/Relu_4_grad/ReluGrad*
_output_shapes

:dd*
transpose_a(*
transpose_b( *
T0*
_class
loc:@MatMul_4
Ё
3training/Adam/gradients/cond_3/Merge_grad/cond_gradSwitch,training/Adam/gradients/MatMul_4_grad/MatMulcond_3/pred_id*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d*
T0*
_class
loc:@MatMul_4
Ѓ
5training/Adam/gradients/cond_3/dropout/mul_grad/ShapeShapecond_3/dropout/div*
T0*%
_class
loc:@cond_3/dropout/mul*
out_type0*
_output_shapes
:
≤
7training/Adam/gradients/cond_3/dropout/mul_grad/Shape_1Shapecond_3/dropout/Floor*
T0*%
_class
loc:@cond_3/dropout/mul*
out_type0*
_output_shapes
:
≤
Etraining/Adam/gradients/cond_3/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/cond_3/dropout/mul_grad/Shape7training/Adam/gradients/cond_3/dropout/mul_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*%
_class
loc:@cond_3/dropout/mul
а
3training/Adam/gradients/cond_3/dropout/mul_grad/MulMul5training/Adam/gradients/cond_3/Merge_grad/cond_grad:1cond_3/dropout/Floor*
T0*%
_class
loc:@cond_3/dropout/mul*'
_output_shapes
:€€€€€€€€€d
Э
3training/Adam/gradients/cond_3/dropout/mul_grad/SumSum3training/Adam/gradients/cond_3/dropout/mul_grad/MulEtraining/Adam/gradients/cond_3/dropout/mul_grad/BroadcastGradientArgs*
T0*%
_class
loc:@cond_3/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Х
7training/Adam/gradients/cond_3/dropout/mul_grad/ReshapeReshape3training/Adam/gradients/cond_3/dropout/mul_grad/Sum5training/Adam/gradients/cond_3/dropout/mul_grad/Shape*
T0*%
_class
loc:@cond_3/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
а
5training/Adam/gradients/cond_3/dropout/mul_grad/Mul_1Mulcond_3/dropout/div5training/Adam/gradients/cond_3/Merge_grad/cond_grad:1*
T0*%
_class
loc:@cond_3/dropout/mul*'
_output_shapes
:€€€€€€€€€d
£
5training/Adam/gradients/cond_3/dropout/mul_grad/Sum_1Sum5training/Adam/gradients/cond_3/dropout/mul_grad/Mul_1Gtraining/Adam/gradients/cond_3/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_3/dropout/mul*
_output_shapes
:
Ы
9training/Adam/gradients/cond_3/dropout/mul_grad/Reshape_1Reshape5training/Adam/gradients/cond_3/dropout/mul_grad/Sum_17training/Adam/gradients/cond_3/dropout/mul_grad/Shape_1*
T0*%
_class
loc:@cond_3/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
Ґ
 training/Adam/gradients/Switch_2SwitchRelu_3cond_3/pred_id*
T0*
_class
loc:@Relu_3*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Я
"training/Adam/gradients/Identity_2Identity"training/Adam/gradients/Switch_2:1*
T0*
_class
loc:@Relu_3*'
_output_shapes
:€€€€€€€€€d
Ь
training/Adam/gradients/Shape_3Shape"training/Adam/gradients/Switch_2:1*
_output_shapes
:*
T0*
_class
loc:@Relu_3*
out_type0
™
%training/Adam/gradients/zeros_2/ConstConst#^training/Adam/gradients/Identity_2*
_class
loc:@Relu_3*
valueB
 *    *
dtype0*
_output_shapes
: 
ќ
training/Adam/gradients/zeros_2Filltraining/Adam/gradients/Shape_3%training/Adam/gradients/zeros_2/Const*
T0*
_class
loc:@Relu_3*

index_type0*'
_output_shapes
:€€€€€€€€€d
ф
=training/Adam/gradients/cond_3/Identity/Switch_grad/cond_gradMerge3training/Adam/gradients/cond_3/Merge_grad/cond_gradtraining/Adam/gradients/zeros_2*
T0*
_class
loc:@Relu_3*
N*)
_output_shapes
:€€€€€€€€€d: 
є
5training/Adam/gradients/cond_3/dropout/div_grad/ShapeShapecond_3/dropout/Shape/Switch:1*
T0*%
_class
loc:@cond_3/dropout/div*
out_type0*
_output_shapes
:
°
7training/Adam/gradients/cond_3/dropout/div_grad/Shape_1Const*%
_class
loc:@cond_3/dropout/div*
valueB *
dtype0*
_output_shapes
: 
≤
Etraining/Adam/gradients/cond_3/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/cond_3/dropout/div_grad/Shape7training/Adam/gradients/cond_3/dropout/div_grad/Shape_1*
T0*%
_class
loc:@cond_3/dropout/div*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
о
7training/Adam/gradients/cond_3/dropout/div_grad/RealDivRealDiv7training/Adam/gradients/cond_3/dropout/mul_grad/Reshapecond_3/dropout/keep_prob*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_3/dropout/div
°
3training/Adam/gradients/cond_3/dropout/div_grad/SumSum7training/Adam/gradients/cond_3/dropout/div_grad/RealDivEtraining/Adam/gradients/cond_3/dropout/div_grad/BroadcastGradientArgs*
T0*%
_class
loc:@cond_3/dropout/div*
_output_shapes
:*
	keep_dims( *

Tidx0
Х
7training/Adam/gradients/cond_3/dropout/div_grad/ReshapeReshape3training/Adam/gradients/cond_3/dropout/div_grad/Sum5training/Adam/gradients/cond_3/dropout/div_grad/Shape*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_3/dropout/div*
Tshape0
≤
3training/Adam/gradients/cond_3/dropout/div_grad/NegNegcond_3/dropout/Shape/Switch:1*
T0*%
_class
loc:@cond_3/dropout/div*'
_output_shapes
:€€€€€€€€€d
м
9training/Adam/gradients/cond_3/dropout/div_grad/RealDiv_1RealDiv3training/Adam/gradients/cond_3/dropout/div_grad/Negcond_3/dropout/keep_prob*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_3/dropout/div
т
9training/Adam/gradients/cond_3/dropout/div_grad/RealDiv_2RealDiv9training/Adam/gradients/cond_3/dropout/div_grad/RealDiv_1cond_3/dropout/keep_prob*
T0*%
_class
loc:@cond_3/dropout/div*'
_output_shapes
:€€€€€€€€€d
З
3training/Adam/gradients/cond_3/dropout/div_grad/mulMul7training/Adam/gradients/cond_3/dropout/mul_grad/Reshape9training/Adam/gradients/cond_3/dropout/div_grad/RealDiv_2*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_3/dropout/div
°
5training/Adam/gradients/cond_3/dropout/div_grad/Sum_1Sum3training/Adam/gradients/cond_3/dropout/div_grad/mulGtraining/Adam/gradients/cond_3/dropout/div_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@cond_3/dropout/div*
_output_shapes
:*
	keep_dims( *

Tidx0
К
9training/Adam/gradients/cond_3/dropout/div_grad/Reshape_1Reshape5training/Adam/gradients/cond_3/dropout/div_grad/Sum_17training/Adam/gradients/cond_3/dropout/div_grad/Shape_1*
_output_shapes
: *
T0*%
_class
loc:@cond_3/dropout/div*
Tshape0
Ґ
 training/Adam/gradients/Switch_3SwitchRelu_3cond_3/pred_id*
T0*
_class
loc:@Relu_3*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Э
"training/Adam/gradients/Identity_3Identity training/Adam/gradients/Switch_3*'
_output_shapes
:€€€€€€€€€d*
T0*
_class
loc:@Relu_3
Ъ
training/Adam/gradients/Shape_4Shape training/Adam/gradients/Switch_3*
T0*
_class
loc:@Relu_3*
out_type0*
_output_shapes
:
™
%training/Adam/gradients/zeros_3/ConstConst#^training/Adam/gradients/Identity_3*
_class
loc:@Relu_3*
valueB
 *    *
dtype0*
_output_shapes
: 
ќ
training/Adam/gradients/zeros_3Filltraining/Adam/gradients/Shape_4%training/Adam/gradients/zeros_3/Const*'
_output_shapes
:€€€€€€€€€d*
T0*
_class
loc:@Relu_3*

index_type0
э
Btraining/Adam/gradients/cond_3/dropout/Shape/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_37training/Adam/gradients/cond_3/dropout/div_grad/Reshape*
N*)
_output_shapes
:€€€€€€€€€d: *
T0*
_class
loc:@Relu_3
€
training/Adam/gradients/AddN_2AddN=training/Adam/gradients/cond_3/Identity/Switch_grad/cond_gradBtraining/Adam/gradients/cond_3/dropout/Shape/Switch_grad/cond_grad*
T0*
_class
loc:@Relu_3*
N*'
_output_shapes
:€€€€€€€€€d
≠
,training/Adam/gradients/Relu_3_grad/ReluGradReluGradtraining/Adam/gradients/AddN_2Relu_3*
T0*
_class
loc:@Relu_3*'
_output_shapes
:€€€€€€€€€d
…
2training/Adam/gradients/BiasAdd_3_grad/BiasAddGradBiasAddGrad,training/Adam/gradients/Relu_3_grad/ReluGrad*
T0*
_class
loc:@BiasAdd_3*
data_formatNHWC*
_output_shapes
:d
т
,training/Adam/gradients/MatMul_3_grad/MatMulMatMul,training/Adam/gradients/Relu_3_grad/ReluGradMatMul_3/ReadVariableOp*
transpose_b(*
T0*
_class
loc:@MatMul_3*'
_output_shapes
:€€€€€€€€€d*
transpose_a( 
а
.training/Adam/gradients/MatMul_3_grad/MatMul_1MatMulcond_2/Merge,training/Adam/gradients/Relu_3_grad/ReluGrad*
T0*
_class
loc:@MatMul_3*
_output_shapes

:dd*
transpose_a(*
transpose_b( 
Ё
3training/Adam/gradients/cond_2/Merge_grad/cond_gradSwitch,training/Adam/gradients/MatMul_3_grad/MatMulcond_2/pred_id*
T0*
_class
loc:@MatMul_3*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Ѓ
5training/Adam/gradients/cond_2/dropout/mul_grad/ShapeShapecond_2/dropout/div*
_output_shapes
:*
T0*%
_class
loc:@cond_2/dropout/mul*
out_type0
≤
7training/Adam/gradients/cond_2/dropout/mul_grad/Shape_1Shapecond_2/dropout/Floor*
T0*%
_class
loc:@cond_2/dropout/mul*
out_type0*
_output_shapes
:
≤
Etraining/Adam/gradients/cond_2/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/cond_2/dropout/mul_grad/Shape7training/Adam/gradients/cond_2/dropout/mul_grad/Shape_1*
T0*%
_class
loc:@cond_2/dropout/mul*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
а
3training/Adam/gradients/cond_2/dropout/mul_grad/MulMul5training/Adam/gradients/cond_2/Merge_grad/cond_grad:1cond_2/dropout/Floor*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_2/dropout/mul
Э
3training/Adam/gradients/cond_2/dropout/mul_grad/SumSum3training/Adam/gradients/cond_2/dropout/mul_grad/MulEtraining/Adam/gradients/cond_2/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_2/dropout/mul*
_output_shapes
:
Х
7training/Adam/gradients/cond_2/dropout/mul_grad/ReshapeReshape3training/Adam/gradients/cond_2/dropout/mul_grad/Sum5training/Adam/gradients/cond_2/dropout/mul_grad/Shape*
T0*%
_class
loc:@cond_2/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
а
5training/Adam/gradients/cond_2/dropout/mul_grad/Mul_1Mulcond_2/dropout/div5training/Adam/gradients/cond_2/Merge_grad/cond_grad:1*
T0*%
_class
loc:@cond_2/dropout/mul*'
_output_shapes
:€€€€€€€€€d
£
5training/Adam/gradients/cond_2/dropout/mul_grad/Sum_1Sum5training/Adam/gradients/cond_2/dropout/mul_grad/Mul_1Gtraining/Adam/gradients/cond_2/dropout/mul_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@cond_2/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Ы
9training/Adam/gradients/cond_2/dropout/mul_grad/Reshape_1Reshape5training/Adam/gradients/cond_2/dropout/mul_grad/Sum_17training/Adam/gradients/cond_2/dropout/mul_grad/Shape_1*
T0*%
_class
loc:@cond_2/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
Ґ
 training/Adam/gradients/Switch_4SwitchRelu_2cond_2/pred_id*
T0*
_class
loc:@Relu_2*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Я
"training/Adam/gradients/Identity_4Identity"training/Adam/gradients/Switch_4:1*'
_output_shapes
:€€€€€€€€€d*
T0*
_class
loc:@Relu_2
Ь
training/Adam/gradients/Shape_5Shape"training/Adam/gradients/Switch_4:1*
T0*
_class
loc:@Relu_2*
out_type0*
_output_shapes
:
™
%training/Adam/gradients/zeros_4/ConstConst#^training/Adam/gradients/Identity_4*
_class
loc:@Relu_2*
valueB
 *    *
dtype0*
_output_shapes
: 
ќ
training/Adam/gradients/zeros_4Filltraining/Adam/gradients/Shape_5%training/Adam/gradients/zeros_4/Const*
T0*
_class
loc:@Relu_2*

index_type0*'
_output_shapes
:€€€€€€€€€d
ф
=training/Adam/gradients/cond_2/Identity/Switch_grad/cond_gradMerge3training/Adam/gradients/cond_2/Merge_grad/cond_gradtraining/Adam/gradients/zeros_4*
T0*
_class
loc:@Relu_2*
N*)
_output_shapes
:€€€€€€€€€d: 
є
5training/Adam/gradients/cond_2/dropout/div_grad/ShapeShapecond_2/dropout/Shape/Switch:1*
T0*%
_class
loc:@cond_2/dropout/div*
out_type0*
_output_shapes
:
°
7training/Adam/gradients/cond_2/dropout/div_grad/Shape_1Const*%
_class
loc:@cond_2/dropout/div*
valueB *
dtype0*
_output_shapes
: 
≤
Etraining/Adam/gradients/cond_2/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/cond_2/dropout/div_grad/Shape7training/Adam/gradients/cond_2/dropout/div_grad/Shape_1*
T0*%
_class
loc:@cond_2/dropout/div*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
о
7training/Adam/gradients/cond_2/dropout/div_grad/RealDivRealDiv7training/Adam/gradients/cond_2/dropout/mul_grad/Reshapecond_2/dropout/keep_prob*
T0*%
_class
loc:@cond_2/dropout/div*'
_output_shapes
:€€€€€€€€€d
°
3training/Adam/gradients/cond_2/dropout/div_grad/SumSum7training/Adam/gradients/cond_2/dropout/div_grad/RealDivEtraining/Adam/gradients/cond_2/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_2/dropout/div
Х
7training/Adam/gradients/cond_2/dropout/div_grad/ReshapeReshape3training/Adam/gradients/cond_2/dropout/div_grad/Sum5training/Adam/gradients/cond_2/dropout/div_grad/Shape*
T0*%
_class
loc:@cond_2/dropout/div*
Tshape0*'
_output_shapes
:€€€€€€€€€d
≤
3training/Adam/gradients/cond_2/dropout/div_grad/NegNegcond_2/dropout/Shape/Switch:1*
T0*%
_class
loc:@cond_2/dropout/div*'
_output_shapes
:€€€€€€€€€d
м
9training/Adam/gradients/cond_2/dropout/div_grad/RealDiv_1RealDiv3training/Adam/gradients/cond_2/dropout/div_grad/Negcond_2/dropout/keep_prob*
T0*%
_class
loc:@cond_2/dropout/div*'
_output_shapes
:€€€€€€€€€d
т
9training/Adam/gradients/cond_2/dropout/div_grad/RealDiv_2RealDiv9training/Adam/gradients/cond_2/dropout/div_grad/RealDiv_1cond_2/dropout/keep_prob*
T0*%
_class
loc:@cond_2/dropout/div*'
_output_shapes
:€€€€€€€€€d
З
3training/Adam/gradients/cond_2/dropout/div_grad/mulMul7training/Adam/gradients/cond_2/dropout/mul_grad/Reshape9training/Adam/gradients/cond_2/dropout/div_grad/RealDiv_2*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_2/dropout/div
°
5training/Adam/gradients/cond_2/dropout/div_grad/Sum_1Sum3training/Adam/gradients/cond_2/dropout/div_grad/mulGtraining/Adam/gradients/cond_2/dropout/div_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@cond_2/dropout/div*
_output_shapes
:*
	keep_dims( *

Tidx0
К
9training/Adam/gradients/cond_2/dropout/div_grad/Reshape_1Reshape5training/Adam/gradients/cond_2/dropout/div_grad/Sum_17training/Adam/gradients/cond_2/dropout/div_grad/Shape_1*
T0*%
_class
loc:@cond_2/dropout/div*
Tshape0*
_output_shapes
: 
Ґ
 training/Adam/gradients/Switch_5SwitchRelu_2cond_2/pred_id*
T0*
_class
loc:@Relu_2*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Э
"training/Adam/gradients/Identity_5Identity training/Adam/gradients/Switch_5*
T0*
_class
loc:@Relu_2*'
_output_shapes
:€€€€€€€€€d
Ъ
training/Adam/gradients/Shape_6Shape training/Adam/gradients/Switch_5*
T0*
_class
loc:@Relu_2*
out_type0*
_output_shapes
:
™
%training/Adam/gradients/zeros_5/ConstConst#^training/Adam/gradients/Identity_5*
_class
loc:@Relu_2*
valueB
 *    *
dtype0*
_output_shapes
: 
ќ
training/Adam/gradients/zeros_5Filltraining/Adam/gradients/Shape_6%training/Adam/gradients/zeros_5/Const*
T0*
_class
loc:@Relu_2*

index_type0*'
_output_shapes
:€€€€€€€€€d
э
Btraining/Adam/gradients/cond_2/dropout/Shape/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_57training/Adam/gradients/cond_2/dropout/div_grad/Reshape*
N*)
_output_shapes
:€€€€€€€€€d: *
T0*
_class
loc:@Relu_2
€
training/Adam/gradients/AddN_3AddN=training/Adam/gradients/cond_2/Identity/Switch_grad/cond_gradBtraining/Adam/gradients/cond_2/dropout/Shape/Switch_grad/cond_grad*
T0*
_class
loc:@Relu_2*
N*'
_output_shapes
:€€€€€€€€€d
≠
,training/Adam/gradients/Relu_2_grad/ReluGradReluGradtraining/Adam/gradients/AddN_3Relu_2*
T0*
_class
loc:@Relu_2*'
_output_shapes
:€€€€€€€€€d
…
2training/Adam/gradients/BiasAdd_2_grad/BiasAddGradBiasAddGrad,training/Adam/gradients/Relu_2_grad/ReluGrad*
T0*
_class
loc:@BiasAdd_2*
data_formatNHWC*
_output_shapes
:d
т
,training/Adam/gradients/MatMul_2_grad/MatMulMatMul,training/Adam/gradients/Relu_2_grad/ReluGradMatMul_2/ReadVariableOp*
T0*
_class
loc:@MatMul_2*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b(
а
.training/Adam/gradients/MatMul_2_grad/MatMul_1MatMulcond_1/Merge,training/Adam/gradients/Relu_2_grad/ReluGrad*
T0*
_class
loc:@MatMul_2*
_output_shapes

:dd*
transpose_a(*
transpose_b( 
Ё
3training/Adam/gradients/cond_1/Merge_grad/cond_gradSwitch,training/Adam/gradients/MatMul_2_grad/MatMulcond_1/pred_id*
T0*
_class
loc:@MatMul_2*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Ѓ
5training/Adam/gradients/cond_1/dropout/mul_grad/ShapeShapecond_1/dropout/div*
T0*%
_class
loc:@cond_1/dropout/mul*
out_type0*
_output_shapes
:
≤
7training/Adam/gradients/cond_1/dropout/mul_grad/Shape_1Shapecond_1/dropout/Floor*
_output_shapes
:*
T0*%
_class
loc:@cond_1/dropout/mul*
out_type0
≤
Etraining/Adam/gradients/cond_1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/cond_1/dropout/mul_grad/Shape7training/Adam/gradients/cond_1/dropout/mul_grad/Shape_1*
T0*%
_class
loc:@cond_1/dropout/mul*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
а
3training/Adam/gradients/cond_1/dropout/mul_grad/MulMul5training/Adam/gradients/cond_1/Merge_grad/cond_grad:1cond_1/dropout/Floor*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_1/dropout/mul
Э
3training/Adam/gradients/cond_1/dropout/mul_grad/SumSum3training/Adam/gradients/cond_1/dropout/mul_grad/MulEtraining/Adam/gradients/cond_1/dropout/mul_grad/BroadcastGradientArgs*
T0*%
_class
loc:@cond_1/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Х
7training/Adam/gradients/cond_1/dropout/mul_grad/ReshapeReshape3training/Adam/gradients/cond_1/dropout/mul_grad/Sum5training/Adam/gradients/cond_1/dropout/mul_grad/Shape*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_1/dropout/mul*
Tshape0
а
5training/Adam/gradients/cond_1/dropout/mul_grad/Mul_1Mulcond_1/dropout/div5training/Adam/gradients/cond_1/Merge_grad/cond_grad:1*
T0*%
_class
loc:@cond_1/dropout/mul*'
_output_shapes
:€€€€€€€€€d
£
5training/Adam/gradients/cond_1/dropout/mul_grad/Sum_1Sum5training/Adam/gradients/cond_1/dropout/mul_grad/Mul_1Gtraining/Adam/gradients/cond_1/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_1/dropout/mul*
_output_shapes
:
Ы
9training/Adam/gradients/cond_1/dropout/mul_grad/Reshape_1Reshape5training/Adam/gradients/cond_1/dropout/mul_grad/Sum_17training/Adam/gradients/cond_1/dropout/mul_grad/Shape_1*
T0*%
_class
loc:@cond_1/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
Ґ
 training/Adam/gradients/Switch_6SwitchRelu_1cond_1/pred_id*
T0*
_class
loc:@Relu_1*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Я
"training/Adam/gradients/Identity_6Identity"training/Adam/gradients/Switch_6:1*'
_output_shapes
:€€€€€€€€€d*
T0*
_class
loc:@Relu_1
Ь
training/Adam/gradients/Shape_7Shape"training/Adam/gradients/Switch_6:1*
T0*
_class
loc:@Relu_1*
out_type0*
_output_shapes
:
™
%training/Adam/gradients/zeros_6/ConstConst#^training/Adam/gradients/Identity_6*
dtype0*
_output_shapes
: *
_class
loc:@Relu_1*
valueB
 *    
ќ
training/Adam/gradients/zeros_6Filltraining/Adam/gradients/Shape_7%training/Adam/gradients/zeros_6/Const*
T0*
_class
loc:@Relu_1*

index_type0*'
_output_shapes
:€€€€€€€€€d
ф
=training/Adam/gradients/cond_1/Identity/Switch_grad/cond_gradMerge3training/Adam/gradients/cond_1/Merge_grad/cond_gradtraining/Adam/gradients/zeros_6*
T0*
_class
loc:@Relu_1*
N*)
_output_shapes
:€€€€€€€€€d: 
є
5training/Adam/gradients/cond_1/dropout/div_grad/ShapeShapecond_1/dropout/Shape/Switch:1*
T0*%
_class
loc:@cond_1/dropout/div*
out_type0*
_output_shapes
:
°
7training/Adam/gradients/cond_1/dropout/div_grad/Shape_1Const*%
_class
loc:@cond_1/dropout/div*
valueB *
dtype0*
_output_shapes
: 
≤
Etraining/Adam/gradients/cond_1/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/cond_1/dropout/div_grad/Shape7training/Adam/gradients/cond_1/dropout/div_grad/Shape_1*
T0*%
_class
loc:@cond_1/dropout/div*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
о
7training/Adam/gradients/cond_1/dropout/div_grad/RealDivRealDiv7training/Adam/gradients/cond_1/dropout/mul_grad/Reshapecond_1/dropout/keep_prob*
T0*%
_class
loc:@cond_1/dropout/div*'
_output_shapes
:€€€€€€€€€d
°
3training/Adam/gradients/cond_1/dropout/div_grad/SumSum7training/Adam/gradients/cond_1/dropout/div_grad/RealDivEtraining/Adam/gradients/cond_1/dropout/div_grad/BroadcastGradientArgs*
T0*%
_class
loc:@cond_1/dropout/div*
_output_shapes
:*
	keep_dims( *

Tidx0
Х
7training/Adam/gradients/cond_1/dropout/div_grad/ReshapeReshape3training/Adam/gradients/cond_1/dropout/div_grad/Sum5training/Adam/gradients/cond_1/dropout/div_grad/Shape*
T0*%
_class
loc:@cond_1/dropout/div*
Tshape0*'
_output_shapes
:€€€€€€€€€d
≤
3training/Adam/gradients/cond_1/dropout/div_grad/NegNegcond_1/dropout/Shape/Switch:1*
T0*%
_class
loc:@cond_1/dropout/div*'
_output_shapes
:€€€€€€€€€d
м
9training/Adam/gradients/cond_1/dropout/div_grad/RealDiv_1RealDiv3training/Adam/gradients/cond_1/dropout/div_grad/Negcond_1/dropout/keep_prob*
T0*%
_class
loc:@cond_1/dropout/div*'
_output_shapes
:€€€€€€€€€d
т
9training/Adam/gradients/cond_1/dropout/div_grad/RealDiv_2RealDiv9training/Adam/gradients/cond_1/dropout/div_grad/RealDiv_1cond_1/dropout/keep_prob*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_1/dropout/div
З
3training/Adam/gradients/cond_1/dropout/div_grad/mulMul7training/Adam/gradients/cond_1/dropout/mul_grad/Reshape9training/Adam/gradients/cond_1/dropout/div_grad/RealDiv_2*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_1/dropout/div
°
5training/Adam/gradients/cond_1/dropout/div_grad/Sum_1Sum3training/Adam/gradients/cond_1/dropout/div_grad/mulGtraining/Adam/gradients/cond_1/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_1/dropout/div
К
9training/Adam/gradients/cond_1/dropout/div_grad/Reshape_1Reshape5training/Adam/gradients/cond_1/dropout/div_grad/Sum_17training/Adam/gradients/cond_1/dropout/div_grad/Shape_1*
_output_shapes
: *
T0*%
_class
loc:@cond_1/dropout/div*
Tshape0
Ґ
 training/Adam/gradients/Switch_7SwitchRelu_1cond_1/pred_id*
T0*
_class
loc:@Relu_1*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Э
"training/Adam/gradients/Identity_7Identity training/Adam/gradients/Switch_7*'
_output_shapes
:€€€€€€€€€d*
T0*
_class
loc:@Relu_1
Ъ
training/Adam/gradients/Shape_8Shape training/Adam/gradients/Switch_7*
_output_shapes
:*
T0*
_class
loc:@Relu_1*
out_type0
™
%training/Adam/gradients/zeros_7/ConstConst#^training/Adam/gradients/Identity_7*
dtype0*
_output_shapes
: *
_class
loc:@Relu_1*
valueB
 *    
ќ
training/Adam/gradients/zeros_7Filltraining/Adam/gradients/Shape_8%training/Adam/gradients/zeros_7/Const*'
_output_shapes
:€€€€€€€€€d*
T0*
_class
loc:@Relu_1*

index_type0
э
Btraining/Adam/gradients/cond_1/dropout/Shape/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_77training/Adam/gradients/cond_1/dropout/div_grad/Reshape*
T0*
_class
loc:@Relu_1*
N*)
_output_shapes
:€€€€€€€€€d: 
€
training/Adam/gradients/AddN_4AddN=training/Adam/gradients/cond_1/Identity/Switch_grad/cond_gradBtraining/Adam/gradients/cond_1/dropout/Shape/Switch_grad/cond_grad*
N*'
_output_shapes
:€€€€€€€€€d*
T0*
_class
loc:@Relu_1
≠
,training/Adam/gradients/Relu_1_grad/ReluGradReluGradtraining/Adam/gradients/AddN_4Relu_1*
T0*
_class
loc:@Relu_1*'
_output_shapes
:€€€€€€€€€d
…
2training/Adam/gradients/BiasAdd_1_grad/BiasAddGradBiasAddGrad,training/Adam/gradients/Relu_1_grad/ReluGrad*
T0*
_class
loc:@BiasAdd_1*
data_formatNHWC*
_output_shapes
:d
т
,training/Adam/gradients/MatMul_1_grad/MatMulMatMul,training/Adam/gradients/Relu_1_grad/ReluGradMatMul_1/ReadVariableOp*
T0*
_class
loc:@MatMul_1*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b(
ё
.training/Adam/gradients/MatMul_1_grad/MatMul_1MatMul
cond/Merge,training/Adam/gradients/Relu_1_grad/ReluGrad*
T0*
_class
loc:@MatMul_1*
_output_shapes

:dd*
transpose_a(*
transpose_b( 
ў
1training/Adam/gradients/cond/Merge_grad/cond_gradSwitch,training/Adam/gradients/MatMul_1_grad/MatMulcond/pred_id*
T0*
_class
loc:@MatMul_1*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
®
3training/Adam/gradients/cond/dropout/mul_grad/ShapeShapecond/dropout/div*
T0*#
_class
loc:@cond/dropout/mul*
out_type0*
_output_shapes
:
ђ
5training/Adam/gradients/cond/dropout/mul_grad/Shape_1Shapecond/dropout/Floor*
T0*#
_class
loc:@cond/dropout/mul*
out_type0*
_output_shapes
:
™
Ctraining/Adam/gradients/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs3training/Adam/gradients/cond/dropout/mul_grad/Shape5training/Adam/gradients/cond/dropout/mul_grad/Shape_1*
T0*#
_class
loc:@cond/dropout/mul*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ў
1training/Adam/gradients/cond/dropout/mul_grad/MulMul3training/Adam/gradients/cond/Merge_grad/cond_grad:1cond/dropout/Floor*
T0*#
_class
loc:@cond/dropout/mul*'
_output_shapes
:€€€€€€€€€d
Х
1training/Adam/gradients/cond/dropout/mul_grad/SumSum1training/Adam/gradients/cond/dropout/mul_grad/MulCtraining/Adam/gradients/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*#
_class
loc:@cond/dropout/mul*
_output_shapes
:
Н
5training/Adam/gradients/cond/dropout/mul_grad/ReshapeReshape1training/Adam/gradients/cond/dropout/mul_grad/Sum3training/Adam/gradients/cond/dropout/mul_grad/Shape*
T0*#
_class
loc:@cond/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
Ў
3training/Adam/gradients/cond/dropout/mul_grad/Mul_1Mulcond/dropout/div3training/Adam/gradients/cond/Merge_grad/cond_grad:1*
T0*#
_class
loc:@cond/dropout/mul*'
_output_shapes
:€€€€€€€€€d
Ы
3training/Adam/gradients/cond/dropout/mul_grad/Sum_1Sum3training/Adam/gradients/cond/dropout/mul_grad/Mul_1Etraining/Adam/gradients/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*#
_class
loc:@cond/dropout/mul*
_output_shapes
:
У
7training/Adam/gradients/cond/dropout/mul_grad/Reshape_1Reshape3training/Adam/gradients/cond/dropout/mul_grad/Sum_15training/Adam/gradients/cond/dropout/mul_grad/Shape_1*'
_output_shapes
:€€€€€€€€€d*
T0*#
_class
loc:@cond/dropout/mul*
Tshape0
Ь
 training/Adam/gradients/Switch_8SwitchRelucond/pred_id*
T0*
_class
	loc:@Relu*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Э
"training/Adam/gradients/Identity_8Identity"training/Adam/gradients/Switch_8:1*
T0*
_class
	loc:@Relu*'
_output_shapes
:€€€€€€€€€d
Ъ
training/Adam/gradients/Shape_9Shape"training/Adam/gradients/Switch_8:1*
_output_shapes
:*
T0*
_class
	loc:@Relu*
out_type0
®
%training/Adam/gradients/zeros_8/ConstConst#^training/Adam/gradients/Identity_8*
_class
	loc:@Relu*
valueB
 *    *
dtype0*
_output_shapes
: 
ћ
training/Adam/gradients/zeros_8Filltraining/Adam/gradients/Shape_9%training/Adam/gradients/zeros_8/Const*
T0*
_class
	loc:@Relu*

index_type0*'
_output_shapes
:€€€€€€€€€d
о
;training/Adam/gradients/cond/Identity/Switch_grad/cond_gradMerge1training/Adam/gradients/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_8*
N*)
_output_shapes
:€€€€€€€€€d: *
T0*
_class
	loc:@Relu
≥
3training/Adam/gradients/cond/dropout/div_grad/ShapeShapecond/dropout/Shape/Switch:1*
T0*#
_class
loc:@cond/dropout/div*
out_type0*
_output_shapes
:
Э
5training/Adam/gradients/cond/dropout/div_grad/Shape_1Const*#
_class
loc:@cond/dropout/div*
valueB *
dtype0*
_output_shapes
: 
™
Ctraining/Adam/gradients/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs3training/Adam/gradients/cond/dropout/div_grad/Shape5training/Adam/gradients/cond/dropout/div_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*#
_class
loc:@cond/dropout/div
ж
5training/Adam/gradients/cond/dropout/div_grad/RealDivRealDiv5training/Adam/gradients/cond/dropout/mul_grad/Reshapecond/dropout/keep_prob*
T0*#
_class
loc:@cond/dropout/div*'
_output_shapes
:€€€€€€€€€d
Щ
1training/Adam/gradients/cond/dropout/div_grad/SumSum5training/Adam/gradients/cond/dropout/div_grad/RealDivCtraining/Adam/gradients/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*#
_class
loc:@cond/dropout/div
Н
5training/Adam/gradients/cond/dropout/div_grad/ReshapeReshape1training/Adam/gradients/cond/dropout/div_grad/Sum3training/Adam/gradients/cond/dropout/div_grad/Shape*
T0*#
_class
loc:@cond/dropout/div*
Tshape0*'
_output_shapes
:€€€€€€€€€d
ђ
1training/Adam/gradients/cond/dropout/div_grad/NegNegcond/dropout/Shape/Switch:1*
T0*#
_class
loc:@cond/dropout/div*'
_output_shapes
:€€€€€€€€€d
д
7training/Adam/gradients/cond/dropout/div_grad/RealDiv_1RealDiv1training/Adam/gradients/cond/dropout/div_grad/Negcond/dropout/keep_prob*'
_output_shapes
:€€€€€€€€€d*
T0*#
_class
loc:@cond/dropout/div
к
7training/Adam/gradients/cond/dropout/div_grad/RealDiv_2RealDiv7training/Adam/gradients/cond/dropout/div_grad/RealDiv_1cond/dropout/keep_prob*
T0*#
_class
loc:@cond/dropout/div*'
_output_shapes
:€€€€€€€€€d
€
1training/Adam/gradients/cond/dropout/div_grad/mulMul5training/Adam/gradients/cond/dropout/mul_grad/Reshape7training/Adam/gradients/cond/dropout/div_grad/RealDiv_2*
T0*#
_class
loc:@cond/dropout/div*'
_output_shapes
:€€€€€€€€€d
Щ
3training/Adam/gradients/cond/dropout/div_grad/Sum_1Sum1training/Adam/gradients/cond/dropout/div_grad/mulEtraining/Adam/gradients/cond/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*#
_class
loc:@cond/dropout/div*
_output_shapes
:
В
7training/Adam/gradients/cond/dropout/div_grad/Reshape_1Reshape3training/Adam/gradients/cond/dropout/div_grad/Sum_15training/Adam/gradients/cond/dropout/div_grad/Shape_1*
T0*#
_class
loc:@cond/dropout/div*
Tshape0*
_output_shapes
: 
Ь
 training/Adam/gradients/Switch_9SwitchRelucond/pred_id*
T0*
_class
	loc:@Relu*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Ы
"training/Adam/gradients/Identity_9Identity training/Adam/gradients/Switch_9*
T0*
_class
	loc:@Relu*'
_output_shapes
:€€€€€€€€€d
Щ
 training/Adam/gradients/Shape_10Shape training/Adam/gradients/Switch_9*
_output_shapes
:*
T0*
_class
	loc:@Relu*
out_type0
®
%training/Adam/gradients/zeros_9/ConstConst#^training/Adam/gradients/Identity_9*
_class
	loc:@Relu*
valueB
 *    *
dtype0*
_output_shapes
: 
Ќ
training/Adam/gradients/zeros_9Fill training/Adam/gradients/Shape_10%training/Adam/gradients/zeros_9/Const*
T0*
_class
	loc:@Relu*

index_type0*'
_output_shapes
:€€€€€€€€€d
ч
@training/Adam/gradients/cond/dropout/Shape/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_95training/Adam/gradients/cond/dropout/div_grad/Reshape*
T0*
_class
	loc:@Relu*
N*)
_output_shapes
:€€€€€€€€€d: 
щ
training/Adam/gradients/AddN_5AddN;training/Adam/gradients/cond/Identity/Switch_grad/cond_grad@training/Adam/gradients/cond/dropout/Shape/Switch_grad/cond_grad*
T0*
_class
	loc:@Relu*
N*'
_output_shapes
:€€€€€€€€€d
І
*training/Adam/gradients/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_5Relu*
T0*
_class
	loc:@Relu*'
_output_shapes
:€€€€€€€€€d
√
0training/Adam/gradients/BiasAdd_grad/BiasAddGradBiasAddGrad*training/Adam/gradients/Relu_grad/ReluGrad*
T0*
_class
loc:@BiasAdd*
data_formatNHWC*
_output_shapes
:d
л
*training/Adam/gradients/MatMul_grad/MatMulMatMul*training/Adam/gradients/Relu_grad/ReluGradMatMul/ReadVariableOp*
T0*
_class
loc:@MatMul*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
÷
,training/Adam/gradients/MatMul_grad/MatMul_1MatMulinput_1*training/Adam/gradients/Relu_grad/ReluGrad*
_output_shapes
:	Аd*
transpose_a(*
transpose_b( *
T0*
_class
loc:@MatMul
U
training/Adam/ConstConst*
value	B	 R*
dtype0	*
_output_shapes
: 
k
!training/Adam/AssignAddVariableOpAssignAddVariableOpAdam/iterationstraining/Adam/Const*
dtype0	
И
training/Adam/ReadVariableOpReadVariableOpAdam/iterations"^training/Adam/AssignAddVariableOp*
dtype0	*
_output_shapes
: 
i
!training/Adam/Cast/ReadVariableOpReadVariableOpAdam/iterations*
dtype0	*
_output_shapes
: 
}
training/Adam/CastCast!training/Adam/Cast/ReadVariableOp*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
X
training/Adam/add/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
b
training/Adam/addAddtraining/Adam/Casttraining/Adam/add/y*
T0*
_output_shapes
: 
d
 training/Adam/Pow/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
n
training/Adam/PowPow training/Adam/Pow/ReadVariableOptraining/Adam/add*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
_output_shapes
: *
T0
Z
training/Adam/Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_2Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_2*
T0*
_output_shapes
: 
Г
training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const_1*
T0*
_output_shapes
: 
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
T0*
_output_shapes
: 
f
"training/Adam/Pow_1/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
r
training/Adam/Pow_1Pow"training/Adam/Pow_1/ReadVariableOptraining/Adam/add*
_output_shapes
: *
T0
Z
training/Adam/sub_1/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
^
training/Adam/ReadVariableOp_1ReadVariableOpAdam/lr*
dtype0*
_output_shapes
: 
p
training/Adam/mulMultraining/Adam/ReadVariableOp_1training/Adam/truediv*
T0*
_output_shapes
: 
t
#training/Adam/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"   d   
^
training/Adam/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ч
training/Adam/zerosFill#training/Adam/zeros/shape_as_tensortraining/Adam/zeros/Const*
T0*

index_type0*
_output_shapes
:	Аd
≈
training/Adam/VariableVarHandleOp*'
shared_nametraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
	container *
shape:	Аd*
dtype0*
_output_shapes
: 
}
7training/Adam/Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable*
_output_shapes
: 
Ц
training/Adam/Variable/AssignAssignVariableOptraining/Adam/Variabletraining/Adam/zeros*)
_class
loc:@training/Adam/Variable*
dtype0
≠
*training/Adam/Variable/Read/ReadVariableOpReadVariableOptraining/Adam/Variable*
dtype0*
_output_shapes
:	Аd*)
_class
loc:@training/Adam/Variable
b
training/Adam/zeros_1Const*
valueBd*    *
dtype0*
_output_shapes
:d
∆
training/Adam/Variable_1VarHandleOp*)
shared_nametraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
	container *
shape:d*
dtype0*
_output_shapes
: 
Б
9training/Adam/Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_1*
_output_shapes
: 
Ю
training/Adam/Variable_1/AssignAssignVariableOptraining/Adam/Variable_1training/Adam/zeros_1*
dtype0*+
_class!
loc:@training/Adam/Variable_1
Ѓ
,training/Adam/Variable_1/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_1*
dtype0*
_output_shapes
:d*+
_class!
loc:@training/Adam/Variable_1
v
%training/Adam/zeros_2/shape_as_tensorConst*
valueB"d   d   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ь
training/Adam/zeros_2Fill%training/Adam/zeros_2/shape_as_tensortraining/Adam/zeros_2/Const*
_output_shapes

:dd*
T0*

index_type0
 
training/Adam/Variable_2VarHandleOp*)
shared_nametraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
	container *
shape
:dd*
dtype0*
_output_shapes
: 
Б
9training/Adam/Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_2*
_output_shapes
: 
Ю
training/Adam/Variable_2/AssignAssignVariableOptraining/Adam/Variable_2training/Adam/zeros_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0
≤
,training/Adam/Variable_2/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*
_output_shapes

:dd
b
training/Adam/zeros_3Const*
valueBd*    *
dtype0*
_output_shapes
:d
∆
training/Adam/Variable_3VarHandleOp*+
_class!
loc:@training/Adam/Variable_3*
	container *
shape:d*
dtype0*
_output_shapes
: *)
shared_nametraining/Adam/Variable_3
Б
9training/Adam/Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_3*
_output_shapes
: 
Ю
training/Adam/Variable_3/AssignAssignVariableOptraining/Adam/Variable_3training/Adam/zeros_3*
dtype0*+
_class!
loc:@training/Adam/Variable_3
Ѓ
,training/Adam/Variable_3/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
dtype0*
_output_shapes
:d
v
%training/Adam/zeros_4/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"d   d   
`
training/Adam/zeros_4/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ь
training/Adam/zeros_4Fill%training/Adam/zeros_4/shape_as_tensortraining/Adam/zeros_4/Const*
T0*

index_type0*
_output_shapes

:dd
 
training/Adam/Variable_4VarHandleOp*
	container *
shape
:dd*
dtype0*
_output_shapes
: *)
shared_nametraining/Adam/Variable_4*+
_class!
loc:@training/Adam/Variable_4
Б
9training/Adam/Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_4*
_output_shapes
: 
Ю
training/Adam/Variable_4/AssignAssignVariableOptraining/Adam/Variable_4training/Adam/zeros_4*+
_class!
loc:@training/Adam/Variable_4*
dtype0
≤
,training/Adam/Variable_4/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_4*
dtype0*
_output_shapes

:dd*+
_class!
loc:@training/Adam/Variable_4
b
training/Adam/zeros_5Const*
dtype0*
_output_shapes
:d*
valueBd*    
∆
training/Adam/Variable_5VarHandleOp*)
shared_nametraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
	container *
shape:d*
dtype0*
_output_shapes
: 
Б
9training/Adam/Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_5*
_output_shapes
: 
Ю
training/Adam/Variable_5/AssignAssignVariableOptraining/Adam/Variable_5training/Adam/zeros_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0
Ѓ
,training/Adam/Variable_5/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0*
_output_shapes
:d
v
%training/Adam/zeros_6/shape_as_tensorConst*
valueB"d   d   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_6/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ь
training/Adam/zeros_6Fill%training/Adam/zeros_6/shape_as_tensortraining/Adam/zeros_6/Const*
_output_shapes

:dd*
T0*

index_type0
 
training/Adam/Variable_6VarHandleOp*)
shared_nametraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
	container *
shape
:dd*
dtype0*
_output_shapes
: 
Б
9training/Adam/Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_6*
_output_shapes
: 
Ю
training/Adam/Variable_6/AssignAssignVariableOptraining/Adam/Variable_6training/Adam/zeros_6*
dtype0*+
_class!
loc:@training/Adam/Variable_6
≤
,training/Adam/Variable_6/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0*
_output_shapes

:dd
b
training/Adam/zeros_7Const*
valueBd*    *
dtype0*
_output_shapes
:d
∆
training/Adam/Variable_7VarHandleOp*
dtype0*
_output_shapes
: *)
shared_nametraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
	container *
shape:d
Б
9training/Adam/Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_7*
_output_shapes
: 
Ю
training/Adam/Variable_7/AssignAssignVariableOptraining/Adam/Variable_7training/Adam/zeros_7*+
_class!
loc:@training/Adam/Variable_7*
dtype0
Ѓ
,training/Adam/Variable_7/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_7*
dtype0*
_output_shapes
:d*+
_class!
loc:@training/Adam/Variable_7
v
%training/Adam/zeros_8/shape_as_tensorConst*
valueB"d   d   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_8/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ь
training/Adam/zeros_8Fill%training/Adam/zeros_8/shape_as_tensortraining/Adam/zeros_8/Const*
_output_shapes

:dd*
T0*

index_type0
 
training/Adam/Variable_8VarHandleOp*
	container *
shape
:dd*
dtype0*
_output_shapes
: *)
shared_nametraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8
Б
9training/Adam/Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_8*
_output_shapes
: 
Ю
training/Adam/Variable_8/AssignAssignVariableOptraining/Adam/Variable_8training/Adam/zeros_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0
≤
,training/Adam/Variable_8/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_8*
dtype0*
_output_shapes

:dd*+
_class!
loc:@training/Adam/Variable_8
b
training/Adam/zeros_9Const*
valueBd*    *
dtype0*
_output_shapes
:d
∆
training/Adam/Variable_9VarHandleOp*)
shared_nametraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
	container *
shape:d*
dtype0*
_output_shapes
: 
Б
9training/Adam/Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_9*
_output_shapes
: 
Ю
training/Adam/Variable_9/AssignAssignVariableOptraining/Adam/Variable_9training/Adam/zeros_9*
dtype0*+
_class!
loc:@training/Adam/Variable_9
Ѓ
,training/Adam/Variable_9/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0*
_output_shapes
:d
k
training/Adam/zeros_10Const*
valueBd*    *
dtype0*
_output_shapes

:d
Ќ
training/Adam/Variable_10VarHandleOp*
shape
:d*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*
	container 
Г
:training/Adam/Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_10*
_output_shapes
: 
Ґ
 training/Adam/Variable_10/AssignAssignVariableOptraining/Adam/Variable_10training/Adam/zeros_10*,
_class"
 loc:@training/Adam/Variable_10*
dtype0
µ
-training/Adam/Variable_10/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*
dtype0*
_output_shapes

:d
c
training/Adam/zeros_11Const*
valueB*    *
dtype0*
_output_shapes
:
…
training/Adam/Variable_11VarHandleOp*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
	container *
shape:
Г
:training/Adam/Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_11*
_output_shapes
: 
Ґ
 training/Adam/Variable_11/AssignAssignVariableOptraining/Adam/Variable_11training/Adam/zeros_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0
±
-training/Adam/Variable_11/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
_output_shapes
:
w
&training/Adam/zeros_12/shape_as_tensorConst*
valueB"   d   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_12/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
†
training/Adam/zeros_12Fill&training/Adam/zeros_12/shape_as_tensortraining/Adam/zeros_12/Const*
T0*

index_type0*
_output_shapes
:	Аd
ќ
training/Adam/Variable_12VarHandleOp*,
_class"
 loc:@training/Adam/Variable_12*
	container *
shape:	Аd*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_12
Г
:training/Adam/Variable_12/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_12*
_output_shapes
: 
Ґ
 training/Adam/Variable_12/AssignAssignVariableOptraining/Adam/Variable_12training/Adam/zeros_12*
dtype0*,
_class"
 loc:@training/Adam/Variable_12
ґ
-training/Adam/Variable_12/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0*
_output_shapes
:	Аd
c
training/Adam/zeros_13Const*
valueBd*    *
dtype0*
_output_shapes
:d
…
training/Adam/Variable_13VarHandleOp**
shared_nametraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
	container *
shape:d*
dtype0*
_output_shapes
: 
Г
:training/Adam/Variable_13/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_13*
_output_shapes
: 
Ґ
 training/Adam/Variable_13/AssignAssignVariableOptraining/Adam/Variable_13training/Adam/zeros_13*,
_class"
 loc:@training/Adam/Variable_13*
dtype0
±
-training/Adam/Variable_13/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
dtype0*
_output_shapes
:d
w
&training/Adam/zeros_14/shape_as_tensorConst*
valueB"d   d   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_14/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Я
training/Adam/zeros_14Fill&training/Adam/zeros_14/shape_as_tensortraining/Adam/zeros_14/Const*
T0*

index_type0*
_output_shapes

:dd
Ќ
training/Adam/Variable_14VarHandleOp**
shared_nametraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
	container *
shape
:dd*
dtype0*
_output_shapes
: 
Г
:training/Adam/Variable_14/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_14*
_output_shapes
: 
Ґ
 training/Adam/Variable_14/AssignAssignVariableOptraining/Adam/Variable_14training/Adam/zeros_14*
dtype0*,
_class"
 loc:@training/Adam/Variable_14
µ
-training/Adam/Variable_14/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*
_output_shapes

:dd
c
training/Adam/zeros_15Const*
valueBd*    *
dtype0*
_output_shapes
:d
…
training/Adam/Variable_15VarHandleOp*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
	container *
shape:d
Г
:training/Adam/Variable_15/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_15*
_output_shapes
: 
Ґ
 training/Adam/Variable_15/AssignAssignVariableOptraining/Adam/Variable_15training/Adam/zeros_15*,
_class"
 loc:@training/Adam/Variable_15*
dtype0
±
-training/Adam/Variable_15/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
dtype0*
_output_shapes
:d
w
&training/Adam/zeros_16/shape_as_tensorConst*
valueB"d   d   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Я
training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*
T0*

index_type0*
_output_shapes

:dd
Ќ
training/Adam/Variable_16VarHandleOp*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
	container *
shape
:dd
Г
:training/Adam/Variable_16/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_16*
_output_shapes
: 
Ґ
 training/Adam/Variable_16/AssignAssignVariableOptraining/Adam/Variable_16training/Adam/zeros_16*,
_class"
 loc:@training/Adam/Variable_16*
dtype0
µ
-training/Adam/Variable_16/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
dtype0*
_output_shapes

:dd
c
training/Adam/zeros_17Const*
dtype0*
_output_shapes
:d*
valueBd*    
…
training/Adam/Variable_17VarHandleOp*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
	container *
shape:d
Г
:training/Adam/Variable_17/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_17*
_output_shapes
: 
Ґ
 training/Adam/Variable_17/AssignAssignVariableOptraining/Adam/Variable_17training/Adam/zeros_17*,
_class"
 loc:@training/Adam/Variable_17*
dtype0
±
-training/Adam/Variable_17/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
dtype0*
_output_shapes
:d
w
&training/Adam/zeros_18/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"d   d   
a
training/Adam/zeros_18/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Я
training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*
T0*

index_type0*
_output_shapes

:dd
Ќ
training/Adam/Variable_18VarHandleOp*
shape
:dd*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
	container 
Г
:training/Adam/Variable_18/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_18*
_output_shapes
: 
Ґ
 training/Adam/Variable_18/AssignAssignVariableOptraining/Adam/Variable_18training/Adam/zeros_18*
dtype0*,
_class"
 loc:@training/Adam/Variable_18
µ
-training/Adam/Variable_18/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_18*
dtype0*
_output_shapes

:dd*,
_class"
 loc:@training/Adam/Variable_18
c
training/Adam/zeros_19Const*
dtype0*
_output_shapes
:d*
valueBd*    
…
training/Adam/Variable_19VarHandleOp*
shape:d*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_19*,
_class"
 loc:@training/Adam/Variable_19*
	container 
Г
:training/Adam/Variable_19/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_19*
_output_shapes
: 
Ґ
 training/Adam/Variable_19/AssignAssignVariableOptraining/Adam/Variable_19training/Adam/zeros_19*,
_class"
 loc:@training/Adam/Variable_19*
dtype0
±
-training/Adam/Variable_19/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_19*,
_class"
 loc:@training/Adam/Variable_19*
dtype0*
_output_shapes
:d
w
&training/Adam/zeros_20/shape_as_tensorConst*
valueB"d   d   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_20/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Я
training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*
T0*

index_type0*
_output_shapes

:dd
Ќ
training/Adam/Variable_20VarHandleOp*
	container *
shape
:dd*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20
Г
:training/Adam/Variable_20/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_20*
_output_shapes
: 
Ґ
 training/Adam/Variable_20/AssignAssignVariableOptraining/Adam/Variable_20training/Adam/zeros_20*,
_class"
 loc:@training/Adam/Variable_20*
dtype0
µ
-training/Adam/Variable_20/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20*
dtype0*
_output_shapes

:dd
c
training/Adam/zeros_21Const*
valueBd*    *
dtype0*
_output_shapes
:d
…
training/Adam/Variable_21VarHandleOp**
shared_nametraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
	container *
shape:d*
dtype0*
_output_shapes
: 
Г
:training/Adam/Variable_21/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_21*
_output_shapes
: 
Ґ
 training/Adam/Variable_21/AssignAssignVariableOptraining/Adam/Variable_21training/Adam/zeros_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0
±
-training/Adam/Variable_21/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0*
_output_shapes
:d
k
training/Adam/zeros_22Const*
valueBd*    *
dtype0*
_output_shapes

:d
Ќ
training/Adam/Variable_22VarHandleOp*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_22*,
_class"
 loc:@training/Adam/Variable_22*
	container *
shape
:d
Г
:training/Adam/Variable_22/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_22*
_output_shapes
: 
Ґ
 training/Adam/Variable_22/AssignAssignVariableOptraining/Adam/Variable_22training/Adam/zeros_22*,
_class"
 loc:@training/Adam/Variable_22*
dtype0
µ
-training/Adam/Variable_22/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_22*,
_class"
 loc:@training/Adam/Variable_22*
dtype0*
_output_shapes

:d
c
training/Adam/zeros_23Const*
dtype0*
_output_shapes
:*
valueB*    
…
training/Adam/Variable_23VarHandleOp**
shared_nametraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
	container *
shape:*
dtype0*
_output_shapes
: 
Г
:training/Adam/Variable_23/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_23*
_output_shapes
: 
Ґ
 training/Adam/Variable_23/AssignAssignVariableOptraining/Adam/Variable_23training/Adam/zeros_23*,
_class"
 loc:@training/Adam/Variable_23*
dtype0
±
-training/Adam/Variable_23/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_23*
dtype0*
_output_shapes
:*,
_class"
 loc:@training/Adam/Variable_23
p
&training/Adam/zeros_24/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_24/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_24Fill&training/Adam/zeros_24/shape_as_tensortraining/Adam/zeros_24/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_24VarHandleOp*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_24*,
_class"
 loc:@training/Adam/Variable_24*
	container *
shape:
Г
:training/Adam/Variable_24/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_24*
_output_shapes
: 
Ґ
 training/Adam/Variable_24/AssignAssignVariableOptraining/Adam/Variable_24training/Adam/zeros_24*
dtype0*,
_class"
 loc:@training/Adam/Variable_24
±
-training/Adam/Variable_24/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_24*,
_class"
 loc:@training/Adam/Variable_24*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_25/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_25/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_25Fill&training/Adam/zeros_25/shape_as_tensortraining/Adam/zeros_25/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_25VarHandleOp**
shared_nametraining/Adam/Variable_25*,
_class"
 loc:@training/Adam/Variable_25*
	container *
shape:*
dtype0*
_output_shapes
: 
Г
:training/Adam/Variable_25/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_25*
_output_shapes
: 
Ґ
 training/Adam/Variable_25/AssignAssignVariableOptraining/Adam/Variable_25training/Adam/zeros_25*
dtype0*,
_class"
 loc:@training/Adam/Variable_25
±
-training/Adam/Variable_25/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_25*,
_class"
 loc:@training/Adam/Variable_25*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_26/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_26/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ы
training/Adam/zeros_26Fill&training/Adam/zeros_26/shape_as_tensortraining/Adam/zeros_26/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_26VarHandleOp*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_26*,
_class"
 loc:@training/Adam/Variable_26*
	container *
shape:
Г
:training/Adam/Variable_26/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_26*
_output_shapes
: 
Ґ
 training/Adam/Variable_26/AssignAssignVariableOptraining/Adam/Variable_26training/Adam/zeros_26*
dtype0*,
_class"
 loc:@training/Adam/Variable_26
±
-training/Adam/Variable_26/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_26*,
_class"
 loc:@training/Adam/Variable_26*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_27/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_27/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ы
training/Adam/zeros_27Fill&training/Adam/zeros_27/shape_as_tensortraining/Adam/zeros_27/Const*
_output_shapes
:*
T0*

index_type0
…
training/Adam/Variable_27VarHandleOp*,
_class"
 loc:@training/Adam/Variable_27*
	container *
shape:*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_27
Г
:training/Adam/Variable_27/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_27*
_output_shapes
: 
Ґ
 training/Adam/Variable_27/AssignAssignVariableOptraining/Adam/Variable_27training/Adam/zeros_27*,
_class"
 loc:@training/Adam/Variable_27*
dtype0
±
-training/Adam/Variable_27/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_27*,
_class"
 loc:@training/Adam/Variable_27*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_28/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_28/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_28Fill&training/Adam/zeros_28/shape_as_tensortraining/Adam/zeros_28/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_28VarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_28*,
_class"
 loc:@training/Adam/Variable_28
Г
:training/Adam/Variable_28/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_28*
_output_shapes
: 
Ґ
 training/Adam/Variable_28/AssignAssignVariableOptraining/Adam/Variable_28training/Adam/zeros_28*,
_class"
 loc:@training/Adam/Variable_28*
dtype0
±
-training/Adam/Variable_28/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_28*
dtype0*
_output_shapes
:*,
_class"
 loc:@training/Adam/Variable_28
p
&training/Adam/zeros_29/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_29/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_29Fill&training/Adam/zeros_29/shape_as_tensortraining/Adam/zeros_29/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_29VarHandleOp*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_29*,
_class"
 loc:@training/Adam/Variable_29*
	container *
shape:
Г
:training/Adam/Variable_29/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_29*
_output_shapes
: 
Ґ
 training/Adam/Variable_29/AssignAssignVariableOptraining/Adam/Variable_29training/Adam/zeros_29*
dtype0*,
_class"
 loc:@training/Adam/Variable_29
±
-training/Adam/Variable_29/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_29*
dtype0*
_output_shapes
:*,
_class"
 loc:@training/Adam/Variable_29
p
&training/Adam/zeros_30/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_30/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_30Fill&training/Adam/zeros_30/shape_as_tensortraining/Adam/zeros_30/Const*
_output_shapes
:*
T0*

index_type0
…
training/Adam/Variable_30VarHandleOp**
shared_nametraining/Adam/Variable_30*,
_class"
 loc:@training/Adam/Variable_30*
	container *
shape:*
dtype0*
_output_shapes
: 
Г
:training/Adam/Variable_30/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_30*
_output_shapes
: 
Ґ
 training/Adam/Variable_30/AssignAssignVariableOptraining/Adam/Variable_30training/Adam/zeros_30*,
_class"
 loc:@training/Adam/Variable_30*
dtype0
±
-training/Adam/Variable_30/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_30*,
_class"
 loc:@training/Adam/Variable_30*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_31/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_31/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_31Fill&training/Adam/zeros_31/shape_as_tensortraining/Adam/zeros_31/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_31VarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_31*,
_class"
 loc:@training/Adam/Variable_31
Г
:training/Adam/Variable_31/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_31*
_output_shapes
: 
Ґ
 training/Adam/Variable_31/AssignAssignVariableOptraining/Adam/Variable_31training/Adam/zeros_31*,
_class"
 loc:@training/Adam/Variable_31*
dtype0
±
-training/Adam/Variable_31/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_31*,
_class"
 loc:@training/Adam/Variable_31*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_32/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_32/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_32Fill&training/Adam/zeros_32/shape_as_tensortraining/Adam/zeros_32/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_32VarHandleOp*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_32*,
_class"
 loc:@training/Adam/Variable_32*
	container *
shape:
Г
:training/Adam/Variable_32/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_32*
_output_shapes
: 
Ґ
 training/Adam/Variable_32/AssignAssignVariableOptraining/Adam/Variable_32training/Adam/zeros_32*,
_class"
 loc:@training/Adam/Variable_32*
dtype0
±
-training/Adam/Variable_32/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_32*
dtype0*
_output_shapes
:*,
_class"
 loc:@training/Adam/Variable_32
p
&training/Adam/zeros_33/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_33/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_33Fill&training/Adam/zeros_33/shape_as_tensortraining/Adam/zeros_33/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_33VarHandleOp*
shape:*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_33*,
_class"
 loc:@training/Adam/Variable_33*
	container 
Г
:training/Adam/Variable_33/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_33*
_output_shapes
: 
Ґ
 training/Adam/Variable_33/AssignAssignVariableOptraining/Adam/Variable_33training/Adam/zeros_33*,
_class"
 loc:@training/Adam/Variable_33*
dtype0
±
-training/Adam/Variable_33/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_33*,
_class"
 loc:@training/Adam/Variable_33*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_34/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_34/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ы
training/Adam/zeros_34Fill&training/Adam/zeros_34/shape_as_tensortraining/Adam/zeros_34/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_34VarHandleOp*
shape:*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_34*,
_class"
 loc:@training/Adam/Variable_34*
	container 
Г
:training/Adam/Variable_34/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_34*
_output_shapes
: 
Ґ
 training/Adam/Variable_34/AssignAssignVariableOptraining/Adam/Variable_34training/Adam/zeros_34*
dtype0*,
_class"
 loc:@training/Adam/Variable_34
±
-training/Adam/Variable_34/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_34*
dtype0*
_output_shapes
:*,
_class"
 loc:@training/Adam/Variable_34
p
&training/Adam/zeros_35/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_35/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_35Fill&training/Adam/zeros_35/shape_as_tensortraining/Adam/zeros_35/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_35VarHandleOp**
shared_nametraining/Adam/Variable_35*,
_class"
 loc:@training/Adam/Variable_35*
	container *
shape:*
dtype0*
_output_shapes
: 
Г
:training/Adam/Variable_35/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_35*
_output_shapes
: 
Ґ
 training/Adam/Variable_35/AssignAssignVariableOptraining/Adam/Variable_35training/Adam/zeros_35*
dtype0*,
_class"
 loc:@training/Adam/Variable_35
±
-training/Adam/Variable_35/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_35*,
_class"
 loc:@training/Adam/Variable_35*
dtype0*
_output_shapes
:
b
training/Adam/ReadVariableOp_2ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
z
"training/Adam/mul_1/ReadVariableOpReadVariableOptraining/Adam/Variable*
dtype0*
_output_shapes
:	Аd
И
training/Adam/mul_1Multraining/Adam/ReadVariableOp_2"training/Adam/mul_1/ReadVariableOp*
_output_shapes
:	Аd*
T0
b
training/Adam/ReadVariableOp_3ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
r
training/Adam/sub_2Subtraining/Adam/sub_2/xtraining/Adam/ReadVariableOp_3*
T0*
_output_shapes
: 
З
training/Adam/mul_2Multraining/Adam/sub_2,training/Adam/gradients/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	Аd
n
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
T0*
_output_shapes
:	Аd
b
training/Adam/ReadVariableOp_4ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
}
"training/Adam/mul_3/ReadVariableOpReadVariableOptraining/Adam/Variable_12*
dtype0*
_output_shapes
:	Аd
И
training/Adam/mul_3Multraining/Adam/ReadVariableOp_4"training/Adam/mul_3/ReadVariableOp*
T0*
_output_shapes
:	Аd
b
training/Adam/ReadVariableOp_5ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_3/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
r
training/Adam/sub_3Subtraining/Adam/sub_3/xtraining/Adam/ReadVariableOp_5*
_output_shapes
: *
T0
v
training/Adam/SquareSquare,training/Adam/gradients/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	Аd
o
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*
_output_shapes
:	Аd
n
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
_output_shapes
:	Аd*
T0
l
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
_output_shapes
:	Аd*
T0
Z
training/Adam/Const_3Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_4Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Ж
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_4*
T0*
_output_shapes
:	Аd
Р
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_3*
T0*
_output_shapes
:	Аd
e
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*
_output_shapes
:	Аd
Z
training/Adam/add_3/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
q
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*
_output_shapes
:	Аd
v
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
_output_shapes
:	Аd*
T0
o
training/Adam/ReadVariableOp_6ReadVariableOpdense_12/kernel*
dtype0*
_output_shapes
:	Аd
}
training/Adam/sub_4Subtraining/Adam/ReadVariableOp_6training/Adam/truediv_1*
T0*
_output_shapes
:	Аd
l
training/Adam/AssignVariableOpAssignVariableOptraining/Adam/Variabletraining/Adam/add_1*
dtype0
Ч
training/Adam/ReadVariableOp_7ReadVariableOptraining/Adam/Variable^training/Adam/AssignVariableOp*
dtype0*
_output_shapes
:	Аd
q
 training/Adam/AssignVariableOp_1AssignVariableOptraining/Adam/Variable_12training/Adam/add_2*
dtype0
Ь
training/Adam/ReadVariableOp_8ReadVariableOptraining/Adam/Variable_12!^training/Adam/AssignVariableOp_1*
dtype0*
_output_shapes
:	Аd
g
 training/Adam/AssignVariableOp_2AssignVariableOpdense_12/kerneltraining/Adam/sub_4*
dtype0
Т
training/Adam/ReadVariableOp_9ReadVariableOpdense_12/kernel!^training/Adam/AssignVariableOp_2*
dtype0*
_output_shapes
:	Аd
c
training/Adam/ReadVariableOp_10ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
w
"training/Adam/mul_6/ReadVariableOpReadVariableOptraining/Adam/Variable_1*
dtype0*
_output_shapes
:d
Д
training/Adam/mul_6Multraining/Adam/ReadVariableOp_10"training/Adam/mul_6/ReadVariableOp*
_output_shapes
:d*
T0
c
training/Adam/ReadVariableOp_11ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_5/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_5Subtraining/Adam/sub_5/xtraining/Adam/ReadVariableOp_11*
T0*
_output_shapes
: 
Ж
training/Adam/mul_7Multraining/Adam/sub_50training/Adam/gradients/BiasAdd_grad/BiasAddGrad*
_output_shapes
:d*
T0
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_12ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
x
"training/Adam/mul_8/ReadVariableOpReadVariableOptraining/Adam/Variable_13*
dtype0*
_output_shapes
:d
Д
training/Adam/mul_8Multraining/Adam/ReadVariableOp_12"training/Adam/mul_8/ReadVariableOp*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_13ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_6/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_6Subtraining/Adam/sub_6/xtraining/Adam/ReadVariableOp_13*
T0*
_output_shapes
: 
w
training/Adam/Square_1Square0training/Adam/gradients/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:d
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes
:d
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes
:d
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes
:d
Z
training/Adam/Const_5Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_6Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Б
%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_6*
T0*
_output_shapes
:d
Л
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_5*
_output_shapes
:d*
T0
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes
:d
Z
training/Adam/add_6/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
T0*
_output_shapes
:d
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes
:d
i
training/Adam/ReadVariableOp_14ReadVariableOpdense_12/bias*
dtype0*
_output_shapes
:d
y
training/Adam/sub_7Subtraining/Adam/ReadVariableOp_14training/Adam/truediv_2*
T0*
_output_shapes
:d
p
 training/Adam/AssignVariableOp_3AssignVariableOptraining/Adam/Variable_1training/Adam/add_4*
dtype0
Ч
training/Adam/ReadVariableOp_15ReadVariableOptraining/Adam/Variable_1!^training/Adam/AssignVariableOp_3*
dtype0*
_output_shapes
:d
q
 training/Adam/AssignVariableOp_4AssignVariableOptraining/Adam/Variable_13training/Adam/add_5*
dtype0
Ш
training/Adam/ReadVariableOp_16ReadVariableOptraining/Adam/Variable_13!^training/Adam/AssignVariableOp_4*
dtype0*
_output_shapes
:d
e
 training/Adam/AssignVariableOp_5AssignVariableOpdense_12/biastraining/Adam/sub_7*
dtype0
М
training/Adam/ReadVariableOp_17ReadVariableOpdense_12/bias!^training/Adam/AssignVariableOp_5*
dtype0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_18ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
|
#training/Adam/mul_11/ReadVariableOpReadVariableOptraining/Adam/Variable_2*
dtype0*
_output_shapes

:dd
К
training/Adam/mul_11Multraining/Adam/ReadVariableOp_18#training/Adam/mul_11/ReadVariableOp*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_19ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_8/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_8Subtraining/Adam/sub_8/xtraining/Adam/ReadVariableOp_19*
T0*
_output_shapes
: 
Й
training/Adam/mul_12Multraining/Adam/sub_8.training/Adam/gradients/MatMul_1_grad/MatMul_1*
T0*
_output_shapes

:dd
o
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_20ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
}
#training/Adam/mul_13/ReadVariableOpReadVariableOptraining/Adam/Variable_14*
dtype0*
_output_shapes

:dd
К
training/Adam/mul_13Multraining/Adam/ReadVariableOp_20#training/Adam/mul_13/ReadVariableOp*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_21ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_9/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_9Subtraining/Adam/sub_9/xtraining/Adam/ReadVariableOp_21*
T0*
_output_shapes
: 
y
training/Adam/Square_2Square.training/Adam/gradients/MatMul_1_grad/MatMul_1*
T0*
_output_shapes

:dd
q
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*
_output_shapes

:dd
o
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*
_output_shapes

:dd
l
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
T0*
_output_shapes

:dd
Z
training/Adam/Const_7Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_8Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Е
%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_8*
T0*
_output_shapes

:dd
П
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_7*
_output_shapes

:dd*
T0
d
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*
_output_shapes

:dd
Z
training/Adam/add_9/yConst*
dtype0*
_output_shapes
: *
valueB
 *Хњ÷3
p
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
_output_shapes

:dd*
T0
v
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*
_output_shapes

:dd
o
training/Adam/ReadVariableOp_22ReadVariableOpdense_13/kernel*
dtype0*
_output_shapes

:dd
~
training/Adam/sub_10Subtraining/Adam/ReadVariableOp_22training/Adam/truediv_3*
T0*
_output_shapes

:dd
p
 training/Adam/AssignVariableOp_6AssignVariableOptraining/Adam/Variable_2training/Adam/add_7*
dtype0
Ы
training/Adam/ReadVariableOp_23ReadVariableOptraining/Adam/Variable_2!^training/Adam/AssignVariableOp_6*
dtype0*
_output_shapes

:dd
q
 training/Adam/AssignVariableOp_7AssignVariableOptraining/Adam/Variable_14training/Adam/add_8*
dtype0
Ь
training/Adam/ReadVariableOp_24ReadVariableOptraining/Adam/Variable_14!^training/Adam/AssignVariableOp_7*
dtype0*
_output_shapes

:dd
h
 training/Adam/AssignVariableOp_8AssignVariableOpdense_13/kerneltraining/Adam/sub_10*
dtype0
Т
training/Adam/ReadVariableOp_25ReadVariableOpdense_13/kernel!^training/Adam/AssignVariableOp_8*
dtype0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_26ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
x
#training/Adam/mul_16/ReadVariableOpReadVariableOptraining/Adam/Variable_3*
dtype0*
_output_shapes
:d
Ж
training/Adam/mul_16Multraining/Adam/ReadVariableOp_26#training/Adam/mul_16/ReadVariableOp*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_27ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_11/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_11Subtraining/Adam/sub_11/xtraining/Adam/ReadVariableOp_27*
T0*
_output_shapes
: 
К
training/Adam/mul_17Multraining/Adam/sub_112training/Adam/gradients/BiasAdd_1_grad/BiasAddGrad*
T0*
_output_shapes
:d
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_28ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_18/ReadVariableOpReadVariableOptraining/Adam/Variable_15*
dtype0*
_output_shapes
:d
Ж
training/Adam/mul_18Multraining/Adam/ReadVariableOp_28#training/Adam/mul_18/ReadVariableOp*
_output_shapes
:d*
T0
c
training/Adam/ReadVariableOp_29ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_12/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
u
training/Adam/sub_12Subtraining/Adam/sub_12/xtraining/Adam/ReadVariableOp_29*
T0*
_output_shapes
: 
y
training/Adam/Square_3Square2training/Adam/gradients/BiasAdd_1_grad/BiasAddGrad*
_output_shapes
:d*
T0
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes
:d
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes
:d
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes
:d
Z
training/Adam/Const_9Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_10Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Г
%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_10*
T0*
_output_shapes
:d
Л
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_9*
T0*
_output_shapes
:d
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
_output_shapes
:d*
T0
[
training/Adam/add_12/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
n
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
T0*
_output_shapes
:d
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
T0*
_output_shapes
:d
i
training/Adam/ReadVariableOp_30ReadVariableOpdense_13/bias*
dtype0*
_output_shapes
:d
z
training/Adam/sub_13Subtraining/Adam/ReadVariableOp_30training/Adam/truediv_4*
T0*
_output_shapes
:d
q
 training/Adam/AssignVariableOp_9AssignVariableOptraining/Adam/Variable_3training/Adam/add_10*
dtype0
Ч
training/Adam/ReadVariableOp_31ReadVariableOptraining/Adam/Variable_3!^training/Adam/AssignVariableOp_9*
dtype0*
_output_shapes
:d
s
!training/Adam/AssignVariableOp_10AssignVariableOptraining/Adam/Variable_15training/Adam/add_11*
dtype0
Щ
training/Adam/ReadVariableOp_32ReadVariableOptraining/Adam/Variable_15"^training/Adam/AssignVariableOp_10*
dtype0*
_output_shapes
:d
g
!training/Adam/AssignVariableOp_11AssignVariableOpdense_13/biastraining/Adam/sub_13*
dtype0
Н
training/Adam/ReadVariableOp_33ReadVariableOpdense_13/bias"^training/Adam/AssignVariableOp_11*
dtype0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_34ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
|
#training/Adam/mul_21/ReadVariableOpReadVariableOptraining/Adam/Variable_4*
dtype0*
_output_shapes

:dd
К
training/Adam/mul_21Multraining/Adam/ReadVariableOp_34#training/Adam/mul_21/ReadVariableOp*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_35ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_14/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_14Subtraining/Adam/sub_14/xtraining/Adam/ReadVariableOp_35*
T0*
_output_shapes
: 
К
training/Adam/mul_22Multraining/Adam/sub_14.training/Adam/gradients/MatMul_2_grad/MatMul_1*
T0*
_output_shapes

:dd
p
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_36ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
}
#training/Adam/mul_23/ReadVariableOpReadVariableOptraining/Adam/Variable_16*
dtype0*
_output_shapes

:dd
К
training/Adam/mul_23Multraining/Adam/ReadVariableOp_36#training/Adam/mul_23/ReadVariableOp*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_37ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_15/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_15Subtraining/Adam/sub_15/xtraining/Adam/ReadVariableOp_37*
T0*
_output_shapes
: 
y
training/Adam/Square_4Square.training/Adam/gradients/MatMul_2_grad/MatMul_1*
T0*
_output_shapes

:dd
r
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*
_output_shapes

:dd
p
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
T0*
_output_shapes

:dd
m
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
_output_shapes

:dd*
T0
[
training/Adam/Const_11Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_12Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
З
%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_12*
T0*
_output_shapes

:dd
Р
training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_11*
_output_shapes

:dd*
T0
d
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
_output_shapes

:dd*
T0
[
training/Adam/add_15/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
r
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
T0*
_output_shapes

:dd
w
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0*
_output_shapes

:dd
o
training/Adam/ReadVariableOp_38ReadVariableOpdense_14/kernel*
dtype0*
_output_shapes

:dd
~
training/Adam/sub_16Subtraining/Adam/ReadVariableOp_38training/Adam/truediv_5*
T0*
_output_shapes

:dd
r
!training/Adam/AssignVariableOp_12AssignVariableOptraining/Adam/Variable_4training/Adam/add_13*
dtype0
Ь
training/Adam/ReadVariableOp_39ReadVariableOptraining/Adam/Variable_4"^training/Adam/AssignVariableOp_12*
dtype0*
_output_shapes

:dd
s
!training/Adam/AssignVariableOp_13AssignVariableOptraining/Adam/Variable_16training/Adam/add_14*
dtype0
Э
training/Adam/ReadVariableOp_40ReadVariableOptraining/Adam/Variable_16"^training/Adam/AssignVariableOp_13*
dtype0*
_output_shapes

:dd
i
!training/Adam/AssignVariableOp_14AssignVariableOpdense_14/kerneltraining/Adam/sub_16*
dtype0
У
training/Adam/ReadVariableOp_41ReadVariableOpdense_14/kernel"^training/Adam/AssignVariableOp_14*
dtype0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_42ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
x
#training/Adam/mul_26/ReadVariableOpReadVariableOptraining/Adam/Variable_5*
dtype0*
_output_shapes
:d
Ж
training/Adam/mul_26Multraining/Adam/ReadVariableOp_42#training/Adam/mul_26/ReadVariableOp*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_43ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_17/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_17Subtraining/Adam/sub_17/xtraining/Adam/ReadVariableOp_43*
T0*
_output_shapes
: 
К
training/Adam/mul_27Multraining/Adam/sub_172training/Adam/gradients/BiasAdd_2_grad/BiasAddGrad*
_output_shapes
:d*
T0
l
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_44ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_28/ReadVariableOpReadVariableOptraining/Adam/Variable_17*
dtype0*
_output_shapes
:d
Ж
training/Adam/mul_28Multraining/Adam/ReadVariableOp_44#training/Adam/mul_28/ReadVariableOp*
_output_shapes
:d*
T0
c
training/Adam/ReadVariableOp_45ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_18/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_18Subtraining/Adam/sub_18/xtraining/Adam/ReadVariableOp_45*
T0*
_output_shapes
: 
y
training/Adam/Square_5Square2training/Adam/gradients/BiasAdd_2_grad/BiasAddGrad*
T0*
_output_shapes
:d
n
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes
:d
l
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes
:d
i
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes
:d
[
training/Adam/Const_13Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_14Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Г
%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_14*
_output_shapes
:d*
T0
М
training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_13*
_output_shapes
:d*
T0
`
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes
:d
[
training/Adam/add_18/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
n
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
_output_shapes
:d*
T0
s
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
_output_shapes
:d*
T0
i
training/Adam/ReadVariableOp_46ReadVariableOpdense_14/bias*
dtype0*
_output_shapes
:d
z
training/Adam/sub_19Subtraining/Adam/ReadVariableOp_46training/Adam/truediv_6*
T0*
_output_shapes
:d
r
!training/Adam/AssignVariableOp_15AssignVariableOptraining/Adam/Variable_5training/Adam/add_16*
dtype0
Ш
training/Adam/ReadVariableOp_47ReadVariableOptraining/Adam/Variable_5"^training/Adam/AssignVariableOp_15*
dtype0*
_output_shapes
:d
s
!training/Adam/AssignVariableOp_16AssignVariableOptraining/Adam/Variable_17training/Adam/add_17*
dtype0
Щ
training/Adam/ReadVariableOp_48ReadVariableOptraining/Adam/Variable_17"^training/Adam/AssignVariableOp_16*
dtype0*
_output_shapes
:d
g
!training/Adam/AssignVariableOp_17AssignVariableOpdense_14/biastraining/Adam/sub_19*
dtype0
Н
training/Adam/ReadVariableOp_49ReadVariableOpdense_14/bias"^training/Adam/AssignVariableOp_17*
dtype0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_50ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
|
#training/Adam/mul_31/ReadVariableOpReadVariableOptraining/Adam/Variable_6*
dtype0*
_output_shapes

:dd
К
training/Adam/mul_31Multraining/Adam/ReadVariableOp_50#training/Adam/mul_31/ReadVariableOp*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_51ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_20/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_20Subtraining/Adam/sub_20/xtraining/Adam/ReadVariableOp_51*
T0*
_output_shapes
: 
К
training/Adam/mul_32Multraining/Adam/sub_20.training/Adam/gradients/MatMul_3_grad/MatMul_1*
T0*
_output_shapes

:dd
p
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_52ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
}
#training/Adam/mul_33/ReadVariableOpReadVariableOptraining/Adam/Variable_18*
dtype0*
_output_shapes

:dd
К
training/Adam/mul_33Multraining/Adam/ReadVariableOp_52#training/Adam/mul_33/ReadVariableOp*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_53ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_21/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_21Subtraining/Adam/sub_21/xtraining/Adam/ReadVariableOp_53*
T0*
_output_shapes
: 
y
training/Adam/Square_6Square.training/Adam/gradients/MatMul_3_grad/MatMul_1*
_output_shapes

:dd*
T0
r
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*
_output_shapes

:dd
p
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
_output_shapes

:dd*
T0
m
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*
_output_shapes

:dd
[
training/Adam/Const_15Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_16Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
З
%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_16*
T0*
_output_shapes

:dd
Р
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_15*
_output_shapes

:dd*
T0
d
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
_output_shapes

:dd*
T0
[
training/Adam/add_21/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
r
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
T0*
_output_shapes

:dd
w
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*
_output_shapes

:dd
o
training/Adam/ReadVariableOp_54ReadVariableOpdense_15/kernel*
dtype0*
_output_shapes

:dd
~
training/Adam/sub_22Subtraining/Adam/ReadVariableOp_54training/Adam/truediv_7*
T0*
_output_shapes

:dd
r
!training/Adam/AssignVariableOp_18AssignVariableOptraining/Adam/Variable_6training/Adam/add_19*
dtype0
Ь
training/Adam/ReadVariableOp_55ReadVariableOptraining/Adam/Variable_6"^training/Adam/AssignVariableOp_18*
dtype0*
_output_shapes

:dd
s
!training/Adam/AssignVariableOp_19AssignVariableOptraining/Adam/Variable_18training/Adam/add_20*
dtype0
Э
training/Adam/ReadVariableOp_56ReadVariableOptraining/Adam/Variable_18"^training/Adam/AssignVariableOp_19*
dtype0*
_output_shapes

:dd
i
!training/Adam/AssignVariableOp_20AssignVariableOpdense_15/kerneltraining/Adam/sub_22*
dtype0
У
training/Adam/ReadVariableOp_57ReadVariableOpdense_15/kernel"^training/Adam/AssignVariableOp_20*
dtype0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_58ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
x
#training/Adam/mul_36/ReadVariableOpReadVariableOptraining/Adam/Variable_7*
dtype0*
_output_shapes
:d
Ж
training/Adam/mul_36Multraining/Adam/ReadVariableOp_58#training/Adam/mul_36/ReadVariableOp*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_59ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_23/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_23Subtraining/Adam/sub_23/xtraining/Adam/ReadVariableOp_59*
T0*
_output_shapes
: 
К
training/Adam/mul_37Multraining/Adam/sub_232training/Adam/gradients/BiasAdd_3_grad/BiasAddGrad*
T0*
_output_shapes
:d
l
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
_output_shapes
:d*
T0
c
training/Adam/ReadVariableOp_60ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_38/ReadVariableOpReadVariableOptraining/Adam/Variable_19*
dtype0*
_output_shapes
:d
Ж
training/Adam/mul_38Multraining/Adam/ReadVariableOp_60#training/Adam/mul_38/ReadVariableOp*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_61ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_24/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_24Subtraining/Adam/sub_24/xtraining/Adam/ReadVariableOp_61*
_output_shapes
: *
T0
y
training/Adam/Square_7Square2training/Adam/gradients/BiasAdd_3_grad/BiasAddGrad*
T0*
_output_shapes
:d
n
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
_output_shapes
:d*
T0
l
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
_output_shapes
:d*
T0
i
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
_output_shapes
:d*
T0
[
training/Adam/Const_17Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_18Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Г
%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_18*
T0*
_output_shapes
:d
М
training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_17*
_output_shapes
:d*
T0
`
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
T0*
_output_shapes
:d
[
training/Adam/add_24/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
n
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
T0*
_output_shapes
:d
s
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
T0*
_output_shapes
:d
i
training/Adam/ReadVariableOp_62ReadVariableOpdense_15/bias*
dtype0*
_output_shapes
:d
z
training/Adam/sub_25Subtraining/Adam/ReadVariableOp_62training/Adam/truediv_8*
T0*
_output_shapes
:d
r
!training/Adam/AssignVariableOp_21AssignVariableOptraining/Adam/Variable_7training/Adam/add_22*
dtype0
Ш
training/Adam/ReadVariableOp_63ReadVariableOptraining/Adam/Variable_7"^training/Adam/AssignVariableOp_21*
dtype0*
_output_shapes
:d
s
!training/Adam/AssignVariableOp_22AssignVariableOptraining/Adam/Variable_19training/Adam/add_23*
dtype0
Щ
training/Adam/ReadVariableOp_64ReadVariableOptraining/Adam/Variable_19"^training/Adam/AssignVariableOp_22*
dtype0*
_output_shapes
:d
g
!training/Adam/AssignVariableOp_23AssignVariableOpdense_15/biastraining/Adam/sub_25*
dtype0
Н
training/Adam/ReadVariableOp_65ReadVariableOpdense_15/bias"^training/Adam/AssignVariableOp_23*
dtype0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_66ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
|
#training/Adam/mul_41/ReadVariableOpReadVariableOptraining/Adam/Variable_8*
dtype0*
_output_shapes

:dd
К
training/Adam/mul_41Multraining/Adam/ReadVariableOp_66#training/Adam/mul_41/ReadVariableOp*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_67ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_26/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
u
training/Adam/sub_26Subtraining/Adam/sub_26/xtraining/Adam/ReadVariableOp_67*
T0*
_output_shapes
: 
К
training/Adam/mul_42Multraining/Adam/sub_26.training/Adam/gradients/MatMul_4_grad/MatMul_1*
_output_shapes

:dd*
T0
p
training/Adam/add_25Addtraining/Adam/mul_41training/Adam/mul_42*
_output_shapes

:dd*
T0
c
training/Adam/ReadVariableOp_68ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
}
#training/Adam/mul_43/ReadVariableOpReadVariableOptraining/Adam/Variable_20*
dtype0*
_output_shapes

:dd
К
training/Adam/mul_43Multraining/Adam/ReadVariableOp_68#training/Adam/mul_43/ReadVariableOp*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_69ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_27/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_27Subtraining/Adam/sub_27/xtraining/Adam/ReadVariableOp_69*
_output_shapes
: *
T0
y
training/Adam/Square_8Square.training/Adam/gradients/MatMul_4_grad/MatMul_1*
_output_shapes

:dd*
T0
r
training/Adam/mul_44Multraining/Adam/sub_27training/Adam/Square_8*
T0*
_output_shapes

:dd
p
training/Adam/add_26Addtraining/Adam/mul_43training/Adam/mul_44*
T0*
_output_shapes

:dd
m
training/Adam/mul_45Multraining/Adam/multraining/Adam/add_25*
_output_shapes

:dd*
T0
[
training/Adam/Const_19Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_20Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
З
%training/Adam/clip_by_value_9/MinimumMinimumtraining/Adam/add_26training/Adam/Const_20*
T0*
_output_shapes

:dd
Р
training/Adam/clip_by_value_9Maximum%training/Adam/clip_by_value_9/Minimumtraining/Adam/Const_19*
_output_shapes

:dd*
T0
d
training/Adam/Sqrt_9Sqrttraining/Adam/clip_by_value_9*
T0*
_output_shapes

:dd
[
training/Adam/add_27/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
r
training/Adam/add_27Addtraining/Adam/Sqrt_9training/Adam/add_27/y*
T0*
_output_shapes

:dd
w
training/Adam/truediv_9RealDivtraining/Adam/mul_45training/Adam/add_27*
_output_shapes

:dd*
T0
o
training/Adam/ReadVariableOp_70ReadVariableOpdense_16/kernel*
dtype0*
_output_shapes

:dd
~
training/Adam/sub_28Subtraining/Adam/ReadVariableOp_70training/Adam/truediv_9*
_output_shapes

:dd*
T0
r
!training/Adam/AssignVariableOp_24AssignVariableOptraining/Adam/Variable_8training/Adam/add_25*
dtype0
Ь
training/Adam/ReadVariableOp_71ReadVariableOptraining/Adam/Variable_8"^training/Adam/AssignVariableOp_24*
dtype0*
_output_shapes

:dd
s
!training/Adam/AssignVariableOp_25AssignVariableOptraining/Adam/Variable_20training/Adam/add_26*
dtype0
Э
training/Adam/ReadVariableOp_72ReadVariableOptraining/Adam/Variable_20"^training/Adam/AssignVariableOp_25*
dtype0*
_output_shapes

:dd
i
!training/Adam/AssignVariableOp_26AssignVariableOpdense_16/kerneltraining/Adam/sub_28*
dtype0
У
training/Adam/ReadVariableOp_73ReadVariableOpdense_16/kernel"^training/Adam/AssignVariableOp_26*
dtype0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_74ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
x
#training/Adam/mul_46/ReadVariableOpReadVariableOptraining/Adam/Variable_9*
dtype0*
_output_shapes
:d
Ж
training/Adam/mul_46Multraining/Adam/ReadVariableOp_74#training/Adam/mul_46/ReadVariableOp*
_output_shapes
:d*
T0
c
training/Adam/ReadVariableOp_75ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_29/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_29Subtraining/Adam/sub_29/xtraining/Adam/ReadVariableOp_75*
T0*
_output_shapes
: 
К
training/Adam/mul_47Multraining/Adam/sub_292training/Adam/gradients/BiasAdd_4_grad/BiasAddGrad*
T0*
_output_shapes
:d
l
training/Adam/add_28Addtraining/Adam/mul_46training/Adam/mul_47*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_76ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_48/ReadVariableOpReadVariableOptraining/Adam/Variable_21*
dtype0*
_output_shapes
:d
Ж
training/Adam/mul_48Multraining/Adam/ReadVariableOp_76#training/Adam/mul_48/ReadVariableOp*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_77ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_30/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_30Subtraining/Adam/sub_30/xtraining/Adam/ReadVariableOp_77*
T0*
_output_shapes
: 
y
training/Adam/Square_9Square2training/Adam/gradients/BiasAdd_4_grad/BiasAddGrad*
T0*
_output_shapes
:d
n
training/Adam/mul_49Multraining/Adam/sub_30training/Adam/Square_9*
T0*
_output_shapes
:d
l
training/Adam/add_29Addtraining/Adam/mul_48training/Adam/mul_49*
T0*
_output_shapes
:d
i
training/Adam/mul_50Multraining/Adam/multraining/Adam/add_28*
_output_shapes
:d*
T0
[
training/Adam/Const_21Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_22Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Д
&training/Adam/clip_by_value_10/MinimumMinimumtraining/Adam/add_29training/Adam/Const_22*
T0*
_output_shapes
:d
О
training/Adam/clip_by_value_10Maximum&training/Adam/clip_by_value_10/Minimumtraining/Adam/Const_21*
T0*
_output_shapes
:d
b
training/Adam/Sqrt_10Sqrttraining/Adam/clip_by_value_10*
_output_shapes
:d*
T0
[
training/Adam/add_30/yConst*
dtype0*
_output_shapes
: *
valueB
 *Хњ÷3
o
training/Adam/add_30Addtraining/Adam/Sqrt_10training/Adam/add_30/y*
T0*
_output_shapes
:d
t
training/Adam/truediv_10RealDivtraining/Adam/mul_50training/Adam/add_30*
T0*
_output_shapes
:d
i
training/Adam/ReadVariableOp_78ReadVariableOpdense_16/bias*
dtype0*
_output_shapes
:d
{
training/Adam/sub_31Subtraining/Adam/ReadVariableOp_78training/Adam/truediv_10*
_output_shapes
:d*
T0
r
!training/Adam/AssignVariableOp_27AssignVariableOptraining/Adam/Variable_9training/Adam/add_28*
dtype0
Ш
training/Adam/ReadVariableOp_79ReadVariableOptraining/Adam/Variable_9"^training/Adam/AssignVariableOp_27*
dtype0*
_output_shapes
:d
s
!training/Adam/AssignVariableOp_28AssignVariableOptraining/Adam/Variable_21training/Adam/add_29*
dtype0
Щ
training/Adam/ReadVariableOp_80ReadVariableOptraining/Adam/Variable_21"^training/Adam/AssignVariableOp_28*
dtype0*
_output_shapes
:d
g
!training/Adam/AssignVariableOp_29AssignVariableOpdense_16/biastraining/Adam/sub_31*
dtype0
Н
training/Adam/ReadVariableOp_81ReadVariableOpdense_16/bias"^training/Adam/AssignVariableOp_29*
dtype0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_82ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
}
#training/Adam/mul_51/ReadVariableOpReadVariableOptraining/Adam/Variable_10*
dtype0*
_output_shapes

:d
К
training/Adam/mul_51Multraining/Adam/ReadVariableOp_82#training/Adam/mul_51/ReadVariableOp*
T0*
_output_shapes

:d
c
training/Adam/ReadVariableOp_83ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_32/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_32Subtraining/Adam/sub_32/xtraining/Adam/ReadVariableOp_83*
T0*
_output_shapes
: 
К
training/Adam/mul_52Multraining/Adam/sub_32.training/Adam/gradients/MatMul_5_grad/MatMul_1*
T0*
_output_shapes

:d
p
training/Adam/add_31Addtraining/Adam/mul_51training/Adam/mul_52*
T0*
_output_shapes

:d
c
training/Adam/ReadVariableOp_84ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
}
#training/Adam/mul_53/ReadVariableOpReadVariableOptraining/Adam/Variable_22*
dtype0*
_output_shapes

:d
К
training/Adam/mul_53Multraining/Adam/ReadVariableOp_84#training/Adam/mul_53/ReadVariableOp*
_output_shapes

:d*
T0
c
training/Adam/ReadVariableOp_85ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_33/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_33Subtraining/Adam/sub_33/xtraining/Adam/ReadVariableOp_85*
T0*
_output_shapes
: 
z
training/Adam/Square_10Square.training/Adam/gradients/MatMul_5_grad/MatMul_1*
T0*
_output_shapes

:d
s
training/Adam/mul_54Multraining/Adam/sub_33training/Adam/Square_10*
T0*
_output_shapes

:d
p
training/Adam/add_32Addtraining/Adam/mul_53training/Adam/mul_54*
T0*
_output_shapes

:d
m
training/Adam/mul_55Multraining/Adam/multraining/Adam/add_31*
T0*
_output_shapes

:d
[
training/Adam/Const_23Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_24Const*
dtype0*
_output_shapes
: *
valueB
 *  А
И
&training/Adam/clip_by_value_11/MinimumMinimumtraining/Adam/add_32training/Adam/Const_24*
T0*
_output_shapes

:d
Т
training/Adam/clip_by_value_11Maximum&training/Adam/clip_by_value_11/Minimumtraining/Adam/Const_23*
T0*
_output_shapes

:d
f
training/Adam/Sqrt_11Sqrttraining/Adam/clip_by_value_11*
T0*
_output_shapes

:d
[
training/Adam/add_33/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
s
training/Adam/add_33Addtraining/Adam/Sqrt_11training/Adam/add_33/y*
T0*
_output_shapes

:d
x
training/Adam/truediv_11RealDivtraining/Adam/mul_55training/Adam/add_33*
T0*
_output_shapes

:d
o
training/Adam/ReadVariableOp_86ReadVariableOpdense_17/kernel*
dtype0*
_output_shapes

:d

training/Adam/sub_34Subtraining/Adam/ReadVariableOp_86training/Adam/truediv_11*
T0*
_output_shapes

:d
s
!training/Adam/AssignVariableOp_30AssignVariableOptraining/Adam/Variable_10training/Adam/add_31*
dtype0
Э
training/Adam/ReadVariableOp_87ReadVariableOptraining/Adam/Variable_10"^training/Adam/AssignVariableOp_30*
dtype0*
_output_shapes

:d
s
!training/Adam/AssignVariableOp_31AssignVariableOptraining/Adam/Variable_22training/Adam/add_32*
dtype0
Э
training/Adam/ReadVariableOp_88ReadVariableOptraining/Adam/Variable_22"^training/Adam/AssignVariableOp_31*
dtype0*
_output_shapes

:d
i
!training/Adam/AssignVariableOp_32AssignVariableOpdense_17/kerneltraining/Adam/sub_34*
dtype0
У
training/Adam/ReadVariableOp_89ReadVariableOpdense_17/kernel"^training/Adam/AssignVariableOp_32*
dtype0*
_output_shapes

:d
c
training/Adam/ReadVariableOp_90ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_56/ReadVariableOpReadVariableOptraining/Adam/Variable_11*
dtype0*
_output_shapes
:
Ж
training/Adam/mul_56Multraining/Adam/ReadVariableOp_90#training/Adam/mul_56/ReadVariableOp*
T0*
_output_shapes
:
c
training/Adam/ReadVariableOp_91ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_35/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_35Subtraining/Adam/sub_35/xtraining/Adam/ReadVariableOp_91*
T0*
_output_shapes
: 
К
training/Adam/mul_57Multraining/Adam/sub_352training/Adam/gradients/BiasAdd_5_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/add_34Addtraining/Adam/mul_56training/Adam/mul_57*
_output_shapes
:*
T0
c
training/Adam/ReadVariableOp_92ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_58/ReadVariableOpReadVariableOptraining/Adam/Variable_23*
dtype0*
_output_shapes
:
Ж
training/Adam/mul_58Multraining/Adam/ReadVariableOp_92#training/Adam/mul_58/ReadVariableOp*
T0*
_output_shapes
:
c
training/Adam/ReadVariableOp_93ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_36/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_36Subtraining/Adam/sub_36/xtraining/Adam/ReadVariableOp_93*
_output_shapes
: *
T0
z
training/Adam/Square_11Square2training/Adam/gradients/BiasAdd_5_grad/BiasAddGrad*
T0*
_output_shapes
:
o
training/Adam/mul_59Multraining/Adam/sub_36training/Adam/Square_11*
T0*
_output_shapes
:
l
training/Adam/add_35Addtraining/Adam/mul_58training/Adam/mul_59*
T0*
_output_shapes
:
i
training/Adam/mul_60Multraining/Adam/multraining/Adam/add_34*
T0*
_output_shapes
:
[
training/Adam/Const_25Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_26Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Д
&training/Adam/clip_by_value_12/MinimumMinimumtraining/Adam/add_35training/Adam/Const_26*
T0*
_output_shapes
:
О
training/Adam/clip_by_value_12Maximum&training/Adam/clip_by_value_12/Minimumtraining/Adam/Const_25*
T0*
_output_shapes
:
b
training/Adam/Sqrt_12Sqrttraining/Adam/clip_by_value_12*
_output_shapes
:*
T0
[
training/Adam/add_36/yConst*
dtype0*
_output_shapes
: *
valueB
 *Хњ÷3
o
training/Adam/add_36Addtraining/Adam/Sqrt_12training/Adam/add_36/y*
T0*
_output_shapes
:
t
training/Adam/truediv_12RealDivtraining/Adam/mul_60training/Adam/add_36*
T0*
_output_shapes
:
i
training/Adam/ReadVariableOp_94ReadVariableOpdense_17/bias*
dtype0*
_output_shapes
:
{
training/Adam/sub_37Subtraining/Adam/ReadVariableOp_94training/Adam/truediv_12*
_output_shapes
:*
T0
s
!training/Adam/AssignVariableOp_33AssignVariableOptraining/Adam/Variable_11training/Adam/add_34*
dtype0
Щ
training/Adam/ReadVariableOp_95ReadVariableOptraining/Adam/Variable_11"^training/Adam/AssignVariableOp_33*
dtype0*
_output_shapes
:
s
!training/Adam/AssignVariableOp_34AssignVariableOptraining/Adam/Variable_23training/Adam/add_35*
dtype0
Щ
training/Adam/ReadVariableOp_96ReadVariableOptraining/Adam/Variable_23"^training/Adam/AssignVariableOp_34*
dtype0*
_output_shapes
:
g
!training/Adam/AssignVariableOp_35AssignVariableOpdense_17/biastraining/Adam/sub_37*
dtype0
Н
training/Adam/ReadVariableOp_97ReadVariableOpdense_17/bias"^training/Adam/AssignVariableOp_35*
dtype0*
_output_shapes
:
µ

training/group_depsNoOp	^loss/mul^metrics/acc/Mean^metrics/f1_score/Mean^training/Adam/ReadVariableOp ^training/Adam/ReadVariableOp_15 ^training/Adam/ReadVariableOp_16 ^training/Adam/ReadVariableOp_17 ^training/Adam/ReadVariableOp_23 ^training/Adam/ReadVariableOp_24 ^training/Adam/ReadVariableOp_25 ^training/Adam/ReadVariableOp_31 ^training/Adam/ReadVariableOp_32 ^training/Adam/ReadVariableOp_33 ^training/Adam/ReadVariableOp_39 ^training/Adam/ReadVariableOp_40 ^training/Adam/ReadVariableOp_41 ^training/Adam/ReadVariableOp_47 ^training/Adam/ReadVariableOp_48 ^training/Adam/ReadVariableOp_49 ^training/Adam/ReadVariableOp_55 ^training/Adam/ReadVariableOp_56 ^training/Adam/ReadVariableOp_57 ^training/Adam/ReadVariableOp_63 ^training/Adam/ReadVariableOp_64 ^training/Adam/ReadVariableOp_65^training/Adam/ReadVariableOp_7 ^training/Adam/ReadVariableOp_71 ^training/Adam/ReadVariableOp_72 ^training/Adam/ReadVariableOp_73 ^training/Adam/ReadVariableOp_79^training/Adam/ReadVariableOp_8 ^training/Adam/ReadVariableOp_80 ^training/Adam/ReadVariableOp_81 ^training/Adam/ReadVariableOp_87 ^training/Adam/ReadVariableOp_88 ^training/Adam/ReadVariableOp_89^training/Adam/ReadVariableOp_9 ^training/Adam/ReadVariableOp_95 ^training/Adam/ReadVariableOp_96 ^training/Adam/ReadVariableOp_97
H

group_depsNoOp	^loss/mul^metrics/acc/Mean^metrics/f1_score/Mean
Z
VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_4*
_output_shapes
: 
]
VarIsInitializedOp_1VarIsInitializedOptraining/Adam/Variable_35*
_output_shapes
: 
\
VarIsInitializedOp_2VarIsInitializedOptraining/Adam/Variable_3*
_output_shapes
: 
\
VarIsInitializedOp_3VarIsInitializedOptraining/Adam/Variable_9*
_output_shapes
: 
]
VarIsInitializedOp_4VarIsInitializedOptraining/Adam/Variable_16*
_output_shapes
: 
S
VarIsInitializedOp_5VarIsInitializedOpdense_16/kernel*
_output_shapes
: 
Z
VarIsInitializedOp_6VarIsInitializedOptraining/Adam/Variable*
_output_shapes
: 
]
VarIsInitializedOp_7VarIsInitializedOptraining/Adam/Variable_30*
_output_shapes
: 
]
VarIsInitializedOp_8VarIsInitializedOptraining/Adam/Variable_10*
_output_shapes
: 
Q
VarIsInitializedOp_9VarIsInitializedOpdense_17/bias*
_output_shapes
: 
R
VarIsInitializedOp_10VarIsInitializedOpdense_16/bias*
_output_shapes
: 
^
VarIsInitializedOp_11VarIsInitializedOptraining/Adam/Variable_24*
_output_shapes
: 
T
VarIsInitializedOp_12VarIsInitializedOpAdam/iterations*
_output_shapes
: 
L
VarIsInitializedOp_13VarIsInitializedOpAdam/lr*
_output_shapes
: 
^
VarIsInitializedOp_14VarIsInitializedOptraining/Adam/Variable_19*
_output_shapes
: 
P
VarIsInitializedOp_15VarIsInitializedOpAdam/beta_1*
_output_shapes
: 
P
VarIsInitializedOp_16VarIsInitializedOpAdam/beta_2*
_output_shapes
: 
^
VarIsInitializedOp_17VarIsInitializedOptraining/Adam/Variable_13*
_output_shapes
: 
^
VarIsInitializedOp_18VarIsInitializedOptraining/Adam/Variable_29*
_output_shapes
: 
O
VarIsInitializedOp_19VarIsInitializedOp
Adam/decay*
_output_shapes
: 
]
VarIsInitializedOp_20VarIsInitializedOptraining/Adam/Variable_1*
_output_shapes
: 
^
VarIsInitializedOp_21VarIsInitializedOptraining/Adam/Variable_25*
_output_shapes
: 
^
VarIsInitializedOp_22VarIsInitializedOptraining/Adam/Variable_34*
_output_shapes
: 
^
VarIsInitializedOp_23VarIsInitializedOptraining/Adam/Variable_14*
_output_shapes
: 
^
VarIsInitializedOp_24VarIsInitializedOptraining/Adam/Variable_21*
_output_shapes
: 
^
VarIsInitializedOp_25VarIsInitializedOptraining/Adam/Variable_12*
_output_shapes
: 
T
VarIsInitializedOp_26VarIsInitializedOpdense_13/kernel*
_output_shapes
: 
^
VarIsInitializedOp_27VarIsInitializedOptraining/Adam/Variable_11*
_output_shapes
: 
T
VarIsInitializedOp_28VarIsInitializedOpdense_14/kernel*
_output_shapes
: 
^
VarIsInitializedOp_29VarIsInitializedOptraining/Adam/Variable_17*
_output_shapes
: 
R
VarIsInitializedOp_30VarIsInitializedOpdense_14/bias*
_output_shapes
: 
T
VarIsInitializedOp_31VarIsInitializedOpdense_15/kernel*
_output_shapes
: 
]
VarIsInitializedOp_32VarIsInitializedOptraining/Adam/Variable_5*
_output_shapes
: 
]
VarIsInitializedOp_33VarIsInitializedOptraining/Adam/Variable_8*
_output_shapes
: 
^
VarIsInitializedOp_34VarIsInitializedOptraining/Adam/Variable_23*
_output_shapes
: 
]
VarIsInitializedOp_35VarIsInitializedOptraining/Adam/Variable_7*
_output_shapes
: 
T
VarIsInitializedOp_36VarIsInitializedOpdense_12/kernel*
_output_shapes
: 
R
VarIsInitializedOp_37VarIsInitializedOpdense_12/bias*
_output_shapes
: 
^
VarIsInitializedOp_38VarIsInitializedOptraining/Adam/Variable_31*
_output_shapes
: 
R
VarIsInitializedOp_39VarIsInitializedOpdense_15/bias*
_output_shapes
: 
R
VarIsInitializedOp_40VarIsInitializedOpdense_13/bias*
_output_shapes
: 
^
VarIsInitializedOp_41VarIsInitializedOptraining/Adam/Variable_18*
_output_shapes
: 
^
VarIsInitializedOp_42VarIsInitializedOptraining/Adam/Variable_28*
_output_shapes
: 
T
VarIsInitializedOp_43VarIsInitializedOpdense_17/kernel*
_output_shapes
: 
]
VarIsInitializedOp_44VarIsInitializedOptraining/Adam/Variable_2*
_output_shapes
: 
^
VarIsInitializedOp_45VarIsInitializedOptraining/Adam/Variable_27*
_output_shapes
: 
^
VarIsInitializedOp_46VarIsInitializedOptraining/Adam/Variable_15*
_output_shapes
: 
^
VarIsInitializedOp_47VarIsInitializedOptraining/Adam/Variable_32*
_output_shapes
: 
]
VarIsInitializedOp_48VarIsInitializedOptraining/Adam/Variable_6*
_output_shapes
: 
^
VarIsInitializedOp_49VarIsInitializedOptraining/Adam/Variable_33*
_output_shapes
: 
^
VarIsInitializedOp_50VarIsInitializedOptraining/Adam/Variable_20*
_output_shapes
: 
^
VarIsInitializedOp_51VarIsInitializedOptraining/Adam/Variable_26*
_output_shapes
: 
^
VarIsInitializedOp_52VarIsInitializedOptraining/Adam/Variable_22*
_output_shapes
: 
ф
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^dense_12/bias/Assign^dense_12/kernel/Assign^dense_13/bias/Assign^dense_13/kernel/Assign^dense_14/bias/Assign^dense_14/kernel/Assign^dense_15/bias/Assign^dense_15/kernel/Assign^dense_16/bias/Assign^dense_16/kernel/Assign^dense_17/bias/Assign^dense_17/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign!^training/Adam/Variable_24/Assign!^training/Adam/Variable_25/Assign!^training/Adam/Variable_26/Assign!^training/Adam/Variable_27/Assign!^training/Adam/Variable_28/Assign!^training/Adam/Variable_29/Assign ^training/Adam/Variable_3/Assign!^training/Adam/Variable_30/Assign!^training/Adam/Variable_31/Assign!^training/Adam/Variable_32/Assign!^training/Adam/Variable_33/Assign!^training/Adam/Variable_34/Assign!^training/Adam/Variable_35/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign"gЯмА4     Q)	срЛҐK6„AJуи
к#ƒ#
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	АР
Ы
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeintИ
E
AssignAddVariableOp
resource
value"dtype"
dtypetypeИ
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
є
DenseToDenseSetOperation	
set1"T	
set2"T
result_indices	
result_values"T
result_shape	"
set_operationstring"
validate_indicesbool("
Ttype:
	2	
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
B
Equal
x"T
y"T
z
"
Ttype:
2	
Р
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	Р
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
;
Minimum
x"T
y"T
z"T"
Ttype:

2	Р
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
0
Round
x"T
y"T"
Ttype:

2	
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
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ
9
VarIsInitializedOp
resource
is_initialized
И*1.12.02v1.12.0-rc2-3-ga6d8ffae09фЇ
П
)Adam/iterations/Initializer/initial_valueConst*"
_class
loc:@Adam/iterations*
value	B	 R *
dtype0	*
_output_shapes
: 
І
Adam/iterationsVarHandleOp* 
shared_nameAdam/iterations*"
_class
loc:@Adam/iterations*
	container *
shape: *
dtype0	*
_output_shapes
: 
o
0Adam/iterations/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/iterations*
_output_shapes
: 
Ч
Adam/iterations/AssignAssignVariableOpAdam/iterations)Adam/iterations/Initializer/initial_value*
dtype0	*"
_class
loc:@Adam/iterations
П
#Adam/iterations/Read/ReadVariableOpReadVariableOpAdam/iterations*
dtype0	*
_output_shapes
: *"
_class
loc:@Adam/iterations
В
!Adam/lr/Initializer/initial_valueConst*
_class
loc:@Adam/lr*
valueB
 *љ7Ж5*
dtype0*
_output_shapes
: 
П
Adam/lrVarHandleOp*
dtype0*
_output_shapes
: *
shared_name	Adam/lr*
_class
loc:@Adam/lr*
	container *
shape: 
_
(Adam/lr/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/lr*
_output_shapes
: 
w
Adam/lr/AssignAssignVariableOpAdam/lr!Adam/lr/Initializer/initial_value*
_class
loc:@Adam/lr*
dtype0
w
Adam/lr/Read/ReadVariableOpReadVariableOpAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 
К
%Adam/beta_1/Initializer/initial_valueConst*
_class
loc:@Adam/beta_1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Ы
Adam/beta_1VarHandleOp*
dtype0*
_output_shapes
: *
shared_nameAdam/beta_1*
_class
loc:@Adam/beta_1*
	container *
shape: 
g
,Adam/beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/beta_1*
_output_shapes
: 
З
Adam/beta_1/AssignAssignVariableOpAdam/beta_1%Adam/beta_1/Initializer/initial_value*
dtype0*
_class
loc:@Adam/beta_1
Г
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 
К
%Adam/beta_2/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@Adam/beta_2*
valueB
 *wЊ?
Ы
Adam/beta_2VarHandleOp*
dtype0*
_output_shapes
: *
shared_nameAdam/beta_2*
_class
loc:@Adam/beta_2*
	container *
shape: 
g
,Adam/beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/beta_2*
_output_shapes
: 
З
Adam/beta_2/AssignAssignVariableOpAdam/beta_2%Adam/beta_2/Initializer/initial_value*
_class
loc:@Adam/beta_2*
dtype0
Г
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 
И
$Adam/decay/Initializer/initial_valueConst*
_class
loc:@Adam/decay*
valueB
 *    *
dtype0*
_output_shapes
: 
Ш

Adam/decayVarHandleOp*
dtype0*
_output_shapes
: *
shared_name
Adam/decay*
_class
loc:@Adam/decay*
	container *
shape: 
e
+Adam/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Adam/decay*
_output_shapes
: 
Г
Adam/decay/AssignAssignVariableOp
Adam/decay$Adam/decay/Initializer/initial_value*
_class
loc:@Adam/decay*
dtype0
А
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: *
_class
loc:@Adam/decay
•
0dense_12/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@dense_12/kernel*
valueB"   d   
Ч
.dense_12/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_12/kernel*
valueB
 *чzXљ*
dtype0*
_output_shapes
: 
Ч
.dense_12/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@dense_12/kernel*
valueB
 *чzX=*
dtype0*
_output_shapes
: 
п
8dense_12/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_12/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	Аd*

seed *
T0*"
_class
loc:@dense_12/kernel*
seed2 
Џ
.dense_12/kernel/Initializer/random_uniform/subSub.dense_12/kernel/Initializer/random_uniform/max.dense_12/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_12/kernel*
_output_shapes
: 
н
.dense_12/kernel/Initializer/random_uniform/mulMul8dense_12/kernel/Initializer/random_uniform/RandomUniform.dense_12/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_12/kernel*
_output_shapes
:	Аd
я
*dense_12/kernel/Initializer/random_uniformAdd.dense_12/kernel/Initializer/random_uniform/mul.dense_12/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_12/kernel*
_output_shapes
:	Аd
∞
dense_12/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_namedense_12/kernel*"
_class
loc:@dense_12/kernel*
	container *
shape:	Аd
o
0dense_12/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_12/kernel*
_output_shapes
: 
Ш
dense_12/kernel/AssignAssignVariableOpdense_12/kernel*dense_12/kernel/Initializer/random_uniform*"
_class
loc:@dense_12/kernel*
dtype0
Ш
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*"
_class
loc:@dense_12/kernel*
dtype0*
_output_shapes
:	Аd
О
dense_12/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:d* 
_class
loc:@dense_12/bias*
valueBd*    
•
dense_12/biasVarHandleOp* 
_class
loc:@dense_12/bias*
	container *
shape:d*
dtype0*
_output_shapes
: *
shared_namedense_12/bias
k
.dense_12/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_12/bias*
_output_shapes
: 
З
dense_12/bias/AssignAssignVariableOpdense_12/biasdense_12/bias/Initializer/zeros*
dtype0* 
_class
loc:@dense_12/bias
Н
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias* 
_class
loc:@dense_12/bias*
dtype0*
_output_shapes
:d
•
0dense_13/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@dense_13/kernel*
valueB"d   d   *
dtype0*
_output_shapes
:
Ч
.dense_13/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_13/kernel*
valueB
 *ђ\1Њ*
dtype0*
_output_shapes
: 
Ч
.dense_13/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@dense_13/kernel*
valueB
 *ђ\1>*
dtype0*
_output_shapes
: 
о
8dense_13/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_13/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:dd*

seed *
T0*"
_class
loc:@dense_13/kernel*
seed2 
Џ
.dense_13/kernel/Initializer/random_uniform/subSub.dense_13/kernel/Initializer/random_uniform/max.dense_13/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes
: 
м
.dense_13/kernel/Initializer/random_uniform/mulMul8dense_13/kernel/Initializer/random_uniform/RandomUniform.dense_13/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes

:dd
ё
*dense_13/kernel/Initializer/random_uniformAdd.dense_13/kernel/Initializer/random_uniform/mul.dense_13/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes

:dd
ѓ
dense_13/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_namedense_13/kernel*"
_class
loc:@dense_13/kernel*
	container *
shape
:dd
o
0dense_13/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_13/kernel*
_output_shapes
: 
Ш
dense_13/kernel/AssignAssignVariableOpdense_13/kernel*dense_13/kernel/Initializer/random_uniform*
dtype0*"
_class
loc:@dense_13/kernel
Ч
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
dtype0*
_output_shapes

:dd*"
_class
loc:@dense_13/kernel
О
dense_13/bias/Initializer/zerosConst* 
_class
loc:@dense_13/bias*
valueBd*    *
dtype0*
_output_shapes
:d
•
dense_13/biasVarHandleOp*
	container *
shape:d*
dtype0*
_output_shapes
: *
shared_namedense_13/bias* 
_class
loc:@dense_13/bias
k
.dense_13/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_13/bias*
_output_shapes
: 
З
dense_13/bias/AssignAssignVariableOpdense_13/biasdense_13/bias/Initializer/zeros* 
_class
loc:@dense_13/bias*
dtype0
Н
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias* 
_class
loc:@dense_13/bias*
dtype0*
_output_shapes
:d
•
0dense_14/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@dense_14/kernel*
valueB"d   d   *
dtype0*
_output_shapes
:
Ч
.dense_14/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_14/kernel*
valueB
 *ђ\1Њ*
dtype0*
_output_shapes
: 
Ч
.dense_14/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@dense_14/kernel*
valueB
 *ђ\1>
о
8dense_14/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_14/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@dense_14/kernel*
seed2 *
dtype0*
_output_shapes

:dd*

seed 
Џ
.dense_14/kernel/Initializer/random_uniform/subSub.dense_14/kernel/Initializer/random_uniform/max.dense_14/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_14/kernel*
_output_shapes
: 
м
.dense_14/kernel/Initializer/random_uniform/mulMul8dense_14/kernel/Initializer/random_uniform/RandomUniform.dense_14/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_14/kernel*
_output_shapes

:dd
ё
*dense_14/kernel/Initializer/random_uniformAdd.dense_14/kernel/Initializer/random_uniform/mul.dense_14/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_14/kernel*
_output_shapes

:dd
ѓ
dense_14/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_namedense_14/kernel*"
_class
loc:@dense_14/kernel*
	container *
shape
:dd
o
0dense_14/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_14/kernel*
_output_shapes
: 
Ш
dense_14/kernel/AssignAssignVariableOpdense_14/kernel*dense_14/kernel/Initializer/random_uniform*"
_class
loc:@dense_14/kernel*
dtype0
Ч
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*"
_class
loc:@dense_14/kernel*
dtype0*
_output_shapes

:dd
О
dense_14/bias/Initializer/zerosConst* 
_class
loc:@dense_14/bias*
valueBd*    *
dtype0*
_output_shapes
:d
•
dense_14/biasVarHandleOp*
shared_namedense_14/bias* 
_class
loc:@dense_14/bias*
	container *
shape:d*
dtype0*
_output_shapes
: 
k
.dense_14/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_14/bias*
_output_shapes
: 
З
dense_14/bias/AssignAssignVariableOpdense_14/biasdense_14/bias/Initializer/zeros* 
_class
loc:@dense_14/bias*
dtype0
Н
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
dtype0*
_output_shapes
:d* 
_class
loc:@dense_14/bias
•
0dense_15/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@dense_15/kernel*
valueB"d   d   *
dtype0*
_output_shapes
:
Ч
.dense_15/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_15/kernel*
valueB
 *ђ\1Њ*
dtype0*
_output_shapes
: 
Ч
.dense_15/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@dense_15/kernel*
valueB
 *ђ\1>*
dtype0*
_output_shapes
: 
о
8dense_15/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_15/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes

:dd*

seed *
T0*"
_class
loc:@dense_15/kernel
Џ
.dense_15/kernel/Initializer/random_uniform/subSub.dense_15/kernel/Initializer/random_uniform/max.dense_15/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_15/kernel*
_output_shapes
: 
м
.dense_15/kernel/Initializer/random_uniform/mulMul8dense_15/kernel/Initializer/random_uniform/RandomUniform.dense_15/kernel/Initializer/random_uniform/sub*
_output_shapes

:dd*
T0*"
_class
loc:@dense_15/kernel
ё
*dense_15/kernel/Initializer/random_uniformAdd.dense_15/kernel/Initializer/random_uniform/mul.dense_15/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_15/kernel*
_output_shapes

:dd
ѓ
dense_15/kernelVarHandleOp* 
shared_namedense_15/kernel*"
_class
loc:@dense_15/kernel*
	container *
shape
:dd*
dtype0*
_output_shapes
: 
o
0dense_15/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_15/kernel*
_output_shapes
: 
Ш
dense_15/kernel/AssignAssignVariableOpdense_15/kernel*dense_15/kernel/Initializer/random_uniform*"
_class
loc:@dense_15/kernel*
dtype0
Ч
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*"
_class
loc:@dense_15/kernel*
dtype0*
_output_shapes

:dd
О
dense_15/bias/Initializer/zerosConst* 
_class
loc:@dense_15/bias*
valueBd*    *
dtype0*
_output_shapes
:d
•
dense_15/biasVarHandleOp*
shared_namedense_15/bias* 
_class
loc:@dense_15/bias*
	container *
shape:d*
dtype0*
_output_shapes
: 
k
.dense_15/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_15/bias*
_output_shapes
: 
З
dense_15/bias/AssignAssignVariableOpdense_15/biasdense_15/bias/Initializer/zeros* 
_class
loc:@dense_15/bias*
dtype0
Н
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias* 
_class
loc:@dense_15/bias*
dtype0*
_output_shapes
:d
•
0dense_16/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@dense_16/kernel*
valueB"d   d   *
dtype0*
_output_shapes
:
Ч
.dense_16/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_16/kernel*
valueB
 *ђ\1Њ*
dtype0*
_output_shapes
: 
Ч
.dense_16/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@dense_16/kernel*
valueB
 *ђ\1>
о
8dense_16/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_16/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:dd*

seed *
T0*"
_class
loc:@dense_16/kernel*
seed2 
Џ
.dense_16/kernel/Initializer/random_uniform/subSub.dense_16/kernel/Initializer/random_uniform/max.dense_16/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@dense_16/kernel
м
.dense_16/kernel/Initializer/random_uniform/mulMul8dense_16/kernel/Initializer/random_uniform/RandomUniform.dense_16/kernel/Initializer/random_uniform/sub*
_output_shapes

:dd*
T0*"
_class
loc:@dense_16/kernel
ё
*dense_16/kernel/Initializer/random_uniformAdd.dense_16/kernel/Initializer/random_uniform/mul.dense_16/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_16/kernel*
_output_shapes

:dd
ѓ
dense_16/kernelVarHandleOp*
	container *
shape
:dd*
dtype0*
_output_shapes
: * 
shared_namedense_16/kernel*"
_class
loc:@dense_16/kernel
o
0dense_16/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_16/kernel*
_output_shapes
: 
Ш
dense_16/kernel/AssignAssignVariableOpdense_16/kernel*dense_16/kernel/Initializer/random_uniform*
dtype0*"
_class
loc:@dense_16/kernel
Ч
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
dtype0*
_output_shapes

:dd*"
_class
loc:@dense_16/kernel
О
dense_16/bias/Initializer/zerosConst* 
_class
loc:@dense_16/bias*
valueBd*    *
dtype0*
_output_shapes
:d
•
dense_16/biasVarHandleOp* 
_class
loc:@dense_16/bias*
	container *
shape:d*
dtype0*
_output_shapes
: *
shared_namedense_16/bias
k
.dense_16/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_16/bias*
_output_shapes
: 
З
dense_16/bias/AssignAssignVariableOpdense_16/biasdense_16/bias/Initializer/zeros* 
_class
loc:@dense_16/bias*
dtype0
Н
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
dtype0*
_output_shapes
:d* 
_class
loc:@dense_16/bias
•
0dense_17/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@dense_17/kernel*
valueB"d      
Ч
.dense_17/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_17/kernel*
valueB
 *B[xЊ*
dtype0*
_output_shapes
: 
Ч
.dense_17/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@dense_17/kernel*
valueB
 *B[x>*
dtype0*
_output_shapes
: 
о
8dense_17/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_17/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:d*

seed *
T0*"
_class
loc:@dense_17/kernel*
seed2 
Џ
.dense_17/kernel/Initializer/random_uniform/subSub.dense_17/kernel/Initializer/random_uniform/max.dense_17/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_17/kernel*
_output_shapes
: 
м
.dense_17/kernel/Initializer/random_uniform/mulMul8dense_17/kernel/Initializer/random_uniform/RandomUniform.dense_17/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_17/kernel*
_output_shapes

:d
ё
*dense_17/kernel/Initializer/random_uniformAdd.dense_17/kernel/Initializer/random_uniform/mul.dense_17/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_17/kernel*
_output_shapes

:d
ѓ
dense_17/kernelVarHandleOp*
shape
:d*
dtype0*
_output_shapes
: * 
shared_namedense_17/kernel*"
_class
loc:@dense_17/kernel*
	container 
o
0dense_17/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_17/kernel*
_output_shapes
: 
Ш
dense_17/kernel/AssignAssignVariableOpdense_17/kernel*dense_17/kernel/Initializer/random_uniform*"
_class
loc:@dense_17/kernel*
dtype0
Ч
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*"
_class
loc:@dense_17/kernel*
dtype0*
_output_shapes

:d
О
dense_17/bias/Initializer/zerosConst* 
_class
loc:@dense_17/bias*
valueB*    *
dtype0*
_output_shapes
:
•
dense_17/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_17/bias* 
_class
loc:@dense_17/bias*
	container *
shape:
k
.dense_17/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_17/bias*
_output_shapes
: 
З
dense_17/bias/AssignAssignVariableOpdense_17/biasdense_17/bias/Initializer/zeros*
dtype0* 
_class
loc:@dense_17/bias
Н
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias* 
_class
loc:@dense_17/bias*
dtype0*
_output_shapes
:
l
input_1Placeholder*
dtype0*(
_output_shapes
:€€€€€€€€€А*
shape:€€€€€€€€€А
f
MatMul/ReadVariableOpReadVariableOpdense_12/kernel*
dtype0*
_output_shapes
:	Аd
И
MatMulMatMulinput_1MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b( 
`
BiasAdd/ReadVariableOpReadVariableOpdense_12/bias*
dtype0*
_output_shapes
:d
{
BiasAddBiasAddMatMulBiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d
G
ReluReluBiasAdd*'
_output_shapes
:€€€€€€€€€d*
T0
\
keras_learning_phase/inputConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 
d
cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
I
cond/switch_tIdentitycond/Switch:1*
T0
*
_output_shapes
: 
G
cond/switch_fIdentitycond/Switch*
T0
*
_output_shapes
: 
O
cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

k
cond/dropout/keep_probConst^cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
m
cond/dropout/ShapeShapecond/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
Х
cond/dropout/Shape/SwitchSwitchRelucond/pred_id*
T0*
_class
	loc:@Relu*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
t
cond/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
t
cond/dropout/random_uniform/maxConst^cond/switch_t*
valueB
 *  А?*
dtype0*
_output_shapes
: 
¶
)cond/dropout/random_uniform/RandomUniformRandomUniformcond/dropout/Shape*

seed *
T0*
dtype0*'
_output_shapes
:€€€€€€€€€d*
seed2 
Й
cond/dropout/random_uniform/subSubcond/dropout/random_uniform/maxcond/dropout/random_uniform/min*
T0*
_output_shapes
: 
§
cond/dropout/random_uniform/mulMul)cond/dropout/random_uniform/RandomUniformcond/dropout/random_uniform/sub*'
_output_shapes
:€€€€€€€€€d*
T0
Ц
cond/dropout/random_uniformAddcond/dropout/random_uniform/mulcond/dropout/random_uniform/min*
T0*'
_output_shapes
:€€€€€€€€€d
~
cond/dropout/addAddcond/dropout/keep_probcond/dropout/random_uniform*
T0*'
_output_shapes
:€€€€€€€€€d
_
cond/dropout/FloorFloorcond/dropout/add*
T0*'
_output_shapes
:€€€€€€€€€d
В
cond/dropout/divRealDivcond/dropout/Shape/Switch:1cond/dropout/keep_prob*
T0*'
_output_shapes
:€€€€€€€€€d
o
cond/dropout/mulMulcond/dropout/divcond/dropout/Floor*
T0*'
_output_shapes
:€€€€€€€€€d
a
cond/IdentityIdentitycond/Identity/Switch*
T0*'
_output_shapes
:€€€€€€€€€d
Р
cond/Identity/SwitchSwitchRelucond/pred_id*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d*
T0*
_class
	loc:@Relu
q

cond/MergeMergecond/Identitycond/dropout/mul*
N*)
_output_shapes
:€€€€€€€€€d: *
T0
g
MatMul_1/ReadVariableOpReadVariableOpdense_13/kernel*
dtype0*
_output_shapes

:dd
П
MatMul_1MatMul
cond/MergeMatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b( 
b
BiasAdd_1/ReadVariableOpReadVariableOpdense_13/bias*
dtype0*
_output_shapes
:d
Б
	BiasAdd_1BiasAddMatMul_1BiasAdd_1/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d
K
Relu_1Relu	BiasAdd_1*
T0*'
_output_shapes
:€€€€€€€€€d
f
cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
M
cond_1/switch_tIdentitycond_1/Switch:1*
T0
*
_output_shapes
: 
K
cond_1/switch_fIdentitycond_1/Switch*
T0
*
_output_shapes
: 
Q
cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
o
cond_1/dropout/keep_probConst^cond_1/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
q
cond_1/dropout/ShapeShapecond_1/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
Э
cond_1/dropout/Shape/SwitchSwitchRelu_1cond_1/pred_id*
T0*
_class
loc:@Relu_1*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
x
!cond_1/dropout/random_uniform/minConst^cond_1/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
x
!cond_1/dropout/random_uniform/maxConst^cond_1/switch_t*
valueB
 *  А?*
dtype0*
_output_shapes
: 
™
+cond_1/dropout/random_uniform/RandomUniformRandomUniformcond_1/dropout/Shape*
T0*
dtype0*'
_output_shapes
:€€€€€€€€€d*
seed2 *

seed 
П
!cond_1/dropout/random_uniform/subSub!cond_1/dropout/random_uniform/max!cond_1/dropout/random_uniform/min*
T0*
_output_shapes
: 
™
!cond_1/dropout/random_uniform/mulMul+cond_1/dropout/random_uniform/RandomUniform!cond_1/dropout/random_uniform/sub*
T0*'
_output_shapes
:€€€€€€€€€d
Ь
cond_1/dropout/random_uniformAdd!cond_1/dropout/random_uniform/mul!cond_1/dropout/random_uniform/min*
T0*'
_output_shapes
:€€€€€€€€€d
Д
cond_1/dropout/addAddcond_1/dropout/keep_probcond_1/dropout/random_uniform*
T0*'
_output_shapes
:€€€€€€€€€d
c
cond_1/dropout/FloorFloorcond_1/dropout/add*'
_output_shapes
:€€€€€€€€€d*
T0
И
cond_1/dropout/divRealDivcond_1/dropout/Shape/Switch:1cond_1/dropout/keep_prob*'
_output_shapes
:€€€€€€€€€d*
T0
u
cond_1/dropout/mulMulcond_1/dropout/divcond_1/dropout/Floor*
T0*'
_output_shapes
:€€€€€€€€€d
e
cond_1/IdentityIdentitycond_1/Identity/Switch*
T0*'
_output_shapes
:€€€€€€€€€d
Ш
cond_1/Identity/SwitchSwitchRelu_1cond_1/pred_id*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d*
T0*
_class
loc:@Relu_1
w
cond_1/MergeMergecond_1/Identitycond_1/dropout/mul*
T0*
N*)
_output_shapes
:€€€€€€€€€d: 
g
MatMul_2/ReadVariableOpReadVariableOpdense_14/kernel*
dtype0*
_output_shapes

:dd
С
MatMul_2MatMulcond_1/MergeMatMul_2/ReadVariableOp*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b( *
T0
b
BiasAdd_2/ReadVariableOpReadVariableOpdense_14/bias*
dtype0*
_output_shapes
:d
Б
	BiasAdd_2BiasAddMatMul_2BiasAdd_2/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d
K
Relu_2Relu	BiasAdd_2*
T0*'
_output_shapes
:€€€€€€€€€d
f
cond_2/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
M
cond_2/switch_tIdentitycond_2/Switch:1*
T0
*
_output_shapes
: 
K
cond_2/switch_fIdentitycond_2/Switch*
_output_shapes
: *
T0

Q
cond_2/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
o
cond_2/dropout/keep_probConst^cond_2/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
q
cond_2/dropout/ShapeShapecond_2/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
Э
cond_2/dropout/Shape/SwitchSwitchRelu_2cond_2/pred_id*
T0*
_class
loc:@Relu_2*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
x
!cond_2/dropout/random_uniform/minConst^cond_2/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
x
!cond_2/dropout/random_uniform/maxConst^cond_2/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  А?
™
+cond_2/dropout/random_uniform/RandomUniformRandomUniformcond_2/dropout/Shape*
T0*
dtype0*'
_output_shapes
:€€€€€€€€€d*
seed2 *

seed 
П
!cond_2/dropout/random_uniform/subSub!cond_2/dropout/random_uniform/max!cond_2/dropout/random_uniform/min*
_output_shapes
: *
T0
™
!cond_2/dropout/random_uniform/mulMul+cond_2/dropout/random_uniform/RandomUniform!cond_2/dropout/random_uniform/sub*
T0*'
_output_shapes
:€€€€€€€€€d
Ь
cond_2/dropout/random_uniformAdd!cond_2/dropout/random_uniform/mul!cond_2/dropout/random_uniform/min*'
_output_shapes
:€€€€€€€€€d*
T0
Д
cond_2/dropout/addAddcond_2/dropout/keep_probcond_2/dropout/random_uniform*
T0*'
_output_shapes
:€€€€€€€€€d
c
cond_2/dropout/FloorFloorcond_2/dropout/add*
T0*'
_output_shapes
:€€€€€€€€€d
И
cond_2/dropout/divRealDivcond_2/dropout/Shape/Switch:1cond_2/dropout/keep_prob*
T0*'
_output_shapes
:€€€€€€€€€d
u
cond_2/dropout/mulMulcond_2/dropout/divcond_2/dropout/Floor*'
_output_shapes
:€€€€€€€€€d*
T0
e
cond_2/IdentityIdentitycond_2/Identity/Switch*
T0*'
_output_shapes
:€€€€€€€€€d
Ш
cond_2/Identity/SwitchSwitchRelu_2cond_2/pred_id*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d*
T0*
_class
loc:@Relu_2
w
cond_2/MergeMergecond_2/Identitycond_2/dropout/mul*
N*)
_output_shapes
:€€€€€€€€€d: *
T0
g
MatMul_3/ReadVariableOpReadVariableOpdense_15/kernel*
dtype0*
_output_shapes

:dd
С
MatMul_3MatMulcond_2/MergeMatMul_3/ReadVariableOp*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€d*
transpose_a( 
b
BiasAdd_3/ReadVariableOpReadVariableOpdense_15/bias*
dtype0*
_output_shapes
:d
Б
	BiasAdd_3BiasAddMatMul_3BiasAdd_3/ReadVariableOp*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d*
T0
K
Relu_3Relu	BiasAdd_3*
T0*'
_output_shapes
:€€€€€€€€€d
f
cond_3/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

M
cond_3/switch_tIdentitycond_3/Switch:1*
_output_shapes
: *
T0

K
cond_3/switch_fIdentitycond_3/Switch*
T0
*
_output_shapes
: 
Q
cond_3/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

o
cond_3/dropout/keep_probConst^cond_3/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
q
cond_3/dropout/ShapeShapecond_3/dropout/Shape/Switch:1*
_output_shapes
:*
T0*
out_type0
Э
cond_3/dropout/Shape/SwitchSwitchRelu_3cond_3/pred_id*
T0*
_class
loc:@Relu_3*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
x
!cond_3/dropout/random_uniform/minConst^cond_3/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
x
!cond_3/dropout/random_uniform/maxConst^cond_3/switch_t*
valueB
 *  А?*
dtype0*
_output_shapes
: 
™
+cond_3/dropout/random_uniform/RandomUniformRandomUniformcond_3/dropout/Shape*
T0*
dtype0*'
_output_shapes
:€€€€€€€€€d*
seed2 *

seed 
П
!cond_3/dropout/random_uniform/subSub!cond_3/dropout/random_uniform/max!cond_3/dropout/random_uniform/min*
_output_shapes
: *
T0
™
!cond_3/dropout/random_uniform/mulMul+cond_3/dropout/random_uniform/RandomUniform!cond_3/dropout/random_uniform/sub*'
_output_shapes
:€€€€€€€€€d*
T0
Ь
cond_3/dropout/random_uniformAdd!cond_3/dropout/random_uniform/mul!cond_3/dropout/random_uniform/min*'
_output_shapes
:€€€€€€€€€d*
T0
Д
cond_3/dropout/addAddcond_3/dropout/keep_probcond_3/dropout/random_uniform*
T0*'
_output_shapes
:€€€€€€€€€d
c
cond_3/dropout/FloorFloorcond_3/dropout/add*'
_output_shapes
:€€€€€€€€€d*
T0
И
cond_3/dropout/divRealDivcond_3/dropout/Shape/Switch:1cond_3/dropout/keep_prob*'
_output_shapes
:€€€€€€€€€d*
T0
u
cond_3/dropout/mulMulcond_3/dropout/divcond_3/dropout/Floor*
T0*'
_output_shapes
:€€€€€€€€€d
e
cond_3/IdentityIdentitycond_3/Identity/Switch*
T0*'
_output_shapes
:€€€€€€€€€d
Ш
cond_3/Identity/SwitchSwitchRelu_3cond_3/pred_id*
T0*
_class
loc:@Relu_3*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
w
cond_3/MergeMergecond_3/Identitycond_3/dropout/mul*
T0*
N*)
_output_shapes
:€€€€€€€€€d: 
g
MatMul_4/ReadVariableOpReadVariableOpdense_16/kernel*
dtype0*
_output_shapes

:dd
С
MatMul_4MatMulcond_3/MergeMatMul_4/ReadVariableOp*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b( *
T0
b
BiasAdd_4/ReadVariableOpReadVariableOpdense_16/bias*
dtype0*
_output_shapes
:d
Б
	BiasAdd_4BiasAddMatMul_4BiasAdd_4/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d
K
Relu_4Relu	BiasAdd_4*'
_output_shapes
:€€€€€€€€€d*
T0
f
cond_4/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
M
cond_4/switch_tIdentitycond_4/Switch:1*
T0
*
_output_shapes
: 
K
cond_4/switch_fIdentitycond_4/Switch*
T0
*
_output_shapes
: 
Q
cond_4/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
o
cond_4/dropout/keep_probConst^cond_4/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
q
cond_4/dropout/ShapeShapecond_4/dropout/Shape/Switch:1*
_output_shapes
:*
T0*
out_type0
Э
cond_4/dropout/Shape/SwitchSwitchRelu_4cond_4/pred_id*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d*
T0*
_class
loc:@Relu_4
x
!cond_4/dropout/random_uniform/minConst^cond_4/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
x
!cond_4/dropout/random_uniform/maxConst^cond_4/switch_t*
valueB
 *  А?*
dtype0*
_output_shapes
: 
™
+cond_4/dropout/random_uniform/RandomUniformRandomUniformcond_4/dropout/Shape*
T0*
dtype0*'
_output_shapes
:€€€€€€€€€d*
seed2 *

seed 
П
!cond_4/dropout/random_uniform/subSub!cond_4/dropout/random_uniform/max!cond_4/dropout/random_uniform/min*
_output_shapes
: *
T0
™
!cond_4/dropout/random_uniform/mulMul+cond_4/dropout/random_uniform/RandomUniform!cond_4/dropout/random_uniform/sub*
T0*'
_output_shapes
:€€€€€€€€€d
Ь
cond_4/dropout/random_uniformAdd!cond_4/dropout/random_uniform/mul!cond_4/dropout/random_uniform/min*'
_output_shapes
:€€€€€€€€€d*
T0
Д
cond_4/dropout/addAddcond_4/dropout/keep_probcond_4/dropout/random_uniform*
T0*'
_output_shapes
:€€€€€€€€€d
c
cond_4/dropout/FloorFloorcond_4/dropout/add*'
_output_shapes
:€€€€€€€€€d*
T0
И
cond_4/dropout/divRealDivcond_4/dropout/Shape/Switch:1cond_4/dropout/keep_prob*
T0*'
_output_shapes
:€€€€€€€€€d
u
cond_4/dropout/mulMulcond_4/dropout/divcond_4/dropout/Floor*
T0*'
_output_shapes
:€€€€€€€€€d
e
cond_4/IdentityIdentitycond_4/Identity/Switch*'
_output_shapes
:€€€€€€€€€d*
T0
Ш
cond_4/Identity/SwitchSwitchRelu_4cond_4/pred_id*
T0*
_class
loc:@Relu_4*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
w
cond_4/MergeMergecond_4/Identitycond_4/dropout/mul*
T0*
N*)
_output_shapes
:€€€€€€€€€d: 
g
MatMul_5/ReadVariableOpReadVariableOpdense_17/kernel*
dtype0*
_output_shapes

:d
С
MatMul_5MatMulcond_4/MergeMatMul_5/ReadVariableOp*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( *
T0
b
BiasAdd_5/ReadVariableOpReadVariableOpdense_17/bias*
dtype0*
_output_shapes
:
Б
	BiasAdd_5BiasAddMatMul_5BiasAdd_5/ReadVariableOp*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€*
T0
O
SoftmaxSoftmax	BiasAdd_5*
T0*'
_output_shapes
:€€€€€€€€€
Д
output_1_targetPlaceholder*
dtype0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*%
shape:€€€€€€€€€€€€€€€€€€
R
ConstConst*
valueB*  А?*
dtype0*
_output_shapes
:
Д
output_1_sample_weightsPlaceholderWithDefaultConst*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
j
(loss/output_1_loss/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Я
loss/output_1_loss/SumSumSoftmax(loss/output_1_loss/Sum/reduction_indices*'
_output_shapes
:€€€€€€€€€*
	keep_dims(*

Tidx0*
T0
x
loss/output_1_loss/truedivRealDivSoftmaxloss/output_1_loss/Sum*
T0*'
_output_shapes
:€€€€€€€€€
]
loss/output_1_loss/ConstConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
]
loss/output_1_loss/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
r
loss/output_1_loss/subSubloss/output_1_loss/sub/xloss/output_1_loss/Const*
T0*
_output_shapes
: 
Щ
(loss/output_1_loss/clip_by_value/MinimumMinimumloss/output_1_loss/truedivloss/output_1_loss/sub*
T0*'
_output_shapes
:€€€€€€€€€
°
 loss/output_1_loss/clip_by_valueMaximum(loss/output_1_loss/clip_by_value/Minimumloss/output_1_loss/Const*'
_output_shapes
:€€€€€€€€€*
T0
q
loss/output_1_loss/LogLog loss/output_1_loss/clip_by_value*
T0*'
_output_shapes
:€€€€€€€€€
x
loss/output_1_loss/mulMuloutput_1_targetloss/output_1_loss/Log*'
_output_shapes
:€€€€€€€€€*
T0
l
*loss/output_1_loss/Sum_1/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
Ѓ
loss/output_1_loss/Sum_1Sumloss/output_1_loss/mul*loss/output_1_loss/Sum_1/reduction_indices*
T0*#
_output_shapes
:€€€€€€€€€*
	keep_dims( *

Tidx0
e
loss/output_1_loss/NegNegloss/output_1_loss/Sum_1*
T0*#
_output_shapes
:€€€€€€€€€
Ю
Gloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeShapeoutput_1_sample_weights*
T0*
out_type0*
_output_shapes
:
И
Floss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
dtype0*
_output_shapes
: *
value	B :
Ь
Floss/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeShapeloss/output_1_loss/Neg*
T0*
out_type0*
_output_shapes
:
З
Eloss/output_1_loss/broadcast_weights/assert_broadcastable/values/rankConst*
dtype0*
_output_shapes
: *
value	B :
З
Eloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar/xConst*
dtype0*
_output_shapes
: *
value	B : 
ь
Closs/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarEqualEloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar/xFloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank*
_output_shapes
: *
T0
Ж
Oloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/SwitchSwitchCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
—
Qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_tIdentityQloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
ѕ
Qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentityOloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
¬
Ploss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
н
Qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1SwitchCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarPloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0
*V
_classL
JHloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : 
Л
oloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualvloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switchxloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
Ц
vloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchEloss/output_1_loss/broadcast_weights/assert_broadcastable/values/rankPloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*X
_classN
LJloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/values/rank*
_output_shapes
: : 
Ъ
xloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchFloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rankPloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*Y
_classO
MKloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank*
_output_shapes
: : 
ш
iloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitcholoss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankoloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
Е
kloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentitykloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
Г
kloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityiloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
И
jloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityoloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
Љ
Вloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
“
~loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsЙloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Вloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
T0*
_output_shapes

:*

Tdim0
∞
Еloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchFloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shapePloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*Y
_classO
MKloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
М
Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchЕloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchjloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*Y
_classO
MKloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
√
Гloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
і
Гloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
ћ
}loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillГloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeГloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*

index_type0*
_output_shapes

:
ѓ
loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
ƒ
zloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2~loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims}loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
Њ
Дloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
ў
Аloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsЛloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Дloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
_output_shapes

:*

Tdim0*
T0
і
Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchGloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapePloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*Z
_classP
NLloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::
С
Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchЗloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchjloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*Z
_classP
NLloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::
Я
Мloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationАloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1zloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*<
_output_shapes*
(:€€€€€€€€€:€€€€€€€€€:*
set_operationa-b*
validate_indices(*
T0
ѕ
Дloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeОloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
•
uloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
Ы
sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualuloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xДloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
ъ
kloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switcholoss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankjloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*В
_classx
vtloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
€
hloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergekloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
¬
Nloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMergehloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeSloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1*
N*
_output_shapes
: : *
T0

І
?loss/output_1_loss/broadcast_weights/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Р
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_1Const*
dtype0*
_output_shapes
: *
valueB Bweights.shape=
Ы
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_2Const*
dtype0*
_output_shapes
: **
value!B Boutput_1_sample_weights:0
П
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
Ъ
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_4Const*)
value B Bloss/output_1_loss/Neg:0*
dtype0*
_output_shapes
: 
М
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
Щ
Lloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitchNloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeNloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
Ћ
Nloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentityNloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
_output_shapes
: *
T0

…
Nloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityLloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
_output_shapes
: *
T0

 
Mloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_idIdentityNloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
£
Jloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOpO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
Е
Xloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependencyIdentityNloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tK^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*a
_classW
USloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
М
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
у
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
ю
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: **
value!B Boutput_1_sample_weights:0
т
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
э
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*)
value B Bloss/output_1_loss/Neg:0*
dtype0*
_output_shapes
: 
п
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
”
Lloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssertSloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
В
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSwitchNloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
_output_shapes
: : *
T0
*a
_classW
USloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge
ю
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchGloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeMloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0*Z
_classP
NLloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::
ь
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchFloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeMloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id* 
_output_shapes
::*
T0*Y
_classO
MKloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape
о
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarMloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0
*V
_classL
JHloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : 
Й
Zloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1IdentityNloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fM^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert*
T0
*a
_classW
USloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
ґ
Kloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/MergeMergeZloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1Xloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
Ў
4loss/output_1_loss/broadcast_weights/ones_like/ShapeShapeloss/output_1_loss/NegL^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
«
4loss/output_1_loss/broadcast_weights/ones_like/ConstConstL^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
valueB
 *  А?*
dtype0*
_output_shapes
: 
в
.loss/output_1_loss/broadcast_weights/ones_likeFill4loss/output_1_loss/broadcast_weights/ones_like/Shape4loss/output_1_loss/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:€€€€€€€€€
Ґ
$loss/output_1_loss/broadcast_weightsMuloutput_1_sample_weights.loss/output_1_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:€€€€€€€€€
Л
loss/output_1_loss/Mul_1Mulloss/output_1_loss/Neg$loss/output_1_loss/broadcast_weights*
T0*#
_output_shapes
:€€€€€€€€€
d
loss/output_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
У
loss/output_1_loss/Sum_2Sumloss/output_1_loss/Mul_1loss/output_1_loss/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
d
loss/output_1_loss/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
Я
loss/output_1_loss/Sum_3Sum$loss/output_1_loss/broadcast_weightsloss/output_1_loss/Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
|
loss/output_1_loss/truediv_1RealDivloss/output_1_loss/Sum_2loss/output_1_loss/Sum_3*
T0*
_output_shapes
: 
b
loss/output_1_loss/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss/output_1_loss/GreaterGreaterloss/output_1_loss/Sum_3loss/output_1_loss/zeros_like*
T0*
_output_shapes
: 
Э
loss/output_1_loss/SelectSelectloss/output_1_loss/Greaterloss/output_1_loss/truediv_1loss/output_1_loss/zeros_like*
T0*
_output_shapes
: 
]
loss/output_1_loss/Const_3Const*
dtype0*
_output_shapes
: *
valueB 
Ф
loss/output_1_loss/MeanMeanloss/output_1_loss/Selectloss/output_1_loss/Const_3*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
O

loss/mul/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
U
loss/mulMul
loss/mul/xloss/output_1_loss/Mean*
T0*
_output_shapes
: 
g
metrics/acc/ArgMax/dimensionConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ш
metrics/acc/ArgMaxArgMaxoutput_1_targetmetrics/acc/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:€€€€€€€€€*

Tidx0
i
metrics/acc/ArgMax_1/dimensionConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ф
metrics/acc/ArgMax_1ArgMaxSoftmaxmetrics/acc/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:€€€€€€€€€*

Tidx0
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*
T0	*#
_output_shapes
:€€€€€€€€€
x
metrics/acc/CastCastmetrics/acc/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:€€€€€€€€€*

DstT0
[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
metrics/acc/MeanMeanmetrics/acc/Castmetrics/acc/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
g
metrics/f1_score/mulMuloutput_1_targetSoftmax*
T0*'
_output_shapes
:€€€€€€€€€
[
metrics/f1_score/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
]
metrics/f1_score/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
У
&metrics/f1_score/clip_by_value/MinimumMinimummetrics/f1_score/mulmetrics/f1_score/Const_1*
T0*'
_output_shapes
:€€€€€€€€€
Ы
metrics/f1_score/clip_by_valueMaximum&metrics/f1_score/clip_by_value/Minimummetrics/f1_score/Const*
T0*'
_output_shapes
:€€€€€€€€€
q
metrics/f1_score/RoundRoundmetrics/f1_score/clip_by_value*
T0*'
_output_shapes
:€€€€€€€€€
i
metrics/f1_score/Const_2Const*
valueB"       *
dtype0*
_output_shapes
:
Л
metrics/f1_score/SumSummetrics/f1_score/Roundmetrics/f1_score/Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
]
metrics/f1_score/Const_3Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
metrics/f1_score/Const_4Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
И
(metrics/f1_score/clip_by_value_1/MinimumMinimumSoftmaxmetrics/f1_score/Const_4*
T0*'
_output_shapes
:€€€€€€€€€
°
 metrics/f1_score/clip_by_value_1Maximum(metrics/f1_score/clip_by_value_1/Minimummetrics/f1_score/Const_3*
T0*'
_output_shapes
:€€€€€€€€€
u
metrics/f1_score/Round_1Round metrics/f1_score/clip_by_value_1*
T0*'
_output_shapes
:€€€€€€€€€
i
metrics/f1_score/Const_5Const*
valueB"       *
dtype0*
_output_shapes
:
П
metrics/f1_score/Sum_1Summetrics/f1_score/Round_1metrics/f1_score/Const_5*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
[
metrics/f1_score/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *Хњ÷3
l
metrics/f1_score/addAddmetrics/f1_score/Sum_1metrics/f1_score/add/y*
T0*
_output_shapes
: 
p
metrics/f1_score/truedivRealDivmetrics/f1_score/Summetrics/f1_score/add*
_output_shapes
: *
T0
i
metrics/f1_score/mul_1Muloutput_1_targetSoftmax*
T0*'
_output_shapes
:€€€€€€€€€
]
metrics/f1_score/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
metrics/f1_score/Const_7Const*
dtype0*
_output_shapes
: *
valueB
 *  А?
Ч
(metrics/f1_score/clip_by_value_2/MinimumMinimummetrics/f1_score/mul_1metrics/f1_score/Const_7*
T0*'
_output_shapes
:€€€€€€€€€
°
 metrics/f1_score/clip_by_value_2Maximum(metrics/f1_score/clip_by_value_2/Minimummetrics/f1_score/Const_6*
T0*'
_output_shapes
:€€€€€€€€€
u
metrics/f1_score/Round_2Round metrics/f1_score/clip_by_value_2*
T0*'
_output_shapes
:€€€€€€€€€
i
metrics/f1_score/Const_8Const*
valueB"       *
dtype0*
_output_shapes
:
П
metrics/f1_score/Sum_2Summetrics/f1_score/Round_2metrics/f1_score/Const_8*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
]
metrics/f1_score/Const_9Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
metrics/f1_score/Const_10Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ъ
(metrics/f1_score/clip_by_value_3/MinimumMinimumoutput_1_targetmetrics/f1_score/Const_10*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
™
 metrics/f1_score/clip_by_value_3Maximum(metrics/f1_score/clip_by_value_3/Minimummetrics/f1_score/Const_9*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0
~
metrics/f1_score/Round_3Round metrics/f1_score/clip_by_value_3*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
j
metrics/f1_score/Const_11Const*
dtype0*
_output_shapes
:*
valueB"       
Р
metrics/f1_score/Sum_3Summetrics/f1_score/Round_3metrics/f1_score/Const_11*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
]
metrics/f1_score/add_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *Хњ÷3
p
metrics/f1_score/add_1Addmetrics/f1_score/Sum_3metrics/f1_score/add_1/y*
_output_shapes
: *
T0
v
metrics/f1_score/truediv_1RealDivmetrics/f1_score/Sum_2metrics/f1_score/add_1*
T0*
_output_shapes
: 
]
metrics/f1_score/mul_2/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
r
metrics/f1_score/mul_2Mulmetrics/f1_score/mul_2/xmetrics/f1_score/truediv*
T0*
_output_shapes
: 
r
metrics/f1_score/mul_3Mulmetrics/f1_score/mul_2metrics/f1_score/truediv_1*
T0*
_output_shapes
: 
t
metrics/f1_score/add_2Addmetrics/f1_score/truedivmetrics/f1_score/truediv_1*
T0*
_output_shapes
: 
]
metrics/f1_score/add_3/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
p
metrics/f1_score/add_3Addmetrics/f1_score/add_2metrics/f1_score/add_3/y*
T0*
_output_shapes
: 
v
metrics/f1_score/truediv_2RealDivmetrics/f1_score/mul_3metrics/f1_score/add_3*
T0*
_output_shapes
: 
\
metrics/f1_score/Const_12Const*
valueB *
dtype0*
_output_shapes
: 
Т
metrics/f1_score/MeanMeanmetrics/f1_score/truediv_2metrics/f1_score/Const_12*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
}
training/Adam/gradients/ShapeConst*
_class
loc:@loss/mul*
valueB *
dtype0*
_output_shapes
: 
Г
!training/Adam/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
_class
loc:@loss/mul*
valueB
 *  А?
ґ
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*
_class
loc:@loss/mul*

index_type0*
_output_shapes
: 
•
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/output_1_loss/Mean*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
Ъ
+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
±
Btraining/Adam/gradients/loss/output_1_loss/Mean_grad/Reshape/shapeConst**
_class 
loc:@loss/output_1_loss/Mean*
valueB *
dtype0*
_output_shapes
: 
У
<training/Adam/gradients/loss/output_1_loss/Mean_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Btraining/Adam/gradients/loss/output_1_loss/Mean_grad/Reshape/shape*
T0**
_class 
loc:@loss/output_1_loss/Mean*
Tshape0*
_output_shapes
: 
©
:training/Adam/gradients/loss/output_1_loss/Mean_grad/ConstConst**
_class 
loc:@loss/output_1_loss/Mean*
valueB *
dtype0*
_output_shapes
: 
Ъ
9training/Adam/gradients/loss/output_1_loss/Mean_grad/TileTile<training/Adam/gradients/loss/output_1_loss/Mean_grad/Reshape:training/Adam/gradients/loss/output_1_loss/Mean_grad/Const*

Tmultiples0*
T0**
_class 
loc:@loss/output_1_loss/Mean*
_output_shapes
: 
≠
<training/Adam/gradients/loss/output_1_loss/Mean_grad/Const_1Const**
_class 
loc:@loss/output_1_loss/Mean*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Н
<training/Adam/gradients/loss/output_1_loss/Mean_grad/truedivRealDiv9training/Adam/gradients/loss/output_1_loss/Mean_grad/Tile<training/Adam/gradients/loss/output_1_loss/Mean_grad/Const_1*
T0**
_class 
loc:@loss/output_1_loss/Mean*
_output_shapes
: 
і
Atraining/Adam/gradients/loss/output_1_loss/Select_grad/zeros_likeConst*,
_class"
 loc:@loss/output_1_loss/Select*
valueB
 *    *
dtype0*
_output_shapes
: 
≥
=training/Adam/gradients/loss/output_1_loss/Select_grad/SelectSelectloss/output_1_loss/Greater<training/Adam/gradients/loss/output_1_loss/Mean_grad/truedivAtraining/Adam/gradients/loss/output_1_loss/Select_grad/zeros_like*
_output_shapes
: *
T0*,
_class"
 loc:@loss/output_1_loss/Select
µ
?training/Adam/gradients/loss/output_1_loss/Select_grad/Select_1Selectloss/output_1_loss/GreaterAtraining/Adam/gradients/loss/output_1_loss/Select_grad/zeros_like<training/Adam/gradients/loss/output_1_loss/Mean_grad/truediv*
T0*,
_class"
 loc:@loss/output_1_loss/Select*
_output_shapes
: 
≥
?training/Adam/gradients/loss/output_1_loss/truediv_1_grad/ShapeConst*/
_class%
#!loc:@loss/output_1_loss/truediv_1*
valueB *
dtype0*
_output_shapes
: 
µ
Atraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/Shape_1Const*/
_class%
#!loc:@loss/output_1_loss/truediv_1*
valueB *
dtype0*
_output_shapes
: 
Џ
Otraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgs?training/Adam/gradients/loss/output_1_loss/truediv_1_grad/ShapeAtraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/Shape_1*
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ч
Atraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/RealDivRealDiv=training/Adam/gradients/loss/output_1_loss/Select_grad/Selectloss/output_1_loss/Sum_3*
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1*
_output_shapes
: 
«
=training/Adam/gradients/loss/output_1_loss/truediv_1_grad/SumSumAtraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/RealDivOtraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/BroadcastGradientArgs*
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1*
_output_shapes
: *
	keep_dims( *

Tidx0
ђ
Atraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/ReshapeReshape=training/Adam/gradients/loss/output_1_loss/truediv_1_grad/Sum?training/Adam/gradients/loss/output_1_loss/truediv_1_grad/Shape*
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1*
Tshape0*
_output_shapes
: 
∞
=training/Adam/gradients/loss/output_1_loss/truediv_1_grad/NegNegloss/output_1_loss/Sum_2*
_output_shapes
: *
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1
щ
Ctraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/RealDiv_1RealDiv=training/Adam/gradients/loss/output_1_loss/truediv_1_grad/Negloss/output_1_loss/Sum_3*
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1*
_output_shapes
: 
€
Ctraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/RealDiv_2RealDivCtraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/RealDiv_1loss/output_1_loss/Sum_3*
_output_shapes
: *
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1
Ъ
=training/Adam/gradients/loss/output_1_loss/truediv_1_grad/mulMul=training/Adam/gradients/loss/output_1_loss/Select_grad/SelectCtraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/RealDiv_2*
_output_shapes
: *
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1
«
?training/Adam/gradients/loss/output_1_loss/truediv_1_grad/Sum_1Sum=training/Adam/gradients/loss/output_1_loss/truediv_1_grad/mulQtraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1
≤
Ctraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/Reshape_1Reshape?training/Adam/gradients/loss/output_1_loss/truediv_1_grad/Sum_1Atraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/Shape_1*
T0*/
_class%
#!loc:@loss/output_1_loss/truediv_1*
Tshape0*
_output_shapes
: 
Ї
Ctraining/Adam/gradients/loss/output_1_loss/Sum_2_grad/Reshape/shapeConst*+
_class!
loc:@loss/output_1_loss/Sum_2*
valueB:*
dtype0*
_output_shapes
:
∞
=training/Adam/gradients/loss/output_1_loss/Sum_2_grad/ReshapeReshapeAtraining/Adam/gradients/loss/output_1_loss/truediv_1_grad/ReshapeCtraining/Adam/gradients/loss/output_1_loss/Sum_2_grad/Reshape/shape*
T0*+
_class!
loc:@loss/output_1_loss/Sum_2*
Tshape0*
_output_shapes
:
ј
;training/Adam/gradients/loss/output_1_loss/Sum_2_grad/ShapeShapeloss/output_1_loss/Mul_1*
T0*+
_class!
loc:@loss/output_1_loss/Sum_2*
out_type0*
_output_shapes
:
Ђ
:training/Adam/gradients/loss/output_1_loss/Sum_2_grad/TileTile=training/Adam/gradients/loss/output_1_loss/Sum_2_grad/Reshape;training/Adam/gradients/loss/output_1_loss/Sum_2_grad/Shape*
T0*+
_class!
loc:@loss/output_1_loss/Sum_2*#
_output_shapes
:€€€€€€€€€*

Tmultiples0
Њ
;training/Adam/gradients/loss/output_1_loss/Mul_1_grad/ShapeShapeloss/output_1_loss/Neg*
_output_shapes
:*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1*
out_type0
ќ
=training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Shape_1Shape$loss/output_1_loss/broadcast_weights*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1*
out_type0*
_output_shapes
:
 
Ktraining/Adam/gradients/loss/output_1_loss/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs;training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Shape=training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Shape_1*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
э
9training/Adam/gradients/loss/output_1_loss/Mul_1_grad/MulMul:training/Adam/gradients/loss/output_1_loss/Sum_2_grad/Tile$loss/output_1_loss/broadcast_weights*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1*#
_output_shapes
:€€€€€€€€€
µ
9training/Adam/gradients/loss/output_1_loss/Mul_1_grad/SumSum9training/Adam/gradients/loss/output_1_loss/Mul_1_grad/MulKtraining/Adam/gradients/loss/output_1_loss/Mul_1_grad/BroadcastGradientArgs*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
©
=training/Adam/gradients/loss/output_1_loss/Mul_1_grad/ReshapeReshape9training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Sum;training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Shape*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1*
Tshape0*#
_output_shapes
:€€€€€€€€€
с
;training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Mul_1Mulloss/output_1_loss/Neg:training/Adam/gradients/loss/output_1_loss/Sum_2_grad/Tile*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1*#
_output_shapes
:€€€€€€€€€
ї
;training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Sum_1Sum;training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Mul_1Mtraining/Adam/gradients/loss/output_1_loss/Mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1*
_output_shapes
:
ѓ
?training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Reshape_1Reshape;training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Sum_1=training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Shape_1*
T0*+
_class!
loc:@loss/output_1_loss/Mul_1*
Tshape0*#
_output_shapes
:€€€€€€€€€
÷
7training/Adam/gradients/loss/output_1_loss/Neg_grad/NegNeg=training/Adam/gradients/loss/output_1_loss/Mul_1_grad/Reshape*
T0*)
_class
loc:@loss/output_1_loss/Neg*#
_output_shapes
:€€€€€€€€€
Њ
;training/Adam/gradients/loss/output_1_loss/Sum_1_grad/ShapeShapeloss/output_1_loss/mul*
T0*+
_class!
loc:@loss/output_1_loss/Sum_1*
out_type0*
_output_shapes
:
©
:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/SizeConst*+
_class!
loc:@loss/output_1_loss/Sum_1*
value	B :*
dtype0*
_output_shapes
: 
ц
9training/Adam/gradients/loss/output_1_loss/Sum_1_grad/addAdd*loss/output_1_loss/Sum_1/reduction_indices:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Size*
_output_shapes
: *
T0*+
_class!
loc:@loss/output_1_loss/Sum_1
К
9training/Adam/gradients/loss/output_1_loss/Sum_1_grad/modFloorMod9training/Adam/gradients/loss/output_1_loss/Sum_1_grad/add:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Size*
T0*+
_class!
loc:@loss/output_1_loss/Sum_1*
_output_shapes
: 
≠
=training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Shape_1Const*+
_class!
loc:@loss/output_1_loss/Sum_1*
valueB *
dtype0*
_output_shapes
: 
∞
Atraining/Adam/gradients/loss/output_1_loss/Sum_1_grad/range/startConst*
dtype0*
_output_shapes
: *+
_class!
loc:@loss/output_1_loss/Sum_1*
value	B : 
∞
Atraining/Adam/gradients/loss/output_1_loss/Sum_1_grad/range/deltaConst*+
_class!
loc:@loss/output_1_loss/Sum_1*
value	B :*
dtype0*
_output_shapes
: 
џ
;training/Adam/gradients/loss/output_1_loss/Sum_1_grad/rangeRangeAtraining/Adam/gradients/loss/output_1_loss/Sum_1_grad/range/start:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/SizeAtraining/Adam/gradients/loss/output_1_loss/Sum_1_grad/range/delta*+
_class!
loc:@loss/output_1_loss/Sum_1*
_output_shapes
:*

Tidx0
ѓ
@training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Fill/valueConst*
dtype0*
_output_shapes
: *+
_class!
loc:@loss/output_1_loss/Sum_1*
value	B :
£
:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/FillFill=training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Shape_1@training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Fill/value*
T0*+
_class!
loc:@loss/output_1_loss/Sum_1*

index_type0*
_output_shapes
: 
†
Ctraining/Adam/gradients/loss/output_1_loss/Sum_1_grad/DynamicStitchDynamicStitch;training/Adam/gradients/loss/output_1_loss/Sum_1_grad/range9training/Adam/gradients/loss/output_1_loss/Sum_1_grad/mod;training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Shape:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Fill*
T0*+
_class!
loc:@loss/output_1_loss/Sum_1*
N*
_output_shapes
:
Ѓ
?training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Maximum/yConst*+
_class!
loc:@loss/output_1_loss/Sum_1*
value	B :*
dtype0*
_output_shapes
: 
†
=training/Adam/gradients/loss/output_1_loss/Sum_1_grad/MaximumMaximumCtraining/Adam/gradients/loss/output_1_loss/Sum_1_grad/DynamicStitch?training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Maximum/y*
T0*+
_class!
loc:@loss/output_1_loss/Sum_1*
_output_shapes
:
Ш
>training/Adam/gradients/loss/output_1_loss/Sum_1_grad/floordivFloorDiv;training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Shape=training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Maximum*
T0*+
_class!
loc:@loss/output_1_loss/Sum_1*
_output_shapes
:
Љ
=training/Adam/gradients/loss/output_1_loss/Sum_1_grad/ReshapeReshape7training/Adam/gradients/loss/output_1_loss/Neg_grad/NegCtraining/Adam/gradients/loss/output_1_loss/Sum_1_grad/DynamicStitch*
T0*+
_class!
loc:@loss/output_1_loss/Sum_1*
Tshape0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
≤
:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/TileTile=training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Reshape>training/Adam/gradients/loss/output_1_loss/Sum_1_grad/floordiv*

Tmultiples0*
T0*+
_class!
loc:@loss/output_1_loss/Sum_1*'
_output_shapes
:€€€€€€€€€
≥
9training/Adam/gradients/loss/output_1_loss/mul_grad/ShapeShapeoutput_1_target*
T0*)
_class
loc:@loss/output_1_loss/mul*
out_type0*
_output_shapes
:
Љ
;training/Adam/gradients/loss/output_1_loss/mul_grad/Shape_1Shapeloss/output_1_loss/Log*
T0*)
_class
loc:@loss/output_1_loss/mul*
out_type0*
_output_shapes
:
¬
Itraining/Adam/gradients/loss/output_1_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9training/Adam/gradients/loss/output_1_loss/mul_grad/Shape;training/Adam/gradients/loss/output_1_loss/mul_grad/Shape_1*
T0*)
_class
loc:@loss/output_1_loss/mul*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
п
7training/Adam/gradients/loss/output_1_loss/mul_grad/MulMul:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Tileloss/output_1_loss/Log*'
_output_shapes
:€€€€€€€€€*
T0*)
_class
loc:@loss/output_1_loss/mul
≠
7training/Adam/gradients/loss/output_1_loss/mul_grad/SumSum7training/Adam/gradients/loss/output_1_loss/mul_grad/MulItraining/Adam/gradients/loss/output_1_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/output_1_loss/mul
Ѓ
;training/Adam/gradients/loss/output_1_loss/mul_grad/ReshapeReshape7training/Adam/gradients/loss/output_1_loss/mul_grad/Sum9training/Adam/gradients/loss/output_1_loss/mul_grad/Shape*
T0*)
_class
loc:@loss/output_1_loss/mul*
Tshape0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
к
9training/Adam/gradients/loss/output_1_loss/mul_grad/Mul_1Muloutput_1_target:training/Adam/gradients/loss/output_1_loss/Sum_1_grad/Tile*'
_output_shapes
:€€€€€€€€€*
T0*)
_class
loc:@loss/output_1_loss/mul
≥
9training/Adam/gradients/loss/output_1_loss/mul_grad/Sum_1Sum9training/Adam/gradients/loss/output_1_loss/mul_grad/Mul_1Ktraining/Adam/gradients/loss/output_1_loss/mul_grad/BroadcastGradientArgs:1*
T0*)
_class
loc:@loss/output_1_loss/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Ђ
=training/Adam/gradients/loss/output_1_loss/mul_grad/Reshape_1Reshape9training/Adam/gradients/loss/output_1_loss/mul_grad/Sum_1;training/Adam/gradients/loss/output_1_loss/mul_grad/Shape_1*'
_output_shapes
:€€€€€€€€€*
T0*)
_class
loc:@loss/output_1_loss/mul*
Tshape0
Л
>training/Adam/gradients/loss/output_1_loss/Log_grad/Reciprocal
Reciprocal loss/output_1_loss/clip_by_value>^training/Adam/gradients/loss/output_1_loss/mul_grad/Reshape_1*
T0*)
_class
loc:@loss/output_1_loss/Log*'
_output_shapes
:€€€€€€€€€
Ъ
7training/Adam/gradients/loss/output_1_loss/Log_grad/mulMul=training/Adam/gradients/loss/output_1_loss/mul_grad/Reshape_1>training/Adam/gradients/loss/output_1_loss/Log_grad/Reciprocal*
T0*)
_class
loc:@loss/output_1_loss/Log*'
_output_shapes
:€€€€€€€€€
а
Ctraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/ShapeShape(loss/output_1_loss/clip_by_value/Minimum*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*
out_type0*
_output_shapes
:
љ
Etraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Shape_1Const*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*
valueB *
dtype0*
_output_shapes
: 
с
Etraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Shape_2Shape7training/Adam/gradients/loss/output_1_loss/Log_grad/mul*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*
out_type0*
_output_shapes
:
√
Itraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *3
_class)
'%loc:@loss/output_1_loss/clip_by_value*
valueB
 *    
÷
Ctraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/zerosFillEtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Shape_2Itraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/zeros/Const*'
_output_shapes
:€€€€€€€€€*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*

index_type0
Е
Jtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/GreaterEqualGreaterEqual(loss/output_1_loss/clip_by_value/Minimumloss/output_1_loss/Const*'
_output_shapes
:€€€€€€€€€*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value
к
Straining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsCtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/ShapeEtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Shape_1*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
€
Dtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/SelectSelectJtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/GreaterEqual7training/Adam/gradients/loss/output_1_loss/Log_grad/mulCtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/zeros*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*'
_output_shapes
:€€€€€€€€€
Б
Ftraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Select_1SelectJtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/GreaterEqualCtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/zeros7training/Adam/gradients/loss/output_1_loss/Log_grad/mul*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*'
_output_shapes
:€€€€€€€€€
Ў
Atraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/SumSumDtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/SelectStraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value
Ќ
Etraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/ReshapeReshapeAtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/SumCtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*
Tshape0
ё
Ctraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Sum_1SumFtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Select_1Utraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/BroadcastGradientArgs:1*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*
_output_shapes
:*
	keep_dims( *

Tidx0
¬
Gtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Reshape_1ReshapeCtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Sum_1Etraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Shape_1*
T0*3
_class)
'%loc:@loss/output_1_loss/clip_by_value*
Tshape0*
_output_shapes
: 
в
Ktraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/ShapeShapeloss/output_1_loss/truediv*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*
out_type0*
_output_shapes
:
Ќ
Mtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Shape_1Const*
dtype0*
_output_shapes
: *;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*
valueB 
П
Mtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Shape_2ShapeEtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Reshape*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*
out_type0*
_output_shapes
:
”
Qtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*
valueB
 *    
ц
Ktraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/zerosFillMtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Shape_2Qtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/zeros/Const*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*

index_type0*'
_output_shapes
:€€€€€€€€€
€
Otraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualloss/output_1_loss/truedivloss/output_1_loss/sub*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*'
_output_shapes
:€€€€€€€€€
К
[training/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/ShapeMtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Shape_1*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
™
Ltraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/SelectSelectOtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/LessEqualEtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/ReshapeKtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/zeros*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*'
_output_shapes
:€€€€€€€€€
ђ
Ntraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Select_1SelectOtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/LessEqualKtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/zerosEtraining/Adam/gradients/loss/output_1_loss/clip_by_value_grad/Reshape*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*'
_output_shapes
:€€€€€€€€€
ш
Itraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/SumSumLtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Select[training/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*
_output_shapes
:
н
Mtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/ReshapeReshapeItraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/SumKtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Shape*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*
Tshape0*'
_output_shapes
:€€€€€€€€€
ю
Ktraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Sum_1SumNtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Select_1]training/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0
в
Otraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeKtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Sum_1Mtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Shape_1*
T0*;
_class1
/-loc:@loss/output_1_loss/clip_by_value/Minimum*
Tshape0*
_output_shapes
: 
≥
=training/Adam/gradients/loss/output_1_loss/truediv_grad/ShapeShapeSoftmax*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*
out_type0*
_output_shapes
:
ƒ
?training/Adam/gradients/loss/output_1_loss/truediv_grad/Shape_1Shapeloss/output_1_loss/Sum*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*
out_type0*
_output_shapes
:
“
Mtraining/Adam/gradients/loss/output_1_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/loss/output_1_loss/truediv_grad/Shape?training/Adam/gradients/loss/output_1_loss/truediv_grad/Shape_1*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Т
?training/Adam/gradients/loss/output_1_loss/truediv_grad/RealDivRealDivMtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/Reshapeloss/output_1_loss/Sum*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*'
_output_shapes
:€€€€€€€€€
Ѕ
;training/Adam/gradients/loss/output_1_loss/truediv_grad/SumSum?training/Adam/gradients/loss/output_1_loss/truediv_grad/RealDivMtraining/Adam/gradients/loss/output_1_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*
_output_shapes
:
µ
?training/Adam/gradients/loss/output_1_loss/truediv_grad/ReshapeReshape;training/Adam/gradients/loss/output_1_loss/truediv_grad/Sum=training/Adam/gradients/loss/output_1_loss/truediv_grad/Shape*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*
Tshape0*'
_output_shapes
:€€€€€€€€€
ђ
;training/Adam/gradients/loss/output_1_loss/truediv_grad/NegNegSoftmax*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*'
_output_shapes
:€€€€€€€€€
В
Atraining/Adam/gradients/loss/output_1_loss/truediv_grad/RealDiv_1RealDiv;training/Adam/gradients/loss/output_1_loss/truediv_grad/Negloss/output_1_loss/Sum*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*'
_output_shapes
:€€€€€€€€€
И
Atraining/Adam/gradients/loss/output_1_loss/truediv_grad/RealDiv_2RealDivAtraining/Adam/gradients/loss/output_1_loss/truediv_grad/RealDiv_1loss/output_1_loss/Sum*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*'
_output_shapes
:€€€€€€€€€
µ
;training/Adam/gradients/loss/output_1_loss/truediv_grad/mulMulMtraining/Adam/gradients/loss/output_1_loss/clip_by_value/Minimum_grad/ReshapeAtraining/Adam/gradients/loss/output_1_loss/truediv_grad/RealDiv_2*'
_output_shapes
:€€€€€€€€€*
T0*-
_class#
!loc:@loss/output_1_loss/truediv
Ѕ
=training/Adam/gradients/loss/output_1_loss/truediv_grad/Sum_1Sum;training/Adam/gradients/loss/output_1_loss/truediv_grad/mulOtraining/Adam/gradients/loss/output_1_loss/truediv_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0
ї
Atraining/Adam/gradients/loss/output_1_loss/truediv_grad/Reshape_1Reshape=training/Adam/gradients/loss/output_1_loss/truediv_grad/Sum_1?training/Adam/gradients/loss/output_1_loss/truediv_grad/Shape_1*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*
Tshape0*'
_output_shapes
:€€€€€€€€€
Ђ
9training/Adam/gradients/loss/output_1_loss/Sum_grad/ShapeShapeSoftmax*
T0*)
_class
loc:@loss/output_1_loss/Sum*
out_type0*
_output_shapes
:
•
8training/Adam/gradients/loss/output_1_loss/Sum_grad/SizeConst*)
_class
loc:@loss/output_1_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 
о
7training/Adam/gradients/loss/output_1_loss/Sum_grad/addAdd(loss/output_1_loss/Sum/reduction_indices8training/Adam/gradients/loss/output_1_loss/Sum_grad/Size*
T0*)
_class
loc:@loss/output_1_loss/Sum*
_output_shapes
: 
В
7training/Adam/gradients/loss/output_1_loss/Sum_grad/modFloorMod7training/Adam/gradients/loss/output_1_loss/Sum_grad/add8training/Adam/gradients/loss/output_1_loss/Sum_grad/Size*
_output_shapes
: *
T0*)
_class
loc:@loss/output_1_loss/Sum
©
;training/Adam/gradients/loss/output_1_loss/Sum_grad/Shape_1Const*)
_class
loc:@loss/output_1_loss/Sum*
valueB *
dtype0*
_output_shapes
: 
ђ
?training/Adam/gradients/loss/output_1_loss/Sum_grad/range/startConst*)
_class
loc:@loss/output_1_loss/Sum*
value	B : *
dtype0*
_output_shapes
: 
ђ
?training/Adam/gradients/loss/output_1_loss/Sum_grad/range/deltaConst*)
_class
loc:@loss/output_1_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 
—
9training/Adam/gradients/loss/output_1_loss/Sum_grad/rangeRange?training/Adam/gradients/loss/output_1_loss/Sum_grad/range/start8training/Adam/gradients/loss/output_1_loss/Sum_grad/Size?training/Adam/gradients/loss/output_1_loss/Sum_grad/range/delta*)
_class
loc:@loss/output_1_loss/Sum*
_output_shapes
:*

Tidx0
Ђ
>training/Adam/gradients/loss/output_1_loss/Sum_grad/Fill/valueConst*)
_class
loc:@loss/output_1_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 
Ы
8training/Adam/gradients/loss/output_1_loss/Sum_grad/FillFill;training/Adam/gradients/loss/output_1_loss/Sum_grad/Shape_1>training/Adam/gradients/loss/output_1_loss/Sum_grad/Fill/value*
T0*)
_class
loc:@loss/output_1_loss/Sum*

index_type0*
_output_shapes
: 
Ф
Atraining/Adam/gradients/loss/output_1_loss/Sum_grad/DynamicStitchDynamicStitch9training/Adam/gradients/loss/output_1_loss/Sum_grad/range7training/Adam/gradients/loss/output_1_loss/Sum_grad/mod9training/Adam/gradients/loss/output_1_loss/Sum_grad/Shape8training/Adam/gradients/loss/output_1_loss/Sum_grad/Fill*
T0*)
_class
loc:@loss/output_1_loss/Sum*
N*
_output_shapes
:
™
=training/Adam/gradients/loss/output_1_loss/Sum_grad/Maximum/yConst*)
_class
loc:@loss/output_1_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 
Ш
;training/Adam/gradients/loss/output_1_loss/Sum_grad/MaximumMaximumAtraining/Adam/gradients/loss/output_1_loss/Sum_grad/DynamicStitch=training/Adam/gradients/loss/output_1_loss/Sum_grad/Maximum/y*
T0*)
_class
loc:@loss/output_1_loss/Sum*
_output_shapes
:
Р
<training/Adam/gradients/loss/output_1_loss/Sum_grad/floordivFloorDiv9training/Adam/gradients/loss/output_1_loss/Sum_grad/Shape;training/Adam/gradients/loss/output_1_loss/Sum_grad/Maximum*
_output_shapes
:*
T0*)
_class
loc:@loss/output_1_loss/Sum
ј
;training/Adam/gradients/loss/output_1_loss/Sum_grad/ReshapeReshapeAtraining/Adam/gradients/loss/output_1_loss/truediv_grad/Reshape_1Atraining/Adam/gradients/loss/output_1_loss/Sum_grad/DynamicStitch*
T0*)
_class
loc:@loss/output_1_loss/Sum*
Tshape0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
™
8training/Adam/gradients/loss/output_1_loss/Sum_grad/TileTile;training/Adam/gradients/loss/output_1_loss/Sum_grad/Reshape<training/Adam/gradients/loss/output_1_loss/Sum_grad/floordiv*'
_output_shapes
:€€€€€€€€€*

Tmultiples0*
T0*)
_class
loc:@loss/output_1_loss/Sum
Й
training/Adam/gradients/AddNAddN?training/Adam/gradients/loss/output_1_loss/truediv_grad/Reshape8training/Adam/gradients/loss/output_1_loss/Sum_grad/Tile*
T0*-
_class#
!loc:@loss/output_1_loss/truediv*
N*'
_output_shapes
:€€€€€€€€€
§
(training/Adam/gradients/Softmax_grad/mulMultraining/Adam/gradients/AddNSoftmax*
T0*
_class
loc:@Softmax*'
_output_shapes
:€€€€€€€€€
°
:training/Adam/gradients/Softmax_grad/Sum/reduction_indicesConst*
_class
loc:@Softmax*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
А
(training/Adam/gradients/Softmax_grad/SumSum(training/Adam/gradients/Softmax_grad/mul:training/Adam/gradients/Softmax_grad/Sum/reduction_indices*
T0*
_class
loc:@Softmax*'
_output_shapes
:€€€€€€€€€*
	keep_dims(*

Tidx0
≈
(training/Adam/gradients/Softmax_grad/subSubtraining/Adam/gradients/AddN(training/Adam/gradients/Softmax_grad/Sum*'
_output_shapes
:€€€€€€€€€*
T0*
_class
loc:@Softmax
≤
*training/Adam/gradients/Softmax_grad/mul_1Mul(training/Adam/gradients/Softmax_grad/subSoftmax*
T0*
_class
loc:@Softmax*'
_output_shapes
:€€€€€€€€€
«
2training/Adam/gradients/BiasAdd_5_grad/BiasAddGradBiasAddGrad*training/Adam/gradients/Softmax_grad/mul_1*
T0*
_class
loc:@BiasAdd_5*
data_formatNHWC*
_output_shapes
:
р
,training/Adam/gradients/MatMul_5_grad/MatMulMatMul*training/Adam/gradients/Softmax_grad/mul_1MatMul_5/ReadVariableOp*
transpose_b(*
T0*
_class
loc:@MatMul_5*'
_output_shapes
:€€€€€€€€€d*
transpose_a( 
ё
.training/Adam/gradients/MatMul_5_grad/MatMul_1MatMulcond_4/Merge*training/Adam/gradients/Softmax_grad/mul_1*
_output_shapes

:d*
transpose_a(*
transpose_b( *
T0*
_class
loc:@MatMul_5
Ё
3training/Adam/gradients/cond_4/Merge_grad/cond_gradSwitch,training/Adam/gradients/MatMul_5_grad/MatMulcond_4/pred_id*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d*
T0*
_class
loc:@MatMul_5
Ѓ
5training/Adam/gradients/cond_4/dropout/mul_grad/ShapeShapecond_4/dropout/div*
T0*%
_class
loc:@cond_4/dropout/mul*
out_type0*
_output_shapes
:
≤
7training/Adam/gradients/cond_4/dropout/mul_grad/Shape_1Shapecond_4/dropout/Floor*
T0*%
_class
loc:@cond_4/dropout/mul*
out_type0*
_output_shapes
:
≤
Etraining/Adam/gradients/cond_4/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/cond_4/dropout/mul_grad/Shape7training/Adam/gradients/cond_4/dropout/mul_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*%
_class
loc:@cond_4/dropout/mul
а
3training/Adam/gradients/cond_4/dropout/mul_grad/MulMul5training/Adam/gradients/cond_4/Merge_grad/cond_grad:1cond_4/dropout/Floor*
T0*%
_class
loc:@cond_4/dropout/mul*'
_output_shapes
:€€€€€€€€€d
Э
3training/Adam/gradients/cond_4/dropout/mul_grad/SumSum3training/Adam/gradients/cond_4/dropout/mul_grad/MulEtraining/Adam/gradients/cond_4/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_4/dropout/mul
Х
7training/Adam/gradients/cond_4/dropout/mul_grad/ReshapeReshape3training/Adam/gradients/cond_4/dropout/mul_grad/Sum5training/Adam/gradients/cond_4/dropout/mul_grad/Shape*
T0*%
_class
loc:@cond_4/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
а
5training/Adam/gradients/cond_4/dropout/mul_grad/Mul_1Mulcond_4/dropout/div5training/Adam/gradients/cond_4/Merge_grad/cond_grad:1*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_4/dropout/mul
£
5training/Adam/gradients/cond_4/dropout/mul_grad/Sum_1Sum5training/Adam/gradients/cond_4/dropout/mul_grad/Mul_1Gtraining/Adam/gradients/cond_4/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_4/dropout/mul*
_output_shapes
:
Ы
9training/Adam/gradients/cond_4/dropout/mul_grad/Reshape_1Reshape5training/Adam/gradients/cond_4/dropout/mul_grad/Sum_17training/Adam/gradients/cond_4/dropout/mul_grad/Shape_1*
T0*%
_class
loc:@cond_4/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
†
training/Adam/gradients/SwitchSwitchRelu_4cond_4/pred_id*
T0*
_class
loc:@Relu_4*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Ы
 training/Adam/gradients/IdentityIdentity training/Adam/gradients/Switch:1*
T0*
_class
loc:@Relu_4*'
_output_shapes
:€€€€€€€€€d
Ъ
training/Adam/gradients/Shape_1Shape training/Adam/gradients/Switch:1*
T0*
_class
loc:@Relu_4*
out_type0*
_output_shapes
:
¶
#training/Adam/gradients/zeros/ConstConst!^training/Adam/gradients/Identity*
dtype0*
_output_shapes
: *
_class
loc:@Relu_4*
valueB
 *    
 
training/Adam/gradients/zerosFilltraining/Adam/gradients/Shape_1#training/Adam/gradients/zeros/Const*
T0*
_class
loc:@Relu_4*

index_type0*'
_output_shapes
:€€€€€€€€€d
т
=training/Adam/gradients/cond_4/Identity/Switch_grad/cond_gradMerge3training/Adam/gradients/cond_4/Merge_grad/cond_gradtraining/Adam/gradients/zeros*
T0*
_class
loc:@Relu_4*
N*)
_output_shapes
:€€€€€€€€€d: 
є
5training/Adam/gradients/cond_4/dropout/div_grad/ShapeShapecond_4/dropout/Shape/Switch:1*
T0*%
_class
loc:@cond_4/dropout/div*
out_type0*
_output_shapes
:
°
7training/Adam/gradients/cond_4/dropout/div_grad/Shape_1Const*
dtype0*
_output_shapes
: *%
_class
loc:@cond_4/dropout/div*
valueB 
≤
Etraining/Adam/gradients/cond_4/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/cond_4/dropout/div_grad/Shape7training/Adam/gradients/cond_4/dropout/div_grad/Shape_1*
T0*%
_class
loc:@cond_4/dropout/div*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
о
7training/Adam/gradients/cond_4/dropout/div_grad/RealDivRealDiv7training/Adam/gradients/cond_4/dropout/mul_grad/Reshapecond_4/dropout/keep_prob*
T0*%
_class
loc:@cond_4/dropout/div*'
_output_shapes
:€€€€€€€€€d
°
3training/Adam/gradients/cond_4/dropout/div_grad/SumSum7training/Adam/gradients/cond_4/dropout/div_grad/RealDivEtraining/Adam/gradients/cond_4/dropout/div_grad/BroadcastGradientArgs*
T0*%
_class
loc:@cond_4/dropout/div*
_output_shapes
:*
	keep_dims( *

Tidx0
Х
7training/Adam/gradients/cond_4/dropout/div_grad/ReshapeReshape3training/Adam/gradients/cond_4/dropout/div_grad/Sum5training/Adam/gradients/cond_4/dropout/div_grad/Shape*
T0*%
_class
loc:@cond_4/dropout/div*
Tshape0*'
_output_shapes
:€€€€€€€€€d
≤
3training/Adam/gradients/cond_4/dropout/div_grad/NegNegcond_4/dropout/Shape/Switch:1*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_4/dropout/div
м
9training/Adam/gradients/cond_4/dropout/div_grad/RealDiv_1RealDiv3training/Adam/gradients/cond_4/dropout/div_grad/Negcond_4/dropout/keep_prob*
T0*%
_class
loc:@cond_4/dropout/div*'
_output_shapes
:€€€€€€€€€d
т
9training/Adam/gradients/cond_4/dropout/div_grad/RealDiv_2RealDiv9training/Adam/gradients/cond_4/dropout/div_grad/RealDiv_1cond_4/dropout/keep_prob*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_4/dropout/div
З
3training/Adam/gradients/cond_4/dropout/div_grad/mulMul7training/Adam/gradients/cond_4/dropout/mul_grad/Reshape9training/Adam/gradients/cond_4/dropout/div_grad/RealDiv_2*
T0*%
_class
loc:@cond_4/dropout/div*'
_output_shapes
:€€€€€€€€€d
°
5training/Adam/gradients/cond_4/dropout/div_grad/Sum_1Sum3training/Adam/gradients/cond_4/dropout/div_grad/mulGtraining/Adam/gradients/cond_4/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_4/dropout/div
К
9training/Adam/gradients/cond_4/dropout/div_grad/Reshape_1Reshape5training/Adam/gradients/cond_4/dropout/div_grad/Sum_17training/Adam/gradients/cond_4/dropout/div_grad/Shape_1*
_output_shapes
: *
T0*%
_class
loc:@cond_4/dropout/div*
Tshape0
Ґ
 training/Adam/gradients/Switch_1SwitchRelu_4cond_4/pred_id*
T0*
_class
loc:@Relu_4*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Э
"training/Adam/gradients/Identity_1Identity training/Adam/gradients/Switch_1*
T0*
_class
loc:@Relu_4*'
_output_shapes
:€€€€€€€€€d
Ъ
training/Adam/gradients/Shape_2Shape training/Adam/gradients/Switch_1*
_output_shapes
:*
T0*
_class
loc:@Relu_4*
out_type0
™
%training/Adam/gradients/zeros_1/ConstConst#^training/Adam/gradients/Identity_1*
dtype0*
_output_shapes
: *
_class
loc:@Relu_4*
valueB
 *    
ќ
training/Adam/gradients/zeros_1Filltraining/Adam/gradients/Shape_2%training/Adam/gradients/zeros_1/Const*
T0*
_class
loc:@Relu_4*

index_type0*'
_output_shapes
:€€€€€€€€€d
э
Btraining/Adam/gradients/cond_4/dropout/Shape/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_17training/Adam/gradients/cond_4/dropout/div_grad/Reshape*
N*)
_output_shapes
:€€€€€€€€€d: *
T0*
_class
loc:@Relu_4
€
training/Adam/gradients/AddN_1AddN=training/Adam/gradients/cond_4/Identity/Switch_grad/cond_gradBtraining/Adam/gradients/cond_4/dropout/Shape/Switch_grad/cond_grad*
N*'
_output_shapes
:€€€€€€€€€d*
T0*
_class
loc:@Relu_4
≠
,training/Adam/gradients/Relu_4_grad/ReluGradReluGradtraining/Adam/gradients/AddN_1Relu_4*
T0*
_class
loc:@Relu_4*'
_output_shapes
:€€€€€€€€€d
…
2training/Adam/gradients/BiasAdd_4_grad/BiasAddGradBiasAddGrad,training/Adam/gradients/Relu_4_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:d*
T0*
_class
loc:@BiasAdd_4
т
,training/Adam/gradients/MatMul_4_grad/MatMulMatMul,training/Adam/gradients/Relu_4_grad/ReluGradMatMul_4/ReadVariableOp*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b(*
T0*
_class
loc:@MatMul_4
а
.training/Adam/gradients/MatMul_4_grad/MatMul_1MatMulcond_3/Merge,training/Adam/gradients/Relu_4_grad/ReluGrad*
_output_shapes

:dd*
transpose_a(*
transpose_b( *
T0*
_class
loc:@MatMul_4
Ё
3training/Adam/gradients/cond_3/Merge_grad/cond_gradSwitch,training/Adam/gradients/MatMul_4_grad/MatMulcond_3/pred_id*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d*
T0*
_class
loc:@MatMul_4
Ѓ
5training/Adam/gradients/cond_3/dropout/mul_grad/ShapeShapecond_3/dropout/div*
T0*%
_class
loc:@cond_3/dropout/mul*
out_type0*
_output_shapes
:
≤
7training/Adam/gradients/cond_3/dropout/mul_grad/Shape_1Shapecond_3/dropout/Floor*
T0*%
_class
loc:@cond_3/dropout/mul*
out_type0*
_output_shapes
:
≤
Etraining/Adam/gradients/cond_3/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/cond_3/dropout/mul_grad/Shape7training/Adam/gradients/cond_3/dropout/mul_grad/Shape_1*
T0*%
_class
loc:@cond_3/dropout/mul*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
а
3training/Adam/gradients/cond_3/dropout/mul_grad/MulMul5training/Adam/gradients/cond_3/Merge_grad/cond_grad:1cond_3/dropout/Floor*
T0*%
_class
loc:@cond_3/dropout/mul*'
_output_shapes
:€€€€€€€€€d
Э
3training/Adam/gradients/cond_3/dropout/mul_grad/SumSum3training/Adam/gradients/cond_3/dropout/mul_grad/MulEtraining/Adam/gradients/cond_3/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_3/dropout/mul*
_output_shapes
:
Х
7training/Adam/gradients/cond_3/dropout/mul_grad/ReshapeReshape3training/Adam/gradients/cond_3/dropout/mul_grad/Sum5training/Adam/gradients/cond_3/dropout/mul_grad/Shape*
T0*%
_class
loc:@cond_3/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
а
5training/Adam/gradients/cond_3/dropout/mul_grad/Mul_1Mulcond_3/dropout/div5training/Adam/gradients/cond_3/Merge_grad/cond_grad:1*
T0*%
_class
loc:@cond_3/dropout/mul*'
_output_shapes
:€€€€€€€€€d
£
5training/Adam/gradients/cond_3/dropout/mul_grad/Sum_1Sum5training/Adam/gradients/cond_3/dropout/mul_grad/Mul_1Gtraining/Adam/gradients/cond_3/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_3/dropout/mul*
_output_shapes
:
Ы
9training/Adam/gradients/cond_3/dropout/mul_grad/Reshape_1Reshape5training/Adam/gradients/cond_3/dropout/mul_grad/Sum_17training/Adam/gradients/cond_3/dropout/mul_grad/Shape_1*
T0*%
_class
loc:@cond_3/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
Ґ
 training/Adam/gradients/Switch_2SwitchRelu_3cond_3/pred_id*
T0*
_class
loc:@Relu_3*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Я
"training/Adam/gradients/Identity_2Identity"training/Adam/gradients/Switch_2:1*
T0*
_class
loc:@Relu_3*'
_output_shapes
:€€€€€€€€€d
Ь
training/Adam/gradients/Shape_3Shape"training/Adam/gradients/Switch_2:1*
T0*
_class
loc:@Relu_3*
out_type0*
_output_shapes
:
™
%training/Adam/gradients/zeros_2/ConstConst#^training/Adam/gradients/Identity_2*
_class
loc:@Relu_3*
valueB
 *    *
dtype0*
_output_shapes
: 
ќ
training/Adam/gradients/zeros_2Filltraining/Adam/gradients/Shape_3%training/Adam/gradients/zeros_2/Const*'
_output_shapes
:€€€€€€€€€d*
T0*
_class
loc:@Relu_3*

index_type0
ф
=training/Adam/gradients/cond_3/Identity/Switch_grad/cond_gradMerge3training/Adam/gradients/cond_3/Merge_grad/cond_gradtraining/Adam/gradients/zeros_2*
T0*
_class
loc:@Relu_3*
N*)
_output_shapes
:€€€€€€€€€d: 
є
5training/Adam/gradients/cond_3/dropout/div_grad/ShapeShapecond_3/dropout/Shape/Switch:1*
T0*%
_class
loc:@cond_3/dropout/div*
out_type0*
_output_shapes
:
°
7training/Adam/gradients/cond_3/dropout/div_grad/Shape_1Const*%
_class
loc:@cond_3/dropout/div*
valueB *
dtype0*
_output_shapes
: 
≤
Etraining/Adam/gradients/cond_3/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/cond_3/dropout/div_grad/Shape7training/Adam/gradients/cond_3/dropout/div_grad/Shape_1*
T0*%
_class
loc:@cond_3/dropout/div*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
о
7training/Adam/gradients/cond_3/dropout/div_grad/RealDivRealDiv7training/Adam/gradients/cond_3/dropout/mul_grad/Reshapecond_3/dropout/keep_prob*
T0*%
_class
loc:@cond_3/dropout/div*'
_output_shapes
:€€€€€€€€€d
°
3training/Adam/gradients/cond_3/dropout/div_grad/SumSum7training/Adam/gradients/cond_3/dropout/div_grad/RealDivEtraining/Adam/gradients/cond_3/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_3/dropout/div*
_output_shapes
:
Х
7training/Adam/gradients/cond_3/dropout/div_grad/ReshapeReshape3training/Adam/gradients/cond_3/dropout/div_grad/Sum5training/Adam/gradients/cond_3/dropout/div_grad/Shape*
T0*%
_class
loc:@cond_3/dropout/div*
Tshape0*'
_output_shapes
:€€€€€€€€€d
≤
3training/Adam/gradients/cond_3/dropout/div_grad/NegNegcond_3/dropout/Shape/Switch:1*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_3/dropout/div
м
9training/Adam/gradients/cond_3/dropout/div_grad/RealDiv_1RealDiv3training/Adam/gradients/cond_3/dropout/div_grad/Negcond_3/dropout/keep_prob*
T0*%
_class
loc:@cond_3/dropout/div*'
_output_shapes
:€€€€€€€€€d
т
9training/Adam/gradients/cond_3/dropout/div_grad/RealDiv_2RealDiv9training/Adam/gradients/cond_3/dropout/div_grad/RealDiv_1cond_3/dropout/keep_prob*
T0*%
_class
loc:@cond_3/dropout/div*'
_output_shapes
:€€€€€€€€€d
З
3training/Adam/gradients/cond_3/dropout/div_grad/mulMul7training/Adam/gradients/cond_3/dropout/mul_grad/Reshape9training/Adam/gradients/cond_3/dropout/div_grad/RealDiv_2*
T0*%
_class
loc:@cond_3/dropout/div*'
_output_shapes
:€€€€€€€€€d
°
5training/Adam/gradients/cond_3/dropout/div_grad/Sum_1Sum3training/Adam/gradients/cond_3/dropout/div_grad/mulGtraining/Adam/gradients/cond_3/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_3/dropout/div
К
9training/Adam/gradients/cond_3/dropout/div_grad/Reshape_1Reshape5training/Adam/gradients/cond_3/dropout/div_grad/Sum_17training/Adam/gradients/cond_3/dropout/div_grad/Shape_1*
T0*%
_class
loc:@cond_3/dropout/div*
Tshape0*
_output_shapes
: 
Ґ
 training/Adam/gradients/Switch_3SwitchRelu_3cond_3/pred_id*
T0*
_class
loc:@Relu_3*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Э
"training/Adam/gradients/Identity_3Identity training/Adam/gradients/Switch_3*
T0*
_class
loc:@Relu_3*'
_output_shapes
:€€€€€€€€€d
Ъ
training/Adam/gradients/Shape_4Shape training/Adam/gradients/Switch_3*
T0*
_class
loc:@Relu_3*
out_type0*
_output_shapes
:
™
%training/Adam/gradients/zeros_3/ConstConst#^training/Adam/gradients/Identity_3*
_class
loc:@Relu_3*
valueB
 *    *
dtype0*
_output_shapes
: 
ќ
training/Adam/gradients/zeros_3Filltraining/Adam/gradients/Shape_4%training/Adam/gradients/zeros_3/Const*
T0*
_class
loc:@Relu_3*

index_type0*'
_output_shapes
:€€€€€€€€€d
э
Btraining/Adam/gradients/cond_3/dropout/Shape/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_37training/Adam/gradients/cond_3/dropout/div_grad/Reshape*
N*)
_output_shapes
:€€€€€€€€€d: *
T0*
_class
loc:@Relu_3
€
training/Adam/gradients/AddN_2AddN=training/Adam/gradients/cond_3/Identity/Switch_grad/cond_gradBtraining/Adam/gradients/cond_3/dropout/Shape/Switch_grad/cond_grad*
N*'
_output_shapes
:€€€€€€€€€d*
T0*
_class
loc:@Relu_3
≠
,training/Adam/gradients/Relu_3_grad/ReluGradReluGradtraining/Adam/gradients/AddN_2Relu_3*'
_output_shapes
:€€€€€€€€€d*
T0*
_class
loc:@Relu_3
…
2training/Adam/gradients/BiasAdd_3_grad/BiasAddGradBiasAddGrad,training/Adam/gradients/Relu_3_grad/ReluGrad*
T0*
_class
loc:@BiasAdd_3*
data_formatNHWC*
_output_shapes
:d
т
,training/Adam/gradients/MatMul_3_grad/MatMulMatMul,training/Adam/gradients/Relu_3_grad/ReluGradMatMul_3/ReadVariableOp*
T0*
_class
loc:@MatMul_3*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b(
а
.training/Adam/gradients/MatMul_3_grad/MatMul_1MatMulcond_2/Merge,training/Adam/gradients/Relu_3_grad/ReluGrad*
_output_shapes

:dd*
transpose_a(*
transpose_b( *
T0*
_class
loc:@MatMul_3
Ё
3training/Adam/gradients/cond_2/Merge_grad/cond_gradSwitch,training/Adam/gradients/MatMul_3_grad/MatMulcond_2/pred_id*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d*
T0*
_class
loc:@MatMul_3
Ѓ
5training/Adam/gradients/cond_2/dropout/mul_grad/ShapeShapecond_2/dropout/div*
T0*%
_class
loc:@cond_2/dropout/mul*
out_type0*
_output_shapes
:
≤
7training/Adam/gradients/cond_2/dropout/mul_grad/Shape_1Shapecond_2/dropout/Floor*
T0*%
_class
loc:@cond_2/dropout/mul*
out_type0*
_output_shapes
:
≤
Etraining/Adam/gradients/cond_2/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/cond_2/dropout/mul_grad/Shape7training/Adam/gradients/cond_2/dropout/mul_grad/Shape_1*
T0*%
_class
loc:@cond_2/dropout/mul*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
а
3training/Adam/gradients/cond_2/dropout/mul_grad/MulMul5training/Adam/gradients/cond_2/Merge_grad/cond_grad:1cond_2/dropout/Floor*
T0*%
_class
loc:@cond_2/dropout/mul*'
_output_shapes
:€€€€€€€€€d
Э
3training/Adam/gradients/cond_2/dropout/mul_grad/SumSum3training/Adam/gradients/cond_2/dropout/mul_grad/MulEtraining/Adam/gradients/cond_2/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_2/dropout/mul
Х
7training/Adam/gradients/cond_2/dropout/mul_grad/ReshapeReshape3training/Adam/gradients/cond_2/dropout/mul_grad/Sum5training/Adam/gradients/cond_2/dropout/mul_grad/Shape*
T0*%
_class
loc:@cond_2/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
а
5training/Adam/gradients/cond_2/dropout/mul_grad/Mul_1Mulcond_2/dropout/div5training/Adam/gradients/cond_2/Merge_grad/cond_grad:1*
T0*%
_class
loc:@cond_2/dropout/mul*'
_output_shapes
:€€€€€€€€€d
£
5training/Adam/gradients/cond_2/dropout/mul_grad/Sum_1Sum5training/Adam/gradients/cond_2/dropout/mul_grad/Mul_1Gtraining/Adam/gradients/cond_2/dropout/mul_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@cond_2/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Ы
9training/Adam/gradients/cond_2/dropout/mul_grad/Reshape_1Reshape5training/Adam/gradients/cond_2/dropout/mul_grad/Sum_17training/Adam/gradients/cond_2/dropout/mul_grad/Shape_1*
T0*%
_class
loc:@cond_2/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
Ґ
 training/Adam/gradients/Switch_4SwitchRelu_2cond_2/pred_id*
T0*
_class
loc:@Relu_2*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Я
"training/Adam/gradients/Identity_4Identity"training/Adam/gradients/Switch_4:1*
T0*
_class
loc:@Relu_2*'
_output_shapes
:€€€€€€€€€d
Ь
training/Adam/gradients/Shape_5Shape"training/Adam/gradients/Switch_4:1*
_output_shapes
:*
T0*
_class
loc:@Relu_2*
out_type0
™
%training/Adam/gradients/zeros_4/ConstConst#^training/Adam/gradients/Identity_4*
_class
loc:@Relu_2*
valueB
 *    *
dtype0*
_output_shapes
: 
ќ
training/Adam/gradients/zeros_4Filltraining/Adam/gradients/Shape_5%training/Adam/gradients/zeros_4/Const*
T0*
_class
loc:@Relu_2*

index_type0*'
_output_shapes
:€€€€€€€€€d
ф
=training/Adam/gradients/cond_2/Identity/Switch_grad/cond_gradMerge3training/Adam/gradients/cond_2/Merge_grad/cond_gradtraining/Adam/gradients/zeros_4*
T0*
_class
loc:@Relu_2*
N*)
_output_shapes
:€€€€€€€€€d: 
є
5training/Adam/gradients/cond_2/dropout/div_grad/ShapeShapecond_2/dropout/Shape/Switch:1*
T0*%
_class
loc:@cond_2/dropout/div*
out_type0*
_output_shapes
:
°
7training/Adam/gradients/cond_2/dropout/div_grad/Shape_1Const*
dtype0*
_output_shapes
: *%
_class
loc:@cond_2/dropout/div*
valueB 
≤
Etraining/Adam/gradients/cond_2/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/cond_2/dropout/div_grad/Shape7training/Adam/gradients/cond_2/dropout/div_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*%
_class
loc:@cond_2/dropout/div
о
7training/Adam/gradients/cond_2/dropout/div_grad/RealDivRealDiv7training/Adam/gradients/cond_2/dropout/mul_grad/Reshapecond_2/dropout/keep_prob*
T0*%
_class
loc:@cond_2/dropout/div*'
_output_shapes
:€€€€€€€€€d
°
3training/Adam/gradients/cond_2/dropout/div_grad/SumSum7training/Adam/gradients/cond_2/dropout/div_grad/RealDivEtraining/Adam/gradients/cond_2/dropout/div_grad/BroadcastGradientArgs*
T0*%
_class
loc:@cond_2/dropout/div*
_output_shapes
:*
	keep_dims( *

Tidx0
Х
7training/Adam/gradients/cond_2/dropout/div_grad/ReshapeReshape3training/Adam/gradients/cond_2/dropout/div_grad/Sum5training/Adam/gradients/cond_2/dropout/div_grad/Shape*
T0*%
_class
loc:@cond_2/dropout/div*
Tshape0*'
_output_shapes
:€€€€€€€€€d
≤
3training/Adam/gradients/cond_2/dropout/div_grad/NegNegcond_2/dropout/Shape/Switch:1*
T0*%
_class
loc:@cond_2/dropout/div*'
_output_shapes
:€€€€€€€€€d
м
9training/Adam/gradients/cond_2/dropout/div_grad/RealDiv_1RealDiv3training/Adam/gradients/cond_2/dropout/div_grad/Negcond_2/dropout/keep_prob*
T0*%
_class
loc:@cond_2/dropout/div*'
_output_shapes
:€€€€€€€€€d
т
9training/Adam/gradients/cond_2/dropout/div_grad/RealDiv_2RealDiv9training/Adam/gradients/cond_2/dropout/div_grad/RealDiv_1cond_2/dropout/keep_prob*
T0*%
_class
loc:@cond_2/dropout/div*'
_output_shapes
:€€€€€€€€€d
З
3training/Adam/gradients/cond_2/dropout/div_grad/mulMul7training/Adam/gradients/cond_2/dropout/mul_grad/Reshape9training/Adam/gradients/cond_2/dropout/div_grad/RealDiv_2*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_2/dropout/div
°
5training/Adam/gradients/cond_2/dropout/div_grad/Sum_1Sum3training/Adam/gradients/cond_2/dropout/div_grad/mulGtraining/Adam/gradients/cond_2/dropout/div_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@cond_2/dropout/div*
_output_shapes
:*
	keep_dims( *

Tidx0
К
9training/Adam/gradients/cond_2/dropout/div_grad/Reshape_1Reshape5training/Adam/gradients/cond_2/dropout/div_grad/Sum_17training/Adam/gradients/cond_2/dropout/div_grad/Shape_1*
_output_shapes
: *
T0*%
_class
loc:@cond_2/dropout/div*
Tshape0
Ґ
 training/Adam/gradients/Switch_5SwitchRelu_2cond_2/pred_id*
T0*
_class
loc:@Relu_2*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Э
"training/Adam/gradients/Identity_5Identity training/Adam/gradients/Switch_5*'
_output_shapes
:€€€€€€€€€d*
T0*
_class
loc:@Relu_2
Ъ
training/Adam/gradients/Shape_6Shape training/Adam/gradients/Switch_5*
_output_shapes
:*
T0*
_class
loc:@Relu_2*
out_type0
™
%training/Adam/gradients/zeros_5/ConstConst#^training/Adam/gradients/Identity_5*
dtype0*
_output_shapes
: *
_class
loc:@Relu_2*
valueB
 *    
ќ
training/Adam/gradients/zeros_5Filltraining/Adam/gradients/Shape_6%training/Adam/gradients/zeros_5/Const*
T0*
_class
loc:@Relu_2*

index_type0*'
_output_shapes
:€€€€€€€€€d
э
Btraining/Adam/gradients/cond_2/dropout/Shape/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_57training/Adam/gradients/cond_2/dropout/div_grad/Reshape*
T0*
_class
loc:@Relu_2*
N*)
_output_shapes
:€€€€€€€€€d: 
€
training/Adam/gradients/AddN_3AddN=training/Adam/gradients/cond_2/Identity/Switch_grad/cond_gradBtraining/Adam/gradients/cond_2/dropout/Shape/Switch_grad/cond_grad*
T0*
_class
loc:@Relu_2*
N*'
_output_shapes
:€€€€€€€€€d
≠
,training/Adam/gradients/Relu_2_grad/ReluGradReluGradtraining/Adam/gradients/AddN_3Relu_2*
T0*
_class
loc:@Relu_2*'
_output_shapes
:€€€€€€€€€d
…
2training/Adam/gradients/BiasAdd_2_grad/BiasAddGradBiasAddGrad,training/Adam/gradients/Relu_2_grad/ReluGrad*
T0*
_class
loc:@BiasAdd_2*
data_formatNHWC*
_output_shapes
:d
т
,training/Adam/gradients/MatMul_2_grad/MatMulMatMul,training/Adam/gradients/Relu_2_grad/ReluGradMatMul_2/ReadVariableOp*
T0*
_class
loc:@MatMul_2*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b(
а
.training/Adam/gradients/MatMul_2_grad/MatMul_1MatMulcond_1/Merge,training/Adam/gradients/Relu_2_grad/ReluGrad*
transpose_b( *
T0*
_class
loc:@MatMul_2*
_output_shapes

:dd*
transpose_a(
Ё
3training/Adam/gradients/cond_1/Merge_grad/cond_gradSwitch,training/Adam/gradients/MatMul_2_grad/MatMulcond_1/pred_id*
T0*
_class
loc:@MatMul_2*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Ѓ
5training/Adam/gradients/cond_1/dropout/mul_grad/ShapeShapecond_1/dropout/div*
T0*%
_class
loc:@cond_1/dropout/mul*
out_type0*
_output_shapes
:
≤
7training/Adam/gradients/cond_1/dropout/mul_grad/Shape_1Shapecond_1/dropout/Floor*
T0*%
_class
loc:@cond_1/dropout/mul*
out_type0*
_output_shapes
:
≤
Etraining/Adam/gradients/cond_1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/cond_1/dropout/mul_grad/Shape7training/Adam/gradients/cond_1/dropout/mul_grad/Shape_1*
T0*%
_class
loc:@cond_1/dropout/mul*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
а
3training/Adam/gradients/cond_1/dropout/mul_grad/MulMul5training/Adam/gradients/cond_1/Merge_grad/cond_grad:1cond_1/dropout/Floor*
T0*%
_class
loc:@cond_1/dropout/mul*'
_output_shapes
:€€€€€€€€€d
Э
3training/Adam/gradients/cond_1/dropout/mul_grad/SumSum3training/Adam/gradients/cond_1/dropout/mul_grad/MulEtraining/Adam/gradients/cond_1/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_1/dropout/mul
Х
7training/Adam/gradients/cond_1/dropout/mul_grad/ReshapeReshape3training/Adam/gradients/cond_1/dropout/mul_grad/Sum5training/Adam/gradients/cond_1/dropout/mul_grad/Shape*
T0*%
_class
loc:@cond_1/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
а
5training/Adam/gradients/cond_1/dropout/mul_grad/Mul_1Mulcond_1/dropout/div5training/Adam/gradients/cond_1/Merge_grad/cond_grad:1*
T0*%
_class
loc:@cond_1/dropout/mul*'
_output_shapes
:€€€€€€€€€d
£
5training/Adam/gradients/cond_1/dropout/mul_grad/Sum_1Sum5training/Adam/gradients/cond_1/dropout/mul_grad/Mul_1Gtraining/Adam/gradients/cond_1/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_1/dropout/mul
Ы
9training/Adam/gradients/cond_1/dropout/mul_grad/Reshape_1Reshape5training/Adam/gradients/cond_1/dropout/mul_grad/Sum_17training/Adam/gradients/cond_1/dropout/mul_grad/Shape_1*
T0*%
_class
loc:@cond_1/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
Ґ
 training/Adam/gradients/Switch_6SwitchRelu_1cond_1/pred_id*
T0*
_class
loc:@Relu_1*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Я
"training/Adam/gradients/Identity_6Identity"training/Adam/gradients/Switch_6:1*'
_output_shapes
:€€€€€€€€€d*
T0*
_class
loc:@Relu_1
Ь
training/Adam/gradients/Shape_7Shape"training/Adam/gradients/Switch_6:1*
_output_shapes
:*
T0*
_class
loc:@Relu_1*
out_type0
™
%training/Adam/gradients/zeros_6/ConstConst#^training/Adam/gradients/Identity_6*
_class
loc:@Relu_1*
valueB
 *    *
dtype0*
_output_shapes
: 
ќ
training/Adam/gradients/zeros_6Filltraining/Adam/gradients/Shape_7%training/Adam/gradients/zeros_6/Const*
T0*
_class
loc:@Relu_1*

index_type0*'
_output_shapes
:€€€€€€€€€d
ф
=training/Adam/gradients/cond_1/Identity/Switch_grad/cond_gradMerge3training/Adam/gradients/cond_1/Merge_grad/cond_gradtraining/Adam/gradients/zeros_6*
T0*
_class
loc:@Relu_1*
N*)
_output_shapes
:€€€€€€€€€d: 
є
5training/Adam/gradients/cond_1/dropout/div_grad/ShapeShapecond_1/dropout/Shape/Switch:1*
_output_shapes
:*
T0*%
_class
loc:@cond_1/dropout/div*
out_type0
°
7training/Adam/gradients/cond_1/dropout/div_grad/Shape_1Const*%
_class
loc:@cond_1/dropout/div*
valueB *
dtype0*
_output_shapes
: 
≤
Etraining/Adam/gradients/cond_1/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/cond_1/dropout/div_grad/Shape7training/Adam/gradients/cond_1/dropout/div_grad/Shape_1*
T0*%
_class
loc:@cond_1/dropout/div*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
о
7training/Adam/gradients/cond_1/dropout/div_grad/RealDivRealDiv7training/Adam/gradients/cond_1/dropout/mul_grad/Reshapecond_1/dropout/keep_prob*
T0*%
_class
loc:@cond_1/dropout/div*'
_output_shapes
:€€€€€€€€€d
°
3training/Adam/gradients/cond_1/dropout/div_grad/SumSum7training/Adam/gradients/cond_1/dropout/div_grad/RealDivEtraining/Adam/gradients/cond_1/dropout/div_grad/BroadcastGradientArgs*
T0*%
_class
loc:@cond_1/dropout/div*
_output_shapes
:*
	keep_dims( *

Tidx0
Х
7training/Adam/gradients/cond_1/dropout/div_grad/ReshapeReshape3training/Adam/gradients/cond_1/dropout/div_grad/Sum5training/Adam/gradients/cond_1/dropout/div_grad/Shape*
T0*%
_class
loc:@cond_1/dropout/div*
Tshape0*'
_output_shapes
:€€€€€€€€€d
≤
3training/Adam/gradients/cond_1/dropout/div_grad/NegNegcond_1/dropout/Shape/Switch:1*
T0*%
_class
loc:@cond_1/dropout/div*'
_output_shapes
:€€€€€€€€€d
м
9training/Adam/gradients/cond_1/dropout/div_grad/RealDiv_1RealDiv3training/Adam/gradients/cond_1/dropout/div_grad/Negcond_1/dropout/keep_prob*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_1/dropout/div
т
9training/Adam/gradients/cond_1/dropout/div_grad/RealDiv_2RealDiv9training/Adam/gradients/cond_1/dropout/div_grad/RealDiv_1cond_1/dropout/keep_prob*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_1/dropout/div
З
3training/Adam/gradients/cond_1/dropout/div_grad/mulMul7training/Adam/gradients/cond_1/dropout/mul_grad/Reshape9training/Adam/gradients/cond_1/dropout/div_grad/RealDiv_2*'
_output_shapes
:€€€€€€€€€d*
T0*%
_class
loc:@cond_1/dropout/div
°
5training/Adam/gradients/cond_1/dropout/div_grad/Sum_1Sum3training/Adam/gradients/cond_1/dropout/div_grad/mulGtraining/Adam/gradients/cond_1/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@cond_1/dropout/div
К
9training/Adam/gradients/cond_1/dropout/div_grad/Reshape_1Reshape5training/Adam/gradients/cond_1/dropout/div_grad/Sum_17training/Adam/gradients/cond_1/dropout/div_grad/Shape_1*
_output_shapes
: *
T0*%
_class
loc:@cond_1/dropout/div*
Tshape0
Ґ
 training/Adam/gradients/Switch_7SwitchRelu_1cond_1/pred_id*
T0*
_class
loc:@Relu_1*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Э
"training/Adam/gradients/Identity_7Identity training/Adam/gradients/Switch_7*'
_output_shapes
:€€€€€€€€€d*
T0*
_class
loc:@Relu_1
Ъ
training/Adam/gradients/Shape_8Shape training/Adam/gradients/Switch_7*
T0*
_class
loc:@Relu_1*
out_type0*
_output_shapes
:
™
%training/Adam/gradients/zeros_7/ConstConst#^training/Adam/gradients/Identity_7*
_class
loc:@Relu_1*
valueB
 *    *
dtype0*
_output_shapes
: 
ќ
training/Adam/gradients/zeros_7Filltraining/Adam/gradients/Shape_8%training/Adam/gradients/zeros_7/Const*
T0*
_class
loc:@Relu_1*

index_type0*'
_output_shapes
:€€€€€€€€€d
э
Btraining/Adam/gradients/cond_1/dropout/Shape/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_77training/Adam/gradients/cond_1/dropout/div_grad/Reshape*
T0*
_class
loc:@Relu_1*
N*)
_output_shapes
:€€€€€€€€€d: 
€
training/Adam/gradients/AddN_4AddN=training/Adam/gradients/cond_1/Identity/Switch_grad/cond_gradBtraining/Adam/gradients/cond_1/dropout/Shape/Switch_grad/cond_grad*
T0*
_class
loc:@Relu_1*
N*'
_output_shapes
:€€€€€€€€€d
≠
,training/Adam/gradients/Relu_1_grad/ReluGradReluGradtraining/Adam/gradients/AddN_4Relu_1*'
_output_shapes
:€€€€€€€€€d*
T0*
_class
loc:@Relu_1
…
2training/Adam/gradients/BiasAdd_1_grad/BiasAddGradBiasAddGrad,training/Adam/gradients/Relu_1_grad/ReluGrad*
T0*
_class
loc:@BiasAdd_1*
data_formatNHWC*
_output_shapes
:d
т
,training/Adam/gradients/MatMul_1_grad/MatMulMatMul,training/Adam/gradients/Relu_1_grad/ReluGradMatMul_1/ReadVariableOp*
T0*
_class
loc:@MatMul_1*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b(
ё
.training/Adam/gradients/MatMul_1_grad/MatMul_1MatMul
cond/Merge,training/Adam/gradients/Relu_1_grad/ReluGrad*
_output_shapes

:dd*
transpose_a(*
transpose_b( *
T0*
_class
loc:@MatMul_1
ў
1training/Adam/gradients/cond/Merge_grad/cond_gradSwitch,training/Adam/gradients/MatMul_1_grad/MatMulcond/pred_id*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d*
T0*
_class
loc:@MatMul_1
®
3training/Adam/gradients/cond/dropout/mul_grad/ShapeShapecond/dropout/div*
_output_shapes
:*
T0*#
_class
loc:@cond/dropout/mul*
out_type0
ђ
5training/Adam/gradients/cond/dropout/mul_grad/Shape_1Shapecond/dropout/Floor*
T0*#
_class
loc:@cond/dropout/mul*
out_type0*
_output_shapes
:
™
Ctraining/Adam/gradients/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs3training/Adam/gradients/cond/dropout/mul_grad/Shape5training/Adam/gradients/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*#
_class
loc:@cond/dropout/mul
Ў
1training/Adam/gradients/cond/dropout/mul_grad/MulMul3training/Adam/gradients/cond/Merge_grad/cond_grad:1cond/dropout/Floor*
T0*#
_class
loc:@cond/dropout/mul*'
_output_shapes
:€€€€€€€€€d
Х
1training/Adam/gradients/cond/dropout/mul_grad/SumSum1training/Adam/gradients/cond/dropout/mul_grad/MulCtraining/Adam/gradients/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*#
_class
loc:@cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Н
5training/Adam/gradients/cond/dropout/mul_grad/ReshapeReshape1training/Adam/gradients/cond/dropout/mul_grad/Sum3training/Adam/gradients/cond/dropout/mul_grad/Shape*
T0*#
_class
loc:@cond/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
Ў
3training/Adam/gradients/cond/dropout/mul_grad/Mul_1Mulcond/dropout/div3training/Adam/gradients/cond/Merge_grad/cond_grad:1*'
_output_shapes
:€€€€€€€€€d*
T0*#
_class
loc:@cond/dropout/mul
Ы
3training/Adam/gradients/cond/dropout/mul_grad/Sum_1Sum3training/Adam/gradients/cond/dropout/mul_grad/Mul_1Etraining/Adam/gradients/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*#
_class
loc:@cond/dropout/mul*
_output_shapes
:
У
7training/Adam/gradients/cond/dropout/mul_grad/Reshape_1Reshape3training/Adam/gradients/cond/dropout/mul_grad/Sum_15training/Adam/gradients/cond/dropout/mul_grad/Shape_1*
T0*#
_class
loc:@cond/dropout/mul*
Tshape0*'
_output_shapes
:€€€€€€€€€d
Ь
 training/Adam/gradients/Switch_8SwitchRelucond/pred_id*
T0*
_class
	loc:@Relu*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Э
"training/Adam/gradients/Identity_8Identity"training/Adam/gradients/Switch_8:1*
T0*
_class
	loc:@Relu*'
_output_shapes
:€€€€€€€€€d
Ъ
training/Adam/gradients/Shape_9Shape"training/Adam/gradients/Switch_8:1*
_output_shapes
:*
T0*
_class
	loc:@Relu*
out_type0
®
%training/Adam/gradients/zeros_8/ConstConst#^training/Adam/gradients/Identity_8*
dtype0*
_output_shapes
: *
_class
	loc:@Relu*
valueB
 *    
ћ
training/Adam/gradients/zeros_8Filltraining/Adam/gradients/Shape_9%training/Adam/gradients/zeros_8/Const*
T0*
_class
	loc:@Relu*

index_type0*'
_output_shapes
:€€€€€€€€€d
о
;training/Adam/gradients/cond/Identity/Switch_grad/cond_gradMerge1training/Adam/gradients/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_8*
T0*
_class
	loc:@Relu*
N*)
_output_shapes
:€€€€€€€€€d: 
≥
3training/Adam/gradients/cond/dropout/div_grad/ShapeShapecond/dropout/Shape/Switch:1*
_output_shapes
:*
T0*#
_class
loc:@cond/dropout/div*
out_type0
Э
5training/Adam/gradients/cond/dropout/div_grad/Shape_1Const*#
_class
loc:@cond/dropout/div*
valueB *
dtype0*
_output_shapes
: 
™
Ctraining/Adam/gradients/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs3training/Adam/gradients/cond/dropout/div_grad/Shape5training/Adam/gradients/cond/dropout/div_grad/Shape_1*
T0*#
_class
loc:@cond/dropout/div*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ж
5training/Adam/gradients/cond/dropout/div_grad/RealDivRealDiv5training/Adam/gradients/cond/dropout/mul_grad/Reshapecond/dropout/keep_prob*
T0*#
_class
loc:@cond/dropout/div*'
_output_shapes
:€€€€€€€€€d
Щ
1training/Adam/gradients/cond/dropout/div_grad/SumSum5training/Adam/gradients/cond/dropout/div_grad/RealDivCtraining/Adam/gradients/cond/dropout/div_grad/BroadcastGradientArgs*
T0*#
_class
loc:@cond/dropout/div*
_output_shapes
:*
	keep_dims( *

Tidx0
Н
5training/Adam/gradients/cond/dropout/div_grad/ReshapeReshape1training/Adam/gradients/cond/dropout/div_grad/Sum3training/Adam/gradients/cond/dropout/div_grad/Shape*
T0*#
_class
loc:@cond/dropout/div*
Tshape0*'
_output_shapes
:€€€€€€€€€d
ђ
1training/Adam/gradients/cond/dropout/div_grad/NegNegcond/dropout/Shape/Switch:1*
T0*#
_class
loc:@cond/dropout/div*'
_output_shapes
:€€€€€€€€€d
д
7training/Adam/gradients/cond/dropout/div_grad/RealDiv_1RealDiv1training/Adam/gradients/cond/dropout/div_grad/Negcond/dropout/keep_prob*'
_output_shapes
:€€€€€€€€€d*
T0*#
_class
loc:@cond/dropout/div
к
7training/Adam/gradients/cond/dropout/div_grad/RealDiv_2RealDiv7training/Adam/gradients/cond/dropout/div_grad/RealDiv_1cond/dropout/keep_prob*
T0*#
_class
loc:@cond/dropout/div*'
_output_shapes
:€€€€€€€€€d
€
1training/Adam/gradients/cond/dropout/div_grad/mulMul5training/Adam/gradients/cond/dropout/mul_grad/Reshape7training/Adam/gradients/cond/dropout/div_grad/RealDiv_2*'
_output_shapes
:€€€€€€€€€d*
T0*#
_class
loc:@cond/dropout/div
Щ
3training/Adam/gradients/cond/dropout/div_grad/Sum_1Sum1training/Adam/gradients/cond/dropout/div_grad/mulEtraining/Adam/gradients/cond/dropout/div_grad/BroadcastGradientArgs:1*
T0*#
_class
loc:@cond/dropout/div*
_output_shapes
:*
	keep_dims( *

Tidx0
В
7training/Adam/gradients/cond/dropout/div_grad/Reshape_1Reshape3training/Adam/gradients/cond/dropout/div_grad/Sum_15training/Adam/gradients/cond/dropout/div_grad/Shape_1*
T0*#
_class
loc:@cond/dropout/div*
Tshape0*
_output_shapes
: 
Ь
 training/Adam/gradients/Switch_9SwitchRelucond/pred_id*
T0*
_class
	loc:@Relu*:
_output_shapes(
&:€€€€€€€€€d:€€€€€€€€€d
Ы
"training/Adam/gradients/Identity_9Identity training/Adam/gradients/Switch_9*
T0*
_class
	loc:@Relu*'
_output_shapes
:€€€€€€€€€d
Щ
 training/Adam/gradients/Shape_10Shape training/Adam/gradients/Switch_9*
T0*
_class
	loc:@Relu*
out_type0*
_output_shapes
:
®
%training/Adam/gradients/zeros_9/ConstConst#^training/Adam/gradients/Identity_9*
_class
	loc:@Relu*
valueB
 *    *
dtype0*
_output_shapes
: 
Ќ
training/Adam/gradients/zeros_9Fill training/Adam/gradients/Shape_10%training/Adam/gradients/zeros_9/Const*'
_output_shapes
:€€€€€€€€€d*
T0*
_class
	loc:@Relu*

index_type0
ч
@training/Adam/gradients/cond/dropout/Shape/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_95training/Adam/gradients/cond/dropout/div_grad/Reshape*
N*)
_output_shapes
:€€€€€€€€€d: *
T0*
_class
	loc:@Relu
щ
training/Adam/gradients/AddN_5AddN;training/Adam/gradients/cond/Identity/Switch_grad/cond_grad@training/Adam/gradients/cond/dropout/Shape/Switch_grad/cond_grad*
T0*
_class
	loc:@Relu*
N*'
_output_shapes
:€€€€€€€€€d
І
*training/Adam/gradients/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_5Relu*
T0*
_class
	loc:@Relu*'
_output_shapes
:€€€€€€€€€d
√
0training/Adam/gradients/BiasAdd_grad/BiasAddGradBiasAddGrad*training/Adam/gradients/Relu_grad/ReluGrad*
T0*
_class
loc:@BiasAdd*
data_formatNHWC*
_output_shapes
:d
л
*training/Adam/gradients/MatMul_grad/MatMulMatMul*training/Adam/gradients/Relu_grad/ReluGradMatMul/ReadVariableOp*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(*
T0*
_class
loc:@MatMul
÷
,training/Adam/gradients/MatMul_grad/MatMul_1MatMulinput_1*training/Adam/gradients/Relu_grad/ReluGrad*
T0*
_class
loc:@MatMul*
_output_shapes
:	Аd*
transpose_a(*
transpose_b( 
U
training/Adam/ConstConst*
value	B	 R*
dtype0	*
_output_shapes
: 
k
!training/Adam/AssignAddVariableOpAssignAddVariableOpAdam/iterationstraining/Adam/Const*
dtype0	
И
training/Adam/ReadVariableOpReadVariableOpAdam/iterations"^training/Adam/AssignAddVariableOp*
dtype0	*
_output_shapes
: 
i
!training/Adam/Cast/ReadVariableOpReadVariableOpAdam/iterations*
dtype0	*
_output_shapes
: 
}
training/Adam/CastCast!training/Adam/Cast/ReadVariableOp*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
X
training/Adam/add/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
b
training/Adam/addAddtraining/Adam/Casttraining/Adam/add/y*
_output_shapes
: *
T0
d
 training/Adam/Pow/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
n
training/Adam/PowPow training/Adam/Pow/ReadVariableOptraining/Adam/add*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
T0*
_output_shapes
: 
Z
training/Adam/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *    
Z
training/Adam/Const_2Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_2*
T0*
_output_shapes
: 
Г
training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const_1*
T0*
_output_shapes
: 
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
T0*
_output_shapes
: 
f
"training/Adam/Pow_1/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
r
training/Adam/Pow_1Pow"training/Adam/Pow_1/ReadVariableOptraining/Adam/add*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
_output_shapes
: *
T0
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
_output_shapes
: *
T0
^
training/Adam/ReadVariableOp_1ReadVariableOpAdam/lr*
dtype0*
_output_shapes
: 
p
training/Adam/mulMultraining/Adam/ReadVariableOp_1training/Adam/truediv*
T0*
_output_shapes
: 
t
#training/Adam/zeros/shape_as_tensorConst*
valueB"   d   *
dtype0*
_output_shapes
:
^
training/Adam/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ч
training/Adam/zerosFill#training/Adam/zeros/shape_as_tensortraining/Adam/zeros/Const*
T0*

index_type0*
_output_shapes
:	Аd
≈
training/Adam/VariableVarHandleOp*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
	container *
shape:	Аd
}
7training/Adam/Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable*
_output_shapes
: 
Ц
training/Adam/Variable/AssignAssignVariableOptraining/Adam/Variabletraining/Adam/zeros*)
_class
loc:@training/Adam/Variable*
dtype0
≠
*training/Adam/Variable/Read/ReadVariableOpReadVariableOptraining/Adam/Variable*
dtype0*
_output_shapes
:	Аd*)
_class
loc:@training/Adam/Variable
b
training/Adam/zeros_1Const*
valueBd*    *
dtype0*
_output_shapes
:d
∆
training/Adam/Variable_1VarHandleOp*
dtype0*
_output_shapes
: *)
shared_nametraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
	container *
shape:d
Б
9training/Adam/Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_1*
_output_shapes
: 
Ю
training/Adam/Variable_1/AssignAssignVariableOptraining/Adam/Variable_1training/Adam/zeros_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0
Ѓ
,training/Adam/Variable_1/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
:d
v
%training/Adam/zeros_2/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"d   d   
`
training/Adam/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ь
training/Adam/zeros_2Fill%training/Adam/zeros_2/shape_as_tensortraining/Adam/zeros_2/Const*
_output_shapes

:dd*
T0*

index_type0
 
training/Adam/Variable_2VarHandleOp*
dtype0*
_output_shapes
: *)
shared_nametraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
	container *
shape
:dd
Б
9training/Adam/Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_2*
_output_shapes
: 
Ю
training/Adam/Variable_2/AssignAssignVariableOptraining/Adam/Variable_2training/Adam/zeros_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0
≤
,training/Adam/Variable_2/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*
_output_shapes

:dd
b
training/Adam/zeros_3Const*
valueBd*    *
dtype0*
_output_shapes
:d
∆
training/Adam/Variable_3VarHandleOp*)
shared_nametraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
	container *
shape:d*
dtype0*
_output_shapes
: 
Б
9training/Adam/Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_3*
_output_shapes
: 
Ю
training/Adam/Variable_3/AssignAssignVariableOptraining/Adam/Variable_3training/Adam/zeros_3*
dtype0*+
_class!
loc:@training/Adam/Variable_3
Ѓ
,training/Adam/Variable_3/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_3*
dtype0*
_output_shapes
:d*+
_class!
loc:@training/Adam/Variable_3
v
%training/Adam/zeros_4/shape_as_tensorConst*
valueB"d   d   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_4/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ь
training/Adam/zeros_4Fill%training/Adam/zeros_4/shape_as_tensortraining/Adam/zeros_4/Const*
_output_shapes

:dd*
T0*

index_type0
 
training/Adam/Variable_4VarHandleOp*
dtype0*
_output_shapes
: *)
shared_nametraining/Adam/Variable_4*+
_class!
loc:@training/Adam/Variable_4*
	container *
shape
:dd
Б
9training/Adam/Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_4*
_output_shapes
: 
Ю
training/Adam/Variable_4/AssignAssignVariableOptraining/Adam/Variable_4training/Adam/zeros_4*+
_class!
loc:@training/Adam/Variable_4*
dtype0
≤
,training/Adam/Variable_4/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_4*
dtype0*
_output_shapes

:dd*+
_class!
loc:@training/Adam/Variable_4
b
training/Adam/zeros_5Const*
valueBd*    *
dtype0*
_output_shapes
:d
∆
training/Adam/Variable_5VarHandleOp*)
shared_nametraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
	container *
shape:d*
dtype0*
_output_shapes
: 
Б
9training/Adam/Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_5*
_output_shapes
: 
Ю
training/Adam/Variable_5/AssignAssignVariableOptraining/Adam/Variable_5training/Adam/zeros_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0
Ѓ
,training/Adam/Variable_5/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0*
_output_shapes
:d
v
%training/Adam/zeros_6/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"d   d   
`
training/Adam/zeros_6/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ь
training/Adam/zeros_6Fill%training/Adam/zeros_6/shape_as_tensortraining/Adam/zeros_6/Const*
T0*

index_type0*
_output_shapes

:dd
 
training/Adam/Variable_6VarHandleOp*+
_class!
loc:@training/Adam/Variable_6*
	container *
shape
:dd*
dtype0*
_output_shapes
: *)
shared_nametraining/Adam/Variable_6
Б
9training/Adam/Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_6*
_output_shapes
: 
Ю
training/Adam/Variable_6/AssignAssignVariableOptraining/Adam/Variable_6training/Adam/zeros_6*
dtype0*+
_class!
loc:@training/Adam/Variable_6
≤
,training/Adam/Variable_6/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0*
_output_shapes

:dd
b
training/Adam/zeros_7Const*
dtype0*
_output_shapes
:d*
valueBd*    
∆
training/Adam/Variable_7VarHandleOp*
dtype0*
_output_shapes
: *)
shared_nametraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
	container *
shape:d
Б
9training/Adam/Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_7*
_output_shapes
: 
Ю
training/Adam/Variable_7/AssignAssignVariableOptraining/Adam/Variable_7training/Adam/zeros_7*+
_class!
loc:@training/Adam/Variable_7*
dtype0
Ѓ
,training/Adam/Variable_7/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
dtype0*
_output_shapes
:d
v
%training/Adam/zeros_8/shape_as_tensorConst*
valueB"d   d   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_8/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ь
training/Adam/zeros_8Fill%training/Adam/zeros_8/shape_as_tensortraining/Adam/zeros_8/Const*
T0*

index_type0*
_output_shapes

:dd
 
training/Adam/Variable_8VarHandleOp*+
_class!
loc:@training/Adam/Variable_8*
	container *
shape
:dd*
dtype0*
_output_shapes
: *)
shared_nametraining/Adam/Variable_8
Б
9training/Adam/Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_8*
_output_shapes
: 
Ю
training/Adam/Variable_8/AssignAssignVariableOptraining/Adam/Variable_8training/Adam/zeros_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0
≤
,training/Adam/Variable_8/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0*
_output_shapes

:dd
b
training/Adam/zeros_9Const*
valueBd*    *
dtype0*
_output_shapes
:d
∆
training/Adam/Variable_9VarHandleOp*)
shared_nametraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
	container *
shape:d*
dtype0*
_output_shapes
: 
Б
9training/Adam/Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_9*
_output_shapes
: 
Ю
training/Adam/Variable_9/AssignAssignVariableOptraining/Adam/Variable_9training/Adam/zeros_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0
Ѓ
,training/Adam/Variable_9/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_9*
dtype0*
_output_shapes
:d*+
_class!
loc:@training/Adam/Variable_9
k
training/Adam/zeros_10Const*
valueBd*    *
dtype0*
_output_shapes

:d
Ќ
training/Adam/Variable_10VarHandleOp**
shared_nametraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*
	container *
shape
:d*
dtype0*
_output_shapes
: 
Г
:training/Adam/Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_10*
_output_shapes
: 
Ґ
 training/Adam/Variable_10/AssignAssignVariableOptraining/Adam/Variable_10training/Adam/zeros_10*,
_class"
 loc:@training/Adam/Variable_10*
dtype0
µ
-training/Adam/Variable_10/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*
dtype0*
_output_shapes

:d
c
training/Adam/zeros_11Const*
dtype0*
_output_shapes
:*
valueB*    
…
training/Adam/Variable_11VarHandleOp*
shape:*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
	container 
Г
:training/Adam/Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_11*
_output_shapes
: 
Ґ
 training/Adam/Variable_11/AssignAssignVariableOptraining/Adam/Variable_11training/Adam/zeros_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0
±
-training/Adam/Variable_11/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_11*
dtype0*
_output_shapes
:*,
_class"
 loc:@training/Adam/Variable_11
w
&training/Adam/zeros_12/shape_as_tensorConst*
valueB"   d   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_12/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
†
training/Adam/zeros_12Fill&training/Adam/zeros_12/shape_as_tensortraining/Adam/zeros_12/Const*
T0*

index_type0*
_output_shapes
:	Аd
ќ
training/Adam/Variable_12VarHandleOp**
shared_nametraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
	container *
shape:	Аd*
dtype0*
_output_shapes
: 
Г
:training/Adam/Variable_12/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_12*
_output_shapes
: 
Ґ
 training/Adam/Variable_12/AssignAssignVariableOptraining/Adam/Variable_12training/Adam/zeros_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0
ґ
-training/Adam/Variable_12/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0*
_output_shapes
:	Аd
c
training/Adam/zeros_13Const*
dtype0*
_output_shapes
:d*
valueBd*    
…
training/Adam/Variable_13VarHandleOp*
	container *
shape:d*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13
Г
:training/Adam/Variable_13/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_13*
_output_shapes
: 
Ґ
 training/Adam/Variable_13/AssignAssignVariableOptraining/Adam/Variable_13training/Adam/zeros_13*
dtype0*,
_class"
 loc:@training/Adam/Variable_13
±
-training/Adam/Variable_13/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
dtype0*
_output_shapes
:d
w
&training/Adam/zeros_14/shape_as_tensorConst*
valueB"d   d   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_14/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Я
training/Adam/zeros_14Fill&training/Adam/zeros_14/shape_as_tensortraining/Adam/zeros_14/Const*
T0*

index_type0*
_output_shapes

:dd
Ќ
training/Adam/Variable_14VarHandleOp*
shape
:dd*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
	container 
Г
:training/Adam/Variable_14/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_14*
_output_shapes
: 
Ґ
 training/Adam/Variable_14/AssignAssignVariableOptraining/Adam/Variable_14training/Adam/zeros_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0
µ
-training/Adam/Variable_14/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*
_output_shapes

:dd
c
training/Adam/zeros_15Const*
valueBd*    *
dtype0*
_output_shapes
:d
…
training/Adam/Variable_15VarHandleOp*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
	container *
shape:d
Г
:training/Adam/Variable_15/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_15*
_output_shapes
: 
Ґ
 training/Adam/Variable_15/AssignAssignVariableOptraining/Adam/Variable_15training/Adam/zeros_15*
dtype0*,
_class"
 loc:@training/Adam/Variable_15
±
-training/Adam/Variable_15/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_15*
dtype0*
_output_shapes
:d*,
_class"
 loc:@training/Adam/Variable_15
w
&training/Adam/zeros_16/shape_as_tensorConst*
valueB"d   d   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Я
training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*
T0*

index_type0*
_output_shapes

:dd
Ќ
training/Adam/Variable_16VarHandleOp*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
	container *
shape
:dd
Г
:training/Adam/Variable_16/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_16*
_output_shapes
: 
Ґ
 training/Adam/Variable_16/AssignAssignVariableOptraining/Adam/Variable_16training/Adam/zeros_16*
dtype0*,
_class"
 loc:@training/Adam/Variable_16
µ
-training/Adam/Variable_16/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
dtype0*
_output_shapes

:dd
c
training/Adam/zeros_17Const*
dtype0*
_output_shapes
:d*
valueBd*    
…
training/Adam/Variable_17VarHandleOp**
shared_nametraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
	container *
shape:d*
dtype0*
_output_shapes
: 
Г
:training/Adam/Variable_17/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_17*
_output_shapes
: 
Ґ
 training/Adam/Variable_17/AssignAssignVariableOptraining/Adam/Variable_17training/Adam/zeros_17*,
_class"
 loc:@training/Adam/Variable_17*
dtype0
±
-training/Adam/Variable_17/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
dtype0*
_output_shapes
:d
w
&training/Adam/zeros_18/shape_as_tensorConst*
valueB"d   d   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_18/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Я
training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*
T0*

index_type0*
_output_shapes

:dd
Ќ
training/Adam/Variable_18VarHandleOp**
shared_nametraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
	container *
shape
:dd*
dtype0*
_output_shapes
: 
Г
:training/Adam/Variable_18/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_18*
_output_shapes
: 
Ґ
 training/Adam/Variable_18/AssignAssignVariableOptraining/Adam/Variable_18training/Adam/zeros_18*,
_class"
 loc:@training/Adam/Variable_18*
dtype0
µ
-training/Adam/Variable_18/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
dtype0*
_output_shapes

:dd
c
training/Adam/zeros_19Const*
valueBd*    *
dtype0*
_output_shapes
:d
…
training/Adam/Variable_19VarHandleOp**
shared_nametraining/Adam/Variable_19*,
_class"
 loc:@training/Adam/Variable_19*
	container *
shape:d*
dtype0*
_output_shapes
: 
Г
:training/Adam/Variable_19/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_19*
_output_shapes
: 
Ґ
 training/Adam/Variable_19/AssignAssignVariableOptraining/Adam/Variable_19training/Adam/zeros_19*,
_class"
 loc:@training/Adam/Variable_19*
dtype0
±
-training/Adam/Variable_19/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_19*,
_class"
 loc:@training/Adam/Variable_19*
dtype0*
_output_shapes
:d
w
&training/Adam/zeros_20/shape_as_tensorConst*
valueB"d   d   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_20/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Я
training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*
T0*

index_type0*
_output_shapes

:dd
Ќ
training/Adam/Variable_20VarHandleOp*
shape
:dd*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20*
	container 
Г
:training/Adam/Variable_20/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_20*
_output_shapes
: 
Ґ
 training/Adam/Variable_20/AssignAssignVariableOptraining/Adam/Variable_20training/Adam/zeros_20*,
_class"
 loc:@training/Adam/Variable_20*
dtype0
µ
-training/Adam/Variable_20/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20*
dtype0*
_output_shapes

:dd
c
training/Adam/zeros_21Const*
valueBd*    *
dtype0*
_output_shapes
:d
…
training/Adam/Variable_21VarHandleOp*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
	container *
shape:d
Г
:training/Adam/Variable_21/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_21*
_output_shapes
: 
Ґ
 training/Adam/Variable_21/AssignAssignVariableOptraining/Adam/Variable_21training/Adam/zeros_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0
±
-training/Adam/Variable_21/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0*
_output_shapes
:d
k
training/Adam/zeros_22Const*
dtype0*
_output_shapes

:d*
valueBd*    
Ќ
training/Adam/Variable_22VarHandleOp*
	container *
shape
:d*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_22*,
_class"
 loc:@training/Adam/Variable_22
Г
:training/Adam/Variable_22/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_22*
_output_shapes
: 
Ґ
 training/Adam/Variable_22/AssignAssignVariableOptraining/Adam/Variable_22training/Adam/zeros_22*,
_class"
 loc:@training/Adam/Variable_22*
dtype0
µ
-training/Adam/Variable_22/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_22*
dtype0*
_output_shapes

:d*,
_class"
 loc:@training/Adam/Variable_22
c
training/Adam/zeros_23Const*
valueB*    *
dtype0*
_output_shapes
:
…
training/Adam/Variable_23VarHandleOp**
shared_nametraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
	container *
shape:*
dtype0*
_output_shapes
: 
Г
:training/Adam/Variable_23/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_23*
_output_shapes
: 
Ґ
 training/Adam/Variable_23/AssignAssignVariableOptraining/Adam/Variable_23training/Adam/zeros_23*
dtype0*,
_class"
 loc:@training/Adam/Variable_23
±
-training/Adam/Variable_23/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_24/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_24/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ы
training/Adam/zeros_24Fill&training/Adam/zeros_24/shape_as_tensortraining/Adam/zeros_24/Const*
_output_shapes
:*
T0*

index_type0
…
training/Adam/Variable_24VarHandleOp**
shared_nametraining/Adam/Variable_24*,
_class"
 loc:@training/Adam/Variable_24*
	container *
shape:*
dtype0*
_output_shapes
: 
Г
:training/Adam/Variable_24/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_24*
_output_shapes
: 
Ґ
 training/Adam/Variable_24/AssignAssignVariableOptraining/Adam/Variable_24training/Adam/zeros_24*,
_class"
 loc:@training/Adam/Variable_24*
dtype0
±
-training/Adam/Variable_24/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_24*,
_class"
 loc:@training/Adam/Variable_24*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_25/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_25/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_25Fill&training/Adam/zeros_25/shape_as_tensortraining/Adam/zeros_25/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_25VarHandleOp*,
_class"
 loc:@training/Adam/Variable_25*
	container *
shape:*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_25
Г
:training/Adam/Variable_25/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_25*
_output_shapes
: 
Ґ
 training/Adam/Variable_25/AssignAssignVariableOptraining/Adam/Variable_25training/Adam/zeros_25*
dtype0*,
_class"
 loc:@training/Adam/Variable_25
±
-training/Adam/Variable_25/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_25*,
_class"
 loc:@training/Adam/Variable_25*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_26/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_26/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_26Fill&training/Adam/zeros_26/shape_as_tensortraining/Adam/zeros_26/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_26VarHandleOp*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_26*,
_class"
 loc:@training/Adam/Variable_26*
	container *
shape:
Г
:training/Adam/Variable_26/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_26*
_output_shapes
: 
Ґ
 training/Adam/Variable_26/AssignAssignVariableOptraining/Adam/Variable_26training/Adam/zeros_26*,
_class"
 loc:@training/Adam/Variable_26*
dtype0
±
-training/Adam/Variable_26/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_26*,
_class"
 loc:@training/Adam/Variable_26*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_27/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_27/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ы
training/Adam/zeros_27Fill&training/Adam/zeros_27/shape_as_tensortraining/Adam/zeros_27/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_27VarHandleOp**
shared_nametraining/Adam/Variable_27*,
_class"
 loc:@training/Adam/Variable_27*
	container *
shape:*
dtype0*
_output_shapes
: 
Г
:training/Adam/Variable_27/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_27*
_output_shapes
: 
Ґ
 training/Adam/Variable_27/AssignAssignVariableOptraining/Adam/Variable_27training/Adam/zeros_27*
dtype0*,
_class"
 loc:@training/Adam/Variable_27
±
-training/Adam/Variable_27/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_27*,
_class"
 loc:@training/Adam/Variable_27*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_28/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_28/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_28Fill&training/Adam/zeros_28/shape_as_tensortraining/Adam/zeros_28/Const*
_output_shapes
:*
T0*

index_type0
…
training/Adam/Variable_28VarHandleOp*,
_class"
 loc:@training/Adam/Variable_28*
	container *
shape:*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_28
Г
:training/Adam/Variable_28/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_28*
_output_shapes
: 
Ґ
 training/Adam/Variable_28/AssignAssignVariableOptraining/Adam/Variable_28training/Adam/zeros_28*
dtype0*,
_class"
 loc:@training/Adam/Variable_28
±
-training/Adam/Variable_28/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_28*,
_class"
 loc:@training/Adam/Variable_28*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_29/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_29/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_29Fill&training/Adam/zeros_29/shape_as_tensortraining/Adam/zeros_29/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_29VarHandleOp*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_29*,
_class"
 loc:@training/Adam/Variable_29*
	container *
shape:
Г
:training/Adam/Variable_29/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_29*
_output_shapes
: 
Ґ
 training/Adam/Variable_29/AssignAssignVariableOptraining/Adam/Variable_29training/Adam/zeros_29*
dtype0*,
_class"
 loc:@training/Adam/Variable_29
±
-training/Adam/Variable_29/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_29*,
_class"
 loc:@training/Adam/Variable_29*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_30/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_30/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_30Fill&training/Adam/zeros_30/shape_as_tensortraining/Adam/zeros_30/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_30VarHandleOp*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_30*,
_class"
 loc:@training/Adam/Variable_30*
	container *
shape:
Г
:training/Adam/Variable_30/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_30*
_output_shapes
: 
Ґ
 training/Adam/Variable_30/AssignAssignVariableOptraining/Adam/Variable_30training/Adam/zeros_30*,
_class"
 loc:@training/Adam/Variable_30*
dtype0
±
-training/Adam/Variable_30/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_30*,
_class"
 loc:@training/Adam/Variable_30*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_31/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_31/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_31Fill&training/Adam/zeros_31/shape_as_tensortraining/Adam/zeros_31/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_31VarHandleOp**
shared_nametraining/Adam/Variable_31*,
_class"
 loc:@training/Adam/Variable_31*
	container *
shape:*
dtype0*
_output_shapes
: 
Г
:training/Adam/Variable_31/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_31*
_output_shapes
: 
Ґ
 training/Adam/Variable_31/AssignAssignVariableOptraining/Adam/Variable_31training/Adam/zeros_31*,
_class"
 loc:@training/Adam/Variable_31*
dtype0
±
-training/Adam/Variable_31/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_31*,
_class"
 loc:@training/Adam/Variable_31*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_32/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_32/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_32Fill&training/Adam/zeros_32/shape_as_tensortraining/Adam/zeros_32/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_32VarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_32*,
_class"
 loc:@training/Adam/Variable_32
Г
:training/Adam/Variable_32/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_32*
_output_shapes
: 
Ґ
 training/Adam/Variable_32/AssignAssignVariableOptraining/Adam/Variable_32training/Adam/zeros_32*
dtype0*,
_class"
 loc:@training/Adam/Variable_32
±
-training/Adam/Variable_32/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_32*,
_class"
 loc:@training/Adam/Variable_32*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_33/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_33/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_33Fill&training/Adam/zeros_33/shape_as_tensortraining/Adam/zeros_33/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_33VarHandleOp*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_33*,
_class"
 loc:@training/Adam/Variable_33*
	container *
shape:
Г
:training/Adam/Variable_33/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_33*
_output_shapes
: 
Ґ
 training/Adam/Variable_33/AssignAssignVariableOptraining/Adam/Variable_33training/Adam/zeros_33*,
_class"
 loc:@training/Adam/Variable_33*
dtype0
±
-training/Adam/Variable_33/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_33*
dtype0*
_output_shapes
:*,
_class"
 loc:@training/Adam/Variable_33
p
&training/Adam/zeros_34/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_34/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_34Fill&training/Adam/zeros_34/shape_as_tensortraining/Adam/zeros_34/Const*
T0*

index_type0*
_output_shapes
:
…
training/Adam/Variable_34VarHandleOp*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_34*,
_class"
 loc:@training/Adam/Variable_34*
	container *
shape:
Г
:training/Adam/Variable_34/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_34*
_output_shapes
: 
Ґ
 training/Adam/Variable_34/AssignAssignVariableOptraining/Adam/Variable_34training/Adam/zeros_34*,
_class"
 loc:@training/Adam/Variable_34*
dtype0
±
-training/Adam/Variable_34/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_34*,
_class"
 loc:@training/Adam/Variable_34*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_35/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_35/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_35Fill&training/Adam/zeros_35/shape_as_tensortraining/Adam/zeros_35/Const*
_output_shapes
:*
T0*

index_type0
…
training/Adam/Variable_35VarHandleOp*,
_class"
 loc:@training/Adam/Variable_35*
	container *
shape:*
dtype0*
_output_shapes
: **
shared_nametraining/Adam/Variable_35
Г
:training/Adam/Variable_35/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_35*
_output_shapes
: 
Ґ
 training/Adam/Variable_35/AssignAssignVariableOptraining/Adam/Variable_35training/Adam/zeros_35*,
_class"
 loc:@training/Adam/Variable_35*
dtype0
±
-training/Adam/Variable_35/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_35*,
_class"
 loc:@training/Adam/Variable_35*
dtype0*
_output_shapes
:
b
training/Adam/ReadVariableOp_2ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
z
"training/Adam/mul_1/ReadVariableOpReadVariableOptraining/Adam/Variable*
dtype0*
_output_shapes
:	Аd
И
training/Adam/mul_1Multraining/Adam/ReadVariableOp_2"training/Adam/mul_1/ReadVariableOp*
T0*
_output_shapes
:	Аd
b
training/Adam/ReadVariableOp_3ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_2/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
r
training/Adam/sub_2Subtraining/Adam/sub_2/xtraining/Adam/ReadVariableOp_3*
_output_shapes
: *
T0
З
training/Adam/mul_2Multraining/Adam/sub_2,training/Adam/gradients/MatMul_grad/MatMul_1*
_output_shapes
:	Аd*
T0
n
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
T0*
_output_shapes
:	Аd
b
training/Adam/ReadVariableOp_4ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
}
"training/Adam/mul_3/ReadVariableOpReadVariableOptraining/Adam/Variable_12*
dtype0*
_output_shapes
:	Аd
И
training/Adam/mul_3Multraining/Adam/ReadVariableOp_4"training/Adam/mul_3/ReadVariableOp*
T0*
_output_shapes
:	Аd
b
training/Adam/ReadVariableOp_5ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_3/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
r
training/Adam/sub_3Subtraining/Adam/sub_3/xtraining/Adam/ReadVariableOp_5*
T0*
_output_shapes
: 
v
training/Adam/SquareSquare,training/Adam/gradients/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	Аd
o
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*
_output_shapes
:	Аd
n
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
T0*
_output_shapes
:	Аd
l
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*
_output_shapes
:	Аd
Z
training/Adam/Const_3Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_4Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Ж
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_4*
_output_shapes
:	Аd*
T0
Р
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_3*
T0*
_output_shapes
:	Аd
e
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*
_output_shapes
:	Аd
Z
training/Adam/add_3/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
q
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*
_output_shapes
:	Аd
v
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
T0*
_output_shapes
:	Аd
o
training/Adam/ReadVariableOp_6ReadVariableOpdense_12/kernel*
dtype0*
_output_shapes
:	Аd
}
training/Adam/sub_4Subtraining/Adam/ReadVariableOp_6training/Adam/truediv_1*
_output_shapes
:	Аd*
T0
l
training/Adam/AssignVariableOpAssignVariableOptraining/Adam/Variabletraining/Adam/add_1*
dtype0
Ч
training/Adam/ReadVariableOp_7ReadVariableOptraining/Adam/Variable^training/Adam/AssignVariableOp*
dtype0*
_output_shapes
:	Аd
q
 training/Adam/AssignVariableOp_1AssignVariableOptraining/Adam/Variable_12training/Adam/add_2*
dtype0
Ь
training/Adam/ReadVariableOp_8ReadVariableOptraining/Adam/Variable_12!^training/Adam/AssignVariableOp_1*
dtype0*
_output_shapes
:	Аd
g
 training/Adam/AssignVariableOp_2AssignVariableOpdense_12/kerneltraining/Adam/sub_4*
dtype0
Т
training/Adam/ReadVariableOp_9ReadVariableOpdense_12/kernel!^training/Adam/AssignVariableOp_2*
dtype0*
_output_shapes
:	Аd
c
training/Adam/ReadVariableOp_10ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
w
"training/Adam/mul_6/ReadVariableOpReadVariableOptraining/Adam/Variable_1*
dtype0*
_output_shapes
:d
Д
training/Adam/mul_6Multraining/Adam/ReadVariableOp_10"training/Adam/mul_6/ReadVariableOp*
_output_shapes
:d*
T0
c
training/Adam/ReadVariableOp_11ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_5/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_5Subtraining/Adam/sub_5/xtraining/Adam/ReadVariableOp_11*
T0*
_output_shapes
: 
Ж
training/Adam/mul_7Multraining/Adam/sub_50training/Adam/gradients/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:d
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_12ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
x
"training/Adam/mul_8/ReadVariableOpReadVariableOptraining/Adam/Variable_13*
dtype0*
_output_shapes
:d
Д
training/Adam/mul_8Multraining/Adam/ReadVariableOp_12"training/Adam/mul_8/ReadVariableOp*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_13ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_6/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_6Subtraining/Adam/sub_6/xtraining/Adam/ReadVariableOp_13*
T0*
_output_shapes
: 
w
training/Adam/Square_1Square0training/Adam/gradients/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:d
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes
:d
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes
:d
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes
:d
Z
training/Adam/Const_5Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_6Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Б
%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_6*
T0*
_output_shapes
:d
Л
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_5*
T0*
_output_shapes
:d
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
_output_shapes
:d*
T0
Z
training/Adam/add_6/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
T0*
_output_shapes
:d
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes
:d
i
training/Adam/ReadVariableOp_14ReadVariableOpdense_12/bias*
dtype0*
_output_shapes
:d
y
training/Adam/sub_7Subtraining/Adam/ReadVariableOp_14training/Adam/truediv_2*
T0*
_output_shapes
:d
p
 training/Adam/AssignVariableOp_3AssignVariableOptraining/Adam/Variable_1training/Adam/add_4*
dtype0
Ч
training/Adam/ReadVariableOp_15ReadVariableOptraining/Adam/Variable_1!^training/Adam/AssignVariableOp_3*
dtype0*
_output_shapes
:d
q
 training/Adam/AssignVariableOp_4AssignVariableOptraining/Adam/Variable_13training/Adam/add_5*
dtype0
Ш
training/Adam/ReadVariableOp_16ReadVariableOptraining/Adam/Variable_13!^training/Adam/AssignVariableOp_4*
dtype0*
_output_shapes
:d
e
 training/Adam/AssignVariableOp_5AssignVariableOpdense_12/biastraining/Adam/sub_7*
dtype0
М
training/Adam/ReadVariableOp_17ReadVariableOpdense_12/bias!^training/Adam/AssignVariableOp_5*
dtype0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_18ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
|
#training/Adam/mul_11/ReadVariableOpReadVariableOptraining/Adam/Variable_2*
dtype0*
_output_shapes

:dd
К
training/Adam/mul_11Multraining/Adam/ReadVariableOp_18#training/Adam/mul_11/ReadVariableOp*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_19ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_8/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_8Subtraining/Adam/sub_8/xtraining/Adam/ReadVariableOp_19*
T0*
_output_shapes
: 
Й
training/Adam/mul_12Multraining/Adam/sub_8.training/Adam/gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:dd*
T0
o
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
_output_shapes

:dd*
T0
c
training/Adam/ReadVariableOp_20ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
}
#training/Adam/mul_13/ReadVariableOpReadVariableOptraining/Adam/Variable_14*
dtype0*
_output_shapes

:dd
К
training/Adam/mul_13Multraining/Adam/ReadVariableOp_20#training/Adam/mul_13/ReadVariableOp*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_21ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_9/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_9Subtraining/Adam/sub_9/xtraining/Adam/ReadVariableOp_21*
T0*
_output_shapes
: 
y
training/Adam/Square_2Square.training/Adam/gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:dd*
T0
q
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*
_output_shapes

:dd
o
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*
_output_shapes

:dd
l
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
_output_shapes

:dd*
T0
Z
training/Adam/Const_7Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_8Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Е
%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_8*
T0*
_output_shapes

:dd
П
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_7*
T0*
_output_shapes

:dd
d
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*
_output_shapes

:dd
Z
training/Adam/add_9/yConst*
dtype0*
_output_shapes
: *
valueB
 *Хњ÷3
p
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
T0*
_output_shapes

:dd
v
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*
_output_shapes

:dd
o
training/Adam/ReadVariableOp_22ReadVariableOpdense_13/kernel*
dtype0*
_output_shapes

:dd
~
training/Adam/sub_10Subtraining/Adam/ReadVariableOp_22training/Adam/truediv_3*
_output_shapes

:dd*
T0
p
 training/Adam/AssignVariableOp_6AssignVariableOptraining/Adam/Variable_2training/Adam/add_7*
dtype0
Ы
training/Adam/ReadVariableOp_23ReadVariableOptraining/Adam/Variable_2!^training/Adam/AssignVariableOp_6*
dtype0*
_output_shapes

:dd
q
 training/Adam/AssignVariableOp_7AssignVariableOptraining/Adam/Variable_14training/Adam/add_8*
dtype0
Ь
training/Adam/ReadVariableOp_24ReadVariableOptraining/Adam/Variable_14!^training/Adam/AssignVariableOp_7*
dtype0*
_output_shapes

:dd
h
 training/Adam/AssignVariableOp_8AssignVariableOpdense_13/kerneltraining/Adam/sub_10*
dtype0
Т
training/Adam/ReadVariableOp_25ReadVariableOpdense_13/kernel!^training/Adam/AssignVariableOp_8*
dtype0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_26ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
x
#training/Adam/mul_16/ReadVariableOpReadVariableOptraining/Adam/Variable_3*
dtype0*
_output_shapes
:d
Ж
training/Adam/mul_16Multraining/Adam/ReadVariableOp_26#training/Adam/mul_16/ReadVariableOp*
_output_shapes
:d*
T0
c
training/Adam/ReadVariableOp_27ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_11/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_11Subtraining/Adam/sub_11/xtraining/Adam/ReadVariableOp_27*
T0*
_output_shapes
: 
К
training/Adam/mul_17Multraining/Adam/sub_112training/Adam/gradients/BiasAdd_1_grad/BiasAddGrad*
T0*
_output_shapes
:d
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_28ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_18/ReadVariableOpReadVariableOptraining/Adam/Variable_15*
dtype0*
_output_shapes
:d
Ж
training/Adam/mul_18Multraining/Adam/ReadVariableOp_28#training/Adam/mul_18/ReadVariableOp*
_output_shapes
:d*
T0
c
training/Adam/ReadVariableOp_29ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_12/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_12Subtraining/Adam/sub_12/xtraining/Adam/ReadVariableOp_29*
T0*
_output_shapes
: 
y
training/Adam/Square_3Square2training/Adam/gradients/BiasAdd_1_grad/BiasAddGrad*
T0*
_output_shapes
:d
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes
:d
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
_output_shapes
:d*
T0
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes
:d
Z
training/Adam/Const_9Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_10Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Г
%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_10*
T0*
_output_shapes
:d
Л
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_9*
_output_shapes
:d*
T0
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes
:d
[
training/Adam/add_12/yConst*
dtype0*
_output_shapes
: *
valueB
 *Хњ÷3
n
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
T0*
_output_shapes
:d
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
T0*
_output_shapes
:d
i
training/Adam/ReadVariableOp_30ReadVariableOpdense_13/bias*
dtype0*
_output_shapes
:d
z
training/Adam/sub_13Subtraining/Adam/ReadVariableOp_30training/Adam/truediv_4*
T0*
_output_shapes
:d
q
 training/Adam/AssignVariableOp_9AssignVariableOptraining/Adam/Variable_3training/Adam/add_10*
dtype0
Ч
training/Adam/ReadVariableOp_31ReadVariableOptraining/Adam/Variable_3!^training/Adam/AssignVariableOp_9*
dtype0*
_output_shapes
:d
s
!training/Adam/AssignVariableOp_10AssignVariableOptraining/Adam/Variable_15training/Adam/add_11*
dtype0
Щ
training/Adam/ReadVariableOp_32ReadVariableOptraining/Adam/Variable_15"^training/Adam/AssignVariableOp_10*
dtype0*
_output_shapes
:d
g
!training/Adam/AssignVariableOp_11AssignVariableOpdense_13/biastraining/Adam/sub_13*
dtype0
Н
training/Adam/ReadVariableOp_33ReadVariableOpdense_13/bias"^training/Adam/AssignVariableOp_11*
dtype0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_34ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
|
#training/Adam/mul_21/ReadVariableOpReadVariableOptraining/Adam/Variable_4*
dtype0*
_output_shapes

:dd
К
training/Adam/mul_21Multraining/Adam/ReadVariableOp_34#training/Adam/mul_21/ReadVariableOp*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_35ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_14/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_14Subtraining/Adam/sub_14/xtraining/Adam/ReadVariableOp_35*
T0*
_output_shapes
: 
К
training/Adam/mul_22Multraining/Adam/sub_14.training/Adam/gradients/MatMul_2_grad/MatMul_1*
_output_shapes

:dd*
T0
p
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
_output_shapes

:dd*
T0
c
training/Adam/ReadVariableOp_36ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
}
#training/Adam/mul_23/ReadVariableOpReadVariableOptraining/Adam/Variable_16*
dtype0*
_output_shapes

:dd
К
training/Adam/mul_23Multraining/Adam/ReadVariableOp_36#training/Adam/mul_23/ReadVariableOp*
_output_shapes

:dd*
T0
c
training/Adam/ReadVariableOp_37ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_15/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_15Subtraining/Adam/sub_15/xtraining/Adam/ReadVariableOp_37*
T0*
_output_shapes
: 
y
training/Adam/Square_4Square.training/Adam/gradients/MatMul_2_grad/MatMul_1*
T0*
_output_shapes

:dd
r
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
_output_shapes

:dd*
T0
p
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
T0*
_output_shapes

:dd
m
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
_output_shapes

:dd*
T0
[
training/Adam/Const_11Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_12Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
З
%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_12*
T0*
_output_shapes

:dd
Р
training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_11*
T0*
_output_shapes

:dd
d
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*
_output_shapes

:dd
[
training/Adam/add_15/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
r
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
T0*
_output_shapes

:dd
w
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
_output_shapes

:dd*
T0
o
training/Adam/ReadVariableOp_38ReadVariableOpdense_14/kernel*
dtype0*
_output_shapes

:dd
~
training/Adam/sub_16Subtraining/Adam/ReadVariableOp_38training/Adam/truediv_5*
_output_shapes

:dd*
T0
r
!training/Adam/AssignVariableOp_12AssignVariableOptraining/Adam/Variable_4training/Adam/add_13*
dtype0
Ь
training/Adam/ReadVariableOp_39ReadVariableOptraining/Adam/Variable_4"^training/Adam/AssignVariableOp_12*
dtype0*
_output_shapes

:dd
s
!training/Adam/AssignVariableOp_13AssignVariableOptraining/Adam/Variable_16training/Adam/add_14*
dtype0
Э
training/Adam/ReadVariableOp_40ReadVariableOptraining/Adam/Variable_16"^training/Adam/AssignVariableOp_13*
dtype0*
_output_shapes

:dd
i
!training/Adam/AssignVariableOp_14AssignVariableOpdense_14/kerneltraining/Adam/sub_16*
dtype0
У
training/Adam/ReadVariableOp_41ReadVariableOpdense_14/kernel"^training/Adam/AssignVariableOp_14*
dtype0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_42ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
x
#training/Adam/mul_26/ReadVariableOpReadVariableOptraining/Adam/Variable_5*
dtype0*
_output_shapes
:d
Ж
training/Adam/mul_26Multraining/Adam/ReadVariableOp_42#training/Adam/mul_26/ReadVariableOp*
_output_shapes
:d*
T0
c
training/Adam/ReadVariableOp_43ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_17/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_17Subtraining/Adam/sub_17/xtraining/Adam/ReadVariableOp_43*
_output_shapes
: *
T0
К
training/Adam/mul_27Multraining/Adam/sub_172training/Adam/gradients/BiasAdd_2_grad/BiasAddGrad*
T0*
_output_shapes
:d
l
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
_output_shapes
:d*
T0
c
training/Adam/ReadVariableOp_44ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_28/ReadVariableOpReadVariableOptraining/Adam/Variable_17*
dtype0*
_output_shapes
:d
Ж
training/Adam/mul_28Multraining/Adam/ReadVariableOp_44#training/Adam/mul_28/ReadVariableOp*
_output_shapes
:d*
T0
c
training/Adam/ReadVariableOp_45ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_18/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
u
training/Adam/sub_18Subtraining/Adam/sub_18/xtraining/Adam/ReadVariableOp_45*
T0*
_output_shapes
: 
y
training/Adam/Square_5Square2training/Adam/gradients/BiasAdd_2_grad/BiasAddGrad*
_output_shapes
:d*
T0
n
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
_output_shapes
:d*
T0
l
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
_output_shapes
:d*
T0
i
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
_output_shapes
:d*
T0
[
training/Adam/Const_13Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_14Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Г
%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_14*
T0*
_output_shapes
:d
М
training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_13*
_output_shapes
:d*
T0
`
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes
:d
[
training/Adam/add_18/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
n
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
T0*
_output_shapes
:d
s
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
T0*
_output_shapes
:d
i
training/Adam/ReadVariableOp_46ReadVariableOpdense_14/bias*
dtype0*
_output_shapes
:d
z
training/Adam/sub_19Subtraining/Adam/ReadVariableOp_46training/Adam/truediv_6*
T0*
_output_shapes
:d
r
!training/Adam/AssignVariableOp_15AssignVariableOptraining/Adam/Variable_5training/Adam/add_16*
dtype0
Ш
training/Adam/ReadVariableOp_47ReadVariableOptraining/Adam/Variable_5"^training/Adam/AssignVariableOp_15*
dtype0*
_output_shapes
:d
s
!training/Adam/AssignVariableOp_16AssignVariableOptraining/Adam/Variable_17training/Adam/add_17*
dtype0
Щ
training/Adam/ReadVariableOp_48ReadVariableOptraining/Adam/Variable_17"^training/Adam/AssignVariableOp_16*
dtype0*
_output_shapes
:d
g
!training/Adam/AssignVariableOp_17AssignVariableOpdense_14/biastraining/Adam/sub_19*
dtype0
Н
training/Adam/ReadVariableOp_49ReadVariableOpdense_14/bias"^training/Adam/AssignVariableOp_17*
dtype0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_50ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
|
#training/Adam/mul_31/ReadVariableOpReadVariableOptraining/Adam/Variable_6*
dtype0*
_output_shapes

:dd
К
training/Adam/mul_31Multraining/Adam/ReadVariableOp_50#training/Adam/mul_31/ReadVariableOp*
_output_shapes

:dd*
T0
c
training/Adam/ReadVariableOp_51ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_20/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_20Subtraining/Adam/sub_20/xtraining/Adam/ReadVariableOp_51*
T0*
_output_shapes
: 
К
training/Adam/mul_32Multraining/Adam/sub_20.training/Adam/gradients/MatMul_3_grad/MatMul_1*
T0*
_output_shapes

:dd
p
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_52ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
}
#training/Adam/mul_33/ReadVariableOpReadVariableOptraining/Adam/Variable_18*
dtype0*
_output_shapes

:dd
К
training/Adam/mul_33Multraining/Adam/ReadVariableOp_52#training/Adam/mul_33/ReadVariableOp*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_53ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_21/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
u
training/Adam/sub_21Subtraining/Adam/sub_21/xtraining/Adam/ReadVariableOp_53*
T0*
_output_shapes
: 
y
training/Adam/Square_6Square.training/Adam/gradients/MatMul_3_grad/MatMul_1*
T0*
_output_shapes

:dd
r
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*
_output_shapes

:dd
p
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
_output_shapes

:dd*
T0
m
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*
_output_shapes

:dd
[
training/Adam/Const_15Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_16Const*
dtype0*
_output_shapes
: *
valueB
 *  А
З
%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_16*
T0*
_output_shapes

:dd
Р
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_15*
T0*
_output_shapes

:dd
d
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
T0*
_output_shapes

:dd
[
training/Adam/add_21/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
r
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
_output_shapes

:dd*
T0
w
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*
_output_shapes

:dd
o
training/Adam/ReadVariableOp_54ReadVariableOpdense_15/kernel*
dtype0*
_output_shapes

:dd
~
training/Adam/sub_22Subtraining/Adam/ReadVariableOp_54training/Adam/truediv_7*
T0*
_output_shapes

:dd
r
!training/Adam/AssignVariableOp_18AssignVariableOptraining/Adam/Variable_6training/Adam/add_19*
dtype0
Ь
training/Adam/ReadVariableOp_55ReadVariableOptraining/Adam/Variable_6"^training/Adam/AssignVariableOp_18*
dtype0*
_output_shapes

:dd
s
!training/Adam/AssignVariableOp_19AssignVariableOptraining/Adam/Variable_18training/Adam/add_20*
dtype0
Э
training/Adam/ReadVariableOp_56ReadVariableOptraining/Adam/Variable_18"^training/Adam/AssignVariableOp_19*
dtype0*
_output_shapes

:dd
i
!training/Adam/AssignVariableOp_20AssignVariableOpdense_15/kerneltraining/Adam/sub_22*
dtype0
У
training/Adam/ReadVariableOp_57ReadVariableOpdense_15/kernel"^training/Adam/AssignVariableOp_20*
dtype0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_58ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
x
#training/Adam/mul_36/ReadVariableOpReadVariableOptraining/Adam/Variable_7*
dtype0*
_output_shapes
:d
Ж
training/Adam/mul_36Multraining/Adam/ReadVariableOp_58#training/Adam/mul_36/ReadVariableOp*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_59ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_23/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_23Subtraining/Adam/sub_23/xtraining/Adam/ReadVariableOp_59*
T0*
_output_shapes
: 
К
training/Adam/mul_37Multraining/Adam/sub_232training/Adam/gradients/BiasAdd_3_grad/BiasAddGrad*
T0*
_output_shapes
:d
l
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_60ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_38/ReadVariableOpReadVariableOptraining/Adam/Variable_19*
dtype0*
_output_shapes
:d
Ж
training/Adam/mul_38Multraining/Adam/ReadVariableOp_60#training/Adam/mul_38/ReadVariableOp*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_61ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_24/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_24Subtraining/Adam/sub_24/xtraining/Adam/ReadVariableOp_61*
_output_shapes
: *
T0
y
training/Adam/Square_7Square2training/Adam/gradients/BiasAdd_3_grad/BiasAddGrad*
T0*
_output_shapes
:d
n
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
T0*
_output_shapes
:d
l
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes
:d
i
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
T0*
_output_shapes
:d
[
training/Adam/Const_17Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_18Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Г
%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_18*
T0*
_output_shapes
:d
М
training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_17*
_output_shapes
:d*
T0
`
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
T0*
_output_shapes
:d
[
training/Adam/add_24/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
n
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
_output_shapes
:d*
T0
s
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
T0*
_output_shapes
:d
i
training/Adam/ReadVariableOp_62ReadVariableOpdense_15/bias*
dtype0*
_output_shapes
:d
z
training/Adam/sub_25Subtraining/Adam/ReadVariableOp_62training/Adam/truediv_8*
T0*
_output_shapes
:d
r
!training/Adam/AssignVariableOp_21AssignVariableOptraining/Adam/Variable_7training/Adam/add_22*
dtype0
Ш
training/Adam/ReadVariableOp_63ReadVariableOptraining/Adam/Variable_7"^training/Adam/AssignVariableOp_21*
dtype0*
_output_shapes
:d
s
!training/Adam/AssignVariableOp_22AssignVariableOptraining/Adam/Variable_19training/Adam/add_23*
dtype0
Щ
training/Adam/ReadVariableOp_64ReadVariableOptraining/Adam/Variable_19"^training/Adam/AssignVariableOp_22*
dtype0*
_output_shapes
:d
g
!training/Adam/AssignVariableOp_23AssignVariableOpdense_15/biastraining/Adam/sub_25*
dtype0
Н
training/Adam/ReadVariableOp_65ReadVariableOpdense_15/bias"^training/Adam/AssignVariableOp_23*
dtype0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_66ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
|
#training/Adam/mul_41/ReadVariableOpReadVariableOptraining/Adam/Variable_8*
dtype0*
_output_shapes

:dd
К
training/Adam/mul_41Multraining/Adam/ReadVariableOp_66#training/Adam/mul_41/ReadVariableOp*
_output_shapes

:dd*
T0
c
training/Adam/ReadVariableOp_67ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_26/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_26Subtraining/Adam/sub_26/xtraining/Adam/ReadVariableOp_67*
T0*
_output_shapes
: 
К
training/Adam/mul_42Multraining/Adam/sub_26.training/Adam/gradients/MatMul_4_grad/MatMul_1*
T0*
_output_shapes

:dd
p
training/Adam/add_25Addtraining/Adam/mul_41training/Adam/mul_42*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_68ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
}
#training/Adam/mul_43/ReadVariableOpReadVariableOptraining/Adam/Variable_20*
dtype0*
_output_shapes

:dd
К
training/Adam/mul_43Multraining/Adam/ReadVariableOp_68#training/Adam/mul_43/ReadVariableOp*
T0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_69ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_27/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_27Subtraining/Adam/sub_27/xtraining/Adam/ReadVariableOp_69*
T0*
_output_shapes
: 
y
training/Adam/Square_8Square.training/Adam/gradients/MatMul_4_grad/MatMul_1*
_output_shapes

:dd*
T0
r
training/Adam/mul_44Multraining/Adam/sub_27training/Adam/Square_8*
T0*
_output_shapes

:dd
p
training/Adam/add_26Addtraining/Adam/mul_43training/Adam/mul_44*
T0*
_output_shapes

:dd
m
training/Adam/mul_45Multraining/Adam/multraining/Adam/add_25*
T0*
_output_shapes

:dd
[
training/Adam/Const_19Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_20Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
З
%training/Adam/clip_by_value_9/MinimumMinimumtraining/Adam/add_26training/Adam/Const_20*
T0*
_output_shapes

:dd
Р
training/Adam/clip_by_value_9Maximum%training/Adam/clip_by_value_9/Minimumtraining/Adam/Const_19*
T0*
_output_shapes

:dd
d
training/Adam/Sqrt_9Sqrttraining/Adam/clip_by_value_9*
T0*
_output_shapes

:dd
[
training/Adam/add_27/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
r
training/Adam/add_27Addtraining/Adam/Sqrt_9training/Adam/add_27/y*
T0*
_output_shapes

:dd
w
training/Adam/truediv_9RealDivtraining/Adam/mul_45training/Adam/add_27*
T0*
_output_shapes

:dd
o
training/Adam/ReadVariableOp_70ReadVariableOpdense_16/kernel*
dtype0*
_output_shapes

:dd
~
training/Adam/sub_28Subtraining/Adam/ReadVariableOp_70training/Adam/truediv_9*
_output_shapes

:dd*
T0
r
!training/Adam/AssignVariableOp_24AssignVariableOptraining/Adam/Variable_8training/Adam/add_25*
dtype0
Ь
training/Adam/ReadVariableOp_71ReadVariableOptraining/Adam/Variable_8"^training/Adam/AssignVariableOp_24*
dtype0*
_output_shapes

:dd
s
!training/Adam/AssignVariableOp_25AssignVariableOptraining/Adam/Variable_20training/Adam/add_26*
dtype0
Э
training/Adam/ReadVariableOp_72ReadVariableOptraining/Adam/Variable_20"^training/Adam/AssignVariableOp_25*
dtype0*
_output_shapes

:dd
i
!training/Adam/AssignVariableOp_26AssignVariableOpdense_16/kerneltraining/Adam/sub_28*
dtype0
У
training/Adam/ReadVariableOp_73ReadVariableOpdense_16/kernel"^training/Adam/AssignVariableOp_26*
dtype0*
_output_shapes

:dd
c
training/Adam/ReadVariableOp_74ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
x
#training/Adam/mul_46/ReadVariableOpReadVariableOptraining/Adam/Variable_9*
dtype0*
_output_shapes
:d
Ж
training/Adam/mul_46Multraining/Adam/ReadVariableOp_74#training/Adam/mul_46/ReadVariableOp*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_75ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_29/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
u
training/Adam/sub_29Subtraining/Adam/sub_29/xtraining/Adam/ReadVariableOp_75*
T0*
_output_shapes
: 
К
training/Adam/mul_47Multraining/Adam/sub_292training/Adam/gradients/BiasAdd_4_grad/BiasAddGrad*
_output_shapes
:d*
T0
l
training/Adam/add_28Addtraining/Adam/mul_46training/Adam/mul_47*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_76ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_48/ReadVariableOpReadVariableOptraining/Adam/Variable_21*
dtype0*
_output_shapes
:d
Ж
training/Adam/mul_48Multraining/Adam/ReadVariableOp_76#training/Adam/mul_48/ReadVariableOp*
T0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_77ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_30/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_30Subtraining/Adam/sub_30/xtraining/Adam/ReadVariableOp_77*
T0*
_output_shapes
: 
y
training/Adam/Square_9Square2training/Adam/gradients/BiasAdd_4_grad/BiasAddGrad*
T0*
_output_shapes
:d
n
training/Adam/mul_49Multraining/Adam/sub_30training/Adam/Square_9*
_output_shapes
:d*
T0
l
training/Adam/add_29Addtraining/Adam/mul_48training/Adam/mul_49*
T0*
_output_shapes
:d
i
training/Adam/mul_50Multraining/Adam/multraining/Adam/add_28*
T0*
_output_shapes
:d
[
training/Adam/Const_21Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_22Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Д
&training/Adam/clip_by_value_10/MinimumMinimumtraining/Adam/add_29training/Adam/Const_22*
T0*
_output_shapes
:d
О
training/Adam/clip_by_value_10Maximum&training/Adam/clip_by_value_10/Minimumtraining/Adam/Const_21*
T0*
_output_shapes
:d
b
training/Adam/Sqrt_10Sqrttraining/Adam/clip_by_value_10*
_output_shapes
:d*
T0
[
training/Adam/add_30/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
o
training/Adam/add_30Addtraining/Adam/Sqrt_10training/Adam/add_30/y*
T0*
_output_shapes
:d
t
training/Adam/truediv_10RealDivtraining/Adam/mul_50training/Adam/add_30*
_output_shapes
:d*
T0
i
training/Adam/ReadVariableOp_78ReadVariableOpdense_16/bias*
dtype0*
_output_shapes
:d
{
training/Adam/sub_31Subtraining/Adam/ReadVariableOp_78training/Adam/truediv_10*
_output_shapes
:d*
T0
r
!training/Adam/AssignVariableOp_27AssignVariableOptraining/Adam/Variable_9training/Adam/add_28*
dtype0
Ш
training/Adam/ReadVariableOp_79ReadVariableOptraining/Adam/Variable_9"^training/Adam/AssignVariableOp_27*
dtype0*
_output_shapes
:d
s
!training/Adam/AssignVariableOp_28AssignVariableOptraining/Adam/Variable_21training/Adam/add_29*
dtype0
Щ
training/Adam/ReadVariableOp_80ReadVariableOptraining/Adam/Variable_21"^training/Adam/AssignVariableOp_28*
dtype0*
_output_shapes
:d
g
!training/Adam/AssignVariableOp_29AssignVariableOpdense_16/biastraining/Adam/sub_31*
dtype0
Н
training/Adam/ReadVariableOp_81ReadVariableOpdense_16/bias"^training/Adam/AssignVariableOp_29*
dtype0*
_output_shapes
:d
c
training/Adam/ReadVariableOp_82ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
}
#training/Adam/mul_51/ReadVariableOpReadVariableOptraining/Adam/Variable_10*
dtype0*
_output_shapes

:d
К
training/Adam/mul_51Multraining/Adam/ReadVariableOp_82#training/Adam/mul_51/ReadVariableOp*
T0*
_output_shapes

:d
c
training/Adam/ReadVariableOp_83ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_32/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_32Subtraining/Adam/sub_32/xtraining/Adam/ReadVariableOp_83*
T0*
_output_shapes
: 
К
training/Adam/mul_52Multraining/Adam/sub_32.training/Adam/gradients/MatMul_5_grad/MatMul_1*
T0*
_output_shapes

:d
p
training/Adam/add_31Addtraining/Adam/mul_51training/Adam/mul_52*
T0*
_output_shapes

:d
c
training/Adam/ReadVariableOp_84ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
}
#training/Adam/mul_53/ReadVariableOpReadVariableOptraining/Adam/Variable_22*
dtype0*
_output_shapes

:d
К
training/Adam/mul_53Multraining/Adam/ReadVariableOp_84#training/Adam/mul_53/ReadVariableOp*
T0*
_output_shapes

:d
c
training/Adam/ReadVariableOp_85ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_33/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_33Subtraining/Adam/sub_33/xtraining/Adam/ReadVariableOp_85*
T0*
_output_shapes
: 
z
training/Adam/Square_10Square.training/Adam/gradients/MatMul_5_grad/MatMul_1*
_output_shapes

:d*
T0
s
training/Adam/mul_54Multraining/Adam/sub_33training/Adam/Square_10*
T0*
_output_shapes

:d
p
training/Adam/add_32Addtraining/Adam/mul_53training/Adam/mul_54*
T0*
_output_shapes

:d
m
training/Adam/mul_55Multraining/Adam/multraining/Adam/add_31*
_output_shapes

:d*
T0
[
training/Adam/Const_23Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_24Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
И
&training/Adam/clip_by_value_11/MinimumMinimumtraining/Adam/add_32training/Adam/Const_24*
_output_shapes

:d*
T0
Т
training/Adam/clip_by_value_11Maximum&training/Adam/clip_by_value_11/Minimumtraining/Adam/Const_23*
_output_shapes

:d*
T0
f
training/Adam/Sqrt_11Sqrttraining/Adam/clip_by_value_11*
T0*
_output_shapes

:d
[
training/Adam/add_33/yConst*
dtype0*
_output_shapes
: *
valueB
 *Хњ÷3
s
training/Adam/add_33Addtraining/Adam/Sqrt_11training/Adam/add_33/y*
T0*
_output_shapes

:d
x
training/Adam/truediv_11RealDivtraining/Adam/mul_55training/Adam/add_33*
T0*
_output_shapes

:d
o
training/Adam/ReadVariableOp_86ReadVariableOpdense_17/kernel*
dtype0*
_output_shapes

:d

training/Adam/sub_34Subtraining/Adam/ReadVariableOp_86training/Adam/truediv_11*
T0*
_output_shapes

:d
s
!training/Adam/AssignVariableOp_30AssignVariableOptraining/Adam/Variable_10training/Adam/add_31*
dtype0
Э
training/Adam/ReadVariableOp_87ReadVariableOptraining/Adam/Variable_10"^training/Adam/AssignVariableOp_30*
dtype0*
_output_shapes

:d
s
!training/Adam/AssignVariableOp_31AssignVariableOptraining/Adam/Variable_22training/Adam/add_32*
dtype0
Э
training/Adam/ReadVariableOp_88ReadVariableOptraining/Adam/Variable_22"^training/Adam/AssignVariableOp_31*
dtype0*
_output_shapes

:d
i
!training/Adam/AssignVariableOp_32AssignVariableOpdense_17/kerneltraining/Adam/sub_34*
dtype0
У
training/Adam/ReadVariableOp_89ReadVariableOpdense_17/kernel"^training/Adam/AssignVariableOp_32*
dtype0*
_output_shapes

:d
c
training/Adam/ReadVariableOp_90ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_56/ReadVariableOpReadVariableOptraining/Adam/Variable_11*
dtype0*
_output_shapes
:
Ж
training/Adam/mul_56Multraining/Adam/ReadVariableOp_90#training/Adam/mul_56/ReadVariableOp*
T0*
_output_shapes
:
c
training/Adam/ReadVariableOp_91ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_35/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_35Subtraining/Adam/sub_35/xtraining/Adam/ReadVariableOp_91*
T0*
_output_shapes
: 
К
training/Adam/mul_57Multraining/Adam/sub_352training/Adam/gradients/BiasAdd_5_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/add_34Addtraining/Adam/mul_56training/Adam/mul_57*
T0*
_output_shapes
:
c
training/Adam/ReadVariableOp_92ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_58/ReadVariableOpReadVariableOptraining/Adam/Variable_23*
dtype0*
_output_shapes
:
Ж
training/Adam/mul_58Multraining/Adam/ReadVariableOp_92#training/Adam/mul_58/ReadVariableOp*
_output_shapes
:*
T0
c
training/Adam/ReadVariableOp_93ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_36/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_36Subtraining/Adam/sub_36/xtraining/Adam/ReadVariableOp_93*
T0*
_output_shapes
: 
z
training/Adam/Square_11Square2training/Adam/gradients/BiasAdd_5_grad/BiasAddGrad*
T0*
_output_shapes
:
o
training/Adam/mul_59Multraining/Adam/sub_36training/Adam/Square_11*
T0*
_output_shapes
:
l
training/Adam/add_35Addtraining/Adam/mul_58training/Adam/mul_59*
_output_shapes
:*
T0
i
training/Adam/mul_60Multraining/Adam/multraining/Adam/add_34*
T0*
_output_shapes
:
[
training/Adam/Const_25Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_26Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Д
&training/Adam/clip_by_value_12/MinimumMinimumtraining/Adam/add_35training/Adam/Const_26*
T0*
_output_shapes
:
О
training/Adam/clip_by_value_12Maximum&training/Adam/clip_by_value_12/Minimumtraining/Adam/Const_25*
T0*
_output_shapes
:
b
training/Adam/Sqrt_12Sqrttraining/Adam/clip_by_value_12*
_output_shapes
:*
T0
[
training/Adam/add_36/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
o
training/Adam/add_36Addtraining/Adam/Sqrt_12training/Adam/add_36/y*
T0*
_output_shapes
:
t
training/Adam/truediv_12RealDivtraining/Adam/mul_60training/Adam/add_36*
T0*
_output_shapes
:
i
training/Adam/ReadVariableOp_94ReadVariableOpdense_17/bias*
dtype0*
_output_shapes
:
{
training/Adam/sub_37Subtraining/Adam/ReadVariableOp_94training/Adam/truediv_12*
T0*
_output_shapes
:
s
!training/Adam/AssignVariableOp_33AssignVariableOptraining/Adam/Variable_11training/Adam/add_34*
dtype0
Щ
training/Adam/ReadVariableOp_95ReadVariableOptraining/Adam/Variable_11"^training/Adam/AssignVariableOp_33*
dtype0*
_output_shapes
:
s
!training/Adam/AssignVariableOp_34AssignVariableOptraining/Adam/Variable_23training/Adam/add_35*
dtype0
Щ
training/Adam/ReadVariableOp_96ReadVariableOptraining/Adam/Variable_23"^training/Adam/AssignVariableOp_34*
dtype0*
_output_shapes
:
g
!training/Adam/AssignVariableOp_35AssignVariableOpdense_17/biastraining/Adam/sub_37*
dtype0
Н
training/Adam/ReadVariableOp_97ReadVariableOpdense_17/bias"^training/Adam/AssignVariableOp_35*
dtype0*
_output_shapes
:
µ

training/group_depsNoOp	^loss/mul^metrics/acc/Mean^metrics/f1_score/Mean^training/Adam/ReadVariableOp ^training/Adam/ReadVariableOp_15 ^training/Adam/ReadVariableOp_16 ^training/Adam/ReadVariableOp_17 ^training/Adam/ReadVariableOp_23 ^training/Adam/ReadVariableOp_24 ^training/Adam/ReadVariableOp_25 ^training/Adam/ReadVariableOp_31 ^training/Adam/ReadVariableOp_32 ^training/Adam/ReadVariableOp_33 ^training/Adam/ReadVariableOp_39 ^training/Adam/ReadVariableOp_40 ^training/Adam/ReadVariableOp_41 ^training/Adam/ReadVariableOp_47 ^training/Adam/ReadVariableOp_48 ^training/Adam/ReadVariableOp_49 ^training/Adam/ReadVariableOp_55 ^training/Adam/ReadVariableOp_56 ^training/Adam/ReadVariableOp_57 ^training/Adam/ReadVariableOp_63 ^training/Adam/ReadVariableOp_64 ^training/Adam/ReadVariableOp_65^training/Adam/ReadVariableOp_7 ^training/Adam/ReadVariableOp_71 ^training/Adam/ReadVariableOp_72 ^training/Adam/ReadVariableOp_73 ^training/Adam/ReadVariableOp_79^training/Adam/ReadVariableOp_8 ^training/Adam/ReadVariableOp_80 ^training/Adam/ReadVariableOp_81 ^training/Adam/ReadVariableOp_87 ^training/Adam/ReadVariableOp_88 ^training/Adam/ReadVariableOp_89^training/Adam/ReadVariableOp_9 ^training/Adam/ReadVariableOp_95 ^training/Adam/ReadVariableOp_96 ^training/Adam/ReadVariableOp_97
H

group_depsNoOp	^loss/mul^metrics/acc/Mean^metrics/f1_score/Mean
Z
VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_4*
_output_shapes
: 
]
VarIsInitializedOp_1VarIsInitializedOptraining/Adam/Variable_35*
_output_shapes
: 
\
VarIsInitializedOp_2VarIsInitializedOptraining/Adam/Variable_3*
_output_shapes
: 
\
VarIsInitializedOp_3VarIsInitializedOptraining/Adam/Variable_9*
_output_shapes
: 
]
VarIsInitializedOp_4VarIsInitializedOptraining/Adam/Variable_16*
_output_shapes
: 
S
VarIsInitializedOp_5VarIsInitializedOpdense_16/kernel*
_output_shapes
: 
Z
VarIsInitializedOp_6VarIsInitializedOptraining/Adam/Variable*
_output_shapes
: 
]
VarIsInitializedOp_7VarIsInitializedOptraining/Adam/Variable_30*
_output_shapes
: 
]
VarIsInitializedOp_8VarIsInitializedOptraining/Adam/Variable_10*
_output_shapes
: 
Q
VarIsInitializedOp_9VarIsInitializedOpdense_17/bias*
_output_shapes
: 
R
VarIsInitializedOp_10VarIsInitializedOpdense_16/bias*
_output_shapes
: 
^
VarIsInitializedOp_11VarIsInitializedOptraining/Adam/Variable_24*
_output_shapes
: 
T
VarIsInitializedOp_12VarIsInitializedOpAdam/iterations*
_output_shapes
: 
L
VarIsInitializedOp_13VarIsInitializedOpAdam/lr*
_output_shapes
: 
^
VarIsInitializedOp_14VarIsInitializedOptraining/Adam/Variable_19*
_output_shapes
: 
P
VarIsInitializedOp_15VarIsInitializedOpAdam/beta_1*
_output_shapes
: 
P
VarIsInitializedOp_16VarIsInitializedOpAdam/beta_2*
_output_shapes
: 
^
VarIsInitializedOp_17VarIsInitializedOptraining/Adam/Variable_13*
_output_shapes
: 
^
VarIsInitializedOp_18VarIsInitializedOptraining/Adam/Variable_29*
_output_shapes
: 
O
VarIsInitializedOp_19VarIsInitializedOp
Adam/decay*
_output_shapes
: 
]
VarIsInitializedOp_20VarIsInitializedOptraining/Adam/Variable_1*
_output_shapes
: 
^
VarIsInitializedOp_21VarIsInitializedOptraining/Adam/Variable_25*
_output_shapes
: 
^
VarIsInitializedOp_22VarIsInitializedOptraining/Adam/Variable_34*
_output_shapes
: 
^
VarIsInitializedOp_23VarIsInitializedOptraining/Adam/Variable_14*
_output_shapes
: 
^
VarIsInitializedOp_24VarIsInitializedOptraining/Adam/Variable_21*
_output_shapes
: 
^
VarIsInitializedOp_25VarIsInitializedOptraining/Adam/Variable_12*
_output_shapes
: 
T
VarIsInitializedOp_26VarIsInitializedOpdense_13/kernel*
_output_shapes
: 
^
VarIsInitializedOp_27VarIsInitializedOptraining/Adam/Variable_11*
_output_shapes
: 
T
VarIsInitializedOp_28VarIsInitializedOpdense_14/kernel*
_output_shapes
: 
^
VarIsInitializedOp_29VarIsInitializedOptraining/Adam/Variable_17*
_output_shapes
: 
R
VarIsInitializedOp_30VarIsInitializedOpdense_14/bias*
_output_shapes
: 
T
VarIsInitializedOp_31VarIsInitializedOpdense_15/kernel*
_output_shapes
: 
]
VarIsInitializedOp_32VarIsInitializedOptraining/Adam/Variable_5*
_output_shapes
: 
]
VarIsInitializedOp_33VarIsInitializedOptraining/Adam/Variable_8*
_output_shapes
: 
^
VarIsInitializedOp_34VarIsInitializedOptraining/Adam/Variable_23*
_output_shapes
: 
]
VarIsInitializedOp_35VarIsInitializedOptraining/Adam/Variable_7*
_output_shapes
: 
T
VarIsInitializedOp_36VarIsInitializedOpdense_12/kernel*
_output_shapes
: 
R
VarIsInitializedOp_37VarIsInitializedOpdense_12/bias*
_output_shapes
: 
^
VarIsInitializedOp_38VarIsInitializedOptraining/Adam/Variable_31*
_output_shapes
: 
R
VarIsInitializedOp_39VarIsInitializedOpdense_15/bias*
_output_shapes
: 
R
VarIsInitializedOp_40VarIsInitializedOpdense_13/bias*
_output_shapes
: 
^
VarIsInitializedOp_41VarIsInitializedOptraining/Adam/Variable_18*
_output_shapes
: 
^
VarIsInitializedOp_42VarIsInitializedOptraining/Adam/Variable_28*
_output_shapes
: 
T
VarIsInitializedOp_43VarIsInitializedOpdense_17/kernel*
_output_shapes
: 
]
VarIsInitializedOp_44VarIsInitializedOptraining/Adam/Variable_2*
_output_shapes
: 
^
VarIsInitializedOp_45VarIsInitializedOptraining/Adam/Variable_27*
_output_shapes
: 
^
VarIsInitializedOp_46VarIsInitializedOptraining/Adam/Variable_15*
_output_shapes
: 
^
VarIsInitializedOp_47VarIsInitializedOptraining/Adam/Variable_32*
_output_shapes
: 
]
VarIsInitializedOp_48VarIsInitializedOptraining/Adam/Variable_6*
_output_shapes
: 
^
VarIsInitializedOp_49VarIsInitializedOptraining/Adam/Variable_33*
_output_shapes
: 
^
VarIsInitializedOp_50VarIsInitializedOptraining/Adam/Variable_20*
_output_shapes
: 
^
VarIsInitializedOp_51VarIsInitializedOptraining/Adam/Variable_26*
_output_shapes
: 
^
VarIsInitializedOp_52VarIsInitializedOptraining/Adam/Variable_22*
_output_shapes
: 
ф
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^dense_12/bias/Assign^dense_12/kernel/Assign^dense_13/bias/Assign^dense_13/kernel/Assign^dense_14/bias/Assign^dense_14/kernel/Assign^dense_15/bias/Assign^dense_15/kernel/Assign^dense_16/bias/Assign^dense_16/kernel/Assign^dense_17/bias/Assign^dense_17/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign!^training/Adam/Variable_24/Assign!^training/Adam/Variable_25/Assign!^training/Adam/Variable_26/Assign!^training/Adam/Variable_27/Assign!^training/Adam/Variable_28/Assign!^training/Adam/Variable_29/Assign ^training/Adam/Variable_3/Assign!^training/Adam/Variable_30/Assign!^training/Adam/Variable_31/Assign!^training/Adam/Variable_32/Assign!^training/Adam/Variable_33/Assign!^training/Adam/Variable_34/Assign!^training/Adam/Variable_35/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign""э8
trainable_variablesе8в8
Г
Adam/iterations:0Adam/iterations/Assign%Adam/iterations/Read/ReadVariableOp:0(2+Adam/iterations/Initializer/initial_value:08
c
	Adam/lr:0Adam/lr/AssignAdam/lr/Read/ReadVariableOp:0(2#Adam/lr/Initializer/initial_value:08
s
Adam/beta_1:0Adam/beta_1/Assign!Adam/beta_1/Read/ReadVariableOp:0(2'Adam/beta_1/Initializer/initial_value:08
s
Adam/beta_2:0Adam/beta_2/Assign!Adam/beta_2/Read/ReadVariableOp:0(2'Adam/beta_2/Initializer/initial_value:08
o
Adam/decay:0Adam/decay/Assign Adam/decay/Read/ReadVariableOp:0(2&Adam/decay/Initializer/initial_value:08
Д
dense_12/kernel:0dense_12/kernel/Assign%dense_12/kernel/Read/ReadVariableOp:0(2,dense_12/kernel/Initializer/random_uniform:08
s
dense_12/bias:0dense_12/bias/Assign#dense_12/bias/Read/ReadVariableOp:0(2!dense_12/bias/Initializer/zeros:08
Д
dense_13/kernel:0dense_13/kernel/Assign%dense_13/kernel/Read/ReadVariableOp:0(2,dense_13/kernel/Initializer/random_uniform:08
s
dense_13/bias:0dense_13/bias/Assign#dense_13/bias/Read/ReadVariableOp:0(2!dense_13/bias/Initializer/zeros:08
Д
dense_14/kernel:0dense_14/kernel/Assign%dense_14/kernel/Read/ReadVariableOp:0(2,dense_14/kernel/Initializer/random_uniform:08
s
dense_14/bias:0dense_14/bias/Assign#dense_14/bias/Read/ReadVariableOp:0(2!dense_14/bias/Initializer/zeros:08
Д
dense_15/kernel:0dense_15/kernel/Assign%dense_15/kernel/Read/ReadVariableOp:0(2,dense_15/kernel/Initializer/random_uniform:08
s
dense_15/bias:0dense_15/bias/Assign#dense_15/bias/Read/ReadVariableOp:0(2!dense_15/bias/Initializer/zeros:08
Д
dense_16/kernel:0dense_16/kernel/Assign%dense_16/kernel/Read/ReadVariableOp:0(2,dense_16/kernel/Initializer/random_uniform:08
s
dense_16/bias:0dense_16/bias/Assign#dense_16/bias/Read/ReadVariableOp:0(2!dense_16/bias/Initializer/zeros:08
Д
dense_17/kernel:0dense_17/kernel/Assign%dense_17/kernel/Read/ReadVariableOp:0(2,dense_17/kernel/Initializer/random_uniform:08
s
dense_17/bias:0dense_17/bias/Assign#dense_17/bias/Read/ReadVariableOp:0(2!dense_17/bias/Initializer/zeros:08
В
training/Adam/Variable:0training/Adam/Variable/Assign,training/Adam/Variable/Read/ReadVariableOp:0(2training/Adam/zeros:08
К
training/Adam/Variable_1:0training/Adam/Variable_1/Assign.training/Adam/Variable_1/Read/ReadVariableOp:0(2training/Adam/zeros_1:08
К
training/Adam/Variable_2:0training/Adam/Variable_2/Assign.training/Adam/Variable_2/Read/ReadVariableOp:0(2training/Adam/zeros_2:08
К
training/Adam/Variable_3:0training/Adam/Variable_3/Assign.training/Adam/Variable_3/Read/ReadVariableOp:0(2training/Adam/zeros_3:08
К
training/Adam/Variable_4:0training/Adam/Variable_4/Assign.training/Adam/Variable_4/Read/ReadVariableOp:0(2training/Adam/zeros_4:08
К
training/Adam/Variable_5:0training/Adam/Variable_5/Assign.training/Adam/Variable_5/Read/ReadVariableOp:0(2training/Adam/zeros_5:08
К
training/Adam/Variable_6:0training/Adam/Variable_6/Assign.training/Adam/Variable_6/Read/ReadVariableOp:0(2training/Adam/zeros_6:08
К
training/Adam/Variable_7:0training/Adam/Variable_7/Assign.training/Adam/Variable_7/Read/ReadVariableOp:0(2training/Adam/zeros_7:08
К
training/Adam/Variable_8:0training/Adam/Variable_8/Assign.training/Adam/Variable_8/Read/ReadVariableOp:0(2training/Adam/zeros_8:08
К
training/Adam/Variable_9:0training/Adam/Variable_9/Assign.training/Adam/Variable_9/Read/ReadVariableOp:0(2training/Adam/zeros_9:08
О
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign/training/Adam/Variable_10/Read/ReadVariableOp:0(2training/Adam/zeros_10:08
О
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign/training/Adam/Variable_11/Read/ReadVariableOp:0(2training/Adam/zeros_11:08
О
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign/training/Adam/Variable_12/Read/ReadVariableOp:0(2training/Adam/zeros_12:08
О
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign/training/Adam/Variable_13/Read/ReadVariableOp:0(2training/Adam/zeros_13:08
О
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign/training/Adam/Variable_14/Read/ReadVariableOp:0(2training/Adam/zeros_14:08
О
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign/training/Adam/Variable_15/Read/ReadVariableOp:0(2training/Adam/zeros_15:08
О
training/Adam/Variable_16:0 training/Adam/Variable_16/Assign/training/Adam/Variable_16/Read/ReadVariableOp:0(2training/Adam/zeros_16:08
О
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign/training/Adam/Variable_17/Read/ReadVariableOp:0(2training/Adam/zeros_17:08
О
training/Adam/Variable_18:0 training/Adam/Variable_18/Assign/training/Adam/Variable_18/Read/ReadVariableOp:0(2training/Adam/zeros_18:08
О
training/Adam/Variable_19:0 training/Adam/Variable_19/Assign/training/Adam/Variable_19/Read/ReadVariableOp:0(2training/Adam/zeros_19:08
О
training/Adam/Variable_20:0 training/Adam/Variable_20/Assign/training/Adam/Variable_20/Read/ReadVariableOp:0(2training/Adam/zeros_20:08
О
training/Adam/Variable_21:0 training/Adam/Variable_21/Assign/training/Adam/Variable_21/Read/ReadVariableOp:0(2training/Adam/zeros_21:08
О
training/Adam/Variable_22:0 training/Adam/Variable_22/Assign/training/Adam/Variable_22/Read/ReadVariableOp:0(2training/Adam/zeros_22:08
О
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign/training/Adam/Variable_23/Read/ReadVariableOp:0(2training/Adam/zeros_23:08
О
training/Adam/Variable_24:0 training/Adam/Variable_24/Assign/training/Adam/Variable_24/Read/ReadVariableOp:0(2training/Adam/zeros_24:08
О
training/Adam/Variable_25:0 training/Adam/Variable_25/Assign/training/Adam/Variable_25/Read/ReadVariableOp:0(2training/Adam/zeros_25:08
О
training/Adam/Variable_26:0 training/Adam/Variable_26/Assign/training/Adam/Variable_26/Read/ReadVariableOp:0(2training/Adam/zeros_26:08
О
training/Adam/Variable_27:0 training/Adam/Variable_27/Assign/training/Adam/Variable_27/Read/ReadVariableOp:0(2training/Adam/zeros_27:08
О
training/Adam/Variable_28:0 training/Adam/Variable_28/Assign/training/Adam/Variable_28/Read/ReadVariableOp:0(2training/Adam/zeros_28:08
О
training/Adam/Variable_29:0 training/Adam/Variable_29/Assign/training/Adam/Variable_29/Read/ReadVariableOp:0(2training/Adam/zeros_29:08
О
training/Adam/Variable_30:0 training/Adam/Variable_30/Assign/training/Adam/Variable_30/Read/ReadVariableOp:0(2training/Adam/zeros_30:08
О
training/Adam/Variable_31:0 training/Adam/Variable_31/Assign/training/Adam/Variable_31/Read/ReadVariableOp:0(2training/Adam/zeros_31:08
О
training/Adam/Variable_32:0 training/Adam/Variable_32/Assign/training/Adam/Variable_32/Read/ReadVariableOp:0(2training/Adam/zeros_32:08
О
training/Adam/Variable_33:0 training/Adam/Variable_33/Assign/training/Adam/Variable_33/Read/ReadVariableOp:0(2training/Adam/zeros_33:08
О
training/Adam/Variable_34:0 training/Adam/Variable_34/Assign/training/Adam/Variable_34/Read/ReadVariableOp:0(2training/Adam/zeros_34:08
О
training/Adam/Variable_35:0 training/Adam/Variable_35/Assign/training/Adam/Variable_35/Read/ReadVariableOp:0(2training/Adam/zeros_35:08"ФШ
cond_contextВШюЧ
Я
cond/cond_textcond/pred_id:0cond/switch_t:0 *й
Relu:0
cond/dropout/Floor:0
cond/dropout/Shape/Switch:1
cond/dropout/Shape:0
cond/dropout/add:0
cond/dropout/div:0
cond/dropout/keep_prob:0
cond/dropout/mul:0
+cond/dropout/random_uniform/RandomUniform:0
!cond/dropout/random_uniform/max:0
!cond/dropout/random_uniform/min:0
!cond/dropout/random_uniform/mul:0
!cond/dropout/random_uniform/sub:0
cond/dropout/random_uniform:0
cond/pred_id:0
cond/switch_t:0 
cond/pred_id:0cond/pred_id:0%
Relu:0cond/dropout/Shape/Switch:1
ћ
cond/cond_text_1cond/pred_id:0cond/switch_f:0*Ц
Relu:0
cond/Identity/Switch:0
cond/Identity:0
cond/pred_id:0
cond/switch_f:0 
cond/pred_id:0cond/pred_id:0 
Relu:0cond/Identity/Switch:0
Ќ
cond_1/cond_textcond_1/pred_id:0cond_1/switch_t:0 *С
Relu_1:0
cond_1/dropout/Floor:0
cond_1/dropout/Shape/Switch:1
cond_1/dropout/Shape:0
cond_1/dropout/add:0
cond_1/dropout/div:0
cond_1/dropout/keep_prob:0
cond_1/dropout/mul:0
-cond_1/dropout/random_uniform/RandomUniform:0
#cond_1/dropout/random_uniform/max:0
#cond_1/dropout/random_uniform/min:0
#cond_1/dropout/random_uniform/mul:0
#cond_1/dropout/random_uniform/sub:0
cond_1/dropout/random_uniform:0
cond_1/pred_id:0
cond_1/switch_t:0)
Relu_1:0cond_1/dropout/Shape/Switch:1$
cond_1/pred_id:0cond_1/pred_id:0
д
cond_1/cond_text_1cond_1/pred_id:0cond_1/switch_f:0*®
Relu_1:0
cond_1/Identity/Switch:0
cond_1/Identity:0
cond_1/pred_id:0
cond_1/switch_f:0$
cond_1/pred_id:0cond_1/pred_id:0$
Relu_1:0cond_1/Identity/Switch:0
Ќ
cond_2/cond_textcond_2/pred_id:0cond_2/switch_t:0 *С
Relu_2:0
cond_2/dropout/Floor:0
cond_2/dropout/Shape/Switch:1
cond_2/dropout/Shape:0
cond_2/dropout/add:0
cond_2/dropout/div:0
cond_2/dropout/keep_prob:0
cond_2/dropout/mul:0
-cond_2/dropout/random_uniform/RandomUniform:0
#cond_2/dropout/random_uniform/max:0
#cond_2/dropout/random_uniform/min:0
#cond_2/dropout/random_uniform/mul:0
#cond_2/dropout/random_uniform/sub:0
cond_2/dropout/random_uniform:0
cond_2/pred_id:0
cond_2/switch_t:0)
Relu_2:0cond_2/dropout/Shape/Switch:1$
cond_2/pred_id:0cond_2/pred_id:0
д
cond_2/cond_text_1cond_2/pred_id:0cond_2/switch_f:0*®
Relu_2:0
cond_2/Identity/Switch:0
cond_2/Identity:0
cond_2/pred_id:0
cond_2/switch_f:0$
Relu_2:0cond_2/Identity/Switch:0$
cond_2/pred_id:0cond_2/pred_id:0
Ќ
cond_3/cond_textcond_3/pred_id:0cond_3/switch_t:0 *С
Relu_3:0
cond_3/dropout/Floor:0
cond_3/dropout/Shape/Switch:1
cond_3/dropout/Shape:0
cond_3/dropout/add:0
cond_3/dropout/div:0
cond_3/dropout/keep_prob:0
cond_3/dropout/mul:0
-cond_3/dropout/random_uniform/RandomUniform:0
#cond_3/dropout/random_uniform/max:0
#cond_3/dropout/random_uniform/min:0
#cond_3/dropout/random_uniform/mul:0
#cond_3/dropout/random_uniform/sub:0
cond_3/dropout/random_uniform:0
cond_3/pred_id:0
cond_3/switch_t:0$
cond_3/pred_id:0cond_3/pred_id:0)
Relu_3:0cond_3/dropout/Shape/Switch:1
д
cond_3/cond_text_1cond_3/pred_id:0cond_3/switch_f:0*®
Relu_3:0
cond_3/Identity/Switch:0
cond_3/Identity:0
cond_3/pred_id:0
cond_3/switch_f:0$
cond_3/pred_id:0cond_3/pred_id:0$
Relu_3:0cond_3/Identity/Switch:0
Ќ
cond_4/cond_textcond_4/pred_id:0cond_4/switch_t:0 *С
Relu_4:0
cond_4/dropout/Floor:0
cond_4/dropout/Shape/Switch:1
cond_4/dropout/Shape:0
cond_4/dropout/add:0
cond_4/dropout/div:0
cond_4/dropout/keep_prob:0
cond_4/dropout/mul:0
-cond_4/dropout/random_uniform/RandomUniform:0
#cond_4/dropout/random_uniform/max:0
#cond_4/dropout/random_uniform/min:0
#cond_4/dropout/random_uniform/mul:0
#cond_4/dropout/random_uniform/sub:0
cond_4/dropout/random_uniform:0
cond_4/pred_id:0
cond_4/switch_t:0$
cond_4/pred_id:0cond_4/pred_id:0)
Relu_4:0cond_4/dropout/Shape/Switch:1
д
cond_4/cond_text_1cond_4/pred_id:0cond_4/switch_f:0*®
Relu_4:0
cond_4/Identity/Switch:0
cond_4/Identity:0
cond_4/pred_id:0
cond_4/switch_f:0$
cond_4/pred_id:0cond_4/pred_id:0$
Relu_4:0cond_4/Identity/Switch:0
ж
Rloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/cond_textRloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0Sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0 *д
Eloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar:0
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:0
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1
Rloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0Ь
Eloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar:0Sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1®
Rloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0Rloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
ЌW
Tloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/cond_text_1Rloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0Sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0*о'
jloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
jloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
kloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
kloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
Оloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
Оloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
Оloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Дloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
Аloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
Жloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
Вloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
Бloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
|loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
Жloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
Еloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
Еloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
wloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
uloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
xloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
zloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Rloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0
Gloss/output_1_loss/broadcast_weights/assert_broadcastable/values/rank:0
Hloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0
Hloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank:0
Iloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0‘
Hloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0∆
Hloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank:0zloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0„
Iloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0√
Gloss/output_1_loss/broadcast_weights/assert_broadcastable/values/rank:0xloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0®
Rloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0Rloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:02Т#
П#
lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textlloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *њ 
Оloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
Оloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
Оloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
Дloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
Аloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
Лloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
Жloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
Вloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
Бloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
|loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
Жloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
Еloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
Еloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
wloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
uloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Hloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0
Iloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0Ф
Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0÷
Hloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1ў
Iloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0Лloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Ш
Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0№
lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:02≈

¬

nloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*т
mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0№
lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0в
qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
Э
Oloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/cond_textOloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0Ploss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0 *§
Zloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency:0
Oloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
Ploss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0Ґ
Oloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0Oloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
’
Qloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/cond_text_1Oloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0Ploss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0*№
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0
Wloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0
Wloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0
Wloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0:0
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1:0
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2:0
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4:0
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5:0
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7:0
\loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1:0
Oloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
Ploss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0
Eloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar:0
Ploss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0
Hloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0
Iloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0†
Eloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar:0Wloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0£
Hloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0Wloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0©
Ploss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0§
Iloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0Wloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0Ґ
Oloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0Oloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0"у8
	variablesе8в8
Г
Adam/iterations:0Adam/iterations/Assign%Adam/iterations/Read/ReadVariableOp:0(2+Adam/iterations/Initializer/initial_value:08
c
	Adam/lr:0Adam/lr/AssignAdam/lr/Read/ReadVariableOp:0(2#Adam/lr/Initializer/initial_value:08
s
Adam/beta_1:0Adam/beta_1/Assign!Adam/beta_1/Read/ReadVariableOp:0(2'Adam/beta_1/Initializer/initial_value:08
s
Adam/beta_2:0Adam/beta_2/Assign!Adam/beta_2/Read/ReadVariableOp:0(2'Adam/beta_2/Initializer/initial_value:08
o
Adam/decay:0Adam/decay/Assign Adam/decay/Read/ReadVariableOp:0(2&Adam/decay/Initializer/initial_value:08
Д
dense_12/kernel:0dense_12/kernel/Assign%dense_12/kernel/Read/ReadVariableOp:0(2,dense_12/kernel/Initializer/random_uniform:08
s
dense_12/bias:0dense_12/bias/Assign#dense_12/bias/Read/ReadVariableOp:0(2!dense_12/bias/Initializer/zeros:08
Д
dense_13/kernel:0dense_13/kernel/Assign%dense_13/kernel/Read/ReadVariableOp:0(2,dense_13/kernel/Initializer/random_uniform:08
s
dense_13/bias:0dense_13/bias/Assign#dense_13/bias/Read/ReadVariableOp:0(2!dense_13/bias/Initializer/zeros:08
Д
dense_14/kernel:0dense_14/kernel/Assign%dense_14/kernel/Read/ReadVariableOp:0(2,dense_14/kernel/Initializer/random_uniform:08
s
dense_14/bias:0dense_14/bias/Assign#dense_14/bias/Read/ReadVariableOp:0(2!dense_14/bias/Initializer/zeros:08
Д
dense_15/kernel:0dense_15/kernel/Assign%dense_15/kernel/Read/ReadVariableOp:0(2,dense_15/kernel/Initializer/random_uniform:08
s
dense_15/bias:0dense_15/bias/Assign#dense_15/bias/Read/ReadVariableOp:0(2!dense_15/bias/Initializer/zeros:08
Д
dense_16/kernel:0dense_16/kernel/Assign%dense_16/kernel/Read/ReadVariableOp:0(2,dense_16/kernel/Initializer/random_uniform:08
s
dense_16/bias:0dense_16/bias/Assign#dense_16/bias/Read/ReadVariableOp:0(2!dense_16/bias/Initializer/zeros:08
Д
dense_17/kernel:0dense_17/kernel/Assign%dense_17/kernel/Read/ReadVariableOp:0(2,dense_17/kernel/Initializer/random_uniform:08
s
dense_17/bias:0dense_17/bias/Assign#dense_17/bias/Read/ReadVariableOp:0(2!dense_17/bias/Initializer/zeros:08
В
training/Adam/Variable:0training/Adam/Variable/Assign,training/Adam/Variable/Read/ReadVariableOp:0(2training/Adam/zeros:08
К
training/Adam/Variable_1:0training/Adam/Variable_1/Assign.training/Adam/Variable_1/Read/ReadVariableOp:0(2training/Adam/zeros_1:08
К
training/Adam/Variable_2:0training/Adam/Variable_2/Assign.training/Adam/Variable_2/Read/ReadVariableOp:0(2training/Adam/zeros_2:08
К
training/Adam/Variable_3:0training/Adam/Variable_3/Assign.training/Adam/Variable_3/Read/ReadVariableOp:0(2training/Adam/zeros_3:08
К
training/Adam/Variable_4:0training/Adam/Variable_4/Assign.training/Adam/Variable_4/Read/ReadVariableOp:0(2training/Adam/zeros_4:08
К
training/Adam/Variable_5:0training/Adam/Variable_5/Assign.training/Adam/Variable_5/Read/ReadVariableOp:0(2training/Adam/zeros_5:08
К
training/Adam/Variable_6:0training/Adam/Variable_6/Assign.training/Adam/Variable_6/Read/ReadVariableOp:0(2training/Adam/zeros_6:08
К
training/Adam/Variable_7:0training/Adam/Variable_7/Assign.training/Adam/Variable_7/Read/ReadVariableOp:0(2training/Adam/zeros_7:08
К
training/Adam/Variable_8:0training/Adam/Variable_8/Assign.training/Adam/Variable_8/Read/ReadVariableOp:0(2training/Adam/zeros_8:08
К
training/Adam/Variable_9:0training/Adam/Variable_9/Assign.training/Adam/Variable_9/Read/ReadVariableOp:0(2training/Adam/zeros_9:08
О
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign/training/Adam/Variable_10/Read/ReadVariableOp:0(2training/Adam/zeros_10:08
О
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign/training/Adam/Variable_11/Read/ReadVariableOp:0(2training/Adam/zeros_11:08
О
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign/training/Adam/Variable_12/Read/ReadVariableOp:0(2training/Adam/zeros_12:08
О
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign/training/Adam/Variable_13/Read/ReadVariableOp:0(2training/Adam/zeros_13:08
О
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign/training/Adam/Variable_14/Read/ReadVariableOp:0(2training/Adam/zeros_14:08
О
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign/training/Adam/Variable_15/Read/ReadVariableOp:0(2training/Adam/zeros_15:08
О
training/Adam/Variable_16:0 training/Adam/Variable_16/Assign/training/Adam/Variable_16/Read/ReadVariableOp:0(2training/Adam/zeros_16:08
О
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign/training/Adam/Variable_17/Read/ReadVariableOp:0(2training/Adam/zeros_17:08
О
training/Adam/Variable_18:0 training/Adam/Variable_18/Assign/training/Adam/Variable_18/Read/ReadVariableOp:0(2training/Adam/zeros_18:08
О
training/Adam/Variable_19:0 training/Adam/Variable_19/Assign/training/Adam/Variable_19/Read/ReadVariableOp:0(2training/Adam/zeros_19:08
О
training/Adam/Variable_20:0 training/Adam/Variable_20/Assign/training/Adam/Variable_20/Read/ReadVariableOp:0(2training/Adam/zeros_20:08
О
training/Adam/Variable_21:0 training/Adam/Variable_21/Assign/training/Adam/Variable_21/Read/ReadVariableOp:0(2training/Adam/zeros_21:08
О
training/Adam/Variable_22:0 training/Adam/Variable_22/Assign/training/Adam/Variable_22/Read/ReadVariableOp:0(2training/Adam/zeros_22:08
О
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign/training/Adam/Variable_23/Read/ReadVariableOp:0(2training/Adam/zeros_23:08
О
training/Adam/Variable_24:0 training/Adam/Variable_24/Assign/training/Adam/Variable_24/Read/ReadVariableOp:0(2training/Adam/zeros_24:08
О
training/Adam/Variable_25:0 training/Adam/Variable_25/Assign/training/Adam/Variable_25/Read/ReadVariableOp:0(2training/Adam/zeros_25:08
О
training/Adam/Variable_26:0 training/Adam/Variable_26/Assign/training/Adam/Variable_26/Read/ReadVariableOp:0(2training/Adam/zeros_26:08
О
training/Adam/Variable_27:0 training/Adam/Variable_27/Assign/training/Adam/Variable_27/Read/ReadVariableOp:0(2training/Adam/zeros_27:08
О
training/Adam/Variable_28:0 training/Adam/Variable_28/Assign/training/Adam/Variable_28/Read/ReadVariableOp:0(2training/Adam/zeros_28:08
О
training/Adam/Variable_29:0 training/Adam/Variable_29/Assign/training/Adam/Variable_29/Read/ReadVariableOp:0(2training/Adam/zeros_29:08
О
training/Adam/Variable_30:0 training/Adam/Variable_30/Assign/training/Adam/Variable_30/Read/ReadVariableOp:0(2training/Adam/zeros_30:08
О
training/Adam/Variable_31:0 training/Adam/Variable_31/Assign/training/Adam/Variable_31/Read/ReadVariableOp:0(2training/Adam/zeros_31:08
О
training/Adam/Variable_32:0 training/Adam/Variable_32/Assign/training/Adam/Variable_32/Read/ReadVariableOp:0(2training/Adam/zeros_32:08
О
training/Adam/Variable_33:0 training/Adam/Variable_33/Assign/training/Adam/Variable_33/Read/ReadVariableOp:0(2training/Adam/zeros_33:08
О
training/Adam/Variable_34:0 training/Adam/Variable_34/Assign/training/Adam/Variable_34/Read/ReadVariableOp:0(2training/Adam/zeros_34:08
О
training/Adam/Variable_35:0 training/Adam/Variable_35/Assign/training/Adam/Variable_35/Read/ReadVariableOp:0(2training/Adam/zeros_35:08а≠[A       ў№2	(]ЉҐK6„A*


batch_lossJѕ0?м£Ј№       `/я#	Ґ^ЉҐK6„A*

	batch_acc  ?аrV1"       x=э	_ЉҐK6„A*

batch_f1_score€€?’.;f"       x=э	…С¬ҐK6„A*

epoch_val_loss^д4?JМёj!       {мц	КТ¬ҐK6„A*

epoch_val_acc   ?
нТ&       sOг 	—Т¬ҐK6„A*

epoch_val_f1_scoreю€€>©&nп       ў№2	У¬ҐK6„A*


epoch_lossJѕ0?їP”°       `/я#	CУ¬ҐK6„A*

	epoch_acc  ?ГYw"       x=э	qУ¬ҐK6„A*

epoch_f1_score€€?Ф|Шѓ        )н©P	їр¬ҐK6„A*


batch_loss Ї0?іНЗy       QKD	oс¬ҐK6„A*

	batch_acc  ?илs~$       B+АM	їс¬ҐK6„A*

batch_f1_score€€?L6∞;$       B+АM	z√ҐK6„A*

epoch_val_lossHя4?ў+h#       ∞ЯwC	ь√ҐK6„A*

epoch_val_acc   ?2’{А(       €pJ	B√ҐK6„A*

epoch_val_f1_scoreю€€>:л’        )н©P	}√ҐK6„A*


epoch_loss Ї0?y=иЌ       QKD	і√ҐK6„A*

	epoch_acc  ?[Wиk$       B+АM	ж√ҐK6„A*

epoch_f1_score€€?g™8c        )н©P	≤G√ҐK6„A*


batch_lossZ¶0?
…№#       QKD	≥H√ҐK6„A*

	batch_acc  ?}фh$       B+АM	I√ҐK6„A*

batch_f1_score€€?Є»A$       B+АM	бZ√ҐK6„A*

epoch_val_lossЏ4?vBјн#       ∞ЯwC	p[√ҐK6„A*

epoch_val_acc   ?фъи(       €pJ	Є[√ҐK6„A*

epoch_val_f1_scoreю€€>љ ЊQ        )н©P	т[√ҐK6„A*


epoch_lossZ¶0? њҐ€       QKD	)\√ҐK6„A*

	epoch_acc  ?C/х$       B+АM	`\√ҐK6„A*

epoch_f1_score€€?Ї€        )н©P	Э√ҐK6„A*


batch_loss1Т0?oўъѓ       QKD	ћЭ√ҐK6„A*

	batch_acc  ?Х†Ј$       B+АM	Ю√ҐK6„A*

batch_f1_score€€?@с8Д$       B+АM	÷∞√ҐK6„A*

epoch_val_loss¬‘4?Z8&е#       ∞ЯwC	С±√ҐK6„A*

epoch_val_acc   ?Ўм[(       €pJ	≤√ҐK6„A*

epoch_val_f1_scoreю€€>¶аХr        )н©P	l≤√ҐK6„A*


epoch_loss1Т0?њ!њ       QKD	ћ≤√ҐK6„A*

	epoch_acc  ?ф:Ѕ$       B+АM	≥√ҐK6„A*

epoch_f1_score€€?Ьаrb        )н©P	Eу√ҐK6„A*


batch_loss~0?6kђ       QKD	Гф√ҐK6„A*

	batch_acc  ?7йРd$       B+АM	х√ҐK6„A*

batch_f1_score€€?™V©ћ$       B+АM	_ƒҐK6„A*

epoch_val_loss,ѕ4?з≈M#       ∞ЯwC	ыƒҐK6„A*

epoch_val_acc   ?)Њ(       €pJ	BƒҐK6„A*

epoch_val_f1_scoreю€€>ЋК≠        )н©P	БƒҐK6„A*


epoch_loss~0?ЃG|       QKD	ЈƒҐK6„A*

	epoch_acc  ?њГfь$       B+АM	оƒҐK6„A*

epoch_f1_score€€?ГЙ!і