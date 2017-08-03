import tensorflow as tf
import numpy as np
from math import ceil
import scipy.misc as sp
try:
  import h5py
except:
  pass
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolWithArgmaxGrad(op, grad):
	return gen_nn_ops._max_pool_grad(op.inputs[0],op.outputs[0],grad,op.get_attr("ksize"),op.get_attr("strides"),padding=op.get_attr("padding"),data_format='NHWC')

class Param_loader():
	def __init__(self,weights_path=None):
		if weights_path is not None:
			self.pretrained=True;
			self.weights_path=weights_path;
			self.file_type=self.weights_path.split('.')[-1];

			if(self.file_type=='npy'):
				self.transposer=[0,1,2,3]
				temp=np.load(weights_path)[()];
				self.layer_names=[];
				for i in temp:
					self.layer_names.append(i[0]);
				self.weight_data=dict()
				for pj in self.layer_names:
					# self.weight_data[pj,'0']=temp[pj]['weights']
					# self.weight_data[pj,'1']=temp[pj]['biases']
					self.weight_data[pj,'0']=temp[pj,'0']
					self.weight_data[pj,'1']=temp[pj,'1']




			elif(self.file_type=='h5'):
				self.transposer=[2,3,1,0]
				self.weight_data=dict();
				f=h5py.File(weights_path,'r');
				self.layer_names=[];
				for i in list(f['data']):
					if(len(f['data'][i])>0):
						self.layer_names.append(i);
				for pj in self.layer_names:
					self.weight_data[pj,'0']=f['data'][pj]['0'][:];
					self.weight_data[pj,'1']=f['data'][pj]['1'][:];

				f.close()
		else:
			self.pretrained=False;

def get_deconv_filter(f_shape):
	width = f_shape[0]
	heigh = f_shape[0]
	f = ceil(width/2.0)
	c = (2 * f - 1 - f % 2) / (2.0 * f)
	bilinear = np.zeros([f_shape[0], f_shape[1]])
	for x in range(width):
		for y in range(heigh):
			value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
			bilinear[x, y] = value
	weights = np.zeros(f_shape)
	for i in range(f_shape[2]):
		weights[:, :, i, i] = bilinear
	return weights


def get_deconv_weights(params,shape,layer_name,var_name,trainable,stddev=1):
	with tf.variable_scope(var_name):
		if(params.pretrained==False):
			rt=get_deconv_filter(shape)
			return tf.get_variable(initializer=tf.constant_initializer(value=rt,dtype=tf.float32),shape=shape,name=var_name)
			# return tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev),shape=shape,name=var_name);
		else:
			if layer_name in params.layer_names:
				print 'loading pretrained weight for ',layer_name,'with shape:',shape;
				wt=tf.get_variable(trainable=trainable,initializer=tf.constant_initializer(params.weight_data[layer_name,'0'].reshape(shape)),shape=shape,name=var_name);
			else:
				rt=get_deconv_filter(shape)
				wt=tf.get_variable(initializer=tf.constant_initializer(value=rt,dtype=tf.float32),shape=shape,name=var_name)
			return wt;


def get_weights(params,shape,layer_name,var_name,trainable,stddev=1):
	params.transposer=[2,3,1,0]
	if(params.pretrained==False):
		return tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev),shape=shape,name=var_name);
	else:
		if layer_name in params.layer_names:
			print 'loading pretrained weight for ',layer_name,'with shape:',shape;
			wt=tf.get_variable(trainable=trainable,initializer=tf.constant_initializer(np.transpose(params.weight_data[layer_name,'0'],params.transposer)),shape=shape,name=var_name);
		else:
			wt=tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=stddev),shape=shape,name=var_name);
		return wt;



def get_biases(params,shape,var_name,layer_name,trainable,val=1.0):
	if(params.pretrained==False):
		return tf.get_variable(initializer=tf.constant_initializer(val),shape=shape,name=var_name);
	else:
		if layer_name in params.layer_names:
			print 'loading pretrained bias for ',layer_name,'with shape:',shape;
			# print params.weight_data[layer_name,'1'].reshape(shape)
			bias=tf.get_variable(trainable=trainable,initializer=tf.constant_initializer(params.weight_data[layer_name,'1'].reshape(shape)),shape=shape,name=var_name)
		else:
			bias=tf.get_variable(initializer=tf.constant_initializer(val),shape=shape,name=var_name);
		return bias;

def max_pool(x,k_shape,strides,name,padding='VALID'):
	with tf.variable_scope(name):
		pooled=tf.nn.max_pool(x,ksize=[1,k_shape[0],k_shape[1],1],strides=[1,strides[0],strides[1],1],padding=padding)
	return pooled;

def lrn(x, radius, alpha, beta, name, bias=1.0):
	return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,beta = beta, bias = bias, name = name)

def dropout(x, keep_prob,phase_train):
	return tf.nn.dropout(x, keep_prob) if phase_train is True else x

def print_shape(obj):
	print obj.name,obj.get_shape().as_list();

def conv_bn(inputT, k_shape, out_channels, stride, name, phase_train, reuse=False,padding='SAME',activation='relu', batch_norm=True,groups=1,params=Param_loader(),trainable=False):
	# Trainable parameter controls if the variable found in weights file should be further trained.
	
	if(activation=='none'):
		activ_function=lambda x: x
	elif(activation=='relu'):
		activ_function=lambda x: tf.nn.relu(x)
	elif(activation=='leakyrelu'):
		activ_function=lambda x: LeakyRelu(x,0.2)
	elif(activation=='prelu'):
		activ_function=lambda x: Parametric_Relu(x,name)
	elif(activation=='sigmoid'):
		activ_function=lambda x: tf.nn.sigmoid(x)
	elif(activation=='tanh'):
		activ_function=lambda x: tf.tanh(x)
	
	in_channels = inputT.get_shape().as_list()[-1];
	with tf.variable_scope(name,reuse=reuse):
		kernel = get_weights(params,var_name='weights', shape=[k_shape[0],k_shape[1],in_channels/groups,out_channels],layer_name=name,trainable=trainable,stddev=1e-1);
		if(groups==1):
			c1=tf.nn.conv2d(inputT,kernel,strides=[1,stride[0],stride[1],1],padding=padding);
		else:
			input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=inputT)
			weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=kernel)
			output_groups = [tf.nn.conv2d(i, k,strides=[1,stride[0],stride[1],1],padding=padding) for i,k in zip(input_groups, weight_groups)]
			c1 = tf.concat(axis = 3, values = output_groups)
		# conv = tf.nn.conv2d(inputT, kernel, [1, stride[0], stride[1], 1], padding='SAME')
		biases = get_biases(params,var_name='biases', shape=[out_channels],layer_name=name,trainable=trainable,val=0.0)
		bias = tf.nn.bias_add(c1, biases)
		if batch_norm is True:
			conv_out = batch_norm_layer(activ_function(bias), phase_train,name,params,reuse=reuse,trainable=trainable)
		else:
			conv_out=activ_function(bias)
	return conv_out

def atrous_conv(inputT, k_shape, out_channels, dilation_rate, name, phase_train, reuse=False,padding='SAME', activation='none',batch_norm=True,params=Param_loader(),trainable=False):
	# Trainable parameter controls if the variable found in weights file should be further trained.
	
	if(activation=='none'):
		activ_function=lambda x: x
	elif(activation=='relu'):
		activ_function=lambda x: tf.nn.relu(x)
	elif(activation=='leakyrelu'):
		activ_function=lambda x: LeakyRelu(x,0.2)
	elif(activation=='prelu'):
		activ_function=lambda x: Parametric_Relu(x,name)
	elif(activation=='sigmoid'):
		activ_function=lambda x: tf.nn.sigmoid(x)
	elif(activation=='tanh'):
		activ_function=lambda x: tf.tanh(x)

	in_channels = inputT.get_shape().as_list()[-1];
	with tf.variable_scope(name,reuse=reuse):
		kernel = get_weights(params,var_name='weights', shape=[k_shape[0],k_shape[1],in_channels,out_channels],layer_name=name,trainable=trainable,stddev=1e-1);
		c1=tf.nn.atrous_conv2d (inputT,kernel,rate=dilation_rate,padding=padding);
		biases = get_biases(params,var_name='biases', shape=[out_channels],layer_name=name,trainable=trainable,val=0.0)
		bias = tf.nn.bias_add(c1, biases)
		if batch_norm is True:
			conv_out = batch_norm_layer(activ_function(bias), phase_train,name,params,reuse=reuse,trainable=trainable)
		else:
			conv_out=activ_function(bias)
	return conv_out

def batch_norm_layer1(inputT, is_training, scope):
	return tf.cond(is_training,
		lambda: tf.contrib.layers.batch_norm(inputT, trainable=True,is_training=True,center=False, updates_collections=None, scope=scope+"_bn"),
		lambda: tf.contrib.layers.batch_norm(inputT, trainable=True,is_training=False,updates_collections=None, center=False, scope=scope+"_bn", reuse = True))


def batch_norm_layer(inputT, is_training, scope,params,reuse,trainable):
	this_name=scope+'_bn';
	if(params.pretrained==False):
		return tf.contrib.layers.batch_norm(inputT,reuse=reuse,is_training=is_training,center=True,scale=True, updates_collections=None, scope=this_name)
	else:
		if(this_name in params.layer_names):
			print 'pretrained BN param', this_name, 'with shape', params.weight_data[this_name,'0'].shape;
			param_initializers=dict()
			param_initializers['gamma']=tf.constant_initializer(params.weight_data[this_name,'0'].reshape([-1]));
			param_initializers['beta']=tf.constant_initializer(params.weight_data[this_name,'1'].reshape([-1]));
			return tf.contrib.layers.batch_norm(inputT, center=True,scale=True, param_initializers=param_initializers,reuse=reuse,trainable=trainable,is_training=is_training, updates_collections=None, scope=this_name) 
		else:
			return tf.contrib.layers.batch_norm(inputT, reuse=reuse,is_training=is_training,center=True,scale=True, updates_collections=None, scope=this_name) 

def batch_normalization(input1,is_training,scope,params,reuse,trainable):
	this_name=scope+'_bn'
	with tf.variable_scope('',reuse=reuse):
		moving_mean=tf.get_variable(trainable=False,name=scope+'moving_mean',shape=input1.get_shape().as_list()[-1],initializer=tf.constant_initializer(0.0))
		moving_variance=tf.get_variable(trainable=False,name=scope+'moving_variance',shape=input1.get_shape().as_list()[-1],initializer=tf.constant_initializer(1.0))
	if(is_training):
		[mu,variance]=tf.nn.moments(x=input1,axes=[0,1,2])
		moving_mean=moving_mean*0.99+0.01*mu
		moving_variance=moving_variance*0.99+0.01*variance
	else:
		mu=moving_mean
		variance=moving_variance

	if(params.pretrained==False):
		with tf.variable_scope('',reuse=reuse):
			beta=tf.get_variableble(name=scope+'beta',shape=[input1.get_shape().as_list()[-1]],initializer=tf.constant_initializer(0.0))
			gamma=tf.get_variable(name=scope+'gamma',shape=[input1.get_shape().as_list()[-1]],initializer=tf.constant_initializer(1.0))
		return tf.nn.batch_normalization(input1,mu,variance,beta,gamma,1e-12)
	else:
		if(this_name in params.layer_names):
			print 'pretrained BN param', this_name, 'with shape',params.weight_data[this_name,'0'].shape
			with tf.variable_scope('',reuse=reuse):
				beta=tf.get_variable(trainable=trainable,name=scope+'beta',initializer=tf.constant_initializer(params.weight_data[this_name,'1'].reshape([-1])),shape=input1.get_shape().as_list()[-1])
				gamma=tf.get_variable(trainable=trainable,name=scope+'gamma',initializer=tf.constant_initializer(params.weight_data[this_name,'0'].reshape([-1])),shape=input1.get_shape().as_list()[-1])
			return tf.nn.batch_normalization(input1,mu,variance,beta,gamma,1e-12)
		else:
			with tf.variable_scope('',reuse=reuse):
				beta=tf.get_variable(name=scope+'beta',shape=[input1.get_shape().as_list()[-1]],initializer=tf.constant_initializer(0.0))
				gamma=tf.get_variable(name=scope+'gamma',shape=[input1.get_shape().as_list()[-1]],initializer=tf.constant_initializer(1.0))
			return tf.nn.batch_normalization(input1,mu,variance,beta,gamma,1e-12)

def LeakyRelu(x,param):
	pos_part=tf.nn.relu(x)
	neg_part=param*(x-tf.abs(x))*0.5
	return pos_part+neg_part

def Parametric_Relu(x,name):
	with tf.variable_scope(name+'_alpha'):
		alpha=tf.get_variable(initializer=tf.constant_initializer(0.0))
	pos_part=tf.nn.relu(x)
	neg_part=alpha*(x-tf.abs(x))*0.5
	return pos_part+neg_part

def upsample(x,k_shape=[2,2],name='upsample',out_shape=None):

	in_shape=x.get_shape().as_list()
	if out_shape is None:
		out_shape=[-1]+[j*i for i,j in zip(in_shape[1:-1],k_shape)]+[-1]
	shape=tf.stack([in_shape[0],out_shape[1],out_shape[2],in_shape[-1]])
	x_indices=tf.where(tf.equal(x,x))
	new_indices=2*x_indices[:,1:-1]
	new_indices=tf.concat([tf.reshape(x_indices[:,0],[-1,1]),new_indices,tf.reshape(x_indices[:,-1],[-1,1])],1)
	# indices=tf.reshape(new_indices,shape=[tf.size(x),4])
	# b=tf.range(comp_shape[0],dtype=tf.int64)
	# b=tf.ones([])
	updates=tf.reshape(x,[-1])
	output=tf.scatter_nd(new_indices,updates,tf.cast(shape,tf.int64))
	return output

def downsample(x,k_shape=[2,2],name='downsample',out_shape=None):
	return x[:,0::k_shape[0],0::k_shape[1],:]
def upsample_with_pool_mask(updates, mask, ksize=[1, 2, 2, 1],out_shape=None,name=None):
	input_shape = updates.get_shape().as_list()
	if out_shape is None:
		out_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
	one_like_mask = tf.ones_like(mask)
	batch_range = tf.reshape(tf.range(out_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
	b = one_like_mask * batch_range
	y = mask // (out_shape[2] * out_shape[3])
	x = mask % (out_shape[2] * out_shape[3]) // out_shape[3]
	feature_range = tf.range(out_shape[3], dtype=tf.int64)
	f = one_like_mask * feature_range
	updates_size = tf.size(updates)
	indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
	values = tf.reshape(updates, [updates_size])
	ret = tf.scatter_nd(indices, values, out_shape)
	return ret


def xavier_initializer(kernel_size,num_filters):
	stddev = math.sqrt(2. / (kernel_size**2 * num_filters))
	return tf.truncated_normal_initializer(stddev=stddev)

def upscore_layer(x,k_shape,stride,out_channels,name,phase_train,reuse=False,out_shape=None,padding='SAME',params=Param_loader(),trainable=False):
	# Trainable parameter controls if the variable found in weights file should be further trained.
	with tf.variable_scope(name,reuse=reuse):
		in_shape=x.get_shape().as_list()
		in_channels=int(x.get_shape().as_list()[-1])
		kernel=get_deconv_weights(params,[k_shape[0],k_shape[1],out_channels,in_channels],var_name='deconv_filters',layer_name=name,stddev=1e-1,trainable=trainable)
		bias=get_biases(params,[out_channels],var_name='deconv_bias',layer_name=name,trainable=trainable);
		if(out_shape==None):
			if(padding=='VALID'):
				h=(in_shape[1]-1)*stride[0]+k_shape[0];
				w=(in_shape[2]-1)*stride[1]+k_shape[1];
				shape=np.stack([in_shape[0],h,w,out_channels]).astype(np.int32)
			elif(padding=='SAME'):
				h=(in_shape[1]-1)*stride[0]+1;
				w=(in_shape[2]-1)*stride[1]+1;
				shape=np.stack([in_shape[0],h,w,out_channels]).astype(np.int32)
		else:
			shape=np.stack([in_shape[0],out_shape[1],out_shape[2],out_channels]).astype(np.int32)
		# if(groups==1):
		c1=tf.nn.conv2d_transpose(x,kernel,output_shape=shape,strides=[1,stride[0],stride[1],1],padding=padding);
		# else:
		# 	input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
		# 	weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=kernel)
		# 	output_groups = [tf.nn.conv2d(i, k,strides=[1,stride[0],stride[1],1],padding=padding) for i,k in zip(input_groups, weight_groups)]
		# 	c1 = tf.concat(axis = 3, values = output_groups)
		bias=tf.nn.bias_add(c1,bias);
		relu=tf.nn.relu(bias);
	return relu


def fc_flatten(x,num_out,name,phase_train,reuse=False,relu=False,params=Param_loader(),trainable=False):
	with tf.variable_scope(name,reuse=reuse):
		shape=x.get_shape().as_list();
		# Unwind before fully connected layer
		x_reshaped=tf.reshape(x,[-1,np.prod(shape[1:])]); #reshape just to be safe
		num_in=np.prod(shape[1:]);
		weights=get_weights(params,var_name='fc_weights',layer_name=name,shape=[num_in,num_out],trainable=trainable)
		biases=get_biases(params,var_name='fc_biases',shape=[num_out],layer_name=name,trainable=trainable)
		added=tf.nn.bias_add(tf.matmul(x_reshaped,weights),biases);
		if(relu==True):
			return tf.nn.relu(added);
		else:
			return added;
		
def fc_convol(x,k_shape,num_out,name,phase_train,reuse=False,relu=False,params=Param_loader(),trainable=False):
	with tf.variable_scope(name,reuse=reuse):
		#Fully connected layer as a convolution
		num_in=x.get_shape().as_list()[-1];	
		weights=get_weights(params,[k_shape[0],k_shape[1],num_in,num_out],var_name='fc_weights',layer_name=name,trainable=trainable)
		bias=get_biases(params,[num_out],var_name='fc_biases',layer_name=name,trainable=trainable)
		c1=tf.nn.conv2d(x,weights,strides=[1,1,1,1],padding='SAME');
		bias=tf.nn.bias_add(c1,bias);
		if(relu==True):
			return tf.nn.relu(bias);
		else:
			return bias;



