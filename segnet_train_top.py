import tensorflow as tf
import numpy as np
import os
from image_reader import *
from utils_mod import *
from argparse import ArgumentParser
# import matplotlib.pyplot as plt
from color_map import *
try:
  import h5py
except:
  pass

class Segnet():
	def __init__(self,keep_prob,num_classes,is_gpu,weights_path=None,pretrained=False):
		self.num_classes = num_classes
		self.keep_prob = keep_prob
		self.is_gpu=is_gpu;
		self.pretrained=pretrained;
		self.params=Param_loader();
		if weights_path is not None:
			self.pretrained=True;
			self.params=Param_loader(weights_path);

	def inference(self,x,is_training,reuse):
		self.x=x;
		self.is_training=is_training;
		self.reuse=reuse;
		self.logits=self.build_network();
		return self.logits;

	def calc_loss(self,logits,labels,num_classes,weighted=False,weights=None):
		epsilon = tf.constant(value=1e-10)
		no_labels=tf.where(tf.logical_and(labels>=0,labels<num_classes))
		labels=tf.gather_nd(labels,no_labels)
		logits=tf.gather_nd(logits,no_labels)
		labels=tf.reshape(tf.one_hot(tf.reshape(labels,[-1]),num_classes),[-1,num_classes]);
		logits=tf.reshape(logits,[-1,num_classes]);
		if(weighted==False):
			cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross_entropy');
			cross_entropy_mean=tf.reduce_mean(cross_entropy);
		else:
			softmax=tf.nn.softmax(logits+epsilon);
			cross_entropy=-1*tf.reduce_sum(tf.mul(labels*tf.log(softmax+epsilon),weights.reshape([num_classes])),axis=1);
			cross_entropy_mean=tf.reduce_mean(cross_entropy);
		tf.add_to_collection('losses', cross_entropy_mean);
		loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		return loss;

	def train(self,learning_rate):
		opt = tf.train.AdamOptimizer(learning_rate);
		gradvar_list=opt.compute_gradients(self.loss);
		self.train_op=opt.apply_gradients(gradvar_list);
	def get_shape(self,x):
		return x.get_shape().as_list();

	def build_network(self):


		# self.inp_tensor = tf.nn.lrn(self.x, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='norm1');
		# with tf.variable_scope('preprocess',dtype=tf.float32):
		# 	mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
		# 	self.inp_tensor = self.x-mean;
		self.rt=dict()
		self.inp_tensor=self.x

		self.rt['data']=self.inp_tensor
		# Layer 1
		conv1_1=conv_bn(self.inp_tensor,[3,3],64,[1,1],name='conv1_1',phase_train=self.is_training,params=self.params,reuse=self.reuse);
		print_shape(conv1_1);
		

		conv1_2=conv_bn(conv1_1,[3,3],64,[1,1],name='conv1_2',phase_train=self.is_training,params=self.params,reuse=self.reuse);
		print_shape(conv1_2);

		if(self.is_gpu==True):
			pool1,pool1_mask=tf.nn.max_pool_with_argmax(conv1_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool1_gpu');
		else:
			pool1=tf.nn.max_pool(conv1_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool1_cpu');
		print_shape(pool1);
		# self.rt['conv1_1']=conv1_1
		# self.rt['conv1_2']=conv1_2
		# self.rt['pool1']=pool1
		# self.rt['pool1_mask']=pool1_mask


		# Layer 2
		conv2_1=conv_bn(pool1,[3,3],128,[1,1],name='conv2_1',phase_train=self.is_training,params=self.params,reuse=self.reuse);
		print_shape(conv2_1);

		conv2_2=conv_bn(conv2_1,[3,3],128,[1,1],name='conv2_2',phase_train=self.is_training,params=self.params,reuse=self.reuse);
		print_shape(conv2_2);

		if(self.is_gpu==True):
			pool2,pool2_mask=tf.nn.max_pool_with_argmax(conv2_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool2_gpu');
		else:
			pool2=tf.nn.max_pool(conv2_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool2_cpu');
		print_shape(pool2);
		# self.rt['conv2_1']=conv2_1
		# self.rt['conv2_2']=conv2_2
		# self.rt['pool2']=pool2
		# self.rt['pool2_mask']=pool2_mask




		# Layer 3
		conv3_1=conv_bn(pool2,[3,3],256,[1,1],name='conv3_1',phase_train=self.is_training,params=self.params,reuse=self.reuse);
		print_shape(conv3_1);

		conv3_2=conv_bn(conv3_1,[3,3],256,[1,1],name='conv3_2',phase_train=self.is_training,params=self.params,reuse=self.reuse);
		print_shape(conv3_2);

		conv3_3=conv_bn(conv3_2,[3,3],256,[1,1],name='conv3_3',phase_train=self.is_training,params=self.params,reuse=self.reuse);
		print_shape(conv3_3);
		
		if(self.is_gpu==True):
			pool3,pool3_mask=tf.nn.max_pool_with_argmax(conv3_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool3_gpu');
		else:
			pool3=tf.nn.max_pool(conv3_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool3_cpu');
		print_shape(pool3);
		# self.rt['conv3_1']=conv3_1
		# self.rt['conv3_2']=conv3_2
		# self.rt['conv3_3']=conv3_3
		# self.rt['pool3']=pool3
		# self.rt['pool3_mask']=pool3_mask



 		# Layer 4
		conv4_1=conv_bn(pool3,[3,3],512,[1,1],name='conv4_1',phase_train=self.is_training,params=self.params,reuse=self.reuse);
		print_shape(conv4_1);

		conv4_2=conv_bn(conv4_1,[3,3],512,[1,1],name='conv4_2',phase_train=self.is_training,params=self.params,reuse=self.reuse);
		print_shape(conv4_2);

		conv4_3=conv_bn(conv4_2,[3,3],512,[1,1],name='conv4_3',phase_train=self.is_training,params=self.params,reuse=self.reuse);
		print_shape(conv4_3);

		if(self.is_gpu==True):
			pool4,pool4_mask=tf.nn.max_pool_with_argmax(conv4_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool4_gpu');
		else:
			pool4=tf.nn.max_pool(conv4_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool4_cpu');
		print_shape(pool4);
		# self.rt['conv4_1']=conv4_1
		# self.rt['conv4_2']=conv4_2
		# self.rt['conv4_3']=conv4_3
		# self.rt['pool4']=pool4
		# self.rt['pool4_mask']=pool4_mask



		# Layer 5
		conv5_1=conv_bn(pool4,[3,3],512,[1,1],name='conv5_1',phase_train=self.is_training,params=self.params,reuse=self.reuse);
		print_shape(conv5_1);

		conv5_2=conv_bn(conv5_1,[3,3],512,[1,1],name='conv5_2',phase_train=self.is_training,params=self.params,reuse=self.reuse);
		print_shape(conv5_2);

		conv5_3=conv_bn(conv5_2,[3,3],512,[1,1],name='conv5_3',phase_train=self.is_training,params=self.params,reuse=self.reuse);
		print_shape(conv5_3);

		if(self.is_gpu==True):
			pool5,pool5_mask=tf.nn.max_pool_with_argmax(conv5_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool5_gpu');
		else:
			pool5=tf.nn.max_pool(conv5_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool5_cpu');
		print_shape(pool5);
		# self.rt['conv5_1']=conv5_1
		# self.rt['conv5_2']=conv5_2
		# self.rt['conv5_3']=conv5_3
		# self.rt['pool5']=pool5
		# self.rt['pool5_mask']=pool5_mask







		# DECODER

		# Upsample 5
		if(self.is_gpu==True):
			pool5_D = upsample_with_pool_mask(pool5, pool5_mask,ksize=[1,2,2,1], out_shape=conv5_1.get_shape().as_list(), name='upsample5_gpu')
		else:
			# upsample5 = upscore_layer(pool5, [2, 2], [2,2] , out_channels=512 , out_shape=tf.shape(conv5_1),name= "upsample5_cpu",phase_train=self.is_training,reuse=self.reuse)
			pool5_D=upsample(pool5,out_shape=conv5_1.get_shape().as_list())
		print_shape(pool5_D);

		# decode 4
		conv5_3_D = conv_bn(pool5_D, [3,3], 512,[1,1], name="conv5_3_D", phase_train=self.is_training,params=self.params,reuse=self.reuse)
		print_shape(conv5_3_D);
		conv5_2_D = conv_bn(conv5_3_D, [3,3], 512,[1,1], name="conv5_2_D", phase_train=self.is_training,params=self.params,reuse=self.reuse)
		print_shape(conv5_2_D);
		conv5_1_D = conv_bn(conv5_2_D, [3,3], 512,[1,1], name="conv5_1_D", phase_train=self.is_training,params=self.params,reuse=self.reuse)
		print_shape(conv5_1_D);

		# self.rt['pool5_D']=pool5_D
		# self.rt['conv5_3_D']=conv5_3_D
		# self.rt['conv5_2_D']=conv5_2_D
		# self.rt['conv5_1_D']=conv5_1_D

	
		# Upsample 4
		if(self.is_gpu==True):
			pool4_D = upsample_with_pool_mask(conv5_1_D, pool4_mask,ksize=[1,2,2,1], out_shape=conv4_1.get_shape().as_list(), name='upsample4_gpu')
		else:
			# upsample4 = upscore_layer(conv5_1_D, [2, 2], [2,2] , out_channels=512 , out_shape=tf.shape(conv4_1),name= "upsample4_cpu",phase_train=self.is_training,reuse=self.reuse)
			pool4_D=upsample(conv5_1_D,out_shape=conv4_1.get_shape().as_list())
		print_shape(pool4_D);

		# decode 4
		conv4_3_D = conv_bn(pool4_D, [3,3], 512,[1,1], name="conv4_3_D", phase_train=self.is_training,params=self.params,reuse=self.reuse)
		print_shape(conv4_3_D);
		conv4_2_D = conv_bn(conv4_3_D, [3,3], 512,[1,1], name="conv4_2_D", phase_train=self.is_training,params=self.params,reuse=self.reuse)
		print_shape(conv4_2_D);
		conv4_1_D = conv_bn(conv4_2_D, [3,3], 256,[1,1], name="conv4_1_D", phase_train=self.is_training,params=self.params,reuse=self.reuse)
		print_shape(conv4_1_D);
		# self.rt['pool4_D']=pool4_D
		# self.rt['conv4_3_D']=conv4_3_D
		# self.rt['conv4_2_D']=conv4_2_D
		# self.rt['conv4_1_D']=conv4_1_D





		# Upsample3
		if(self.is_gpu==True):
			pool3_D = upsample_with_pool_mask(conv4_1_D, pool3_mask,ksize=[1,2,2,1], out_shape=conv3_1.get_shape().as_list(), name='upsample3_gpu')
		else:
			# upsample3 = upscore_layer(conv4_1_D, [2, 2], [2,2] , out_channels=256 , out_shape=tf.shape(conv3_1),name= "upsample3_cpu",phase_train=self.is_training,reuse=self.reuse)
			pool3_D=upsample(conv4_1_D,out_shape=conv3_1.get_shape().as_list())
		print_shape(pool3_D);

		# decode 4
		conv3_3_D = conv_bn(pool3_D, [3,3], 256,[1,1], name="conv3_3_D", phase_train=self.is_training,params=self.params,reuse=self.reuse)
		print_shape(conv3_3_D);
		conv3_2_D = conv_bn(conv3_3_D, [3,3], 256,[1,1], name="conv3_2_D", phase_train=self.is_training,params=self.params,reuse=self.reuse)
		print_shape(conv3_2_D);
		conv3_1_D = conv_bn(conv3_2_D, [3,3], 128,[1,1], name="conv3_1_D", phase_train=self.is_training,params=self.params,reuse=self.reuse)
		print_shape(conv3_1_D);
		# self.rt['pool3_D']=pool3_D
		# self.rt['conv3_3_D']=conv3_3_D
		# self.rt['conv3_2_D']=conv3_2_D
		# self.rt['conv3_1_D']=conv3_1_D

	
	    


	    # Upsample2
		if(self.is_gpu==True):
			pool2_D = upsample_with_pool_mask(conv3_1_D, pool2_mask,ksize=[1,2,2,1], out_shape=conv2_1.get_shape().as_list(), name='upsample2_gpu')
		else:
			# upsample2 = upscore_layer(conv3_1_D, [2, 2], [2,2] , out_channels=128 , out_shape=tf.shape(conv2_1),name= "upsample2_cpu",phase_train=self.is_training,reuse=self.reuse)
			pool2_D=upsample(conv3_1_D,out_shape=conv2_1.get_shape().as_list())

		print_shape(pool2_D);

		# decode 4
		conv2_2_D = conv_bn(pool2_D, [3,3], 128,[1,1], name="conv2_2_D", phase_train=self.is_training,params=self.params,reuse=self.reuse)
		print_shape(conv2_2_D);
		conv2_1_D = conv_bn(conv2_2_D, [3,3], 64,[1,1], name="conv2_1_D", phase_train=self.is_training,params=self.params,reuse=self.reuse)
		print_shape(conv2_1_D);
		# deconv2_2 = conv_bn(deconv2_1, [3,3], 64,[1,1], name="deconv2_2", phase_train=self.is_training,params=self.params)
		# print_shape(deconv2_2);
		# self.rt['pool2_D']=pool2_D
		# self.rt['conv2_2_D']=conv2_2_D
		# self.rt['conv2_1_D']=conv2_1_D


	    

	    # Upsample1
		if(self.is_gpu==True):
			pool1_D = upsample_with_pool_mask(conv2_1_D, pool1_mask,ksize=[1,2,2,1], out_shape=conv1_1.get_shape().as_list(), name='upsample1_gpu')
		else:
			# upsample1 = upscore_layer(conv2_1_D, [2, 2], [2,2] , out_channels=64 , out_shape=tf.shape(conv1_1),name= "upsample1_cpu",phase_train=self.is_training,reuse=self.reuse)
			pool1_D=upsample(conv2_1_D,out_shape=conv1_1.get_shape().as_list())

		print_shape(pool1_D);

		# decode 4
		conv1_2_D = conv_bn(pool1_D, [3,3], 64,[1,1], name="conv1_2_D_retrain", phase_train=self.is_training,params=self.params,reuse=self.reuse,trainable=True)
		print_shape(conv1_2_D);
		conv1_1_D = conv_bn(conv1_2_D, [3,3], self.num_classes,[1,1], name="conv1_1_D_retrain", phase_train=self.is_training,batch_norm=False,params=self.params,reuse=self.reuse,trainable=True)
		print_shape(conv1_1_D);
		# deconv1_2 = conv_bn(deconv1_1, [3,3], self.num_classes,[1,1], name="deconv1_2", phase_train=self.is_training,params=self.params)
		# print_shape(deconv1_2);
		# self.rt['pool1_D']=pool1_D
		# self.rt['conv1_2_D']=conv1_2_D
		# self.rt['conv1_1_D']=conv1_1_D



		# Fully connected layer
		# self.logits=fc_convol(conv_decode1,[1,1],self.num_classes,name='fc_classify1',params=self.params);  
		return conv1_1_D;

def train_segnet():
	num_classes=8
	n_epochs=100
	batch_size_train=3
	batch_size_valid=1
	lr_decay_every=5
	validate_every=5
	save_every=10
	base_lr=1e-6
	img_size=[360,480]

	train_data_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/training_set/images/')
	train_label_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/training_set/new_labels/')
	test_data_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/val_set/images/')
	test_label_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/val_set/new_labels/')
	
	# train_data_dir=os.path.join(BASE_DIR,'SegNet-Tutorial/CamVid/train/')
	# train_label_dir=os.path.join(BASE_DIR,'SegNet-Tutorial/CamVid/trainannot/')
	# test_data_dir=os.path.join(BASE_DIR,'SegNet-Tutorial/CamVid/test/')
	# test_label_dir=os.path.join(BASE_DIR,'SegNet-Tutorial/CamVid/testannot/')

	reader=image_reader(train_data_dir,train_label_dir,batch_size_train,image_size=[360,480,3]);
	reader_valid=image_reader(test_data_dir,test_label_dir,batch_size_valid,image_size=[360,480,3]);
	image_size=reader.image_size;
	n_batches=reader.n_batches;
	f_train=open('train_log_file','w+');
	sess=tf.Session();
	train_data=tf.placeholder(tf.float32,shape=[batch_size_train,image_size[0],image_size[1],image_size[2]]);
	train_labels=tf.placeholder(tf.int64, shape=[batch_size_train, image_size[0], image_size[1]]);
	valid_data=tf.placeholder(tf.float32,shape=[batch_size_valid,image_size[0],image_size[1],image_size[2]]);
	valid_labels=tf.placeholder(tf.int64, shape=[batch_size_valid, image_size[0], image_size[1]]);
	count=tf.placeholder(tf.int32,shape=[]);
	
	net=Segnet(keep_prob=0.5,num_classes=num_classes,is_gpu=True,weights_path='segnet_road.npy');
	train_logits=net.inference(train_data, is_training=True,reuse=False)
	valid_logits=net.inference(valid_data, is_training=False,reuse=True)
	print 'built network';

	file=open(os.path.join('match_labels.txt'))
	match_labels=file.readlines()
	file.close()
	match_labels=[line.splitlines()[0].split(' ') for line in match_labels]
	net.match_labels=match_labels

	net.loss=net.calc_loss(train_logits,train_labels,net.num_classes);
	learning_rate=tf.train.exponential_decay(base_lr,count,1,0.5)
	net.train(learning_rate);
	prediction_train=tf.argmax(train_logits,axis=3);
	prediction_valid=tf.argmax(valid_logits,axis=3);
	# accuracy=tf.size(tf.where(prediction==train_labels)[0]);

	print 'built loss graph';
	saver=tf.train.Saver(tf.trainable_variables())
	sess.run(tf.global_variables_initializer());
	# saver.restore(sess,'segnet_model')
	print 'initialized vars';
	cnt=0;


	while(reader.epoch<n_epochs):	
		while(reader.batch_num<reader.n_batches):
			[train_data_batch,train_label_batch]=reader.next_batch();
			feed_dict_train={train_data:train_data_batch,train_labels:train_label_batch,count:cnt//lr_decay_every};
			[pred,_]=sess.run([prediction_train,net.train_op],feed_dict=feed_dict_train);
			[corr,total_pix]=transform_labels(pred,train_label_batch,match_labels,num_classes)
			acc=corr*1.0/total_pix
			print 'Learning_rate:',sess.run(learning_rate,feed_dict={count:cnt//lr_decay_every}),'epoch:',reader.epoch+1,', Batch:',reader.batch_num, ', correct pixels:', corr, ', Accuracy:',acc
			f_train.write('Training'+' learning_rate:'+str(sess.run(learning_rate,feed_dict={count:cnt//lr_decay_every}))+' epoch:'+str(reader.epoch+1)+' Batch:'+str(reader.batch_num)+' Accuracy:'+str(acc)+'\n');


		if((reader.epoch+1)%save_every==0):
			saver.save(sess,'segnet_arlmodel',global_step=(reader.epoch+1))

		if((reader.epoch+1)%validate_every==0):
			reader_valid.reset_reader()
			print 'validating..';
			while(reader_valid.batch_num<reader_valid.n_batches):
				[valid_data_batch,valid_label_batch]=reader_valid.next_batch();
				feed_dict_validate={valid_data:valid_data_batch,valid_labels:valid_label_batch};
				pred_valid=sess.run(prediction_valid,feed_dict=feed_dict_validate);
				[corr,total_pix]=transform_labels(pred_valid,valid_label_batch,match_labels,num_classes)
				acc=corr*1.0/total_pix
				print 'epoch:',reader_valid.epoch+1,', Batch:',reader_valid.batch_num, ', correct pixels:', corr, ', Accuracy:',acc

		reader.epoch=reader.epoch+1
		reader.batch_num=0
		cnt=cnt+1		

def transform_labels(pred1,label_img,match_labels,num_classes):
	valid_labels=np.where(np.logical_and(label_img>=0,label_img<num_classes))
	pred=pred1[valid_labels]
	label_img=label_img[valid_labels]
	
	modpred_img=pred[:]
	for cl in range(num_classes):
		t=np.where(pred==cl)
		modpred_img[t]=int(match_labels[cl][-1])
	corr_pix=np.where(modpred_img==label_img)[0].size
	# non_exist=np.where(label_img==11)[0].size
	total_pix=modpred_img.size-non_exist
	return [corr_pix,total_pix]


def test_segnet():

	num_classes=12
	batch_size_test=2
	train_data_dir='SegNet-Tutorial/CamVid/train/'
	train_label_dir='SegNet-Tutorial/CamVid/trainannot/'
	test_data_dir='SegNet-Tutorial/CamVid/test/'
	test_label_dir='SegNet-Tutorial/CamVid/testannot/'

	# test_data_dir='/home/sriram/intern/datasets/pascal/VOC2011/jpeg_images/'
	# test_label_dir='/home/sriram/intern/datasets/pascal/VOC2011/labels/'
	reader_test=image_reader(train_data_dir,train_label_dir,batch_size_test,image_size=[360,480,3]);
	image_size=reader_test.image_size;

	sess=tf.Session();
	test_data = tf.placeholder(tf.float32,shape=[batch_size_test, image_size[0], image_size[1], image_size[2]])
	test_labels = tf.placeholder(tf.int64, shape=[batch_size_test, image_size[0], image_size[1]])

	net=Segnet(keep_prob=0.5,num_classes=num_classes,is_gpu=True,weights_path='segnet_road.npy')
	test_logits=net.inference(test_data,is_training=False,reuse=False);
	print 'built network';
	prediction=tf.argmax(test_logits,axis=3);
	net.rt['argmax']=prediction
	# accuracy=tf.size(tf.where(prediction==test_labels)[0]);
	print 'built loss graph'
	sess.run(tf.global_variables_initializer());
	print 'initialized vars';
	# plt.ion()
	file=open(os.path.join('match_labels.txt'))
	match_labels=file.readlines()
	file.close()
	match_labels=[line.splitlines()[0].split(' ') for line in match_labels]
	while(reader_test.batch_num<reader_test.n_batches):
		[test_data_batch,test_label_batch]=reader_test.next_batch();
		feed_dict={test_data:test_data_batch,test_labels:test_label_batch};
		pred=sess.run(prediction,feed_dict=feed_dict);
		[corr,total_pix]=transform_labels(pred,test_label_batch,match_labels,num_classes)
		acc=corr*1.0/total_pix
		print 'epoch:',reader_test.epoch+1,', Batch:',reader_test.batch_num, ', correct pixels:', corr, ', Accuracy:',acc

	

def viz(pred,test_label_batch,num_classes):
	viz=np.zeros([pred.shape[0]]+image_size)
	colors=color_map(num_classes)
	for cl in range(num_classes):
		t=np.where(pred==cl)
		viz[t]=colors[cl,:]
	plt.figure(1)
	plt.imshow(viz[0,:])
	plt.figure(2)
	plt.imshow(test_label_batch[0,:])
	plt.show()
	plt.pause(0.05)	

def predict_segnet():
	num_classes=12
	batch_size_test=2
	all_dir='SegNet-Tutorial/CamVid/allcamvid/'
	save_folder='predictions_camvid'
	save_to=os.path.dirname(os.path.abspath(all_dir))
	reader_test=single_reader(all_dir,batch_size_test,image_size=[360,480,3]);
	image_size=reader_test.image_size

	sess=tf.Session();
	test_data = tf.placeholder(tf.float32,shape=[batch_size_test, image_size[0], image_size[1], image_size[2]])
	test_labels = tf.placeholder(tf.int64, shape=[batch_size_test, image_size[0], image_size[1]])

	net=Segnet(keep_prob=0.5,num_classes=num_classes,is_gpu=True,weights_path='segnet_road.npy')
	test_logits=net.inference(test_data,is_training=False,reuse=False);
	print 'built network';
	
	prediction=tf.argmax(test_logits,axis=3);
	net.rt['argmax']=prediction
	accuracy=tf.size(tf.where(prediction==test_labels)[0]);
	print 'built loss graph'
	
	sess.run(tf.global_variables_initializer());
	print 'initialized vars';
	plt.ion()
	path=os.path.abspath(os.path.join(save_to,save_folder))

	if not os.path.exists(path):
		os.makedirs(os.path.join(save_to,save_folder))

	while(reader_test.batch_num<reader_test.n_batches):
		test_data_batch=reader_test.next_batch();
		file_names=[j.split('.')[-2] for j in reader_test.chunk_data]
		file_names=[j.split('/')[-1] for j in file_names]
		feed_dict={test_data:test_data_batch};
		pred=sess.run(prediction,feed_dict=feed_dict);
		for i in range(pred.shape[0]):
			sp.imsave(os.path.join(path,file_names[i]+'.png'),pred[i,:].astype('uint8'))
		print 'epoch:',reader_test.epoch+1,', Batch:',reader_test.batch_num

	
def evaluate_segnet_camvid():   
	#gives 87.7% accuracy on 11 common classes
	
	pred_path='/home/sriram/intern/datasets/SegNet_Output/CamVid_Output/'
	labels_path='/home/sriram/intern/datasets/CamVid/'
	num_classes=12
	file=open(os.path.join(pred_path,'match_labels.txt'))
	match_labels=file.readlines()
	file.close()
	match_labels=[line.splitlines()[0].split(' ') for line in match_labels]
	count=0;s=0
	for i in [f for f in os.listdir(pred_path) if os.path.isdir(os.path.join(pred_path,f))]:
		path1=os.path.join(pred_path,i)
		for prediction in fnmatch.filter(os.listdir(path1),'*.png'):
			pred_img=sp.imread(os.path.join(path1,prediction))
			label_img=sp.imread(os.path.join(labels_path,i+'annot',prediction))
			modpred_img=pred_img[:]
			for cl in range(num_classes):
				t=np.where(pred_img==cl)
				modpred_img[t]=int(match_labels[cl][-1])
			corr_pix=np.where(modpred_img==label_img)[0].size
			non_exist=np.where(label_img==11)[0].size
			total_pix=modpred_img.shape[0]*modpred_img.shape[1]-non_exist
			accuracy=corr_pix*1.0/total_pix
			s=s+accuracy
			count=count+1
			agg_acc=s/count
			print i,prediction,'img accuracy:',accuracy, 'aggregate accuracy:',agg_acc


def evaluate_segnet_arl(absent_classes):  
	#55% on lej15 and 45% on b507 after removing 5 absent classes. 
	#Accuracy suffers slightly upon adding those classes as a general object
	# Retrain segnet on ira7 dataset

	pred_path='/home/sriram/intern/datasets/data/data-with-labels/b507/predictions/'
	labels_path='/home/sriram/intern/datasets/data/data-with-labels/b507/new_labels/'
	# absent_classes=[7,8,10,11]
	num_classes=12
	file=open(os.path.join('/home/sriram/intern/datasets/data/data-with-labels/b507/','match_labels.txt'))
	match_labels=file.readlines()
	file.close()
	match_labels=[line.splitlines()[0].split(' ') for line in match_labels]
	count=0;s=0 
	for label in fnmatch.filter(os.listdir(labels_path),'*.png'):
		label_img=sp.imread(os.path.join(labels_path,label))
		pred_img=sp.imresize(sp.imread(os.path.join(pred_path,label)),label_img.shape,interp='nearest')
		modpred_img=-1*np.ones(pred_img.shape)
		for cl in range(num_classes):
			if cl in absent_classes:
				continue
			t=np.where(pred_img==cl)
			modpred_img[t]=int(match_labels[cl][-1])
		cond=np.where(np.logical_and(label_img!=255,modpred_img!=-1))
		corr_pix=np.where(modpred_img[cond]==label_img[cond])[0].size
		total_pix=cond[0].size
		accuracy=corr_pix*1.0/total_pix
		s=s+accuracy
		count=count+1
		agg_acc=s/count
		print label,'img accuracy:',accuracy, 'aggregate accuracy:',agg_acc

def save_hdf5(sess,var_list):
	file=h5py.File('segnet_model.h5','w')
	file.create_dataset()


if __name__=="__main__":
	parser = ArgumentParser()
	parser.add_argument('-devbox',type=int,default=0)
	args = parser.parse_args()
	
	if args.devbox:
	  BASE_DIR = '/root/segnet_vgg16'
	  os.environ['CUDA_VISIBLE_DEVICES']="0";
	else:
	  BASE_DIR = '/home/sriram/intern'
	  os.environ['CUDA_VISIBLE_DEVICES']="";
  
  train_segnet()
