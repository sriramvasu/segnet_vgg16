import tensorflow as tf
import numpy as np
import os
from image_reader_test import *
from utils_mod import *
from argparse import ArgumentParser
import fnmatch
from color_map import *
try:
  import h5py
  import matplotlib.pyplot as plt
  from color_map import *
except:
  pass

class Segnet():
	def __init__(self,keep_prob,num_classes,is_gpu,weights_path=None,pretrained=False):
		self.num_classes = num_classes
		self.keep_prob = keep_prob
		self.is_gpu=is_gpu
		self.pretrained=pretrained
		self.params=Param_loader()
		if weights_path is not None:
			self.pretrained=True
			self.params=Param_loader(weights_path)

	def inference(self,x,is_training,reuse):
		self.x=x
		self.is_training=is_training
		self.reuse=reuse
		self.logits=self.build_network()
		return self.logits

	def calc_loss(self,logits,labels,num_classes,weighted=False,weights=None):
		epsilon = tf.constant(value=1e-10)
		no_labels=tf.where(tf.logical_and(labels>=0,labels<num_classes))
		labels=tf.gather_nd(labels,no_labels)
		logits=tf.gather_nd(logits,no_labels)
		labels=tf.reshape(tf.one_hot(tf.reshape(labels,[-1]),num_classes),[-1,num_classes])
		logits=tf.reshape(logits,[-1,num_classes])
		if(weighted==False):
			cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross_entropy')
			cross_entropy_mean=tf.reduce_mean(cross_entropy)
		else:
			softmax=tf.nn.softmax(logits+epsilon)
			cross_entropy=-1*tf.reduce_sum(tf.mul(labels*tf.log(softmax+epsilon),weights.reshape([num_classes])),axis=1)
			cross_entropy_mean=tf.reduce_mean(cross_entropy)
		tf.add_to_collection('losses', cross_entropy_mean)
		loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		return loss

	def train(self,learning_rate):
		opt = tf.train.AdamOptimizer(learning_rate)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		# with tf.control_dependencies(update_ops):
		gradvar_list=opt.compute_gradients(self.loss)
		self.train_op=opt.apply_gradients(gradvar_list)

	def get_shape(self,x):
		return x.get_shape().as_list()

	def build_network(self):


		# self.inp_tensor = tf.nn.lrn(self.x, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='norm1');
		# with tf.variable_scope('preprocess',dtype=tf.float32):
		# 	mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
		# 	self.inp_tensor = self.x-mean;
		with tf.device('/cpu:0'):

			self.rt=dict()
			self.inp_tensor=self.x

			self.rt['data']=self.inp_tensor
			# Layer 1
			conv1_1=conv_bn(self.inp_tensor,[3,3],64,[1,1],name='conv1_1',phase_train=self.is_training,params=self.params,reuse=self.reuse);
			print_shape(conv1_1)
			

			conv1_2=conv_bn(conv1_1,[3,3],64,[1,1],name='conv1_2',phase_train=self.is_training,params=self.params,reuse=self.reuse);
			print_shape(conv1_2)

			if(self.is_gpu==True):
				with tf.device('/gpu:0'):
					pool1,pool1_mask=tf.nn.max_pool_with_argmax(conv1_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool1_gpu');
			else:
				pool1=tf.nn.max_pool(conv1_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool1_cpu');
			pool1=dropout(pool1,self.keep_prob,self.is_training)

			print_shape(pool1)
			# self.rt['conv1_1']=conv1_1
			# self.rt['conv1_2']=conv1_2
			# self.rt['pool1']=pool1
			# self.rt['pool1_mask']=pool1_mask


			# Layer 2
			conv2_1=conv_bn(pool1,[3,3],128,[1,1],name='conv2_1',phase_train=self.is_training,params=self.params,reuse=self.reuse);
			print_shape(conv2_1)

			conv2_2=conv_bn(conv2_1,[3,3],128,[1,1],name='conv2_2',phase_train=self.is_training,params=self.params,reuse=self.reuse);
			print_shape(conv2_2)

			if(self.is_gpu==True):
				with tf.device('/gpu:0'):
					pool2,pool2_mask=tf.nn.max_pool_with_argmax(conv2_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool2_gpu');
			else:
				pool2=tf.nn.max_pool(conv2_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool2_cpu');
			pool2=dropout(pool2,self.keep_prob,self.is_training)

			print_shape(pool2)
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
				with tf.device('/gpu:0'):
					pool3,pool3_mask=tf.nn.max_pool_with_argmax(conv3_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool3_gpu')
			else:
				pool3=tf.nn.max_pool(conv3_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool3_cpu')
			pool3=dropout(pool3,self.keep_prob,self.is_training)

			print_shape(pool3)
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
				with tf.device('/gpu:0'):
					pool4,pool4_mask=tf.nn.max_pool_with_argmax(conv4_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool4_gpu')
			else:
				pool4=tf.nn.max_pool(conv4_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool4_cpu')
			
			pool4=dropout(pool4,self.keep_prob,self.is_training)
			print_shape(pool4)

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
				with tf.device('/gpu:0'):
					pool5,pool5_mask=tf.nn.max_pool_with_argmax(conv5_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool5_gpu');
			else:
				pool5=tf.nn.max_pool(conv5_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool5_cpu');

			pool5=dropout(pool5,self.keep_prob,self.is_training)
			print_shape(pool5)

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
			
			conv5_1_D=dropout(conv5_1_D,self.keep_prob,self.is_training)
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
			print_shape(pool4_D)

			# decode 4
			conv4_3_D = conv_bn(pool4_D, [3,3], 512,[1,1], name="conv4_3_D", phase_train=self.is_training,params=self.params,reuse=self.reuse)
			print_shape(conv4_3_D);
			conv4_2_D = conv_bn(conv4_3_D, [3,3], 512,[1,1], name="conv4_2_D", phase_train=self.is_training,params=self.params,reuse=self.reuse)
			print_shape(conv4_2_D);
			conv4_1_D = conv_bn(conv4_2_D, [3,3], 256,[1,1], name="conv4_1_D", phase_train=self.is_training,params=self.params,reuse=self.reuse)
			print_shape(conv4_1_D);

			conv4_1_D=dropout(conv4_1_D,self.keep_prob,self.is_training)
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

			conv3_1_D=dropout(conv3_1_D,self.keep_prob,self.is_training)
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
			conv2_2_D = conv_bn(pool2_D, [3,3], 128,[1,1], name="conv2_2_D_retrain", phase_train=self.is_training,params=self.params,reuse=self.reuse)
			print_shape(conv2_2_D);
			conv2_1_D = conv_bn(conv2_2_D, [3,3], 64,[1,1], name="conv2_1_D_retrain", phase_train=self.is_training,params=self.params,reuse=self.reuse)
			print_shape(conv2_1_D);

			conv2_1_D=dropout(conv2_1_D,self.keep_prob,self.is_training)
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

def train_segnet(modelfile_name,logfile_name,train_data_dir,train_label_dir):
	num_classes=8
	n_epochs=100
	batch_size_train=3
	batch_size_valid=1
	lr_decay_every=5
	validate_every=1
	save_every=33
	base_lr=5e-5
	img_size=[360,480]
	#modelfile_name='retrain_4layer_moment'
	#logfile_name='lej_retrain_4layer_moment'
	#train_data_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/training_set/images/')
	#train_label_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/training_set/new_labels/')
	# modelfile_name='retrain_4layer_moment'
	# logfile_name='lej_retrain_4layer_moment'
	# train_data_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/training_set/images/')
	# train_label_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/training_set/new_labels/')
	test_data_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/val_set/images/')
	test_label_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/val_set/new_labels/')
	
	#train_data_dir=os.path.join(BASE_DIR,'SegNet-Tutorial/CamVid/train/')
	#train_label_dir=os.path.join(BASE_DIR,'SegNet-Tutorial/CamVid/trainannot/')
	#test_data_dir=os.path.join(BASE_DIR,'SegNet-Tutorial/CamVid/test/')
	#test_label_dir=os.path.join(BASE_DIR,'SegNet-Tutorial/CamVid/testannot/')

	reader=image_reader(train_data_dir,train_label_dir,batch_size_train,image_size=[360,480,3])
	reader_valid=image_reader(test_data_dir,test_label_dir,batch_size_valid,image_size=[360,480,3])
	image_size=reader.image_size
	n_batches=reader.n_batches
	f_train=open(logfile_name,'w+')
	sess=tf.Session()
	
	train_data=tf.placeholder(tf.float32,shape=[batch_size_train,image_size[0],image_size[1],image_size[2]])
	train_labels=tf.placeholder(tf.int64, shape=[batch_size_train, image_size[0], image_size[1]])
	valid_data=tf.placeholder(tf.float32,shape=[batch_size_valid,image_size[0],image_size[1],image_size[2]])
	valid_labels=tf.placeholder(tf.int64, shape=[batch_size_valid, image_size[0], image_size[1]])
	count=tf.placeholder(tf.int32,shape=[])
	rate=tf.placeholder(dtype=tf.float32,shape=[])
	learning_rate=tf.placeholder(dtype=tf.float32,shape=[])
	
	net=Segnet(keep_prob=0.7,num_classes=num_classes,is_gpu=True,weights_path='segnet_road.npy')
	train_logits=net.inference(train_data, is_training=True,reuse=False)
	valid_logits=net.inference(valid_data, is_training=False,reuse=True)
	print 'built network'

	file=open(os.path.join('camvid_match_labels.txt'))
	match_labels=file.readlines()
	file.close()
	match_labels=[line.splitlines()[0].split(' ') for line in match_labels]
	net.match_labels=match_labels

	net.loss=net.calc_loss(train_logits,train_labels,net.num_classes)
	#learning_rate=tf.train.exponential_decay(base_lr,count,1,1.0/rate)
	# learning_rate = base_lr*(1/rate)^count


	net.train(learning_rate)
	prediction_train=tf.argmax(train_logits,axis=3)
	prediction_valid=tf.argmax(valid_logits,axis=3)
	# accuracy=tf.size(tf.where(prediction==train_labels)[0]);

	print 'built loss graph'
	moment_vars=[]
        for var in tf.global_variables():
                if('moving_mean' in var.name or 'moving_variance' in var.name):
                        moment_vars.append(var)

	saver=tf.train.Saver(tf.trainable_variables()+moment_vars)
	sess.run(tf.global_variables_initializer())
	# saver.restore(sess,'segnet_model')
	print 'initialized vars'
	cnt=0
	dec=4
	colors=color_map(num_classes)
	lr_calc=base_lr

	while(reader.epoch<n_epochs):
		s_train=0;cnt_train=1	
		while(reader.batch_num<reader.n_batches):
			[train_data_batch,train_label_batch]=reader.next_batch()
			
			feed_dict_train={train_data:train_data_batch,train_labels:train_label_batch,learning_rate:lr_calc}
			[pred,_]=sess.run([prediction_train,net.train_op],feed_dict=feed_dict_train)
			[corr,total_pix]=transform_labels(pred,train_label_batch,match_labels,num_classes)
			acc=corr*1.0/total_pix
			s_train=s_train+acc
			print 'Learning_rate:',lr_calc,'epoch:',reader.epoch+1,', Batch:',reader.batch_num, ', correct pixels:', corr, ', Accuracy:',acc,'aggregate_acc:',s_train*1.0/cnt_train
			f_train.write('Training'+' learning_rate:'+str(lr_calc)+' epoch:'+str(reader.epoch+1)+' Batch:'+str(reader.batch_num)+' Accuracy:'+str(acc)+' aggregate_acc'+str(s_train*1.0/cnt_train)+'\n')
			cnt_train+=1

		if((reader.epoch+1)%save_every==0):
			saver.save(sess,modelfile_name,global_step=(reader.epoch+1))

		if((reader.epoch+1)%validate_every==0):
			reader_valid.reset_reader()
			s_valid=0;cnt_valid=1
			print 'validating..'
			while(reader_valid.batch_num<reader_valid.n_batches):
				[valid_data_batch,valid_label_batch]=reader_valid.next_batch()
				feed_dict_validate={valid_data:valid_data_batch,valid_labels:valid_label_batch}
				pred_valid=sess.run(prediction_valid,feed_dict=feed_dict_validate);
				[corr,total_pix]=transform_labels(pred_valid,valid_label_batch,match_labels,num_classes)
				acc=corr*1.0/total_pix
				s_valid=s_valid+acc
				

				print 'epoch:',reader_valid.epoch+1,', Batch:',reader_valid.batch_num, ', correct pixels:', corr, ', Accuracy:',acc,' aggregate acc:',s_valid*1.0/cnt_valid
				f_train.write('Validation'+' epoch:'+str(reader_valid.epoch+1)+' Batch:'+str(reader_valid.batch_num)+' Accuracy:'+str(acc)+' aggregate_acc'+str(s_valid*1.0/cnt_valid)+'\n')
				cnt_valid+=1
			#print 'Change lr/Save model?'
			#char=raw_input()
			#if(char=='c'):
			#	print 'Current lr:',lr_calc,'Enter new lr:'
			#	lr_calc=float(raw_input())
			#elif(char=='s'):
			#	saver.save(sess,modelfile_name,global_step=(reader.epoch+1))

				
		reader.epoch=reader.epoch+1
		reader.batch_num=0

def transform_labels(pred1,label_img,match_labels,num_classes):
	valid_labels=np.where(np.logical_and(label_img>=0,label_img<num_classes))
	pred=pred1[valid_labels]
	label_img=label_img[valid_labels]
	non_labels=[]
	modpred_img=-1*np.ones(pred.shape)
	#modpred_img=pred[:]
	non_exist=0
	#for cl in range(num_classes):
	#	t=np.where(pred==cl)
	#	modpred_img[t]=int(match_labels[cl][-1])
	corr_pix=np.where(pred==label_img)[0].size
	#per_class1=np.zeros([num_classes])
	#per_class2=np.zeros([num_classes])
	#for cl in range(num_classes):
	#	per_class1[cl]=np.where(np.logical_and(pred==cl,label_img==cl))[0].size*1.0/(np.where(label_img==cl)[0].size+1e-10)
	#	per_class2[cl]=np.where(label_img==cl)[0].size
	
	# su=0;matr=np.zeros([num_classes,num_classes])
	# for cl in range(num_classes):
	# 	if(cl not in non_labels):
	# 		t1=label_img==cl
	# 		for pl in range(num_classes):
	# 			t=np.where(np.logical_and(t1,modpred_img==pl))
	# 			matr[cl,pl]=t[0].size

	for cl in non_labels:
		non_exist=non_exist+np.where(label_img==cl)[0].size
	total_pix=modpred_img.size-non_exist
	return [corr_pix,total_pix]

def transform_labels_perclass(pred1,label_img,match_labels,num_classes):
 	valid_labels=np.where(np.logical_and(label_img>=0,label_img<num_classes))
        pred=pred1[valid_labels]
        label_img=label_img[valid_labels]
        non_labels=[]
        modpred_img=-1*np.ones(pred.shape)
        #modpred_img=pred[:]
        non_exist=0
        #for cl in range(num_classes):
        #       t=np.where(pred==cl)
        #       modpred_img[t]=int(match_labels[cl][-1])
        corr_pix=np.where(pred==label_img)[0].size
        per_class1=np.zeros([num_classes])
        per_class2=np.zeros([num_classes])
        for cl in range(num_classes):
               per_class1[cl]=np.where(np.logical_and(pred==cl,label_img==cl))[0].size*1.0/(np.where(label_img==cl)[0].size+1e-10)
               per_class2[cl]=np.where(label_img==cl)[0].size

        # su=0;matr=np.zeros([num_classes,num_classes])
        # for cl in range(num_classes):
        #       if(cl not in non_labels):
        #               t1=label_img==cl
        #               for pl in range(num_classes):
        #                       t=np.where(np.logical_and(t1,modpred_img==pl))
        #                       matr[cl,pl]=t[0].size

        for cl in non_labels:
                non_exist=non_exist+np.where(label_img==cl)[0].size
        total_pix=modpred_img.size-non_exist
        return [corr_pix,total_pix,per_class1,per_class2]
def test_segnet():

	num_classes=8
	batch_size_test=1
	train_data_dir='SegNet-Tutorial/CamVid/train/'
	train_label_dir='SegNet-Tutorial/CamVid/trainannot/'
	#test_data_dir='SegNet-Tutorial/CamVid/test/'
	#test_label_dir='SegNet-Tutorial/CamVid/testannot/'
	test_data_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/testing_set/images/')
	test_label_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/testing_set/new_labels/')

	modelfile_name='retrain_models/retrain_4layer_moment'
	epoch_number=99
	# test_data_dir='/home/sriram/intern/datasets/pascal/VOC2011/jpeg_images/'
	# test_label_dir='/home/sriram/intern/datasets/pascal/VOC2011/labels/'
	reader_test=image_reader(test_data_dir,test_label_dir,batch_size_test,image_size=[360,480,3])
	image_size=reader_test.image_size

	sess=tf.Session()
	test_data = tf.placeholder(tf.float32,shape=[batch_size_test, image_size[0], image_size[1], image_size[2]])
	test_labels = tf.placeholder(tf.int64, shape=[batch_size_test, image_size[0], image_size[1]])

	net=Segnet(keep_prob=0.5,num_classes=num_classes,is_gpu=True,weights_path='/home/sriram/intern/segnet_road.h5')
	test_logits=net.inference(test_data,is_training=False,reuse=False);
	print 'built network';
	prediction=tf.argmax(test_logits,axis=3);
	#net.rt['argmax']=prediction
	# accuracy=tf.size(tf.where(prediction==test_labels)[0]);
	print 'built loss graph'
	sess.run(tf.global_variables_initializer())
	print 'initialized vars'
	moment_vars=[]
	for var in tf.global_variables():
		if('moving_mean' in var.name or 'moving_variance' in var.name):
			moment_vars.append(var)

	saver=tf.train.Saver(tf.trainable_variables()+moment_vars)

	# saver.restore(sess,modelfile_name+'-'+str(epoch_number))
	print 'restored model'
	# plt.ion()
	file=open(os.path.join('camvid_match_labels.txt'))
	match_labels=file.readlines()
	file.close()
	match_labels=[line.splitlines()[0].split(' ') for line in match_labels]
	
	colors=color_map(num_classes)
	s_test=0;count_test=1
	while(reader_test.batch_num<reader_test.n_batches):
		[test_data_batch,test_label_batch]=reader_test.next_batch();
		feed_dict={test_data:test_data_batch,test_labels:test_label_batch};
		pred=sess.run(prediction,feed_dict=feed_dict)
		[corr,total_pix,per_class1,per_class2]=transform_labels_perclass(pred,test_label_batch,match_labels,num_classes)

		viz=np.zeros([pred.shape[0]]+image_size)
                for cl in range(num_classes):
                	t=np.where(pred==cl)
                	viz[t]=colors[cl,:]

		acc=corr*1.0/total_pix
		s_test=s_test+acc
		agg_acc=s_test/count_test
		count_test+=1
		print per_class1
		print per_class2
		print 'epoch:',reader_test.epoch+1,', Batch:',reader_test.batch_num, ', correct pixels:', corr, ', Accuracy:',acc,'Aggregate_acc:',agg_acc
		print '\n'
		sp.imsave('outimgs-'+'retrain_4layer_moment'+'-%d-%f.png'%(reader_test.batch_num,acc),viz[0,:])
		sp.imsave('outimgs_real-'+'retrain_4layer_moment'+'-%d-%f.png'%(reader_test.batch_num,acc),test_data_batch[0,:])
	

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
	file=open('camvid_match_labels.txt')
	match_labels=file.readlines()
	file.close()
	match_labels=[line.splitlines()[0].split(' ') for line in match_labels]
	count=0;s=0
	train_matr=np.zeros([num_classes,num_classes])
	val_matr=np.zeros([num_classes,num_classes])
	test_matr=np.zeros([num_classes,num_classes])

	for i in [f for f in os.listdir(pred_path) if os.path.isdir(os.path.join(pred_path,f))]:
		path1=os.path.join(pred_path,i)
		for prediction in fnmatch.filter(os.listdir(path1),'*.png'):
			pred_img=sp.imread(os.path.join(path1,prediction))
			label_img=sp.imread(os.path.join(labels_path,i+'annot',prediction))


			corr_pix,total_pix,matr=transform_labels(pred_img,label_img,match_labels,num_classes)
			if(i=='train'):
				train_matr=train_matr+matr
			elif(i=='val'):
				val_matr=val_matr+matr
			elif(i=='test'):
				test_matr=test_matr+matr
			# modpred_img=pred_img[:]
			# for cl in range(num_classes):
			# 	t=np.where(pred_img==cl)
			# 	modpred_img[t]=int(match_labels[cl][-1])
			# corr_pix=np.where(modpred_img==label_img)[0].size
			# non_exist=np.where(label_img==11)[0].size
			# total_pix=modpred_img.shape[0]*modpred_img.shape[1]-non_exist
			accuracy=corr_pix*1.0/total_pix
			s=s+accuracy
			count=count+1
			agg_acc=s/count
			# print i,prediction,'img accuracy:',accuracy, 'aggregate accuracy:',agg_acc
			# print '\n'
	# print 'train_matrix',train_matr
	# print 'val_matrix',val_matr
	# print 'test_matrix',test_matr
	np.save(os.path.join(pred_path,'train_conf_matrix.npy'),train_matr)
	np.save(os.path.join(pred_path,'val_conf_matrix.npy'),val_matr)
	np.save(os.path.join(pred_path,'test_conf_matrix.npy'),test_matr)

def evaluate_segnet_camvid_small():
        #gives 87.7% accuracy on 11 common classes

        pred_path='./predictions_camvid_another/'
        labels_path='./SegNet-Tutorial/CamVid/trainannot/'
        num_classes=12
        file=open('camvid_match_labels.txt')
        match_labels=file.readlines()
        file.close()
        match_labels=[line.splitlines()[0].split(' ') for line in match_labels]
        count=0;s=0
        train_matr=np.zeros([num_classes,num_classes])
        val_matr=np.zeros([num_classes,num_classes])
        test_matr=np.zeros([num_classes,num_classes])

        
        path1=pred_path
        for prediction in fnmatch.filter(os.listdir(path1),'*.png'):
                        pred_img=sp.imread(os.path.join(path1,prediction))
                        label_img=sp.imread(os.path.join(labels_path,prediction))


                        corr_pix,total_pix=transform_labels(pred_img,label_img,match_labels,num_classes)
                        
                        accuracy=corr_pix*1.0/total_pix
                        s=s+accuracy
                        count=count+1
                        agg_acc=s/count
                        print prediction,'img accuracy:',accuracy, 'aggregate accuracy:',agg_acc
                        # print '\n'
        # print 'train_matrix',train_matr
        # print 'val_matrix',val_matr
        # print 'test_matrix',test_matr
        #np.save(os.path.join(pred_path,'train_conf_matrix.npy'),train_matr)
        #np.save(os.path.join(pred_path,'val_conf_matrix.npy'),val_matr)
        #np.save(os.path.join(pred_path,'test_conf_matrix.npy'),test_matr)

def evaluate_segnet_arl(absent_classes=[]):  
	#55% on lej15 and 45% on b507 after removing 5 absent classes. 
	#Accuracy suffers slightly upon adding those classes as a general object
	# Retrain segnet on ira7 dataset
	se=['training_set','val_set','testing_set']
	num_classes=12

	train_matr=np.zeros([num_classes,num_classes])
	val_matr=np.zeros([num_classes,num_classes])
	test_matr=np.zeros([num_classes,num_classes])
	file=open('lej_match_labels.txt')
	match_labels=file.readlines()
	file.close()
	match_labels=[line.splitlines()[0].split(' ') for line in match_labels]
	count_train=0;s_train=0
	count_val=0;s_val=0
	count_test=0;s_test=0

	for se_elem in se:
		pred_path='/home/sriram/intern/datasets/data/data-with-labels/lej15/'+se_elem+'/predictions/'
		labels_path='/home/sriram/intern/datasets/data/data-with-labels/lej15/'+se_elem+'/new_labels/'
		# absent_classes=[7,8,10,11]


		for label in fnmatch.filter(os.listdir(labels_path),'*.png'):
			label_img=sp.imread(os.path.join(labels_path,label))
			pred_img=sp.imresize(sp.imread(os.path.join(pred_path,label)),label_img.shape,interp='nearest')

			[corr_pix,total_pix,matr]=transform_labels(pred_img,label_img,match_labels,num_classes)
			accuracy=corr_pix*1.0/total_pix
			if(se_elem=='training_set'):
				train_matr=train_matr+matr
				s_train=s_train+accuracy
				count_train=count_train+1
				agg_acc_train=s_train/count_train
				print se_elem,label,'img accuracy:',accuracy, 'aggregate accuracy:',agg_acc_train

			elif(se_elem=='val_set'):
				val_matr=val_matr+matr
				s_val=s_val+accuracy
				count_val=count_val+1
				agg_acc_val=s_val/count_val
				print se_elem,label,'img accuracy:',accuracy, 'aggregate accuracy:',agg_acc_val

			elif(se_elem=='testing_set'):
				test_matr=test_matr+matr
				s_test=s_test+accuracy
				count_test=count_test+1
				agg_acc_test=s_test/count_test
				print se_elem,label,'img accuracy:',accuracy, 'aggregate accuracy:',agg_acc_test

	print train_matr
	print val_matr
	print test_matr
	print count_train
	print count_val
	print count_test
	np.save(os.path.join(os.path.join(pred_path,'..'),'train_conf_matrix.npy'),train_matr)
	np.save(os.path.join(os.path.join(pred_path,'..'),'val_conf_matrix.npy'),val_matr)
	np.save(os.path.join(os.path.join(pred_path,'..'),'test_conf_matrix.npy'),test_matr)


def save_hdf5(sess,var_list):
	file=h5py.File('segnet_model.h5','w')
	file.create_dataset()


if __name__=="__main__":
	parser = ArgumentParser()
	parser.add_argument('-devbox',type=int,default=0)
	parser.add_argument('-ngpu',type=int,default=0)
	args = parser.parse_args()
	
	if args.devbox:
		BASE_DIR = '/root/segnet_vgg16'
		os.environ['CUDA_VISIBLE_DEVICES']=str(args.ngpu)
		print os.system('echo CUDA_VISBLE_DEVICES')
	else:
	  BASE_DIR = '/home/sriram/intern'
	  os.environ['CUDA_VISIBLE_DEVICES']=""
	lis=['full','half','third','quarter']
	mfile_names=['retrain_'+i+'_moment'for i in lis]
	lfile_names=['lej_retrain_'+i+'_moment' for i in lis]

        train_data_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/training_set/images/')
        train_label_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/training_set/new_labels/')
        test_data_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/val_set/images/')
        test_label_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/val_set/new_labels/')
	

	total=len(os.listdir(train_data_dir))
	tr_num=[total,int(total/2),int(total/3),int(total/4)]
	for modelfile_name,logfile_name,n_train_samples in zip(mfile_names,lfile_names,total):
		if(not os.isdir('./scratch_train_data_dir')):
                	os.makedirs('./scratch_train_data_dir')
       
        	if(not os.isdir('./scratch_train_label_dir')):
                	os.makedirs('./scratch_train_label_dir')
		scratch_train_data_dir=os.path.abspath('./scratch_train_data_dir')
        	scratch_train_label_dir=os.path.abspath('./scratch_train_label_dir')
		
		rand_index=np.random.randint(0,total,[n_train_samples])
		image_filenames=fnmatch.filter(os.listdir(train_data_dir),'*.png')[rand_index]
		for name in image_filenames:
			shutil.copy(os.path.join(train_data_dir,name),scratch_train_data_dir)
			shutil.copy(os.path.join(train_label_dir,name),scratch_train_label_dir)
		
		train_segnet(modelfile_name,logfile_name,scratch_train_data_dir,scratch_train_label_dir)
		os.rmtree(scratch_train_data_dir)
		os.rmtree(scratch_train_label_dir)
	#test_segnet()
	#train_segnet()
		BASE_DIR = '/home/sriram/intern'
		os.environ['CUDA_VISIBLE_DEVICES']="0"
	# lis=['1layer','2layer','3layer','4layer']
	# mfile_names=['retrain_'+i+'_moment'for i in lis]
	# lfile_names=['lej_retrain_'+i+'_moment' for i in lis]

	# train_data_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/training_set/images/')
	# train_label_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/training_set/new_labels/')
	# test_data_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/val_set/images/')
	# test_label_dir=os.path.join(BASE_DIR,'datasets/data/data-with-labels/lej15/val_set/new_labels/')
	
	# if(not os.isdir('./scratch_train_data_dir'):
	# 	os.makedirs('./scratch_train_data_dir')
	# if(not os.isdir('./scratch_test_data_dir'):
	# 	os.makedirs('./scratch_test_data_dir')
	# if(not os.isdir('./scratch_train_label_dir'):
	# 	os.makedirs('./scratch_train_label_dir')
	# if(not os.isdir('./scratch_test_label_dir'):
	# 	os.makedirs('./scratch_test_label_dir')

	# scratch_train_data_dir=os.path.abspath('./scratch_train_data_dir')
	# scratch_train_label_dir=os.path.abspath('./scratch_train_label_dir')

	# total=len(os.listdir(train_data_dir))
	# tr_num=[total,int(total/2),int(total/4)]
	# for n in tr_num:
	# 	rand_index=np.random.randint(0,tr_num,[tr_num])
	# 	image_names=os.listdir(train_data_dir)[rand_index]
	# 	for name in image_names:
	# 		shutil.copy(os.path.join(train_data_dir,name),scratch_train_data_dir)
	# 		shutil.copy(os.path.join(train_label_dir,name),scratch_train_label_dir)


	# for modelfile_name,logfile_name in zip(mfile_names,lfile_names):
	# 	print modelfile_name,logfile_name
	# 	train_segnet(modelfile_name,logfile_name,scratch_train_data_dir,scratch_train_label_dir)
	test_segnet()
	# train_segnet()
	#evaluate_segnet_camvid_small()
	# evaluate_segnet_arl()
