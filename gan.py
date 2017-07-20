import tensorflow as tf
import numpy as np
from image_reader import image_reader
import os
from utils_mod import *


class DCGAN():
	def __init__(self,keep_prob=0.5,is_gpu=False,weights_path=None,pretrained=False):

		self.keep_prob = keep_prob
		self.is_gpu=is_gpu;
		self.pretrained=pretrained;
		self.params=Param_loader();
		if weights_path is not None:
			self.pretrained=True;
			self.params=Param_loader(weights_path);

	def inference(self,x,z,is_training):
		self.x=x;
		self.z=z;
		self.is_training=is_training;
		self.z_shape=tf.shape(z);
		self.x_shape=tf.shape(x);
		gen_out=self.build_Generator()
		disc_out=self.build_Discriminator()
		gen_loss=calc_loss_generator()
		disc_loss=calc_loss_discriminator()



	def train_discriminator():
		
	def train_generator():


	def calc_loss_generator(self,gen_out,labels,num_classes,weighted=False,weights=None):
		epsilon = tf.constant(value=1e-10)
		labels=tf.reshape(tf.one_hot(tf.reshape(labels,[-1]),num_classes),[-1,num_classes]);
		logits=tf.reshape(logits,[-1,num_classes]);
		if(weighted==False):
			# cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross_entropy');
			# cross_entropy_mean=tf.reduce_mean(cross_entropy);
			softmax=tf.nn.softmax(logits+epsilon);
			cross_entropy=-1*tf.reduce_sum(tf.mul(labels*tf.log(softmax+epsilon),weights.reshape([num_classes])),axis=1);
			cross_entropy_mean=tf.reduce_mean(cross_entropy);

		else:
			softmax=tf.nn.softmax(logits+epsilon);
			cross_entropy=-1*tf.reduce_sum(tf.mul(labels*tf.log(softmax+epsilon),weights.reshape([num_classes])),axis=1);
			cross_entropy_mean=tf.reduce_mean(cross_entropy);
		tf.add_to_collection('losses', cross_entropy_mean);
		loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		return loss;


	def calc_loss_discriminator(self,disc_out,disc_labels,num_classes,weighted=False,weights=None):
		epsilon = tf.constant(value=1e-10)
		disc_labels=tf.reshape(tf.one_hot(tf.reshape(disc_labels,[-1]),num_classes),[-1,num_classes]);
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


	def build_Generator(self):
		with tf.variable_scope('generator'):
			z_proj=fc_flatten(self.z, np.prod([4,4,1024]), 'reshape', phase_train=self.is_training)
			ups1_1=upscore_layer(z_proj, [2,2], [2,2], 512, 'deconv1_1');
			ups1_2=conv_bn(ups1_1, [3,3], 512, [1,1], 'conv1_2', self.is_training)
			ups1_3=conv_bn(ups1_2, [3,3], 512, [1,1], 'conv1_3', self.is_training);

			# layer 2
			ups2_1=upscore_layer(ups1_3, [2,2], [2,2], 256, 'deconv2_1');
			ups2_2=conv_bn(ups2_1, [3,3], 256, [1,1], 'conv2_2', self.is_training);
			ups2_3=conv_bn(ups2_2, [3,3], 256, [1,1], 'conv2_3', self.is_training);

			ups2_1=upscore_layer(ups1_3, [2,2], [2,2], 128, 'deconv3_1');
			ups2_2=conv_bn(ups2_1, [3,3], 128, [1,1], 'conv2_2', self.is_training);
			ups2_3=conv_bn(ups2_2, [3,3], 128, [1,1], 'conv2_3', self.is_training);

			ups3_1=upscore_layer(ups1_3, [2,2], [2,2], 3, 'deconv4_1');
			ups3_2=conv_bn(ups2_1, [3,3], 3, [1,1], 'conv2_2', self.is_training);
			ups3_3=conv_bn(ups2_2, [3,3], 3, [1,1], 'conv2_3', self.is_training);
			self.gen_out=ups3_3
		return gen_out


	def build_Discriminator(self):
		with tf.variable_scope('discriminator'):
			conv1_1=conv_bn(ups1_1, [3,3], 512, [1,1], 'conv1_1', self.is_training)
			conv1_2=conv_bn(ups1_2, [3,3], 512, [1,1], 'conv1_2', self.is_training);

			# layer 2
			conv2_1=conv_bn(ups2_1, [3,3], 256, [1,1], 'conv2_1', self.is_training);
			conv2_2=conv_bn(ups2_2, [3,3], 256, [1,1], 'conv2_2', self.is_training);

			conv3_1=conv_bn(ups2_1, [3,3], 128, [1,1], 'conv3_1', self.is_training);
			conv3_2=conv_bn(ups2_2, [3,3], 128, [1,1], 'conv3_2', self.is_training);

			ups3_1=upscore_layer(ups1_3, [2,2], [2,2], 3, 'deconv4_1');
			ups3_2=conv_bn(ups2_1, [3,3], 3, [1,1], 'conv2_2', self.is_training);
			ups3_3=conv_bn(ups2_2, [3,3], 3, [1,1], 'conv2_3', self.is_training);
			self.gen_out=ups3_3
		return gen_out





def train_segnet():
	# max_steps=FLAGS.max_steps;
	# batch_size=FLAGS.batch_size;
	# train_dir=FLAGS.train_dir;
	# test_dir=FLAGS.test_dir;
	# image_size=FLAGS.img_size;

	# num_classes=12;
	# n_epochs=100;
	# batch_size=6;
	# batch_size_valid=1;
	# save_every=6;
	# validate_every=3;
	# base_lr=1e-4;
	reader=image_reader(train_data_dir,train_label_dir,batch_size);
	reader_valid=image_reader(test_data_dir,test_label_dir,batch_size_valid);

	image_size=reader.image_size;
	n_batches=reader.n_batches;


	sess=tf.Session();


	phase_train = tf.placeholder(tf.bool,name='phase_train');
	count=tf.placeholder(tf.int32,shape=[]);
	
	net=DCGAN(keep_prob=0.5,is_gpu=False,weights_path=None);
	net.inference(train_data,phase_train);
	# valid_logits=net.inference(valid_data,phase_train);
	print 'built network';

	net.loss=net.calc_loss(net.logits,train_labels,net.num_classes);
	learning_rate=tf.train.exponential_decay(base_lr,count,n_epochs*n_batches,0.4)
	net.train(learning_rate);
	prediction=tf.argmax(net.logits,axis=3);
	# accuracy=tf.size(tf.where(prediction==train_labels)[0]);
	saver=tf.train.Saver(tf.trainable_variables())

	sess.run(tf.global_variables_initializer());
	print 'initialized vars';
	cnt=0;
	while(reader.epoch<n_epochs):


		[train_data_batch,train_label_batch]=reader.next_batch();
		feed_dict_train={phase_train:True,train_data:train_data_batch,train_labels:train_label_batch,count:cnt};
		[logits,loss,pred]=sess.run([net.logits,net.loss,prediction],feed_dict=feed_dict_train);
		corr=np.where(train_label_batch==pred)[0].size;
		acc=corr*1.0/(np.prod(image_size[:-1])*batch_size);
		sess.run(net.train_op,feed_dict=feed_dict_train);
		
		if((reader.epoch+1)%save_every==0):
			saver.save(sess,'segnet_model',global_step=(reader.epoch+1))
		print 'learning rate:', sess.run(learning_rate,feed_dict={count:cnt}),'epoch:',reader.epoch+1,'Batch:',reader.batch_num, 'correct pixels:', corr, 'Accuracy:',acc;
		
		if((reader.epoch+1)%validate_every==0):
			print 'validating..';
			while(reader_valid.epoch==0):
				[valid_data_batch,valid_label_batch]=reader_valid.next_batch();
				feed_dict_validate={phase_train:False,train_data:valid_data_batch,train_labels:valid_label_batch};
				pred_valid=sess.run([prediction],feed_dict=feed_dict_validate);
				corr_valid=np.where(valid_label_batch==pred_valid)[0].size;
				acc_valid=corr_valid*1.0/(np.prod(image_size[:-1])*batch_size_valid);
				print 'Validation ','Batch:',reader_valid.batch_num, 'correct pixels:', corr_valid, 'Accuracy:',acc_valid;

		cnt=cnt+1


def test_segnet():
	# max_steps=FLAGS.max_steps;
	# batch_size=FLAGS.batch_size;
	# train_dir=FLAGS.train_dir;
	# test_dir=FLAGS.test_dir;
	# image_size=FLAGS.img_size;
	num_classes=12;
	n_epochs=10;
	batch_size=6;
	train_data_dir='SegNet-Tutorial/CamVid/train/'
	train_label_dir='SegNet-Tutorial/CamVid/trainannot/'
	test_data_dir='SegNet-Tutorial/CamVid/test/'
	test_label_dir='SegNet-Tutorial/CamVid/testannot/'
	reader=image_reader(test_data_dir,test_label_dir,batch_size);
	image_size=reader.image_size;

	sess=tf.Session();
	test_data = tf.placeholder(tf.float32,shape=[batch_size, image_size[0], image_size[1], image_size[2]])
	test_labels = tf.placeholder(tf.int64, shape=[batch_size, image_size[0], image_size[1]])
	phase_train = tf.placeholder(tf.bool, name='phase_train');

	net=Segnet(keep_prob=0.5,num_classes=num_classes,is_gpu=False,weights_path='vgg16.npy');
	net.inference(test_data,phase_train);
	print 'built network';
	prediction=tf.argmax(net.logits,axis=3);
	# accuracy=tf.size(tf.where(prediction==test_labels)[0]);
	sess.run(tf.global_variables_initializer());
	print 'initialized vars';
	while(reader.epoch==0):
		[test_data_batch,test_label_batch]=reader.next_batch();

		feed_dict={phase_train:False,test_data:test_data_batch,test_labels:test_label_batch};
		[logits,pred]=sess.run([net.logits,prediction],feed_dict=feed_dict);
		corr=np.where(test_label_batch==pred)[0].size;
		acc=corr*1.0/(np.prod(image_size[:-1])*batch_size);

		print 'epoch:',reader.epoch+1,'Batch:',reader.batch_num, 'correct pixels:', corr, 'Accuracy:',acc


os.environ['CUDA_VISIBLE_DEVICES']="";
train_segnet()
