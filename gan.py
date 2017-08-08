import tensorflow as tf
import numpy as np
from image_reader import image_reader
import os
from utils_mod import *
import h5py
import fnmatch

class DCGAN():
	def __init__(self,keep_prob=0.5,is_gpu=False,weights_path=None,pretrained=False):

		self.keep_prob = keep_prob
		self.is_gpu=is_gpu;
		self.pretrained=pretrained;
		self.params=Param_loader();
		if weights_path is not None:
			self.pretrained=True;
			self.params=Param_loader(weights_path);

	def inference(self,z,x=None,is_training=False,reuse=False):
		# self.x=x;
		self.z=z
		self.is_training=is_training
		self.batch_size_z=self.z.get_shape().as_list()[0]
		self.rand_vect_size=self.z.get_shape().as_list()[1]
		self.reuse=reuse
		self.gen_out=self.Generator(self.z)
		self.disc_out_generated=self.Discriminator(self.gen_out)
		if x is not None:
			self.x=x
			self.disc_out_images=self.Discriminator(self.x)
		# self.disc_out_images

	def train(self,x,z,is_training=True,reuse=False,gen_lr=1e-5,disc_lr=1e-5):
		self.x=x
		self.batch_size_x=self.x.get_shape().as_list()[0]
		self.gen_lr=gen_lr
		self.disc_lr=disc_lr
		self.z=z
		self.batch_size_z=self.z.get_shape().as_list()[0]
		self.rand_vect_size=self.z.get_shape().as_list()[1]
		self.is_training=is_training
		self.reuse=reuse

		self.gen_out=self.Generator(self.z,reuse=self.reuse)
		self.disc_out_generated=self.Discriminator(self.gen_out,reuse=self.reuse)

		self.disc_out_images=self.Discriminator(self.x,reuse=True)
		self.gen_loss=self.calc_loss_generator()
		self.disc_loss=self.calc_loss_discriminator()

		disc_var_names=fnmatch.filter([v.name.split(':')[0] for v in tf.trainable_variables()],'discriminator*')
		gen_var_names=fnmatch.filter([v.name.split(':')[0] for v in tf.trainable_variables()],'generator*')

		with tf.variable_scope('',reuse=True):
			self.disc_vars=[tf.get_variable(name=name) for name in disc_var_names]
		with tf.variable_scope('',reuse=True):
			self.gen_vars=[tf.get_variable(name=name) for name in gen_var_names]

		# self.gen_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator')
		# self.disc_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')
		self.train_discriminator()
		self.train_generator()



	def train_discriminator(self):
		opt = tf.train.AdamOptimizer(self.disc_lr)
		gradvar_list=opt.compute_gradients(self.disc_loss,var_list=self.disc_vars)
		self.disc_train_op=opt.apply_gradients(gradvar_list)

		
	def train_generator(self):
		opt = tf.train.AdamOptimizer(self.gen_lr)
		gradvar_list=opt.compute_gradients(self.gen_loss,var_list=self.gen_vars)
		self.gen_train_op=opt.apply_gradients(gradvar_list)


	def calc_loss_generator(self,num_classes=2,weighted=False,weights=None):
		epsilon = tf.constant(value=1e-10)
		gen_labels=tf.ones([self.batch_size_z],dtype=tf.int32)
		gen_logits=self.disc_out_generated

		gen_labels=tf.reshape(tf.one_hot(tf.reshape(gen_labels,[-1]),num_classes),[-1,num_classes]);
		gen_logits=tf.reshape(gen_logits,[-1,num_classes]);
		if(weighted==False):
			cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=gen_labels,logits=gen_logits);
			cross_entropy_mean=tf.reduce_mean(cross_entropy);
			# softmax=tf.nn.softmax(gen_logits+epsilon)
			# cross_entropy=-1*tf.reduce_sum(gen_labels*tf.log(softmax+epsilon),axis=1)
			# cross_entropy_mean=tf.reduce_mean(cross_entropy)

		else:
			softmax=tf.nn.softmax(gen_logits+epsilon)
			cross_entropy=-1*tf.reduce_sum(tf.multiply(gen_labels*tf.log(softmax+epsilon),weights.reshape([num_classes])),axis=1)
			cross_entropy_mean=tf.reduce_mean(cross_entropy)
		tf.add_to_collection('gen_losses', cross_entropy_mean)
		loss = tf.add_n(tf.get_collection('gen_losses'), name='total_loss')
		return loss


	def calc_loss_discriminator(self,num_classes=2,weighted=False,weights=None):
		epsilon = tf.constant(value=1e-10)
		disc_labels=tf.concat([tf.zeros([self.batch_size_z],dtype=tf.int32),tf.ones([self.batch_size_x],dtype=tf.int32)],0)
		disc_logits=tf.concat([self.disc_out_generated,self.disc_out_images],0)

		disc_labels=tf.reshape(tf.one_hot(tf.reshape(disc_labels,[-1]),num_classes),[-1,num_classes])
		disc_logits=tf.reshape(disc_logits,[-1,num_classes])
		if(weighted==False):
			cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=disc_labels,logits=disc_logits);
			cross_entropy_mean=tf.reduce_mean(cross_entropy);
			# softmax=tf.nn.softmax(disc_logits+epsilon)
			# cross_entropy=-1*tf.reduce_sum(disc_labels*tf.log(softmax+epsilon),axis=1)
			# cross_entropy_mean=tf.reduce_mean(cross_entropy)

		else:
			softmax=tf.nn.softmax(disc_logits+epsilon)
			cross_entropy=-1*tf.reduce_sum(tf.multiply(disc_labels*tf.log(softmax+epsilon),weights.reshape([num_classes])),axis=1)
			cross_entropy_mean=tf.reduce_mean(cross_entropy)
		tf.add_to_collection('disc_losses', cross_entropy_mean)
		loss = tf.add_n(tf.get_collection('disc_losses'), name='total_loss')
		return loss;


	def Generator(self,gen_input,reuse):
		self.gen_input=tf.reshape(gen_input,[-1,self.rand_vect_size])
		# generator input shape bsxrand_vect_size
		# generator output shape bsxhxwx1
		with tf.variable_scope('generator'):
			print_shape(self.gen_input)
			z_proj=fc_flatten(self.gen_input, np.prod([7,7,256]),name='fc_layer', phase_train=self.is_training,reuse=self.reuse)
			z_img=tf.reshape(z_proj,[-1,7,7,256])
			ups1_1=upscore_layer(z_img, [2,2], [2,2], 256, 'deconv1_1',phase_train=self.is_training,reuse=reuse,out_shape=[0,14,14,0])
			# ups1_1=upsample(z_img)
			ups1_2=conv_bn(ups1_1, [3,3], 256, [1,1], 'conv1_2', self.is_training,reuse=reuse,activation='leakyrelu')
			ups1_3=conv_bn(ups1_2, [3,3], 256, [1,1], 'conv1_3', self.is_training,reuse=reuse,activation='leakyrelu')
			print_shape(ups1_3)
			# layer 2
			# ups2_1=upscore_layer(ups1_3, [2,2], [2,2], 256, 'deconv2_1');
		
			# ups2_1=upsample(ups1_3)
			ups2_1=ups1_3
			ups2_2=conv_bn(ups2_1, [3,3], 128, [1,1], 'conv2_2', self.is_training,reuse=reuse,activation='leakyrelu')
			ups2_3=conv_bn(ups2_2, [3,3], 128, [1,1], 'conv2_3', self.is_training,reuse=reuse,activation='leakyrelu')
			print_shape(ups2_3)


			ups3_1=upscore_layer(ups2_3, [2,2], [2,2], 128, 'deconv3_1',phase_train=self.is_training,reuse=reuse,out_shape=[0,28,28,0])
			# ups3_1=upsample(ups2_3)
			ups3_2=conv_bn(ups3_1, [3,3], 64, [1,1], 'conv3_2', self.is_training,reuse=reuse,activation='leakyrelu')
			ups3_3=conv_bn(ups3_2, [3,3], 64, [1,1], 'conv3_3', self.is_training,reuse=reuse,activation='leakyrelu')
			print_shape(ups3_3)

			# ups4_1=upscore_layer(ups3_3, [2,2], [2,2], 3, 'deconv4_1');
			# ups4_1=upsample(ups3_3)
			ups4_1=ups3_3
			ups4_2=conv_bn(ups4_1, [3,3], 32, [1,1], 'conv4_2', self.is_training,reuse=reuse,activation='leakyrelu')
			ups4_3=conv_bn(ups4_2, [3,3], 1, [1,1], 'conv4_3', self.is_training,reuse=reuse,batch_norm=False,activation='tanh')
			print_shape(ups4_3)

			self.gen_output=ups4_3
			print_shape(self.gen_output)
		return self.gen_output


	def Discriminator(self,disc_input,reuse):
		self.disc_input=tf.reshape(disc_input,[-1,28,28,1])
		# disc input shape bsxhxwx1
		# disc output shape bsx2
		with tf.variable_scope('discriminator'):
			conv1_1=conv_bn(disc_input, [3,3], 256, [1,1], 'conv1_1', self.is_training,reuse=reuse,activation='leakyrelu')
			conv1_2=conv_bn(conv1_1, [3,3], 256, [1,1], 'conv1_2', self.is_training,reuse=reuse,activation='leakyrelu')
			conv1_2=tf.nn.max_pool(conv1_2,[1,2,2,1],[1,2,2,1],'SAME')
			print_shape(conv1_2)

			# layer 2
			conv2_1=conv_bn(conv1_2, [3,3], 128, [1,1], 'conv2_1', self.is_training,reuse=reuse,activation='leakyrelu')
			conv2_2=conv_bn(conv2_1, [3,3], 128, [1,1], 'conv2_2', self.is_training,reuse=reuse,activation='leakyrelu')
			conv2_2=tf.nn.max_pool(conv2_2,[1,2,2,1],[1,2,2,1],'SAME')
			print_shape(conv2_2)


			conv3_1=conv_bn(conv2_2, [3,3], 64, [1,1], 'conv3_1', self.is_training,reuse=reuse,activation='leakyrelu')
			conv3_2=conv_bn(conv3_1, [3,3], 64, [1,1], 'conv3_2', self.is_training,reuse=reuse,activation='leakyrelu')
			conv3_2=tf.nn.max_pool(conv3_2,[1,2,2,1],[1,2,2,1],'SAME')
			print_shape(conv3_2)


			conv4_1=conv_bn(conv3_2, [3,3], 32, [1,1], 'conv4_2', self.is_training,reuse=reuse)
			conv4_2=conv_bn(conv4_1, [3,3], 32, [1,1], 'conv4_3', self.is_training,reuse=reuse)
			conv4_2=tf.nn.max_pool(conv4_2,[1,2,2,1],[1,2,2,1],'SAME')
			print_shape(conv4_2)


			flat1_1=fc_flatten(conv4_2,100,'fc1',self.is_training,reuse=reuse)
			flat1_2=fc_flatten(flat1_1,2,'fc2',self.is_training,reuse=reuse)
			self.disc_output=flat1_2
			print_shape(self.disc_output)

		return self.disc_output


def train_DCGAN():
	f=h5py.File('mnist.h5','r')
	x_train=f['x_train'][:]
	y_train=f['y_train'][:]
	x_train=(2*x_train.astype(np.float32)-127)/127
	batch_size=50
	save_every=1
	rand_vect_size=100
	gen_base_lr=1e-6
	disc_base_lr=1e-5
	train_data=tf.placeholder(dtype=tf.float32,shape=[batch_size,28,28,1])
	random_vector=tf.placeholder(dtype=tf.float32,shape=[batch_size,rand_vect_size])
	train_labels=tf.placeholder(dtype=tf.int32,shape=[batch_size])
	count=tf.placeholder(dtype=tf.int32)

	gan=DCGAN()
	gen_lr=tf.train.exponential_decay(gen_base_lr,count,1,0.5)
	disc_lr=tf.train.exponential_decay(disc_base_lr,count,1,0.5)
	gan.train(train_data,random_vector,gen_lr=gen_lr,disc_lr=disc_lr,is_training=False,reuse=False)
	n_epochs=100
	n_batches=x_train.shape[0]/batch_size
	sess=tf.Session()
	sess.run(tf.global_variables_initializer())
	# save_npy(gan.gen_vars,sess,'gen_vars',0)
	# save_npy(gan.disc_vars,sess,'disc_vars',0)
	for i in range(n_epochs):
		for j in range(n_batches):
			feed_dict={random_vector:np.random.standard_normal([batch_size,rand_vect_size]),train_data:next_batch(x_train,batch_size),count:0}
			[_,g_loss]=sess.run([gan.gen_train_op,gan.gen_loss],feed_dict)
			[_,d_loss]=sess.run([gan.disc_train_op,gan.disc_loss],feed_dict)

			feed_dict[random_vector]=np.random.standard_normal([batch_size,rand_vect_size])
			[_,g_loss]=sess.run([gan.gen_train_op,gan.gen_loss],feed_dict)
			print 'epoch:',i,'batch:',j,'gen_loss:',g_loss,'disc_loss:',d_loss
		if((i+1)%save_every==0):
			save_npy(gan.gen_vars,sess,'gen_vars',i+1)
			save_npy(gan.disc_vars,sess,'disc_vars',i+1)

	
def save_npy(list_vars,sess,base_name,suffix):
	rt=dict()
	for i in list_vars:
		rt[i.name]=sess.run(i)
	filename=base_name+str(suffix)+'.npy'
	np.save(filename,rt)
			
def save_hdf5(list_vars,list_names,base_name,suffix):
	filename=base_name+str(suffix)+'.h5'
	f=h5py.File(filename,'w')
	for i,name in zip(list_vars,list_names):
		f.create_dataset(name,data=i)
	f.close()

def load_npy(list_vars,sess,base_name,suffix):
	filename=base_name+str(suffix)+'.npy'
	rt=np.load(filename)[()]
	
	
def read_hdf5(base_name,suffix):
	filename=base_name+str(suffix)+'.h5'
	f=h5py.File(filename,'r')
	keys=f.keys()
	for i in keys:
		print f[i][:].shape
	f.close()

def get_varnames(list_vars):
	for i in list_vars:
		print i.name


def next_batch(x_train,batch_size):
	index=np.random.randint(0,x_train.shape[0],batch_size)
	return x_train[index,:].reshape([batch_size,28,28,1])

if __name__=="__main__":
	parser = ArgumentParser()
	parser.add_argument('-devbox',type=int,default=0)
	args = parser.parse_args()
	
	if args.devbox:
	  BASE_DIR = '/root/segnet_vgg16'
	  os.environ['CUDA_VISIBLE_DEVICES']="1";
	else:
	  BASE_DIR = '/home/sriram/intern'
	  os.environ['CUDA_VISIBLE_DEVICES']="";
  
	train_DCGAN()
