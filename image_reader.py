import numpy as np
import scipy.misc as sp
import glob
from math import floor, ceil

class image_reader():
	def shuffle_data(self):
		self.indices=np.arange(self.size);
		np.random.shuffle(self.indices);
		self.data_files=self.data_files[self.indices];self.label_files=self.label_files[self.indices];

	def __init__(self,data_dir,label_dir,batch_size,image_size=None):
		self.data_dir=data_dir;
		self.label_dir=label_dir;
		self.batch_size=batch_size;
		self.data_files=np.sort(glob.glob(self.data_dir+'*'));
		self.label_files=np.sort(glob.glob(self.label_dir+'*'));
		if(image_size==None):
			self.image_size=sp.imread(self.data_files[0]).shape
		else:
			self.image_size=image_size;
		self.data_siz=np.size(self.data_files);
		self.label_siz=np.size(self.label_files);
		self.size=min(self.data_siz,self.label_siz);
		self.n_batches=ceil(self.size*1.0/self.batch_size);

		self.data_files=self.data_files[:self.size];
		self.label_files=self.label_files[:self.size];\
		self.reset_reader();
		# self.shuffle_data();
	def print_content():
		while(raw_input()!='q'):
			print 'epoch',self.epoch;
			print 'batch:', self.batch_num;
			b=self.next_batch();
			print b[0].shape;
			print b[1].shape;
			print '\n';
	def reset_reader(self):
		self.cursor=0;
		self.epoch=0;
		self.batch_num=0;

	# def next_batch1(self):
	# 	if(self.cursor+self.batch_size<self.size):
	# 		self.chunk_data=self.data_files[self.cursor:self.cursor+self.batch_size];
	# 		self.chunk_labels=self.label_files[self.cursor:self.cursor+self.batch_size];
	# 	else:
	# 		residue=self.size-self.cursor;
	# 		self.chunk_data=np.concatenate([self.data_files[-1*residue:],self.data_files[:self.batch_size-residue]])
	# 		self.chunk_labels=np.concatenate([self.label_files[-1*residue:],self.label_files[:self.batch_size-residue]])
	# 		self.epoch=self.epoch+1;
	# 		self.batch_num=0;

	# 	# data=[[sp.imread(i),sp.imread(j)] for i,j in zip(chunk_data,chunk_labels)];
	# 	data=[sp.imresize(sp.imread(i, mode='RGB'),[self.image_size[0],self.image_size[1]]) for i in self.chunk_data];
	# 	lab=[sp.imresize(sp.imread(i),[self.image_size[0],self.image_size[1]]) for i in self.chunk_labels];
	# 	self.cursor=(self.cursor+self.batch_size)%self.size;
	# 	self.batch_num=self.batch_num+1;
	# 	return [np.stack(data),np.stack(lab)];

	def next_batch(self):
		if(self.batch_num<self.n_batches-1):
			self.chunk_data=self.data_files[self.cursor:self.cursor+self.batch_size];
			self.chunk_labels=self.label_files[self.cursor:self.cursor+self.batch_size];
			self.cursor=(self.cursor+self.batch_size)%self.size;


		elif(self.batch_num==self.n_batches-1):
			residue=self.size-self.cursor;
			self.chunk_data=np.concatenate([self.data_files[-1*residue:],self.data_files[:self.batch_size-residue]])
			self.chunk_labels=np.concatenate([self.label_files[-1*residue:],self.label_files[:self.batch_size-residue]])
			self.cursor=0;
			self.shuffle_data();
			# self.epoch=self.epoch+1;
			# self.batch_num=0;
		self.batch_num=self.batch_num+1

		data=[sp.imresize(sp.imread(i, mode='RGB'),[self.image_size[0],self.image_size[1]]) for i in self.chunk_data];
		lab=[sp.imresize(sp.imread(i),[self.image_size[0],self.image_size[1]],interp='nearest') for i in self.chunk_labels];
		
		return [np.stack(data),np.stack(lab)];


class single_reader():
	def shuffle_data(self):
		self.indices=np.arange(self.size);
		np.random.shuffle(self.indices);
		self.data_files=self.data_files[self.indices];

	def __init__(self,data_dir,batch_size,image_size=None):
		self.data_dir=data_dir;
		self.batch_size=batch_size;
		self.data_files=np.sort(glob.glob(self.data_dir+'*'));
		if(image_size==None):
			self.image_size=sp.imread(self.data_files[0]).shape
		else:
			self.image_size=image_size;
		self.size=np.size(self.data_files);
		self.n_batches=ceil(self.size*1.0/self.batch_size);

		self.data_files=self.data_files[:self.size];
		self.reset_reader();
		self.shuffle_data();
	def reset_reader(self):
		self.cursor=0;
		self.epoch=0;
		self.batch_num=0;

	def next_batch(self):
		if(self.batch_num<self.n_batches-1):
			self.chunk_data=self.data_files[self.cursor:self.cursor+self.batch_size];
			self.cursor=(self.cursor+self.batch_size)%self.size;


		elif(self.batch_num==self.n_batches-1):
			residue=self.size-self.cursor;
			self.chunk_data=np.concatenate([self.data_files[-1*residue:],self.data_files[:self.batch_size-residue]])
			self.cursor=0;
			self.shuffle_data();
			# self.epoch=self.epoch+1;
			# self.batch_num=0;
		self.batch_num=self.batch_num+1

		data=[sp.imresize(sp.imread(i, mode='RGB'),[self.image_size[0],self.image_size[1]],interp='bicubic') for i in self.chunk_data];
		return np.stack(data);



