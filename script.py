import os
import glob
import fnmatch
import numpy as np
import scipy.misc as sp

def convert_labels():
	path=os.getcwd()
	color_map='color_map.txt'
	fil=open(color_map)
	t=fil.readlines()
	classes=[]
	for i in t:
		classes.append(i.splitlines()[0].split('\t'))
	info=np.zeros([len(classes),3])
	for i in classes:
		info[int(i[0])]=np.array([int(pj) for pj in i[1:]])
	images=glob.glob(path+'/*.png')
	# for i in images:
	# 	if('label' in i.split('_.')):
	# 		del i;
	for i in images:
		rt=sp.imread(i)
		rt_new=np.zeros([rt.shape[0],rt.shape[1]])
		for j in range(info.shape[0]-1):
			rt_new[np.where(np.logical_and(np.logical_and(rt[:,:,0]==info[j,0],rt[:,:,1]==info[j,1]),rt[:,:,2]==info[j,2]))]=j
		rt_new[np.where(np.logical_and(np.logical_and(rt[:,:,0]==info[-1,0],rt[:,:,1]==info[-1,1]),rt[:,:,2]==info[-1,2]))]=255

		lol=i.split('/')[-1].split('.')
		sp.imsave(os.getcwd()+'/labels/'+lol[0]+'_label'+'.'+lol[1],rt_new.astype('uint8'))




def correct_file(path):
	f=open(path)
	lines=f.readlines()
	f.close()
	f=open(path,'w')
	for line in lines:
		f.write(line[1:])
	f.close()


def show_labels():

	cl=np.zeros([50])
	path='/home/sriram/intern/datasets/gtFine_trainvaltest/gtFine/train'
	for pa in os.listdir(path):
		images=glob.glob(os.path.join(path,pa+'/labels/*.png'))
		for name in images:
			img=sp.imread(name)
			[vals,cnts]=np.unique(img,return_counts=True)
			for i,j in zip(vals,cnts):
				cl[i]=cl[i]+j

def rename_files():
	import shutil,fnmatch
	path='/home/sriram/intern/datasets/SegNet_Output/'
	r=[]
	for root,dirs,files in os.walk(path):
		files=fnmatch.filter(files, '*_pred.png')
		for name in files:
			r.append(os.path.join(root,name))
	for file in r:
		new_name=file.split('.')[0][:-5]+'.png'
		shutil.move(file, new_name)

def compare_labels():
	path='/home/sriram/intern/datasets/SegNet_Output/CamVid_Output/train/'
	gt_path='/home/sriram/intern/SegNet-Tutorial/CamVid/trainannot/'
	names=os.listdir(path)
	g_names=os.listdir(gt_path)
	for i in names:
		pred=sp.imread(os.path.join(path,i))
		gt=sp.imread(os.path.join(gt_path,i))

		print pred.shape
		print gt.shape
		accur=np.where(pred==gt)[0].size*1.0/(pred.shape[0]*pred.shape[1])
		print accur

def copy_files():
	path='/home/sriram/intern/datasets/gtFine_trainvaltest/gtFine/'
	dirs=os.listdir(path)
	for i in dirs:
		path1=os.path.join(path,i)
		dirs=os.listdir(path1)
		for j in dirs:
			path2=os.path.join(path1,j)
			labels=os.path.join(path2,'labels')
			instances=os.path.join(path2,'instances')
			colors=os.path.join(path2,'colors')
			polygons=os.path.join(path2,'polygons')

			if not os.path.exists(labels):
				os.makedirs(os.path.join(path2,'labels'))
			if not os.path.exists(instances):
				os.makedirs(os.path.join(path2,'instances'))
			if not os.path.exists(polygons):
				os.makedirs(os.path.join(path2,'polygons'))
			if not os.path.exists(colors):
				os.makedirs(os.path.join(path2,'colors'))

			for names in glob.glob(os.path.join(path2,'*labelIds.png')):
				shutil.move(names,labels)
			for names in glob.glob(os.path.join(path2,'*instanceIds.png')):
				shutil.move(names,instances)
			for names in glob.glob(os.path.join(path2,'*polygons.json')):
				shutil.move(names,polygons)
			for names in glob.glob(os.path.join(path2,'*color.png')):
				shutil.move(names,colors)

def copy_outputs():
	path='/home/sriram/intern/datasets/SegNet_Output/ARL_Output/ira3/train/train/*.png'
	dest='/home/sriram/intern/datasets/data/data-with-labels/ira3/training_set/predictions/'
	for i in glob.glob(path):
		shutil.copy(i,dest)

	path='/home/sriram/intern/datasets/SegNet_Output/ARL_Output/ira3/test/eval/*.png'
	dest='/home/sriram/intern/datasets/data/data-with-labels/ira3/testing_set/predictions/'
	for i in glob.glob(path):
		shutil.copy(i,dest)

	path='/home/sriram/intern/datasets/SegNet_Output/ARL_Output/lej15/*.png'
	dest='/home/sriram/intern/datasets/data/data-with-labels/lej15/predictions/'
	for i in glob.glob(path):
		shutil.copy(i,dest)

	path='/home/sriram/intern/datasets/SegNet_Output/ARL_Output/b507/*.png'
	dest='/home/sriram/intern/datasets/data/data-with-labels/b507/predictions/'
	for i in glob.glob(path):
		shutil.copy(i,dest)

def color_from_label():
	num_classes=12
	factor=20
	path='/home/sriram/intern/datasets/CamVid/'
	dirs=fnmatch.filter(os.listdir(path),'*annot')
	for i in dirs:
		if(not os.path.isdir(os.path.join(path,i+'_color'))):
			os.makedirs(os.path.join(path,i+'_color'))
		path1=os.path.join(path,i)
		path2=os.path.join(path,i+'_color')
		for name in fnmatch.filter(os.listdir(path1),'*.png'):
			label_img=sp.imread(os.path.join(path1,name))
			img_size=label_img.shape
			color_img=factor*label_img
			sp.imsave(os.path.join(path2,name),color_img.astype('uint8'))


def walk_all_labels():
	path='/home/sriram/intern/datasets/gtFine_trainvaltest/gtFine'
	r=[]
	for i in os.listdir(path):
		path1=os.path.join(path,i)
		for city in os.listdir(path1):
			path2=os.path.join(path1,city,'labels')
			for file in os.listdir(path2):
				img=sp.imread(os.path.join(path2,file))
				r.append(np.unique(img))
	array=np.concatenate(r)
	print array


def train_test_split():
	import shutil
	split_ratio=0.15
	path='/home/sriram/intern/datasets/data/data-with-labels/lej15/'
	
	if(not os.path.isdir(os.path.join(path,'training_set'))):
		os.makedirs(os.path.join(path,'training_set'))
	if(not os.path.isdir(os.path.join(path,'testing_set'))):
		os.makedirs(os.path.join(path,'testing_set'))
	if(not os.path.isdir(os.path.join(path,'val_set'))):
		os.makedirs(os.path.join(path,'val_set'))
	l=os.listdir(os.path.join(path,'new_labels'))
	l.sort()
	n=len(l)
	a=np.array(range(n))
	np.random.shuffle(a)
	test_l=l[:int(split_ratio*n)+1]
	val_l=l[int(split_ratio*n)+1:int(2*split_ratio*n)+1]
	train_l=l[int(2*split_ratio*n)+1:]
	folders=['annotated_images','images','labels','new_labels']

	

	path_m=os.path.join(path,'training_set')
	for f in folders:
		path1=os.path.join(path,f)
		path2=os.path.join(path_m,f)
		if(not os.path.isdir(path2)):
			os.makedirs(path2)
		if(f=='labels'):
			for filename in [f.split('.')[0]+'.txt' for f in train_l]:
				shutil.copy(os.path.join(path1,filename), os.path.join(path2,filename))
			continue
		for filename in train_l:
			shutil.copy(os.path.join(path1,filename), os.path.join(path2,filename))

	path_m=os.path.join(path,'val_set')
	for f in folders:
		path1=os.path.join(path,f)
		path2=os.path.join(path_m,f)
		if(not os.path.isdir(path2)):
			os.makedirs(path2)
		if(f=='labels'):
			for filename in [f.split('.')[0]+'.txt' for f in val_l]:
				shutil.copy(os.path.join(path1,filename), os.path.join(path2,filename))
			continue

		for filename in val_l:
			shutil.copy(os.path.join(path1,filename), os.path.join(path2,filename))

	path_m=os.path.join(path,'testing_set')
	for f in folders:
		path1=os.path.join(path,f)
		path2=os.path.join(path_m,f)
		if(not os.path.isdir(path2)):
			os.makedirs(path2)
		if(f=='labels'):
			for filename in [f.split('.')[0]+'.txt' for f in test_l]:
				shutil.copy(os.path.join(path1,filename), os.path.join(path2,filename))
			continue

		for filename in test_l:
			shutil.copy(os.path.join(path1,filename), os.path.join(path2,filename))

def make_labelimg_fromtxt():
	path='/home/sriram/intern/datasets/data/data-with-labels/'
	dirs=[f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]
	for i in dirs:
		if(i!='ira3'):
			img_size=[275,344]

			path1=os.path.join(path,i,'labels/')
			if(not os.path.isdir(os.path.join(path,i,'new_labels'))):
				os.makedirs(os.path.join(path,i,'new_labels'))
			for j in os.listdir(path1):
				file_path=os.path.join(path1,j)

				f=open(file_path,'r')
				if(f.readline()[0]==' '):
					correct_file(file_path)
					f.close()
				f=open(file_path,'r')
				count=0;r=[]
				while(count<img_size[0]):
					line=f.readline()
					count=count+1
					r.append(np.array(line.splitlines()[0].split(' ')).astype('int'))
					
				f.close()
				img=np.stack(r).reshape(img_size)
				img[np.where(img==-1)]=255
				sp.imsave(os.path.join(path,i,'new_labels',j.split('.')[0]+'.png'),img.astype('uint8'))
		if(i=='ira3'):
			img_size=[240,320]
			for k in ['training_set','testing_set']:
				path1=os.path.join(path,i,k,'labels/')
				if(not os.path.isdir(os.path.join(path,i,k,'new_labels'))):
					os.makedirs(os.path.join(path,i,k,'new_labels'))
				for j in os.listdir(path1):
					file_path=os.path.join(path1,j)

					# f=open(file_path,'r')
					# if(f.readline()[0]==' '):
					# 	correct_file(file_path)
					# 	f.close()
					f=open(file_path,'r')
					count=0;r=[]
					while(count<img_size[0]):
						line=f.readline()
						count=count+1
						r.append(np.array(line.splitlines()[0].split(' ')).astype('int'))
						
					f.close()
					img=np.stack(r).reshape(img_size)
					img[np.where(img==-1)]=255
					sp.imsave(os.path.join(path,i,k,'new_labels',j.split('.')[0]+'.png'),img.astype('uint8'))


	



