import yaml
import cv2
import numpy as np
import random
import sys
from sklearn.utils import shuffle


#Labels:
#	GreenStraightRight: 3
#	off: 726
#	GreenStraightLeft: 1
#	GreenStraight: 20
#	occluded: 170
#	GreenRight: 13
#	Yellow: 444
#	RedStraightLeft: 1
#	RedStraight: 9
#	Green: 5207
#	GreenLeft: 178
#	RedRight: 5
#	RedLeft: 1092
#	Red: 3057


class generator(object):

	def __init__(self, batch_size, gen_type):

		self.batch_size = batch_size
		self.gen_type = gen_type

		# bosch data
		self.bosch_data_path ="data/bosch_data/"
		if self.gen_type == "train":
			self.bosch_yaml_path = self.bosch_data_path + "train.yaml"
			data = self.load_data(self.bosch_yaml_path)
			self.images, self.labels = self.bosch_preprocess(data)
			print("done processing  # images, labels", len(self.images), len(self.labels) )

		else:
			print("unknown type ",gen_type)
			return
	
		# other data / gentypes
 
	def load_data(self,path):
		data = yaml.load(open(path, 'rb').read())
		yaml.dump(data[:3], stream=sys.stdout)
		return data

	def bosch_preprocess(self,data):

		drop_list = ["occluded"]  #not the occluded field in dict, the light description is meant

		x = list();
		y = list();

		#restrict size for test
		#data = data[:20]

		for i, image_dict in enumerate(data):

			image_path = image_dict['path']
			image_path = self.bosch_data_path + image_path[2:]
			x.append(image_path)

			item_list = list()
			for box in image_dict['boxes']:
				xmin = box['x_min']
				ymin = box['y_min']
				xmax = box['x_max']
				ymax = box['y_max']
				label = box['label']
				occluded = box['occluded']

				if not (label in drop_list):
					item_list.append([ label , [xmin,ymin,xmax,ymax]])
					#label.convert_to()

			y.append(item_list)

		return x,y

	def augment(self,image,label):
		return image, label

	def stream(self):
	
		x = self.images
		y = self.labels
		batch_size = self.batch_size

		#print("streaming ", len(self.images), len(self.labels) , "   batch size = ", batch_size)
		assert len(x)==len(y)
		num_samples = len(x)

		x,y = shuffle(x,y)

		while 1: # Loop forever so the generator never terminates

			# pick the datasets for the batch
			which_ones = random.sample(range(num_samples),batch_size)

			# pull the images and prepare the datasets
			images = []
			labels = []
			for i in range(batch_size):

				# load the image
				index = which_ones[i]
				label = y[index]

				pathname = x[index]
				print(pathname)
				image = cv2.imread(pathname)
				print(image.shape)

				# training generator augments the image
				# validation generator doesn't augment image
				if self.gen_type == "train":
					image, label = self.augment(image,label)

				#collect the results
				images.append(image)
				#print(len(images))
				labels.append(label)
				#print(len(labels))

			batch_x = images[0]
			print(batch_x.shape)
			batch_x = batch_x.reshape(-1,images[0].shape[0],images[0].shape[1],images[0].shape[2])

			for i,im in enumerate(images[1:]):
				tim = im.reshape(-1,im.shape[0],im.shape[1],im.shape[2])
				what = (batch_x,tim)
				batch_x = np.concatenate(what,axis=0)
				batch_y = labels

			yield batch_x, batch_y
