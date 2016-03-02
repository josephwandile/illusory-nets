import numpy as np
import sys, traceback, os
caffe_root_invert = '/home/bill/Libraries/caffe_invert_alexnet/'
caffe_root = '/home/bill/Libraries/caffe/'
import Image

import pdb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as spio

sys.path.insert(0, '/home/bill/Dropbox/Cox_Lab/General_Scripts/')
sys.path.insert(0, '/home/bill/Dropbox/Cox_Lab/caffe/scripts')

import hickle as hkl

base_dir = '/home/bill/Dropbox/Cox_Lab/Illusions/'

def get_features(im_file, layer):

	sys.path.insert(0, caffe_root+'python/')
	import caffe

	PRETRAINED = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
	MODEL_FILE = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
	net = caffe.Classifier(MODEL_FILE, PRETRAINED,
		mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
		channel_swap=(2,1,0),
		raw_scale=255,
		image_dims=(227, 227))

	im = caffe.io.load_image(im_file)
	pdb.set_trace()
	feats = net.predict([im], oversample=False)
	feats = net.blobs[layer].data
	return feats


def get_recon(feats, layer):

	sys.path.insert(0, caffe_root_invert+'python/')
	import caffe

	PRETRAINED = '/home/bill/Libraries/caffe_invert_alexnet/'+layer+'/invert_alexnet_'+layer+'.caffemodel'
	MODEL_FILE = '/home/bill/Libraries/caffe_invert_alexnet/'+layer+'/invert_alexnet_'+layer+'_deploy_from_features.prototxt'
	net = caffe.Classifier(MODEL_FILE, PRETRAINED)
	pdb.set_trace()
	out = net.predict(feats, is_features=True)
	pdb.set_trace()
	im = out[0].transpose((1,2,0))
	im2 = np.copy(im)
	im2[:,:,0] = im[:,:,2]
	im2[:,:,2] = im[:,:,0]
	im = im2
	im = np.dot(im[...,:3], [0.299, 0.587, 0.144])

	return im


def get_recon_from_im(im_file, layer, save_name, gray=True):

	sys.path.insert(0, caffe_root_invert+'python/')
	import caffe

	PRETRAINED = '/home/bill/Libraries/caffe_invert_alexnet/'+layer+'/invert_alexnet_'+layer+'.caffemodel'
	MODEL_FILE = '/home/bill/Libraries/caffe_invert_alexnet/'+layer+'/invert_alexnet_'+layer+'_deploy.prototxt'
	net = caffe.Classifier(MODEL_FILE, PRETRAINED,
		mean_file=caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy',
		channel_swap=(2,1,0),
		input_scale=255,
		image_dims=(227, 227))
	im = caffe.io.load_image(im_file)
	out = net.predict([im], is_features=False)
	recon_im = out[0].transpose((1,2,0))
	im2 = np.copy(im)
	im2[:,:,0] = im[:,:,2]
	im2[:,:,2] = im[:,:,0]
	im = im2
	if gray:
		recon_im = np.dot(recon_im[...,:3], [0.299, 0.587, 0.144])
		plt.imshow(recon_im, cmap='Greys_r')
	else:
		print 'to implement'
	plt.savefig(save_name)




def test():

	#im_file = caffe_root+'examples/images/cat.jpg'
	im_file = '/home/bill/Dropbox/Cox_Lab/Illusions/images/T_illusion.jpg'

	layer1 = 'pool2'
	layer2 = 'conv2'
	save_file = '/home/bill/Dropbox/Cox_Lab/Illusions/misc/T_feats2_'+layer1+'_notoversample.hkl'
	save_file2 = '/home/bill/Dropbox/Cox_Lab/Illusions/misc/T_recon_'+layer1+'_0-2tran.jpg'

	if not os.path.isfile(save_file):
		feats = get_features(im_file, layer1)
		hkl.dump(feats, open(save_file, 'w'))
	else:
		feats = hkl.load(open(save_file))
		#feats = feats.reshape((10,256,6,6))

	recon_im = get_recon(feats, layer2)
	#	img = Image.fromarray(recon_im, 'RGB')
	pdb.set_trace()
	#recon_im[recon_im<0] = 0
	#recon_im = recon_im/255
	plt.imshow(recon_im) #, cmap='Greys_r')
	plt.show(block=False)
	plt.savefig(save_file2)
	pdb.set_trace()


def test2():

	#im_file = '/home/bill/Dropbox/Cox_Lab/Illusions/images/adelson_illusion.jpg'
	#im_file = '/home/bill/Dropbox/Cox_Lab/Illusions/images/T_illusion.jpg'
	path,_,files = os.walk(base_dir+'processed_illusions/').next()
	#im_file = '/home/bill/Dropbox/Cox_Lab/Illusions/raw_illusions/T_illusion_download.png'
	save_dir = base_dir+'reconstructions/'
	for layer in ['conv1']: #['conv2', 'conv5', 'fc7']:
		for f in files:
			e = f[f.rfind('.'):]
			b = f[:f.rfind('.')]
			save_name = save_dir+b+'_'+layer+e
			get_recon_from_im(path+'/'+f, layer, save_name)





if __name__=='__main__':
    try:
		test2()

    except:
		ty, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
