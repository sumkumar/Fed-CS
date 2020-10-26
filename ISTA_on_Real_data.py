####FISTA###########

import copy
import numpy as np
import argparse
import os
from math import ceil
import tensorflow as tf
from scipy.io import loadmat
# Shrinkage Function
def shrink(V, theta):
	
	###########################################################################
	# V ==> The input vector                                                  #   
	# theta ==> The value in between the vector is shrinked and thresholded   #
	#                                                                         #
	# The algorithm runs element-wise on the vector V                         #
	###########################################################################
	
	h = np.sign(V) * np.maximum(np.abs(V)-theta, 0)
	return h

def run_ISTA_Real(config):#M,N,F,A) :
    from utils.cs import imread_CS_py, img2col_py, col2im_CS_py

    from skimage.io import imsave
    """Load dictionary and sensing matrix."""
    Phi = np.load (config.sensing) ['A']
    # D   = np.load (config.dict)
    # D = D["arr_0"]
    D = loadmat(config.dict)['D']
    D = D.astype(np.float32)
    A = np.matmul(Phi,D)
    # loading compressive sensing settings
    M = Phi.shape [0]
    F = Phi.shape [1]
    N = D.shape [1]
    assert M == config.M and F == config.F and N == config.N
    patch_size = int (np.sqrt (F))
    assert patch_size ** 2 == F

    L = 1.001 * np.linalg.norm(A, ord=2)**2
    lamda = 0.2
    e = 0.05

    Th = lamda/(L)

    # x = np.zeros((x_.shape[0],x_.shape[1]))
    # calculate average NMSE and PSRN on test images
    test_dir = './data/test_images/'
    test_files = os.listdir (test_dir)
    avg_nmse = 0.0
    avg_psnr = 0.0
    overlap = 0
    stride = patch_size - overlap
    out_dir = "./data/recon_images"
    sample_rate = ceil(M / F * 100.0)
    for test_fn in test_files :
        # read in image
        out_fn = test_fn[:-4] + "_recon_{}.png".format(sample_rate)
        out_fn = os.path.join(out_dir, out_fn)
        test_fn = os.path.join (test_dir, test_fn)
        test_im, H, W, test_im_pad, H_pad, W_pad = \
                imread_CS_py (test_fn, patch_size, stride)
        test_fs = img2col_py (test_im_pad, patch_size, stride)

        # remove dc from features
        test_dc = np.mean (test_fs, axis=0, keepdims=True)
        test_cfs = test_fs - test_dc
        test_cfs = np.asarray (test_cfs) / 255.0

        # sensing signals
        test_ys = (np.matmul (Phi, test_cfs)).astype(np.float32)
        num_patch = test_ys.shape [1]

        x = np.zeros((512,num_patch)).astype(np.float32)
        for i in range(10):
          Thr = copy.deepcopy(Th)
          xold = copy.deepcopy(x)
          z = xold + (1/L)*A.T@(test_ys - A@(xold)) 
          x = shrink(z, Thr)

        fhs_ = D@x
        rec_fs  = fhs_ * 255.0 + test_dc

        # patch-level NMSE
        patch_err = np.sum (np.square (rec_fs - test_fs))
        patch_denom = np.sum (np.square (test_fs))
        avg_nmse += 10.0 * np.log10 (patch_err / patch_denom)

        rec_im = col2im_CS_py (rec_fs, patch_size, stride,
                                H, W, H_pad, W_pad)

        import cv2
        cv2.imwrite('%s'%out_fn, np.clip(rec_im, 0.0, 255.0))

        # image-level PSNR
        image_mse = np.mean (np.square (np.clip(rec_im, 0.0, 255.0) - test_im))
        avg_psnr += 10.0 * np.log10 (255.**2 / image_mse)

    num_test_ims = len (test_files)
    print ('Average Patch-level NMSE is {}'.format (avg_nmse / num_test_ims))
    print ('Average Image-level PSNR is {}'.format (avg_psnr / num_test_ims))
    return (avg_psnr / num_test_ims)
    # end of cs_testing
# PSNR = run_ISTA_Real(M,N,F,A)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--sensing", type=str, help="Sensing matrix file. Instance of Problem class.")
parser.add_argument(
    '-M', '--M', type=int, default=250,
    help="Dimension of measurements.")
parser.add_argument(
    '-N', '--N', type=int, default=500,
    help="Dimension of sparse codes.")
parser.add_argument(
    '-F', '--F', type=int, default=256,
    help='Number of features of extracted patches.')
parser.add_argument(
    '-dc', '--dict', type=str, default=None,
    help="Dictionary file. Numpy array instance stored as npy file.")

if __name__ == "__main__":
    config, unparsed = parser.parse_known_args()
    PSNR = run_ISTA_Real(config)    

  
