import numpy as np
from scipy import misc
import argparse
import time
from matplotlib import pyplot as plt
import cv2

from frame_interp import interpolate_frame
from frame_interp import decompose

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img1', type=str, help='Path to first frame.')
    parser.add_argument('img2', type=str, help='Path to second frame.')
    parser.add_argument('--n_frames', '-n', type=int, default=1, help='Number of new frames.')
    parser.add_argument('--show', '-sh', type=int, default=0, help='Display result.')
    parser.add_argument('--save', '-s', type=int, default=0, help='Save interpolated images.')
    parser.add_argument('--save_path', '-p', type=str, default='', help='Output path.')
    args = parser.parse_args()
    xp = np

    img1 = misc.imread(args.img1)
    img2 = misc.imread(args.img2)

    print('start')
    start = time.time()
    print(cv2.__version__)
    cap = cv2.VideoCapture('syn_ball.avi')
    vidHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vidWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    nChannels = 3
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    fr_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    nOrientations = 8
    tWidth = 1
    limit = 0.2
    min_size = 15
    max_levels = 23
    py_level = 4
    scale = 0.5**(1/py_level)
    n_scales = min(np.ceil(np.log2(min((vidHeight, vidWidth))) / np.log2(1. / scale) -
                           (np.log2(min_size) / np.log2(1 / scale))).astype('int'), max_levels)
    motion_freq_es = 10/3
    time_interval = 1/4 * 1/motion_freq_es
    amp_factor = 5
    # motionAMP
    frame_interval = np.ceil(frame_rate*time_interval).astype(int)
    windowSize = 2*frame_interval
    norder = windowSize*2

    #TEMPKERNEL - INT

    signalLen = 2*windowSize
    sigma = frame_interval/2
    x = np.linspace(-signalLen/2, signalLen/2, signalLen+1)
    kernel = np.zeros(x.shape, dtype=float)

    INT_kernel = kernel
    INT_kernel[frame_interval] = 0.5
    INT_kernel[2*frame_interval] = -1
    INT_kernel[3*frame_interval] = 0.5
    kernel = -INT_kernel / sum(abs(INT_kernel))

    ret, im = cap.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    im_stru = decompose(im, n_scales, nOrientations, tWidth, scale, n_scales, xp)
    phase_im_1 = [item for i in im_stru['phase'] for it in i for itm in it for item in itm]
    #phase_im = np.matlib.repmat(phase_im_1, 1, norder+1)
    phase_im=np.tile(phase_im_1, (norder+1, 1)).transpose()
    cv2.imwrite("izhod.png", im)

    for ii in range(2, 50):
        ret, im = cap.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
        im_stru = decompose(im, n_scales, nOrientations, tWidth, scale, n_scales, xp)

        fac = 1.5
        reshp = np.reshape(im_stru['phase'],(-1))
        phase_im = np.append(phase_im, reshp)
        phase_im[:,0] = []
        cv2.imshow('frame', im)
        cv2.waitKey(1)

    cap.release()
    print('Took %.2fm' % ((time.time() - start) / 60.))

    '''if args.save:
        import os
        for i in range(args.n_frames):
            misc.imsave(os.path.join(args.save_path, 'output%d.jpg' % (i+1)), new_frames[i])

    if args.show:
        plt.figure(0)
        plt.subplot(1, args.n_frames+2, 1)
        plt.imshow(img1)
        for i in range(args.n_frames):
            plt.subplot(1, args.n_frames+2, i+2)
            plt.imshow(new_frames[i])
        plt.subplot(1, args.n_frames+2, args.n_frames+2)
        plt.imshow(img2)
        plt.show()'''