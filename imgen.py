from cv2tools import vis,filt
import cv2

import numpy as np

import qrcode

def qr(string,box_size=8,border=2):
    # qr section

    qrgen = qrcode.QRCode(
        version=None,
        error_correction = qrcode.constants.ERROR_CORRECT_H,
        box_size = box_size,
        border = border,
    )

    qrgen.add_data(string)
    qrgen.make(fit=True)

    qrimg = qrgen.make_image()

    qrimgarr = np.fromstring(
        qrimg.get_image().convert('L').tobytes(),dtype='uint8'
    ) # grascale 8-bit -> nparray

    qrimgarr.shape = (qrimg.size[1],qrimg.size[0],1)

    # print('qrimgarr',qrimgarr.shape,'box_size',box_size,'border',border)

    # uv section

    uvimg = np.zeros(qrimgarr.shape[0:2]+(4,),dtype='float32')

    iw = uvimg.shape[0] # image width
    bw = box_size*border # border width in pixels
    cw = iw - bw*2 # code width in pixels

    # for row in range(uvimg.shape[0]):
    #     for col in range(uvimg.shape[1]):
    #         u = (col - bw) / cw
    #         v = 1 - ((row - bw) / cw)
    #         a = 1 if 0<u<1 and 0<v<1 else 0
    #         uvimg[row,col] = np.array([u,v,a])

    rowimg,colimg = np.mgrid[0:iw,0:iw]

    u = (colimg - bw + 0.5)/cw
    v = 1 - ((rowimg - bw + 0.5)/cw)
    a = (0<u) * (u<1) * (0<v) * (v<1)

    uvimg[:,:,0] = u # ucoord
    uvimg[:,:,1] = v # vcoord
    uvimg[:,:,2] = 1 # qrsection
    uvimg[:,:,3] = 1 # alpha channel

    return qrimgarr,uvimg

def r(i=1):
    return np.random.uniform() * i

def tcify(i,channels=3): # three channelify
    if(len(i.shape)==2 and channels==1):
        i.shape+=(1,)
        return i

    a = np.zeros((i.shape[0],i.shape[1],channels),dtype='float32')
    a[:,:] = i[:,:]
    if(channels>3):
        a[:,:,3:] = 1
    return a

def random_qr_pipeline(ol=128,dirty=True):
    randomstring = str(r())+str(r())

    qrimg, uvimg = qr(
        randomstring,
        box_size=int(3+r(2)),
        border=int(r(3))
    )

    qrimg = qrimg / 255. # normalize

    # bias and gain
    randombias = r(0.5) - .25
    randomgain = r(.5)+.1
    qrimg = ((qrimg - 0.5) * randomgain + randombias) + 0.5

    # 4-channel
    qrimg = tcify(qrimg,4)

    # rotate and scale
    randomangle = r(360)
    randomscale = .5 + r(1)

    # translation
    randomx = int(r(30)-15)
    randomy = int(r(30)-15)

    # ol = 128
    ol = ol
    dol = int(ol*2)
    # output limit
    # some random distortion here

    qr_rotated = filt.rotate_scale(qrimg,randomangle)
    ui_rotated = filt.rotate_scale(uvimg,randomangle)

    # noisy background
    noisy = np.random.uniform(size=(dol,dol,3)).astype('float32')

    # alpha_composite
    offset = int(ol-qr_rotated.shape[0]/2)
    offset = [offset+randomy,offset+randomx]

    uv_noisy = filt.alpha_composite(
        bg = noisy*0,
        fg = ui_rotated,
        offset = offset,
    )

    qr_noisy = filt.alpha_composite(
        bg = noisy,
        fg = qr_rotated,
        offset = offset,
    )

    uvout = vis.resize_linear(uv_noisy,ol,ol)
    qrout = vis.resize_linear(qr_noisy,ol,ol)

    # output channel number adjustment
    uvout = uvout[:,:,0:3] # 3-channel
    qrout = np.mean(qrout,axis=2,keepdims=True) # 1-channel

    if dirty:
        # distort
        qrout,uvout = distort(qrout,uvout,k=31,scale=150)

        # add low freq pollution
        qrout = pollute(qrout,k=5,scale=1.0)
        # qrout = pollute(qrout,k=11,scale=1.5)
        qrout = pollute(qrout,k=21,scale=2.5)

    return uvout,qrout

def pollute(i,k=11,scale=1.0):
    noise = pollution(i,k,scale)
    return i+noise

def pollution(i,k=11,scale=1.0,loc=0.0):
    noise = np.random.normal(scale = scale, loc=loc ,size=i.shape).astype('float32')
    noise = cv2.blur(noise,(k,k))
    noise = cv2.blur(noise,(k,k))
    noise = tcify(noise,1)
    return noise

def distort(i1,i2,k=11,scale=1.0): # displace with grace
    hi1 = cv2.pyrUp(i1)
    hi1 = tcify(hi1,1)
    # res1 = hi1.copy()

    hi2 = cv2.pyrUp(i2)
    # res2 = hi2.copy()

    displace_x = pollution(hi1,k,scale)
    displace_y = pollution(hi1,k,scale) # H W C

    displace_x.shape = displace_x.shape[0:2] # H W
    displace_y.shape = displace_y.shape[0:2]

    ih,iw = hi1.shape[0:2]

    rowimg,colimg = np.mgrid[0:ih,0:iw]

    # def clampy(j):
    #     return min(max(j,0),hi1.shape[0]-1)
    # def clampx(j):
    #     return min(max(j,0),hi1.shape[1]-1)

    rowimg += displace_y.astype('int32')
    colimg += displace_x.astype('int32')

    # def old_displace_algor():
    #     rowimg = np.clip(rowimg,a_max=hi1.shape[0]-1,a_min=0).astype('uint32')
    #     colimg = np.clip(colimg,a_max=hi1.shape[1]-1,a_min=0).astype('uint32')
    #
    #     for row in range(hi1.shape[0]):
    #         for col in range(hi1.shape[1]):
    #             # dx = displace_x[row,col,0]
    #             # dy = displace_y[row,col,0]
    #
    #             dx = colimg[row,col,0]
    #             dy = rowimg[row,col,0]
    #
    #             res1[row,col,:] = hi1[dy,dx,:]
    #             res2[row,col,:] = hi2[dy,dx,:]

    rowimg = np.clip(rowimg,a_max=hi1.shape[0]-1,a_min=0)
    colimg = np.clip(colimg,a_max=hi1.shape[1]-1,a_min=0)

    res1 = hi1[rowimg,colimg]
    res2 = hi2[rowimg,colimg]

    # print(res1.shape,res2.shape)

    return tcify(cv2.pyrDown(res1),1), cv2.pyrDown(res2)

def qrtest():

    uvout,qrout = random_qr_pipeline(128,dirty=True)

    print('uvout',uvout.shape,'qrout',qrout.shape)

    # qrimg, uvimg = [pipeline(i) for i in [qrimg,uvimg]]

    vis.show_autoscaled(qrout,name='qr')
    vis.show_autoscaled(uvout,name='uv')

    # overlaid = uvimg.copy()
    # overlaid[:,:,2] = overlaid[:,:,3]
    # vis.show_autoscaled(overlaid,name='overlay')

import time
from collections import deque

dek = deque()

def fill():
    while True:
        if len(dek) < 200:
            uvout,qrout = random_qr_pipeline(128,dirty=True)

            l = [(np.rot90(uvout,i),np.rot90(qrout,i))
            for i in [1,2,3]]

            dek.append((uvout,qrout))
            [dek.append(t) for t in l]

        else:
            return

def pourone():
    while True:
        if len(dek)==0:
            time.sleep(0.5)
        else:
            return dek.popleft()

def pour(num=20):
    startThreads()

    data = [pourone() for i in range(num)]
    uvouts = [data[i][0] for i in range(num)]
    qrouts = [data[i][1] for i in range(num)]

    return np.array(uvouts),np.array(qrouts)

import threading as th
samplethread = [None,None,None,None]

def startThreads():
    for i in range(len(samplethread)):
        if samplethread[i] is None:
            samplethread[i] = th.Thread(target=fill)
            samplethread[i].start()
        if not samplethread[i].is_alive():
            samplethread[i] = th.Thread(target=fill)
            samplethread[i].start()

def testPour():
    u,q = pour(16)

    vis.show_batch_autoscaled(u,name='uv')
    vis.show_batch_autoscaled(q,name='qr')

    print('deque len:',len(dek))

if __name__ == '__main__':
    qrtest()
else:
    startThreads()
