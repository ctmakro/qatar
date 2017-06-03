from imgen import pour
import tensorflow as tf
import canton as ct
from canton import *

def buildmodel():
    c = Can()
    
    
    [c.add(can) for can in
    [
    Conv2D(1,32,k=5,usebias=True), Act('relu'),
    Conv2D(32,32,k=5,usebias=False), Act('relu'),
    AvgPool2D(k=2,std=2),
    Conv2D(32,32,k=5,usebias=False), Act('relu'),
    AvgPool2D(k=2,std=2),
    Conv2D(32,32,k=5,usebias=False), Act('relu'),
    AvgPool2D(k=2,std=2),
    Conv2D(32,32,k=5,usebias=False), Act('relu'),
    #AvgPool2D(k=2,std=2),
    Conv2D(32,32,k=5,usebias=False), Act('relu'),
    Conv2D(32,32,k=5,usebias=False), Act('relu'),
    #Up2D(scale=2),
    Conv2D(32,32,k=5,usebias=False), Act('relu'),
    Up2D(scale=2),
    Conv2D(32,32,k=5,usebias=False), Act('relu'),
    Up2D(scale=2),
    Conv2D(32,32,k=5,usebias=False), Act('relu'),
    Up2D(scale=2),
    Conv2D(32,32,k=5,usebias=False), Act('relu'),
    Conv2D(32,32,k=5,usebias=False), Act('relu'),
    Conv2D(32,3,k=5,usebias=True),
    ]
    ]
    
    '''
    [c.add(can) for can in 
    [
    Conv2D(1,32,k=5), Act('relu'),
    ResConv(32,32),
    ResConv(32,32),
    ResConv(32,64),
    ResConv(64,64),
    ResConv(64,64),
    ResConv(64,64),
    ResConv(64,64),
    ResConv(64,64),
    ResConv(64,64),
    Conv2D(64,3,k=5),
    ]
    ]
    '''
    c.chain()
    return c

model = buildmodel()

def gen():

    x = ph([None,None,1]) # grayscale
    gt = ph([None,None,3]) # UVA

    y = model(x-0.5)

    sigmred = tf.nn.sigmoid(y[:,:,:,2:3])
    
    importance = gt[:,:,:,2:3]
    # ratios = tf.reduce_mean(importance,axis=[1,2,3],keep_dims=True)
    # scaled_importance = importance/ratios

    sqrdiff = (y[:,:,:,0:2]-gt[:,:,:,0:2])**2
    
    #loss = tf.reduce_mean(sqrdiff * importance) + \
    #   ct.mean_sigmoid_cross_entropy_loss(y[:,:,:,2:3],gt[:,:,:,2:3]) * 0.01
    rmsloss = tf.reduce_mean(sqrdiff * importance)
    celoss = ct.mean_sigmoid_cross_entropy_loss(y[:,:,:,2:3],importance)
    loss = celoss + rmsloss
    
    opt = tf.train.AdamOptimizer(1e-3)

    train_step = opt.minimize(loss,var_list=model.get_weights())

    sess = ct.get_session()
    def feed(qr,uv):
        res = sess.run([train_step,loss],feed_dict={
            x:qr,gt:uv
        })
        return res[1]

    def test(qr):
        res = sess.run([y,sigmred],feed_dict={
            x:qr
        })
        return res

    return feed,test

feed,test = gen()

get_session().run(gvi())

def r(ep=1):
    for i in range(ep):
        print('ep',i)

        uv,qr = pour(20)

        loss = feed(qr,uv)
        # vis.show_batch_autoscaled(y,name='output')    
        
        print('loss',loss)
        
        if i%10==0:
            t()

from cv2tools import vis
    
def t():
    uv,qr = pour(10)

    y,sigmred = test(qr)
    
    y[:,:,:,2:3] = sigmred

    vis.show_batch_autoscaled(y,name='output')
    vis.show_batch_autoscaled(qr,name='input')
    vis.show_batch_autoscaled(uv,name='gt')

