from imgen import pour
import tensorflow as tf
import canton as ct
from canton import *

def buildmodel():
    c = Can()

    [c.add(can) for can in
    [
    Conv2D(1,32,k=5,usebias=True),
    Conv2D(32,32,k=5,usebias=False),
    # AvgPool2D(k=2,std=1),
    Conv2D(32,32,k=5,usebias=False),
    Conv2D(32,32,k=5,usebias=False),
    # Conv2D(32,32,k=5,usebias=False),
    Conv2D(32,32,k=5,usebias=False),
    # Up2D(scale=2),
    Conv2D(32,32,k=5,usebias=False),
    Conv2D(32,3,k=5,usebias=True),
    ]
    ]

    c.chain()
    return c

model = buildmodel()

def gen():

    x = ph([None,None,1]) # grayscale
    gt = ph([None,None,3]) # UVA

    y = model(x-0.5)

    importance = gt[:,:,2:3]
    # ratios = tf.reduce_mean(importance,axis=[1,2,3],keep_dims=True)
    # scaled_importance = importance/ratios

    loss = tf.reduce_mean((y[:,:,0:2]-gt[:,:,0:2]) ** 2 * importance)+ \
        tf.reduce_mean((y[:,:,2:3]-importance)**2)

    opt = tf.train.AdamOptimizer()

    train_step = opt.minimize(loss,var_list=model.get_weights())

    sess = ct.get_session()
    def feed(qr,uv):
        res = sess.run([train_step,loss],feed_dict={
            x:qr,gt:uv
        })
        return res[1]

    def test(qr):
        res = sess.run([y],feed_dict={
            x:qr
        })[0]
        return res

    return feed,test

feed,test = gen()

get_session().run(gvi())

def r(ep=1):
    for i in range(ep):
        print('ep',i)

        uv,qr = pour(20)

        loss = feed(qr,uv)
        print('loss',loss)

from cv2tools import vis
def t():
    uv,qr = pour(10)

    res = test(qr)

    vis.show_batch_autoscaled(res,name='test')
    vis.show_batch_autoscaled(qr,name='input')
    vis.show_batch_autoscaled(uv,name='gt')
