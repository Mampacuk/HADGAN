import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from skimage.segmentation import slic
from skimage.util import img_as_float
import time

from .SuperGraph import supergraph

def myRGAE(X, SG, lambda_, n_hid):
    N,B=X.shape  # number of pixels
    lr=0.01  # learning rate 
    epochs=1200  # epochs
    batch_num=10
    batch_size=N/batch_num

    print("Training RGAE: epochs=%d, batch_size=%d, lr=%d"%(epochs,batch_size,lr))

    W1 = tf.Variable(0.01*np.random.randn(B, n_hid), dtype=tf.float32)
    b1 = tf.Variable(np.random.randn(1, n_hid), dtype=tf.float32)
    W2 = tf.Variable(0.01*np.random.randn(n_hid, B), dtype=tf.float32)
    b2 = tf.Variable(np.random.randn(1, B), dtype=tf.float32)

    opt=tf.keras.optimizers.Adam(learning_rate=lr)

    X=tf.convert_to_tensor(X,dtype=tf.float32)
    SG=tf.convert_to_tensor(SG,dtype=tf.float32)

    start=time.time()
    for epoch in range(epochs):
        idx=np.random.permutation(N) # shuffle the data
        for j in range(batch_num):  
            batch_idx=idx[j*int(batch_size):(j+1)*int(batch_size)]
            
            # (batch_size,B)
            xb=tf.gather(X, batch_idx) # get the batch data

            # (batch_size,batch_size) 
            L=tf.gather(tf.gather(SG, batch_idx, axis=0), batch_idx, axis=1) # get the corresponding graph Laplacian
            
            with tf.GradientTape() as tape:
                # forward pass
                H=tf.nn.sigmoid(tf.matmul(xb,W1)+b1) # hidden layer (batch_size,n_hid)
                x_hat=tf.nn.sigmoid(tf.matmul(H,W2)+b2) # output layer (batch_size,B)

                error_matrix=(xb-x_hat) # (batch_size,B)
                res=tf.math.sqrt(tf.reduce_sum(tf.square(error_matrix),axis=1)+1e-8) # (batch_size,)
                rec_loss=tf.reduce_sum(res)

                # graph regularization loss
                graph_loss = tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(H), L), H))    

                loss = (1/(2*batch_size))*rec_loss + (lambda_/batch_size)*graph_loss
            
            grads=tape.gradient(loss, [W1, b1, W2, b2])
            opt.apply_gradients(zip(grads, [W1, b1, W2, b2]))

        if epoch%100==0:
            elapsed=time.time()-start
            print("Epoch: %d, Loss=%.4f, Time=%.2fs"%(epoch,float(loss.numpy()),elapsed))

    np.savez("rgae_sal.npz", W1=W1.numpy(), b1=b1.numpy(), W2=W2.numpy(), b2=b2.numpy())

    Z=tf.nn.sigmoid(tf.matmul(X,W1)+b1) # final hidden representation (N,n_hid)
    X_hat=tf.nn.sigmoid(tf.matmul(Z,W2)+b2) # final output (N,B)
    return tf.math.sqrt(tf.reduce_sum((X-X_hat)**2,axis=1)).numpy() # return the final map

def RGAE(hsi, S, n_hid, lambda_):
    SG,_,_= supergraph(hsi,S) # SG is the graph Laplacian of size (H*W, H*W)

    X_new=hsi.reshape(-1,hsi.shape[2]) # (H*W,B)

    y_tmp=myRGAE(X_new,SG,lambda_,n_hid)

    return y_tmp