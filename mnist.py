import os, struct
from array import array as pyarray

from numpy import *
from pylab import *
import numpy as np
from sknn.mlp import Classifier, Layer
from sklearn.externals import joblib
from sklearn.metrics import classification_report,confusion_matrix

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = 't10k-images-idx3-ubyte' #os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = 't10k-labels-idx1-ubyte' #os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows,cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
		images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols));
		labels[i] = lbl[ind[i]];

    return images, labels
	
	
images, labels = load_mnist('testing')#,digits=[3,8,5,6,9])
#imshow(images.mean(axis=0), cmap=cm.gray)
#show()

xd=images[:5000]; yd=labels[:5000];
#xt=images[6000:7000]; yt=labels[6000:7000];

#xt=images[10000:12000]; yt=labels[10000:12000];

xt,yt = load_mnist('testing')


#yd=yd.reshape(1,len(yd))
print xd.shape,", ",yd.shape
xd=xd.astype('float'); xt=xt.astype('float');
xd/=255; xt/=255;

#print xd[0]

for img in xd:
	for r in xrange( img.shape[0] ):
		for c in xrange( img.shape[1] ):
			print chr(int(255*round(img[r,c],2))),
		print ""



nn = Classifier(
    layers=[
        Layer("Rectifier", units=200),
		#Layer("Rectifier", units=100),
		#Layer("Rectifier", units=100),
        Layer("Softmax")],
    learning_rate=0.01,
	dropout_rate=0.3,
	valid_size=0.1,
	learning_momentum=0.7,
	batch_size=20, # actually number of batches to split in
	#learning_rule='adam',
    n_iter=30,
	verbose=True)
"""

96 % 5000 of testing
layers=[
        Layer("Rectifier", units=200),
		Layer("Rectifier", units=100),
        Layer("Softmax")],
    learning_rate=0.005,
	dropout_rate=0.3,
	valid_size=0.05,
	learning_momentum=0.7,
    n_iter=30,

	
	

nn=Classifier(
        layers=[
            # Convolution("Rectifier", channels=10, pool_shape=(2,2), kernel_shape=(3, 3)),
            Layer('Rectifier', units=100),
            Layer('Softmax')],
        learning_rate=0.01,
        #learning_rule='nesterov',
        n_iter=10,
        verbose=True)
"""
"""
nn.fit(xd,yd)	

res=nn.score(xt,yt)
print res


yres = nn.predict(xt)

  
print("\tReport:")
print(classification_report(yt,yres))
print '\nConfusion matrix:\n',confusion_matrix(yt, yres)
"""
#joblib.dump(nn, 'nn300-200_lr005_lmom07_drop03_bs10.pkl') 
#clf = joblib.load('filename.pkl') 


	
"""
def unpickle(file):
import cPickle
fo = open(file, 'rb')
dict = cPickle.load(fo)
fo.close()
return dict
"""




"""

**************RESULTS********************************

__________________________________________________________________
using:

nn = Classifier(
    layers=[
        Layer("Rectifier", units=200),
		Layer("Rectifier", units=100),
        Layer("Softmax")],
    learning_rate=0.005,
	dropout_rate=0.3,
	valid_size=0.05,
	learning_momentum=0.7,
	batch_size=10,
	#learning_rule='sgd',
    n_iter=30,
	verbose=True)
	
__________________________________________________________________	

C:\Users\DPY\Desktop\code>python mnist.py
(50000, 784) ,  (50000, 1)
D:\Programs\Anaconda2\lib\site-packages\theano\tensor\signal\downsample.py:6: UserWarning: downsample module has been moved to the theano.tensor.signal.pool module.
  "downsample module has been moved to the theano.tensor.signal.pool module.")
Initializing neural network with 3 layers, 784 inputs and 10 outputs.
  - Dense: Rectifier  Units:  200
  - Dense: Rectifier  Units:  100
  - Dense: Softmax    Units:  10

Training on dataset of 50,000 samples with 39,700,000 total size.
  - Train: 47,500     Valid: 2,500
  - Using `dropout` for regularization.
  - Terminating loop after 30 total iterations.
  - Early termination after 10 stable iterations.

Epoch       Training Error       Validation Error       Time
------------------------------------------------------------
    1          9.132e-01             3.474e-01         32.0s
    2          4.939e-01             2.749e-01         11.6s
    3          4.058e-01             2.297e-01         11.6s
    4          3.525e-01             2.002e-01         11.4s
    5          3.228e-01             1.797e-01         11.4s
    6          2.948e-01             1.649e-01         11.4s
    7          2.705e-01             1.532e-01         11.3s
    8          2.567e-01             1.404e-01         11.4s
    9          2.408e-01             1.307e-01         11.4s
   10          2.281e-01             1.227e-01         11.4s
   11          2.184e-01             1.176e-01         11.4s
   12          2.079e-01             1.113e-01         11.3s
   13          2.024e-01             1.065e-01         11.3s
   14          1.935e-01             1.019e-01         11.2s
   15          1.860e-01             1.008e-01         11.4s
   16          1.830e-01             9.496e-02         11.3s
   17          1.740e-01             9.473e-02         11.3s
   18          1.711e-01             8.987e-02         11.3s
   19          1.691e-01             8.631e-02         11.3s
   20          1.647e-01             8.348e-02         11.4s
   21          1.600e-01             8.511e-02         11.3s
   22          1.550e-01             8.112e-02         11.3s
   23          1.495e-01             7.934e-02         11.3s
   24          1.523e-01             7.853e-02         11.8s
   25          1.491e-01             7.585e-02         15.8s
   26          1.438e-01             7.457e-02         18.1s
   27          1.442e-01             7.430e-02         17.6s
   28          1.390e-01             7.467e-02         17.8s
   29          1.330e-01             6.895e-02         18.6s
   30          1.309e-01             7.306e-02         18.3s

Terminating after specified 30 total iterations.
[(10000, 10)]
0.9767
[(10000, 10)]
        Report:
             precision    recall  f1-score   support

          0       0.98      0.99      0.98       980
          1       0.99      0.99      0.99      1135
          2       0.97      0.97      0.97      1032
          3       0.98      0.97      0.98      1010
          4       0.98      0.98      0.98       982
          5       0.98      0.97      0.98       892
          6       0.98      0.98      0.98       958
          7       0.97      0.97      0.97      1028
          8       0.96      0.97      0.97       974
          9       0.97      0.97      0.97      1009

avg / total       0.98      0.98      0.98     10000














_________________________________________________________________________
using:

nn = Classifier(
    layers=[
        Layer("Rectifier", units=300),
		Layer("Rectifier", units=200),
        Layer("Softmax")],
    learning_rate=0.005,
	dropout_rate=0.3,
	valid_size=0.05,
	learning_momentum=0.7,
	batch_size=10,
	#learning_rule='sgd',
    n_iter=100,
	verbose=True)

________________________________________________________________________________





C:\Users\DPY\Desktop\code>python mnist.py
(50000, 784) ,  (50000, 1)
D:\Programs\Anaconda2\lib\site-packages\theano\tensor\signal\downsample.py:6: UserWarning: downsample module has been moved to the theano.tensor.signal.pool module.
  "downsample module has been moved to the theano.tensor.signal.pool module.")
Initializing neural network with 3 layers, 784 inputs and 10 outputs.
  - Dense: Rectifier  Units:  300
  - Dense: Rectifier  Units:  200
  - Dense: Softmax    Units:  10

Training on dataset of 50,000 samples with 39,700,000 total size.
  - Train: 47,500     Valid: 2,500
  - Using `dropout` for regularization.
  - Terminating loop after 100 total iterations.
  - Early termination after 10 stable iterations.

Epoch       Training Error       Validation Error       Time
------------------------------------------------------------
    1          8.181e-01             3.264e-01         20.4s
    2          4.532e-01             2.461e-01         20.7s
    3          3.661e-01             1.964e-01         20.2s
    4          3.204e-01             1.717e-01         22.7s
    5          2.924e-01             1.523e-01         20.6s
    6          2.651e-01             1.367e-01         23.3s
    7          2.465e-01             1.249e-01         20.3s
    8          2.282e-01             1.168e-01         21.8s
    9          2.204e-01             1.109e-01         20.7s
   10          2.069e-01             1.016e-01         21.4s
   11          1.944e-01             9.936e-02         21.8s
   12          1.848e-01             9.176e-02         22.4s
   13          1.752e-01             9.000e-02         20.0s
   14          1.713e-01             8.699e-02         19.5s
   15          1.645e-01             8.494e-02         18.8s
   16          1.629e-01             8.164e-02         35.5s
   17          1.523e-01             8.192e-02         20.6s
   18          1.520e-01             7.665e-02         25.8s
   19          1.469e-01             7.560e-02         21.5s
   20          1.431e-01             7.522e-02         21.5s
   21          1.396e-01             7.239e-02         20.9s
   22          1.345e-01             7.022e-02         20.0s
   23          1.311e-01             6.870e-02         20.9s
   24          1.327e-01             6.751e-02         22.9s
   25          1.289e-01             6.669e-02         22.8s
   26          1.244e-01             6.798e-02         19.6s
   27          1.239e-01             6.540e-02         24.6s
   28          1.181e-01             6.681e-02         19.8s
   29          1.199e-01             6.516e-02         21.6s
   30          1.161e-01             6.214e-02         25.5s
   31          1.141e-01             6.470e-02         20.0s
   32          1.107e-01             5.996e-02         19.2s
   33          1.068e-01             6.042e-02         17.9s
   34          1.089e-01             5.943e-02         19.1s
   35          1.065e-01             6.050e-02         19.1s
   36          1.063e-01             5.802e-02         18.8s
   37          1.037e-01             5.716e-02         20.4s
   38          1.002e-01             5.884e-02         18.1s
   39          9.887e-02             5.756e-02         18.1s
   40          9.664e-02             5.714e-02         19.0s
   41          9.532e-02             5.690e-02         19.9s
   42          9.649e-02             5.617e-02         19.5s
   43          9.452e-02             5.320e-02         19.3s
   44          9.296e-02             5.346e-02         20.1s
   45          9.308e-02             5.369e-02         19.0s
   46          8.849e-02             5.251e-02         18.0s
   47          9.072e-02             5.323e-02         18.1s
   48          8.943e-02             5.285e-02         21.7s
   49          8.580e-02             5.056e-02         19.5s
   50          8.949e-02             5.219e-02         20.9s
   51          8.560e-02             5.132e-02         21.2s
   52          8.541e-02             5.201e-02         20.7s
   53          8.423e-02             5.176e-02         21.9s
   54          8.215e-02             5.150e-02         22.0s
   55          8.076e-02             4.973e-02         20.9s
   56          8.127e-02             4.998e-02         20.7s
   57          8.033e-02             4.974e-02         21.0s
   58          8.164e-02             4.911e-02         21.9s
   59          8.019e-02             4.730e-02         19.8s
   60          7.681e-02             4.782e-02         18.7s
   61          7.772e-02             5.094e-02         18.6s
   62          7.602e-02             4.897e-02         19.1s
   63          7.737e-02             4.844e-02         22.0s
   64          7.398e-02             5.033e-02         20.0s
   65          7.470e-02             4.874e-02         20.4s
   66          7.371e-02             4.761e-02         21.9s
   67          7.346e-02             4.864e-02         19.1s
   68          7.256e-02             4.707e-02         20.3s
   69          7.268e-02             4.828e-02         20.2s
   70          7.203e-02             4.775e-02         19.5s
   71          7.078e-02             5.043e-02         21.8s
   72          7.021e-02             4.658e-02         21.5s
   73          6.797e-02             4.650e-02         19.8s
   74          6.995e-02             4.839e-02         20.2s
   75          6.948e-02             4.511e-02         19.1s
   76          7.071e-02             4.519e-02         20.3s
   77          6.684e-02             4.576e-02         20.2s
   78          6.858e-02             4.479e-02         20.4s
   79          6.568e-02             4.729e-02         19.4s
   80          6.710e-02             4.635e-02         19.3s
   81          6.679e-02             4.491e-02         20.6s
   82          6.297e-02             4.622e-02         21.5s
   83          6.453e-02             4.501e-02         24.3s
   84          6.375e-02             4.572e-02         20.6s
   85          6.462e-02             4.515e-02         20.3s
   86          6.231e-02             4.488e-02         18.3s
   87          6.196e-02             4.608e-02         21.3s
   88          6.260e-02             4.528e-02         21.3s

Early termination condition fired at 88 iterations.
[(10000, 10)]
0.9837
[(10000, 10)]
        Report:
             precision    recall  f1-score   support

          0       0.98      0.99      0.99       980
          1       0.99      0.99      0.99      1135
          2       0.98      0.99      0.99      1032
          3       0.98      0.98      0.98      1010
          4       0.99      0.98      0.99       982
          5       0.99      0.97      0.98       892
          6       0.99      0.99      0.99       958
          7       0.98      0.98      0.98      1028
          8       0.98      0.98      0.98       974
          9       0.98      0.98      0.98      1009

avg / total       0.98      0.98      0.98     10000


Confusion matrix:
[[ 975    1    1    0    0    0    0    1    2    0]
 [   0 1128    2    1    0    1    2    0    1    0]
 [   3    0 1018    2    1    0    0    5    3    0]
 [   2    0    2  992    0    2    0    6    4    2]
 [   2    0    2    0  963    0    4    2    0    9]
 [   2    0    0    9    2  865    6    0    5    3]
 [   5    2    0    1    2    2  944    0    2    0]
 [   1    3    8    1    0    0    0 1006    2    7]
 [   3    0    2    5    0    0    2    3  956    3]
 [   3    4    0    2    4    1    0    3    2  990]]


"""


	
