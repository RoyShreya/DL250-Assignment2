import numpy as np
import matplotlib



#matplotlib.use('Agg')
import matplotlib.pyplot as plt

loss=np.load('TrainLossWithMultilayerNet.npy')
plt.figure()
print(loss)
plt.plot(np.arange(0,loss.shape[0],1),loss)
#plt.switch_backend('agg')
plt.show()

loss=np.load('TrainLossWithConvNet.npy')
plt.figure()
print(loss)
plt.plot(np.arange(0,loss.shape[0],1),loss)
#plt.switch_backend('agg')
plt.show()


loss=np.load('ValAccWithMultilayerNet.npy')
plt.figure()
print(loss)
plt.plot(np.arange(0,loss.shape[0],1),loss)
#plt.switch_backend('agg')
plt.show()

loss=np.load('ValAccWithConvNet.npy')
plt.figure()
print(loss)
plt.plot(np.arange(0,loss.shape[0],1),loss)
#plt.switch_backend('agg')
plt.show()



loss=np.load('ValLossWithMultilayerNet.npy')
plt.figure()
print(loss)
plt.plot(np.arange(0,loss.shape[0],1),loss)
#plt.switch_backend('agg')
plt.show()

loss=np.load('ValLossWithConvNet.npy')
plt.figure()
print(loss)
plt.plot(np.arange(0,loss.shape[0],1),loss)
#plt.switch_backend('agg')
plt.show()

