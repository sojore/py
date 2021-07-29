# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 22:37:44 2021

@author: sojore
"""

#first we need to handle all the important libraries we will need in this project
#note that different versions of python instalation will not support most of the libraries such as tensorflow.contrib
#so it is for this reason we need to install only the python version which is compatible with this library
#loading all the modules neccesary to run our algorithms
#make sure to install all the below libraries for better compatibility and perfomance
#keras 2.1.4
#tensorflow 1.14.0
#tensorflow_probablity 0.7.0
#scipy
#matplotlib
#python 3.6 or 3.7

from models.BoundedFullyNonLinear import *
from models.Merton import *
from models.NoLeverage import *
from models.OneAsset import *
from models.MongeAmpere import *
from models.SemiBounded import *
from models.SemiUnbounded import *
from models.Burgers import *

from networks.FeedForwardU import *
from networks.FeedForwardUDU import *
from networks.FeedForwardUZ import *
from networks.FeedForwardUZDZ import *
from networks.FeedForwardGam import *

from solvers.FullNonLinearMEDBDP  import *
from solvers.FullNonLinearM2DBDPGPU  import *
from solvers.FullNonLinearMDBDPGPU import *
from solvers.FullNonLinearBaseGPU import *
from solvers.FullNonLinearPWG import *
from solvers.SemiLinearBaseGPU import *
from solvers.Splitting import *
from solvers.HPW import *
from solvers.SemiHJE import *
import matplotlib
from matplotlib import pyplot as plt




#TRANSFER LEARNING IMPLEMENTATION  ON MODEL 1.Mertonxxx
##Make sure to run each model-type code at a time

import numpy as np
import tensorflow as tf
import os
import solvers  as solv
import networks as net
import models as mod
import multiprocessing
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_MULTIPROCESSING_CORE_COUNT = multiprocessing.cpu_count()
print("args", sys.argv)

d = 1
xInit= np.ones(d,dtype=np.float32) 
nbLayer= 2  
rescal=1.
T=1.

batchSize= 1000
batchSizeVal= 10000
num_epoch=400 
num_epochExtNoLast =10
num_epochExtLast= 200
initialLearningRateLast = 1e-2
initialLearningRateNoLast = 1e-3
nbOuterLearning =10
nTest = 1

lamb = np.array([1.5], dtype=np.float32)
eta = 0.5
theta = np.array([0.4], dtype=np.float32) 
gamma = np.array([0.2], dtype=np.float32) 
kappa = np.array([1.], dtype=np.float32)
sigma = np.array([1.], dtype=np.float32)
nbNeuron = d + 10
sigScal =   np.array([1.], dtype=np.float32)

muScal = np.array([np.sum(theta*lamb)])
layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32) 

# create the model
model = mod.ModelMerton(xInit, muScal, sigScal, T, theta, lamb, eta)

print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")
print("REAL IS ", model.Sol(0.,xInit), " DERIV", model.derSol(0.,xInit))

theNetwork = net.FeedForwardUZ(d,layerSize,tf.nn.tanh)
theNetworkGam = net.FeedForwardGam(d,layerSize,tf.nn.tanh)

ndt = [ (120,3) ] 

print("PDE Merton MDBDP  Dim ", d, " layerSize " , layerSize,  " rescal " ,rescal, "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast, "VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.PDEFNLSolve2OptZGPU(model, T, indt[0], indt[1], theNetwork , theNetworkGam, initialLearningRateLast=initialLearningRateLast, initialLearningRateNoLast = initialLearningRateNoLast)
        
    baseFile = "MertonMDBDPd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt[0])+str(indt[1])+"eta"+str(int(eta*100))
    plotFile = "C:\\Users\\sojore\\Pictures/"+baseFile
    saveFolder = "save/"

    Y0List = []
    for i in range(nTest):
        # train
        t0 = time.time()
        Y0, Z0, Gamma0    = resol.BuildAndtrainML( batchSize, batchSizeVal, num_epochExtNoLast=num_epochExtNoLast, num_epochExtLast= num_epochExtLast ,num_epoch=num_epoch,  nbOuterLearning=nbOuterLearning, thePlot= plotFile , baseFile = baseFile, saveDir= saveFolder)
        t1 = time.time()
        print(" NBSTEP", indt, " EstimMC Val is " , Y0,  " REAL IS ", model.Sol(0.,xInit),    " Z0 ", Z0," DERREAL IS  ",model.derSol(0.,xInit), "Gamma0 " , Gamma0, t1-t0)

        Y0List.append(Y0)
        print(Y0List)

    print("Y0", Y0List)
    yList = np.array(Y0List)
    yMean = np.mean(yList)
    print(" DNT ", indt , "MeanVal ", yMean, " Etyp ", np.sqrt(np.mean(np.power(yList-yMean,2.))))

#1.Mertonxxx

#-in this groups of algorithms ,we gonna train one model e.g MertonMDBDP,we then save its weights -train another model e.g MertonM2DBDP ,freeze its hidden layers so it does not train again, -transfer this model to load the saved weights from MertonMDBDP and compare its perfomance
#lets save the weights,and the above trained model 
#saving the weights/parameters
model.save_weights('MertonMDBDP_model_types_weights.h5')#this will save the weights of the MertonMDBDP model trained above
#which we want
#to load them using a different algorithm such as MertonM2DBDP and see how it perfoms of these weights

#we now going to train another model MertonM2DBDP ,save the model,load it again,freeze the hidden layers of the model,
#add an outer dense layer (output neurons=2) ,use this new model and load the MertonMDBDP_model_types_weights, to see its
#perfomance

# we are first executing the code below so as to get a MertonM2DBDP_model,which we can then save it,freeze the hidden layers
#then transfer the new model to load the MertonMDBDP_model_types_weights
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import solvers  as solv
import networks as net
import models as mod
import multiprocessing
import sys
import time

_MULTIPROCESSING_CORE_COUNT = multiprocessing.cpu_count()
print("args", sys.argv)

d = 1 
xInit= np.ones(d,dtype=np.float32) 
nbLayer= 2  
print("nbLayer " ,nbLayer)
rescal=1.
T=1.

batchSize= 1000
batchSizeVal= 10000
num_epoch=400 
num_epochExtNoLast =10
num_epochExtLast= 200
initialLearningRateLast = 1e-2
initialLearningRateNoLast = 1e-3
nbOuterLearning =10
nTest = 1

lamb = np.array([1.5], dtype=np.float32)
eta = 0.5
theta = np.array([0.4], dtype=np.float32) 
gamma = np.array([0.2], dtype=np.float32) 
kappa = np.array([1.], dtype=np.float32)
sigma = np.array([1.], dtype=np.float32) 
nbNeuron = d + 10
sigScal =   np.array([1.], dtype=np.float32)

muScal = np.array([np.sum(theta*lamb)])
layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32) 
# create the model
model = mod.ModelMerton(xInit, muScal, sigScal, T, theta, lamb, eta)

print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")
print("REAL IS ", model.Sol(0.,xInit), " DERIV", model.derSol(0.,xInit))

theNetwork = net.FeedForwardUZ(d,layerSize,tf.nn.tanh)
theNetworkGam = net.FeedForwardGam(d,layerSize,tf.nn.tanh)

ndt = [ (120,30) ] 


print("PDE Merton M2DBDP  Dim ", d, " layerSize " , layerSize,  " rescal " ,rescal, "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast,"VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.PDEFNLSolve2OptGPU(model, T, indt[0], indt[1], theNetwork , theNetworkGam, initialLearningRateLast=initialLearningRateLast, initialLearningRateNoLast = initialLearningRateNoLast)

    baseFile = "MertonM2DBDPd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt[0])+str(indt[1])+"eta"+str(int(eta*100))
    plotFile = "pictures/"+baseFile
    saveFolder = "save/"

    Y0List = []
    for i in range(nTest):
        # train
        t0 = time.time()
        Y0, Z0, Gamma0  = resol.BuildAndtrainML( batchSize, batchSizeVal, num_epochExtNoLast=num_epochExtNoLast, num_epochExtLast= num_epochExtLast ,num_epoch=num_epoch,  nbOuterLearning=nbOuterLearning, thePlot= plotFile , baseFile = baseFile, saveDir= saveFolder)
        t1 = time.time()
        print(" NBSTEP", indt, " EstimMC Val is " , Y0,  " REAL IS ", model.Sol(0.,xInit),    " Z0 ", Z0," DERREAL IS  ",model.derSol(0.,xInit), "Gamma0 " , Gamma0,t1-t0)

        Y0List.append(Y0)
        print(Y0List)

    print("Y0", Y0List)
    yList = np.array(Y0List)
    yMean = np.mean(yList)
    print(" DNT ", indt , "MeanVal ", yMean, " Etyp ", np.sqrt(np.mean(np.power(yList-yMean,2.))))


#saving the model
model.save('MertonM2DBDPd_model.h5')

#loading the saved model
from keras.models import load_model
new_MertonMDBDP_model=load_model('MertonM2DBDPd_model.h5')

#getting the summary of the model
new_MertonMDBDP_model.summary()

#implementing transfer_learn by freezing the inner layers of this new_MertonMDBDP_model ,so we can train it on the  
#MertonMDBDP_model_types_weights weights

type(new_MertonMDBDP_model)

#we always convert the functional models into sequential model and also remove the last layer (output layer)
MertonM2DBDPd_model_new=Sequential()
for layer in new_MertonMDBDP_model.layers[:-1]:#this code removes the last layer of the model
    MertonM2DBDPd_model_new.add(layer)
    
#we can confirm the type of the model is Sequential or functional
type(MertonM2DBDPd_model_new)   

MertonM2DBDPd_model_new.summary()

##freezing the layers so we dont train them again
for layer in MertonM2DBDPd_model_new.layers:
    layer.trainable=False #we set the hidden layers untrainable so we dont train them again
    
#we now add a custom  last layer(Dense) to our transfered MertonM2DBDPd_model_new model with 2 output classes
MertonM2DBDPd_model_new.add(Dense(2,activation='softmax'))

#validation of the added last layer
MertonM2DBDPd_model_new.summary()

#we finnaly use this transefered MertonM2DBDPd_model_new model to load the already saved weights from  MertonMDBDP model
#thus we compare it perfomance
#we do the same for algorithms 1.MertonEMDBDP
#                              2.MertonPWG
#for this group Mertonxxx we implement the frozen models and load  the MertonMDBDP_model_types_weights

#transfer  MertonM2DBDPd_model and train it on the MertonMDBDP_model_types_weights
MertonM2DBDPd_model_new.load_weights('MertonMDBDP_model_types_weights')










#TRANSFER LEARNING IMPLEMENTATION  ON MODEL 2.NoLeveragexxx
##Make sure to run each model-type code at a time

#2.NoLeveragexxx
#-in this groups of algorithms ,we gonna train one model e.g NoLeverageM2DBDP,we then save its weights -train another model e.g NoLeverageEMDBDP ,freeze its hidden layers so it does not train again, -transfer this model to load the saved weights from NoLeverageEMDBDP and compare its perfomance

#we now going to train another model NoLeverageM2DBDP ,save the model,load it again,freeze the hidden layers of the model,
#add an outer dense layer (output neurons=2) ,use this new model and load the NoLeverageEMDBDP_model_types_weights, to see its
#perfomance

import numpy as np
import tensorflow as tf
import os
import solvers  as solv
import networks as net
import models as mod
import multiprocessing
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_MULTIPROCESSING_CORE_COUNT = multiprocessing.cpu_count()
print("args", sys.argv)


nbLayer= 2  
print("nbLayer " ,nbLayer)
rescal=1.
T=1.

batchSize= 1000
batchSizeVal= 10000
num_epoch= 400 
num_epochExtNoLast =10
num_epochExtLast= 200
initialLearningRateLast = 1e-2
initialLearningRateNoLast = 1e-3
nbOuterLearning = 10
nTest = 10

lamb = np.array([1.5, 1.1, 2., 0.8, 0.5, 1.7, 0.9, 1., 0.9,1.5], dtype=np.float32)[0:9]
eta = 0.5
theta = np.array([0.1, 0.2, 0.3, 0.4, 0.25, 0.15, 0.18, 0.08, 0.91,0.4], dtype=np.float32)[0:9]
gamma = np.array([0.2, 0.15, 0.25, 0.31, 0.4, 0.35, 0.22, 0.4, 0.15,0.2], dtype=np.float32)[0:9]
kappa = np.array([1., 0.8, 1.1, 1.3, 0.95, 0.99, 1.02, 1.06, 1.6,1.], dtype=np.float32)[0:9]
sigma = np.ones(10, dtype=np.float32)[0:9]
d = lamb.shape[0] + 1

nbNeuron = d + 10
layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32)

sigScal =  np.concatenate([np.array([1.], dtype=np.float32),gamma]).reshape((d))
muScal = np.concatenate([np.array([np.sum(theta*lamb)]),np.zeros((d-1), dtype=np.float32)]).reshape((d))
xyInit= np.concatenate([np.array([1.]),theta])   

# create the model
model = mod.ModelNoLeverage(xyInit, muScal, sigScal, T, theta, sigma, lamb, eta, gamma, kappa)

print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")
print("REAL IS ", model.Sol(0.,xyInit.reshape(1,d)), " DERIV", model.derSol(0.,xyInit.reshape(1,d)), " GAMMA", model.der2Sol(0.,xyInit.reshape(1,d)))

theNetwork = net.FeedForwardUZDZ(d,layerSize,tf.nn.tanh)

ndt = [120]


print("PDE No Leverage EMDBDP  Dim ", d, " layerSize " , layerSize,  " rescal " ,rescal, "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast,"VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.PDEFNLSolveSimpleLSExp(model, T, indt, theNetwork , initialLearningRateLast=initialLearningRateLast, initialLearningRateNoLast = initialLearningRateNoLast)
        
    baseFile = "NoLeverageEMDBDPd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt)+"eta"+str(int(eta*100))
    plotFile = "pictures/"+baseFile
    saveFolder = "save/"

    Y0List = []
    for i in range(nTest):
        # train
        t0 = time.time()
        Y0, Z0, Gamma0  = resol.BuildAndtrainML( batchSize, batchSizeVal, num_epochExtNoLast=num_epochExtNoLast, num_epochExtLast= num_epochExtLast ,num_epoch=num_epoch,  nbOuterLearning=nbOuterLearning, thePlot= plotFile ,  baseFile = baseFile, saveDir= saveFolder)
        t1 = time.time()
        print(" NBSTEP", indt, " EstimMC Val is " , Y0,  " REAL IS ", model.Sol(0.,xyInit.reshape((1,d))),    " Z0 ", Z0," DERREAL IS  ",model.derSol(0.,xyInit.reshape((1,d))), "Gamma0 " , Gamma0,  " GAMMA", model.der2Sol(0.,xyInit.reshape(1,d)),t1-t0)

        Y0List.append(Y0)
        print(Y0List)

    print("Y0", Y0List)
    yList = np.array(Y0List)
    yMean = np.mean(yList)
    print(" DNT ", indt , "MeanVal ", yMean, " Etyp ", np.sqrt(np.mean(np.power(yList-yMean,2.))))


#lets save the weights,and the above trained model 
#saving the weights/parameters
model.save_weights('NoLeverageEMDBDP_model_types_weights.h5')#this will save the weights of the NoLeverageEMDBDP model trained above
#which we want
#to load them using a different algorithm such as NoLeverageEMDBDP and see how it perfoms of these weights

# we are first executing the code below so as to get a NoLeverageM2DBDP_model,which we can then save it,freeze the hidden layers
#then transfer the new model to load the NoLeverageEMDBDP_model_types_weights
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import solvers  as solv
import networks as net
import models as mod
import multiprocessing
import sys
import time

_MULTIPROCESSING_CORE_COUNT = multiprocessing.cpu_count()
print("args", sys.argv)

nbLayer= 2  
print("nbLayer " ,nbLayer)
sig = 1 
print("Sig ",  sig)
rescal=1.
muScal =0.
T=1.

batchSize= 1000
batchSizeVal= 10000
num_epoch=400 
num_epochExtNoLast =10
num_epochExtLast= 200
initialLearningRateLast = 1e-2
initialLearningRateNoLast = 1e-3
nbOuterLearning =10
nTest = 10


lamb = np.array([1.5, 1.1, 2., 0.8, 0.5, 1.7, 0.9, 1., 0.9, 1.5], dtype=np.float32)[0:9]
eta = 0.5
theta = np.array([0.1, 0.2, 0.3, 0.4, 0.25, 0.15, 0.18, 0.08, 0.91, 0.4], dtype=np.float32)[0:9]
gamma = np.array([0.2, 0.15, 0.25, 0.31, 0.4, 0.35, 0.22, 0.4, 0.15, 0.2], dtype=np.float32)[0:9]
kappa = np.array([1., 0.8, 1.1, 1.3, 0.95, 0.99, 1.02, 1.06, 1.6, 0.1], dtype=np.float32)[0:9]
sigma = np.ones(9, dtype=np.float32)[0:9]
d = lamb.shape[0] + 1
nbNeuron = d + 10
sigScal =  np.concatenate([np.array([1.], dtype=np.float32),gamma]).reshape((d))

muScal = np.concatenate([np.array([np.sum(theta*lamb)]),np.zeros((d-1), dtype=np.float32)]).reshape((d))
layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32)

xyInit= np.concatenate([np.array([1]),theta])   

# create the model
model = mod.ModelNoLeverage(xyInit, muScal, sigScal, T, theta, sigma, lamb, eta, gamma, kappa)

print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")
print("REAL IS ", model.Sol(0.,xyInit.reshape(1,d)), " DERIV", model.derSol(0.,xyInit.reshape(1,d)), " GAMMA", model.der2Sol(0.,xyInit.reshape(1,d)))

theNetwork = net.FeedForwardUZ(d,layerSize,tf.nn.tanh)
theNetworkGam = net.FeedForwardGam(d,layerSize,tf.nn.tanh)

ndt = [ (120,30) ] 

print("PDE No Leverage M2DBDP  Dim ", d, " layerSize " , layerSize,  " rescal " ,rescal, "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast,  "VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.PDEFNLSolve2OptGPU(model, T, indt[0], indt[1], theNetwork , theNetworkGam, initialLearningRateLast=initialLearningRateLast, initialLearningRateNoLast = initialLearningRateNoLast)
        
    baseFile = "NoLeverageM2DBDPd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt[0])+str(indt[1])+"eta"+str(int(eta*100))
    plotFile = "pictures/"+baseFile
    saveFolder = "save/"

    Y0List = []
    for i in range(nTest):
        # train
        t0 = time.time()
        Y0, Z0, Gamma0  = resol.BuildAndtrainML( batchSize, batchSizeVal, num_epochExtNoLast=num_epochExtNoLast, num_epochExtLast= num_epochExtLast ,num_epoch=num_epoch,  nbOuterLearning=nbOuterLearning, thePlot= plotFile , baseFile = baseFile, saveDir= saveFolder)
        t1 = time.time()
        print(" NBSTEP", indt, " EstimMC Val is " , Y0,  " REAL IS ", model.Sol(0.,xyInit.reshape(1,d)),    " Z0 ", Z0," DERREAL IS  ",model.derSol(0.,xyInit.reshape(1,d)), "Gamma0 " , Gamma0, " GAMMA", model.der2Sol(0.,xyInit.reshape(1,d)),t1-t0)

        Y0List.append(Y0)
        print(Y0List)

    print("Y0", Y0List)
    yList = np.array(Y0List)
    yMean = np.mean(yList)
    print(" DNT ", indt , "MeanVal ", yMean, " Etyp ", np.sqrt(np.mean(np.power(yList-yMean,2.))))

#saving the model
model.save('NoLeverageM2DBDP_model.h5')

#loading the saved model
from keras.models import load_model
new_NoLeverageM2DBDP_model=load_model('NoLeverageM2DBDP_model.h5')

#we always convert the functional models into sequential model and also remove the last layer (output layer)
NoLeverageM2DBDP_model_new=Sequential()
for layer in new_NoLeverageM2DBDP_model.layers[:-1]:#this code removes the last layer of the model
    NoLeverageM2DBDP_model_new.add(layer)

##freezing the layers so we dont train them again
for layer in NoLeverageM2DBDP_model_new.layers:
    layer.trainable=False #we set the hidden layers untrainable so we dont train them again
    
#we now add a custom  last layer(Dense) to our transfered NoLeverageM2DBD_model_new model with 2 output classes
NoLeverageM2DBDP_model_new.add(Dense(2,activation='softmax'))    

#we finnaly use this transefered NoLeverageM2DBDP_model_new model to load the already saved weights from  NoLeverageEMDBDP model
#thus we compare it perfomance
#we do the same for algorithms 1.NoLeverageMDBDP
#                              2.NoLeveragePWG
#for this group NoLeverageMxxx we implement the frozen models and load  the NoLeverageEMDBDP_model_types_weights

#transfer  NoLeverageM2DBDP_model and train it on the NoLeverageEMDBDP_model_types_weights
NoLeverageM2DBDP_model_new.load_weights('NoLeverageEMDBDP_model_types_weights')










#TRANSFER LEARNING IMPLEMENTATION  ON MODEL 3.OneAssetxxx
##Make sure to run each model-type code at a time

#3.OneAssetxxx
#-in this groups of algorithms ,we gonna train one model e.g OneAssetEMDBDP,we then save its weights -train another model e.g OneAssetM2DBDP ,freeze its hidden layers so it does not train again, -transfer this model to load the saved weights from OneAssetEMDBDP and compare its perfomance

import numpy as np
import tensorflow as tf
import os
import solvers  as solv
import networks as net
import models as mod
import multiprocessing
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_MULTIPROCESSING_CORE_COUNT = multiprocessing.cpu_count()
print("args", sys.argv)

nbLayer= 2  
print("nbLayer " ,nbLayer)
rescal=1.
T=1.

batchSize= 1000
batchSizeVal= 10000
num_epoch= 400 
num_epochExtNoLast =10
num_epochExtLast= 200
initialLearningRateLast = 1e-2
initialLearningRateNoLast = 1e-3
nbOuterLearning = 10
nTest = 10

lamb = np.array([1.], dtype=np.float32)
eta = 0.5
theta = np.array([0.4], dtype=np.float32) 
gamma = np.array([0.4], dtype=np.float32) 
kappa = np.array([1.], dtype=np.float32) 
sigma = np.array([1.], dtype=np.float32)
rho = np.array([-0.7], dtype=np.float32)
d = lamb.shape[0] + 1
nbNeuron = d + 10
sigScal =  np.concatenate([np.array([1.], dtype=np.float32),gamma]).reshape((d))
muScal = np.concatenate([np.array([np.sum(theta*lamb)]),np.zeros((d-1), dtype=np.float32)]).reshape((d))

layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32)
xyInit= np.concatenate([np.array([1]),theta])  

# create the model
model = mod.ModelOneAsset(xyInit, muScal, sigScal, T, theta, sigma, lamb, eta, gamma, kappa, rho)

print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")
print("REAL IS ", model.Sol(0.,xyInit.reshape(1,d)), " DERIV", model.derSol(0.,xyInit.reshape(1,d)), " GAMMA", model.der2Sol(0.,xyInit.reshape(1,d)))

theNetwork = net.FeedForwardUZDZ(d,layerSize,tf.nn.tanh)

ndt = [120]


print("PDE OneAsset EMDBDP  Dim ", d, " layerSize " , layerSize,  " rescal " ,rescal, "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast,"VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.PDEFNLSolveSimpleLSExp(model, T, indt, theNetwork , initialLearningRateLast=initialLearningRateLast, initialLearningRateNoLast = initialLearningRateNoLast)
        
    baseFile = "OneAssetEMDBDPd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt)+"eta"+str(int(eta*100))
    plotFile = "pictures/"+baseFile
    saveFolder = "save/"

    Y0List = []
    for i in range(nTest):
        # train
        t0 = time.time()
        Y0, Z0, Gamma0  = resol.BuildAndtrainML( batchSize, batchSizeVal, num_epochExtNoLast=num_epochExtNoLast, num_epochExtLast= num_epochExtLast ,num_epoch=num_epoch,  nbOuterLearning=nbOuterLearning, thePlot= plotFile ,  baseFile = baseFile, saveDir= saveFolder)
        t1 = time.time()
        print(" NBSTEP", indt, " EstimMC Val is " , Y0,  " REAL IS ", model.Sol(0.,xyInit.reshape((1,d))),    " Z0 ", Z0," DERREAL IS  ",model.derSol(0.,xyInit.reshape((1,d))), "Gamma0 " , Gamma0," GAMMA", model.der2Sol(0.,xyInit.reshape(1,d)),t1-t0)

        Y0List.append(Y0)
        print(Y0List)

    print("Y0", Y0List)
    yList = np.array(Y0List)
    yMean = np.mean(yList)
    print(" DNT ", indt , "MeanVal ", yMean, " Etyp ", np.sqrt(np.mean(np.power(yList-yMean,2.))))

#lets save the weights,and the above trained model 
#saving the weights/parameters
model.save_weights('OneAssetEMDBDP_model_types_weights.h5')#this will save the weights of the OneAssetEMDBDP model trained above
#which we want
#to load them using a different algorithm such as OneAssetM2DBDP and see how it perfoms of these weights

# we are first executing the code below so as to get a OneAssetM2DBDP_model,which we can then save it,freeze the hidden layers
#then transfer the new model to load the OneAssetEMDBDP_model_types_weights
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import solvers  as solv
import networks as net
import models as mod
import multiprocessing
import sys
import time

 

_MULTIPROCESSING_CORE_COUNT = multiprocessing.cpu_count()
print("args", sys.argv)

nbLayer= 2  
print("nbLayer " ,nbLayer)
sig = 1 
print("Sig ",  sig)
rescal=1.
T=1.

batchSize= 1000
batchSizeVal= 10000
num_epoch=400 
num_epochExtNoLast =10
num_epochExtLast= 200
initialLearningRateLast = 1e-2
initialLearningRateNoLast = 1e-3
nbOuterLearning =10
nTest = 10

lamb = np.array([1.], dtype=np.float32)
eta = 0.5
theta = np.array([0.4], dtype=np.float32) 
gamma = np.array([0.4], dtype=np.float32) 
kappa = np.array([1.], dtype=np.float32) 
sigma = np.array([1.], dtype=np.float32)
rho = np.array([-0.7], dtype=np.float32)
d = lamb.shape[0] + 1
nbNeuron = d + 10
sigScal =  np.concatenate([np.array([1.], dtype=np.float32),gamma]).reshape((d))

muScal = np.concatenate([np.array([np.sum(theta*lamb)]),np.zeros((d-1), dtype=np.float32)]).reshape((d))

layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32)
xyInit= np.concatenate([np.array([1]),theta])  

# create the model
model = mod.ModelOneAsset(xyInit, muScal, sigScal, T, theta, sigma, lamb, eta, gamma, kappa, rho)

print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")
print("REAL IS ", model.Sol(0.,xyInit.reshape(1,d)), " DERIV", model.derSol(0.,xyInit.reshape(1,d)), " GAMMA", model.der2Sol(0.,xyInit.reshape(1,d)))

theNetwork = net.FeedForwardUZ(d,layerSize,tf.nn.tanh)
theNetworkGam = net.FeedForwardGam(d,layerSize,tf.nn.tanh)

ndt = [(120,30) ] 
print("PDE OneAsset M2DBDP  Dim ", d, " layerSize " , layerSize,  " rescal " ,rescal, "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast,  "VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.PDEFNLSolve2OptGPU(model, T, indt[0], indt[1], theNetwork , theNetworkGam, initialLearningRateLast=initialLearningRateLast, initialLearningRateNoLast = initialLearningRateNoLast)
        
    baseFile = "OneAssetM2DBDPd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt[0])+str(indt[1])+"eta"+str(int(eta*100))
    plotFile = "pictures/"+baseFile
    saveFolder = "save/"

    Y0List = []
    for i in range(nTest):
        # train
        t0 = time.time()
        Y0, Z0, Gamma0  = resol.BuildAndtrainML( batchSize, batchSizeVal, num_epochExtNoLast=num_epochExtNoLast, num_epochExtLast= num_epochExtLast ,num_epoch=num_epoch,  nbOuterLearning=nbOuterLearning, thePlot= plotFile , baseFile = baseFile, saveDir= saveFolder)
        t1 = time.time()
        print(" NBSTEP", indt, " EstimMC Val is " , Y0,  " REAL IS ", model.Sol(0.,xyInit.reshape(1,d)),    " Z0 ", Z0," DERREAL IS  ",model.derSol(0.,xyInit.reshape(1,d)), "Gamma0 " , Gamma0, " GAMMA", model.der2Sol(0.,xyInit.reshape(1,d)),t1-t0)

        Y0List.append(Y0)
        print(Y0List)

    print("Y0", Y0List)
    yList = np.array(Y0List)
    yMean = np.mean(yList)
    print(" DNT ", indt , "MeanVal ", yMean, " Etyp ", np.sqrt(np.mean(np.power(yList-yMean,2.))))

#saving the model
model.save('OneAssetM2DBDP_model.h5')

#loading the saved model
from keras.models import load_model
new_OneAssetM2DBDP_model=load_model('OneAssetM2DBDP_model.h5')

#we always convert the functional models into sequential model and also remove the last layer (output layer)
OneAssetM2DBDP_model_new=Sequential()
for layer in new_OneAssetM2DBDP_model.layers[:-1]:#this code removes the last layer of the model
    OneAssetM2DBDP_model_new.add(layer)
    
##freezing the layers so we dont train them again
for layer in OneAssetM2DBDP_model_new.layers:
    layer.trainable=False #we set the hidden layers untrainable so we dont train them again
    
#we now add a custom  last layer(Dense) to our transfered OneAssetM2DBDP_model_new model with 2 output classes
OneAssetM2DBDP_model_new.add(Dense(2,activation='softmax'))

#we finnaly use this transefered OneAssetM2DBDP_model_new model to load the already saved weights from  OneAssetEMDBDP model
#thus we compare it perfomance
#we do the same for algorithms 1.OneAssetMDBDP
#                              2.OneAssetPWG

#transfer  OneAssetM2DBDP_model and train it on the OneAssetEMDBDP_model_types_weights
OneAssetM2DBDP_model_new.load_weights('OneAssetEMDBDP_model_types_weights')








#TRANSFER LEARNING IMPLEMENTATION  ON MODEL 4.Burgersxxx
##Make sure to run each model-type code at a time

#4.Burgersxxx
#-in this group of algorithms we gonna train BurgersHJE model,save its weights
#-we then train BurgersHPW model,freeze its hidden layers,then transfer it to load the saved weights
#-also compare its perfomance

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import solvers  as solv
import networks as net
import models as mod
import multiprocessing
import sys
import time

_MULTIPROCESSING_CORE_COUNT = multiprocessing.cpu_count()
print("args", sys.argv)

nbLayer= 2  
print("nbLayer " ,nbLayer)

rescal=1.
T=1.

batchSize= 1000
batchSizeVal= 50000 
num_epoch= 50 
num_epochExtNoLast = 200
initialLearningRateLast = 1e-2
initialLearningRateNoLast = 1e-3
nbOuterLearning =10
nTest = 10

d = 1
nbNeuron = 11
sigScal = np.ones(d, dtype=np.float32)
muScal = np.zeros(d, dtype=np.float32)
nu = 1

layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32)

xInit = np.ones(d, dtype=np.float32) 

# create the model
model = mod.Burgers(xInit, muScal, sigScal, T, nu)

print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")

print("REAL IS ", model.Sol(0.,xInit.reshape(1,d),1000000), " DERIV", model.derSol(0.,xInit.reshape(1,d),1000000))

theNetwork = net.FeedForwardUZ(d,layerSize,tf.nn.tanh)

ndt = [ 120 ] 


print("PDE Burgers HJE  Dim ", d, " layerSize " , layerSize,  " rescal " ,rescal, "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  ,"VOL " , sigScal, "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.SemiHJE(model, T, indt, theNetwork, initialLearningRateNoLast = initialLearningRateNoLast)
    
    baseFile = "BurgersHJEd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt)
    plotFile = "pictures/"+baseFile
    saveFolder = "save/"

    Y0List = []
    for i in range(nTest):
        # train
        t0 = time.time()
        Y0, Z0  = resol.BuildAndtrainML( batchSize, batchSizeVal, num_epochExtNoLast=num_epochExtNoLast ,num_epoch=num_epoch,  nbOuterLearning=nbOuterLearning, thePlot= plotFile , baseFile = baseFile, saveDir= saveFolder)
        t1 = time.time()
        print(" NBSTEP", indt, " EstimMC Val is " , Y0,  " REAL IS ", model.Sol(0.,xInit.reshape(1,d)),    " Z0 ", Z0," DERREAL IS  ",model.derSol(0.,xInit.reshape(1,d)),t1-t0)

        Y0List.append(Y0)
        print(Y0List)

    print("Y0", Y0List)
    yList = np.array(Y0List)
    yMean = np.mean(yList)
    print(" DNT ", indt , "MeanVal ", yMean, " Etyp ", np.sqrt(np.mean(np.power(yList-yMean,2.))))

#lets save the weights,and the above trained model 
#saving the weights/parameters
model.save_weights('BurgersHJE_model_types_weights.h5')#this will save the weights of the BurgersHJE model trained above
#which we want
#to load them using a different algorithm such as BurgersHPW and see how it perfoms of these weights

# we are then executing the code below so as to get a BurgersHPW_model,which we can then save it,freeze the hidden layers
#then transfer the new model to load the BurgersHJE_model_types_weights
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import solvers  as solv
import networks as net
import models as mod
import multiprocessing
import sys
import time


_MULTIPROCESSING_CORE_COUNT = multiprocessing.cpu_count()
print("args", sys.argv)

nbLayer= 2
print("nbLayer " ,nbLayer)

T=1.

batchSize= 1000 
batchSizeVal= 50000 
num_epoch= 50 
num_epochExtNoLast = 100
num_epochExtLast= 1000 
initialLearningRateLast = 1e-2 
initialLearningRateNoLast = 1e-3
nbOuterLearning =10
nTest = 10

d = 1
nbNeuron = 11
sigScal = np.ones(d)
muScal = np.zeros(d)
nu = 1

layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32)

xInit = np.ones(d) #np.ones(d) 

# create the model
model = mod.Burgers(xInit, muScal, sigScal, T, nu)

print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")
print("REAL IS ", model.Sol(0.,xInit.reshape(1,d),1000000), " DERIV", model.derSol(0.,xInit.reshape(1,d),1000000))

theNetworkUZ = net.FeedForwardUZ(d,layerSize,tf.nn.tanh)

ndt = [120] 

print("PDE Burgers HPW Dim ", d, " layerSize " , layerSize,   "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast,  "VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.HPW(xInit, model, T, indt, theNetworkUZ, initialLearningRate= initialLearningRateLast,initialLearningRateStep =  initialLearningRateNoLast)
        
    baseFile = "BurgersHPW"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt)
    plotFile = "C:\\Users\\sojore\\Documents\\pyhthon s/"+baseFile #make sure to change the directory 
    saveFolder = "save/"

    Y0List = []
    for i in range(nTest):
        # train
        t0 = time.time()
        Y0, Z0  = resol.BuildAndtrainML( batchSize, batchSizeVal, num_epoch=num_epoch, num_epochExt=num_epochExtNoLast,  nbOuterLearning=nbOuterLearning, thePlot= plotFile , baseFile = baseFile, saveDir= saveFolder)
        t1 = time.time()
        print(" NBSTEP", indt, " EstimMC Val is " , Y0,  " REAL IS ", model.Sol(0.,xInit.reshape(1,d),1000000),    " Z0 ", Z0,t1-t0)

        Y0List.append(Y0)
        print(Y0List)

    print("Y0", Y0List)
    yList = np.array(Y0List)
    yMean = np.mean(yList)
    print(" DNT ", indt , "MeanVal ", yMean, " Etyp ", np.sqrt(np.mean(np.power(yList-yMean,2.))))

#saving the model
model.save('BurgersHPW_model.h5')

#loading the saved model
from keras.models import load_model
new_BurgersHPW_model=load_model('BurgersHPW_model.h5')

#we always convert the functional models into sequential model and also remove the last layer (output layer)
BurgersHPW_model_new=Sequential()
for layer in new_BurgersHPW_model.layers[:-1]:#this code removes the last layer of the model
    BurgersHPW_model_new.add(layer)
    
##freezing the layers so we dont train them again
for layer in BurgersHPW_model_new.layers:
    layer.trainable=False #we set the hidden layers untrainable so we dont train them again

#we now add a custom  last layer(Dense) to our transfered BurgersHPW_model_new model with 2 output classes
BurgersHPW_model_new.add(Dense(2,activation='softmax'))
    
#transfer  BurgersHPW_model and train it on the BurgersHJE_model_types_weights
BurgersHPW_model_new.load_weights('BurgersHJE_model_types_weights')






#TRANSFER LEARNING IMPLEMENTATION  ON MODEL 5.BoundedFNLxxx
##Make sure to run each model-type code at a time    
#5.BoundedFNLxxx
#-in this group of algorithms we gonna train BoundedFNLEMDBDP model,save its weights -we then train BoundedFNLPWG model,freeze its hidden layers,then transfer it to load the saved weights -also compare its perfomance

#we now going to train another model BoundedFNLPWG ,save the model,load it again,freeze the hidden layers of the model,
#add an outer dense layer (output neurons=2) ,use this new model and load the BoundedFNLEMDBDP_model_types_weights, to see its
#perfomance

import numpy as np
import tensorflow as tf
import os
import solvers  as solv
import networks as net
import models as mod
import multiprocessing
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_MULTIPROCESSING_CORE_COUNT = multiprocessing.cpu_count()
print("args", sys.argv)

d = 1
xInit= np.ones(d,dtype=np.float32) 
# nb neuron
nbNeuron = 10 +d 
print("nbNeuron " , nbNeuron)
# nb layer
nbLayer= 2  
print("nbLayer " ,nbLayer)
layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32)
sig = np.array([float(1/np.sqrt(float(d))) ], dtype=np.float32)
print("Sig ",  sig)
rescal=1.
muScal =0.
sigScal=1
alpha= 1 
T=1.

batchSize= 1000
batchSizeVal= 10000
num_epoch=400
num_epochExtNoLast =10
num_epochExtLast= 200
initialLearningRateLast = 1e-2
initialLearningRateNoLast = 1e-3
nbOuterLearning =10
nTest = 10

# create the model
model = mod.BoundedFNL(xInit, muScal, sigScal, rescal, alpha ,d,T)

print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")
print("REAL IS ", model.Sol(0.,xInit), " DERIV", model.derSol(0.,xInit))


theNetwork = net.FeedForwardUZDZ(d,layerSize,tf.nn.tanh)
ndt = [120]

print("PDE BoundedFNL EMDBDP  Dim ", d, " layerSize " , layerSize,  " rescal " ,rescal, "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast,   "alpha ", alpha ,"VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.PDEFNLSolveSimpleLSExp(model, T, indt, theNetwork , initialLearningRateLast=initialLearningRateLast, initialLearningRateNoLast = initialLearningRateNoLast)
        

    baseFile = "BoundedFNLEMDBDPd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt)+"Alpha"+str(int(alpha*100))
    plotFile = "pictures/"+baseFile
    
    Y0List = []
    for i in range(nTest):
        # train
        t0 = time.time()
        Y0, Z0, Gamma0    = resol.BuildAndtrainML( batchSize, batchSizeVal, num_epochExtNoLast=num_epochExtNoLast, num_epochExtLast= num_epochExtLast ,num_epoch=num_epoch,  nbOuterLearning=nbOuterLearning, thePlot= plotFile ,  baseFile = baseFile, saveDir= "save/")
        t1 = time.time()
        print(" NBSTEP", indt, " EstimMC Val is " , Y0,  " REAL IS ", model.Sol(0.,xInit),    " Z0 ", Z0," DERREAL IS  ",model.derSol(0.,xInit), "Gamma0 " , Gamma0,t1-t0)

        Y0List.append(Y0)
        print(Y0List)

    print("Y0", Y0List)
    yList = np.array(Y0List)
    yMean = np.mean(yList)
    print(" DNT ", indt , "MeanVal ", yMean, " Etyp ", np.sqrt(np.mean(np.power(yList-yMean,2.))))


#lets save the weights,and the above trained model 
#saving the weights/parameters
model.save_weights('BoundedFNLEMDBDP_model_types_weights.h5')#this will save the weights of the BoundedFNLEMDBDP model trained above
#which we want
#to load them using a different algorithm such as BoundedFNLEMDBDP and see how it perfoms of these weights


# we are first executing the code below so as to get a BoundedFNLPWG_model,which we can then save it,freeze the hidden layers
#then transfer the new model to load the BoundedFNLPWG_model_types_weights
import numpy as np
import tensorflow as tf
import os
import solvers  as solv
import networks as net
import models as mod
import multiprocessing
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_MULTIPROCESSING_CORE_COUNT = multiprocessing.cpu_count()
print("args", sys.argv)

d = 1
xInit= np.ones(d,dtype=np.float32) 
# nb neuron
nbNeuron = 10 +d 
print("nbNeuron " , nbNeuron)
# nb layer
nbLayer= 2  
print("nbLayer " ,nbLayer)
layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32)
sig = np.array([float(1/np.sqrt(float(d))) ], dtype=np.float32)
print("Sig ",  sig)
rescal=1.
muScal =0.
sigScal=1
alpha= 1 
T=1.

batchSize= 1000
batchSizeVal= 10000
num_epoch= 400 
num_epochExtNoLast =10
num_epochExtLast= 200
initialLearningRateLast = 1e-2
initialLearningRateNoLast = 1e-3
nbOuterLearning =10
nTest = 10

# create the model
model = mod.BoundedFNL(xInit, muScal, sigScal, rescal, alpha ,d,T)
print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")
print("REAL IS ", model.Sol(0.,xInit), " DERIV", model.derSol(0.,xInit))

theNetwork = net.FeedForwardUZ(d,layerSize,tf.nn.tanh)
ndt = [120]


print("PDE Bounded FNL PWG  Dim ", d, " layerSize " , layerSize,  " rescal " ,rescal, "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast,   "alpha ", alpha ,"VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.PDEFullNLExplicitGamAdapt(xInit,model, T, indt, theNetwork , initialLearningRate=initialLearningRateLast, initialLearningRateStep = initialLearningRateNoLast)
        

    baseFile = "BoundedFNLPWGd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt)+"Alpha"+str(int(alpha*100))
    plotFile = "pictures/"+baseFile
    
    Y0List = []
    for i in range(nTest):
        # train
        t0 = time.time()
        Y0, Z0    = resol.BuildAndtrainML( batchSize, batchSizeVal, num_epochExt=num_epochExtNoLast, num_epoch=num_epoch,  nbOuterLearning=nbOuterLearning, thePlot= plotFile ,  baseFile = baseFile, saveDir= "save/")
        t1 = time.time()
        print(" NBSTEP", indt, " EstimMC Val is " , Y0,  " REAL IS ", model.Sol(0.,xInit),    " Z0 ", Z0," DERREAL IS  ",model.derSol(0.,xInit),t1-t0)

        Y0List.append(Y0)
        print(Y0List)

    print("Y0", Y0List)
    yList = np.array(Y0List)
    yMean = np.mean(yList)
    print(" DNT ", indt , "MeanVal ", yMean, " Etyp ", np.sqrt(np.mean(np.power(yList-yMean,2.))))

#saving the model
model.save('BoundedFNLPWG_model.h5')

#loading the saved model
from keras.models import load_model
new_BoundedFNLPWG_model=load_model('BoundedFNLPWG_model.h5')

#getting the summary of the model
new_BoundedFNLPWG_model.summary()

#we always convert the functional models into sequential model and also remove the last layer (output layer)
BoundedFNLPWG_model_new=Sequential()
for layer in new_BoundedFNLPWG_model.layers[:-1]:#this code removes the last layer of the model
    BoundedFNLPWG_model_new.add(layer)
       
#we can confirm the type of the model is Sequential or functional
type(BoundedFNLPWG_model_new)

##freezing the layers so we dont train them again
for layer in BoundedFNLPWG_model_new.layers:
    layer.trainable=False #we set the hidden layers untrainable so we dont train them again

#we now add a custom  last layer(Dense) to our transfered BoundedFNLPWG_model_new model with 2 output classes
BoundedFNLPWG_model_new.add(Dense(2,activation='softmax'))

#we finnaly use this transefered BoundedFNLPWG_model_new model to load the already saved weights from  BoundedFNLEMDBDP model
#thus we compare it perfomance
#we do the same for algorithms 1.BoundedFNLM2DBDP
#                              2.BoundedFNLMDBDP
#for this group BoundedFNLxxx we implement the frozen models and load  the BoundedFNLEMDBDP_model_types_weights

#transfer  BoundedFNLPWG_model and train it on the BoundedFNLEMDBDP_model_types_weights
BoundedFNLPWG_model_new.load_weights('BoundedFNLEMDBDP_model_types_weights')



###NOTE THAT THE ABOVE PROCEDURES CAN BE USED TO TRANSFER ANY MODEL TO LOAD DIFFERENT WEIGHTS








