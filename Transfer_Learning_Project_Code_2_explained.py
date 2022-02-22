# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 00:49:48 2021

@author: sojore
"""

#2rd approach we can implement TL




#we gonna implement transfer learning on each model,where we gonna be comparing their perfomances based on the getValues function
#note that these models are not sequential models but functional models and as a result we have the following commands for which
#we will use to test the perfomance of every model when loaded with new weights
#VERY IMPORTANT note that the network module contains the functions which does save the weights in different types of formats
#this implies that we load models on different types of network_weights_archtectures and that each network stores the different types of weights
#so it means that we can run the same model on different weights then we test their perfomance(in other words we are
#transfering a model to load different types of weights which are basically stored in the network module ) then we compare 
#its performance
#to campare how these models perfoms we gonna use the below functions
#1.model.getValues() --this gives an array of values associated with the loaded weights -similar to get_weights in sequential models
#2.model.Sol() --this gives an approximation solution of the given model



# so we gonna implement TL on 3 stages
#1. we load the model (feed the model with one type of weights archtecture)
#then we test its perfoamance
#2. we load the same model but with different types of weights archtecture
#then we test its perfomance
#3 we load another model in the same class but load different types of weights archtecture
#then we test its perfomance and compare the results


#Note also that,these models dont have the same number of dense layers 
#so with TL we apply this to models with more than 2 dense layers,where we can get the inner layer and feed it with diffrent
#types of weights as explained above
#the outer layer will act as our custom layer (as in Sequential models) which has an output of 2
#models with only one dense layer wont be tranfered since they are already modified to perfom that specific task with 
#no inner layers in them---this means we dont have any layers we can freeze before we transfer the model


#these models will have their weights before we transfer them ,buh we apply different type of weight archteture on them to see if we can improve their perfomance
#IMPLEMENTATION OF TL ON 1.Mertonxxx

#we are first going to test the perfomance of this model

#First stage

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


#this model has been fed the weights in FeedForwardUZ approach
#so we can test its perfomance by 
model.getValues()--similar to get weights
#and
model.Sol()

 
#next we implement TL by loading the same model buh with diffrent types of weights archtecture
#so i have done some minor modifications on the code by loading the weights archtecture to the inner layers of the same model

#Second stage

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


#so we alter the inner layer so as to load weights in a FeedForwardU approach
theNetwork = net.FeedForwardU(d,layerSize,tf.nn.tanh)
#we retain the outer layer since it acts as a custom layer (similar to Sequential) models
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


#lets test its peromance as well
model.getValues()--similar to get weights
#and
model.Sol()

#compare its perfomance with the model in first stage

#Third stage

#we get a different model and load different types of weights

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

#note that we are loading different types of weights to the inner layers of this model 
theNetwork = net.FeedForwardUZDZ(d,layerSize,tf.nn.tanh)
#we also retain the outer layer 
theNetworkGam = net.FeedForwardGam(d,layerSize,tf.nn.tanh)

ndt = [ (120,30) ] 

print("PDE Merton MDBDP  Dim ", d, " layerSize " , layerSize,  " rescal " ,rescal, "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast, "VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.PDEFNLSolve2OptZGPU(model, T, indt[0], indt[1], theNetwork , theNetworkGam, initialLearningRateLast=initialLearningRateLast, initialLearningRateNoLast = initialLearningRateNoLast)
        
    baseFile = "MertonMDBDPd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt[0])+str(indt[1])+"eta"+str(int(eta*100))
    plotFile = "pictures/"+baseFile
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

#lets test its peromance as well
model.getValues()--similar to get weights
#and
model.Sol()

#compare its perfomance with the model in first stage




# 2. NoLeveragexxx

#we repeat the same precedure as described above (the 3 stages)
#where we first train a default model,get its weights,train the same model on different weights archtecture,
#get its weights,n lastly train a different model in the same class with different types of weights archtecture

# 1st stage
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

#lets test its peromance as well
model.getValues()--similar to get weights
#and
model.Sol()


#2rd stage
#use the same model but loading different types of weights archtecture

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

theNetwork = net.FeedForwardUZDZ(d,layerSize,tf.nn.tanh)
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

#lets test its peromance as well
model.getValues()--similar to get weights
#and
model.Sol()

#3rd stage
#train a different model with different types of weights

import numpy as np
import tensorflow as tf
import os
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
batchSizeVal= 10000
num_epoch=400 
num_epochExtNoLast =10
num_epochExtLast= 200
initialLearningRateLast = 1e-2
initialLearningRateNoLast = 1e-3
nbOuterLearning =10
nTest = 10

lamb = np.array([1.5, 1.1, 2., 0.8, 0.5, 1.7, 0.9, 1., 0.9,1.5], dtype=np.float32)[0:9]
eta = 0.5
theta = np.array([0.1, 0.2, 0.3, 0.4, 0.25, 0.15, 0.18, 0.08, 0.91,0.4], dtype=np.float32)[0:9]
gamma = np.array([0.2, 0.15, 0.25, 0.31, 0.4, 0.35, 0.22, 0.4, 0.15,0.2], dtype=np.float32)[0:9]
kappa = np.array([1., 0.8, 1.1, 1.3, 0.95, 0.99, 1.02, 1.06, 1.6,1.], dtype=np.float32)[0:9]
sigma = np.ones(9, dtype=np.float32)[0:9]
d = lamb.shape[0] + 1
nbNeuron = d + 10
sigScal =  np.concatenate([np.array([1.], dtype=np.float32),gamma]).reshape((d))

muScal = np.concatenate([np.array([np.sum(theta*lamb)]),np.zeros((d-1), dtype=np.float32)]).reshape((d))
layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32)

xyInit= np.concatenate([np.array([1.]),theta])   

# create the model
model = mod.ModelNoLeverage(xyInit, muScal, sigScal, T, theta, sigma, lamb, eta, gamma, kappa)

print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")
print("REAL IS ", model.Sol(0.,xyInit.reshape(1,d)), " DERIV", model.derSol(0.,xyInit.reshape(1,d)), " GAMMA", model.der2Sol(0.,xyInit.reshape(1,d)))

theNetwork = net.FeedForwardUDU(d,layerSize,tf.nn.tanh)
theNetworkGam = net.FeedForwardGam(d,layerSize,tf.nn.tanh)

ndt = [ (120,120) ] 


print("PDE No Leverage MDBDP  Dim ", d, " layerSize " , layerSize,  " rescal " ,rescal, "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast, "VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.PDEFNLSolve2OptZGPU(model, T, indt[0], indt[1], theNetwork , theNetworkGam, initialLearningRateLast=initialLearningRateLast, initialLearningRateNoLast = initialLearningRateNoLast)
 
    baseFile = "NoLeverageMDBDPd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt[0])+str(indt[1])+"eta"+str(int(eta*100))
    plotFile = "pictures/"+baseFile
    saveFolder = "save/"

    Y0List = []
    for i in range(nTest):
        # train
        t0 = time.time()
        Y0, Z0, Gamma0    = resol.BuildAndtrainML( batchSize, batchSizeVal, num_epochExtNoLast=num_epochExtNoLast, num_epochExtLast= num_epochExtLast ,num_epoch=num_epoch,  nbOuterLearning=nbOuterLearning, thePlot= plotFile , baseFile = baseFile, saveDir= saveFolder)
        t1 = time.time()
        print(" NBSTEP", indt, " EstimMC Val is " , Y0,  " REAL IS ", model.Sol(0.,xyInit.reshape(1,d)),    " Z0 ", Z0," DERREAL IS  ",model.derSol(0.,xyInit.reshape(1,d)), "Gamma0 " , Gamma0, " GAMMA", model.der2Sol(0.,xyInit.reshape(1,d)), t1-t0)

        Y0List.append(Y0)
        print(Y0List)

    print("Y0", Y0List)
    yList = np.array(Y0List)
    yMean = np.mean(yList)
    print(" DNT ", indt , "MeanVal ", yMean, " Etyp ", np.sqrt(np.mean(np.power(yList-yMean,2.))))


#lets test its peromance as well and compare with 1st stage 
model.getValues()--similar to get weights
#and
model.Sol()



# 3. OneAssetxxx

#we repeat the same precedure as described above (the 3 stages)
#where we first train a default model,get its weights,train the same model on different weights archtecture,
#get its weights,n lastly train a different model in the same class with different types of weights archtecture

# 1st stage
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


model.getValues()--similar to get weights
#and
model.Sol()



#2rd stage
#use the same model but loading different types of weights archtecture

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

theNetwork = net.FeedForwardU(d,layerSize,tf.nn.tanh)
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


model.getValues()--similar to get weights
#and
model.Sol()


#3rd stage
#train a different model with different types of weights
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

theNetwork = net.FeedForwardUDU(d,layerSize,tf.nn.tanh)
theNetworkGam = net.FeedForwardGam(d,layerSize,tf.nn.tanh)

ndt = [ (120,30) ] 


print("PDE OneAsset MDBDP  Dim ", d, " layerSize " , layerSize,  " rescal " ,rescal, "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast, "VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.PDEFNLSolve2OptZGPU(model, T, indt[0], indt[1], theNetwork , theNetworkGam, initialLearningRateLast=initialLearningRateLast, initialLearningRateNoLast = initialLearningRateNoLast)
        
    baseFile = "OneAssetMDBDPd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt[0])+str(indt[1])+"eta"+str(int(eta*100))
    plotFile = "pictures/"+baseFile
    saveFolder = "save/"

    Y0List = []
    for i in range(nTest):
        # train
        t0 = time.time()
        Y0, Z0, Gamma0    = resol.BuildAndtrainML( batchSize, batchSizeVal, num_epochExtNoLast=num_epochExtNoLast, num_epochExtLast= num_epochExtLast ,num_epoch=num_epoch,  nbOuterLearning=nbOuterLearning, thePlot= plotFile , baseFile = baseFile, saveDir= saveFolder)
        t1 = time.time()
        print(" NBSTEP", indt, " EstimMC Val is " , Y0,  " REAL IS ", model.Sol(0.,xyInit.reshape(1,d)),    " Z0 ", Z0," DERREAL IS  ",model.derSol(0.,xyInit.reshape(1,d)), "Gamma0 " , Gamma0, " GAMMA", model.der2Sol(0.,xyInit.reshape(1,d)), t1-t0)

        Y0List.append(Y0)
        print(Y0List)

    print("Y0", Y0List)
    yList = np.array(Y0List)
    yMean = np.mean(yList)
    print(" DNT ", indt , "MeanVal ", yMean, " Etyp ", np.sqrt(np.mean(np.power(yList-yMean,2.))))

model.getValues()--similar to get weights
#and
model.Sol()
#compare results with the 1st stage



# 4. OneAssetxxx

#we repeat the same precedure as described above (the 3 stages)
#where we first train a default model,get its weights,train the same model on different weights archtecture,
#get its weights,n lastly train a different model in the same class with different types of weights archtecture


# 1st stage
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

nbLayer= 3 
print("nbLayer " ,nbLayer)
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

d = 8
nbNeuron = d + 10
sigScal =  0.5*np.ones(d)
muScal = np.array([0])
lamb = 0.5

layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32)

xyInit = np.ones(d) 

# create the model
model = mod.ModelMongeAmpere(xyInit, muScal, sigScal, T, lamb, d)

print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")
print("REAL IS ", model.Sol(0.,xyInit.reshape(1,d)), " DERIV", model.derSol(0.,xyInit.reshape(1,d)), " GAMMA", model.der2Sol(0.,xyInit.reshape(1,d)))

theNetwork = net.FeedForwardUZ(d,layerSize,tf.nn.tanh)
theNetworkGam = net.FeedForwardGam(d,layerSize,tf.nn.tanh)

ndt = [(120,60)  ] 

print("PDE Monge Ampere M2DBDP Dim ", d, " layerSize " , layerSize,   "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast,  "VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.PDEFNLSolve2OptGPU(model, T, indt[0], indt[1], theNetwork , theNetworkGam, initialLearningRateLast=initialLearningRateLast, initialLearningRateNoLast = initialLearningRateNoLast)
        
    baseFile = "MongeAmpereM2DBDPd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt[0])+str(indt[1])
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

model.getValues()--similar to get weights
#and
model.Sol()



#2rd stage
#use the same model but loading different types of weights archtecture

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

nbLayer= 3 
print("nbLayer " ,nbLayer)
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

d = 8
nbNeuron = d + 10
sigScal =  0.5*np.ones(d)
muScal = np.array([0])
lamb = 0.5

layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32)

xyInit = np.ones(d) 

# create the model
model = mod.ModelMongeAmpere(xyInit, muScal, sigScal, T, lamb, d)

print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")
print("REAL IS ", model.Sol(0.,xyInit.reshape(1,d)), " DERIV", model.derSol(0.,xyInit.reshape(1,d)), " GAMMA", model.der2Sol(0.,xyInit.reshape(1,d)))

theNetwork = net.FeedForwardUZDZ(d,layerSize,tf.nn.tanh)
theNetworkGam = net.FeedForwardGam(d,layerSize,tf.nn.tanh)

ndt = [(120,60)  ] 

print("PDE Monge Ampere M2DBDP Dim ", d, " layerSize " , layerSize,   "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast,  "VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.PDEFNLSolve2OptGPU(model, T, indt[0], indt[1], theNetwork , theNetworkGam, initialLearningRateLast=initialLearningRateLast, initialLearningRateNoLast = initialLearningRateNoLast)
        
    baseFile = "MongeAmpereM2DBDPd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt[0])+str(indt[1])
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


model.getValues()--similar to get weights
#and
model.Sol()



#3rd stage
#train a different model with different types of weights

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

# 2 optimization version
# Non linearity in   "uD2u"
# suppress verbose
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  

_MULTIPROCESSING_CORE_COUNT = multiprocessing.cpu_count()
print("args", sys.argv)

nbLayer= 3 
print("nbLayer " ,nbLayer)
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

d = 15
nbNeuron = d + 10
sigScal =  0.5*np.ones(d)
muScal = np.zeros(d)
lamb = 0.5

layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32)

xyInit = np.ones(d) 

# create the model
model = mod.ModelMongeAmpere(xyInit, muScal, sigScal, T, lamb, d)

print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")
print("REAL IS ", model.Sol(0.,xyInit.reshape(1,d)), " DERIV", model.derSol(0.,xyInit.reshape(1,d)), " GAMMA", model.der2Sol(0.,xyInit.reshape(1,d)))

theNetwork = net.FeedForwardUDU(d,layerSize,tf.nn.tanh)
theNetworkGam = net.FeedForwardGam(d,layerSize,tf.nn.tanh)

ndt = [(120,30)  ] 

print("PDE Monge Ampere MDBDP  Dim ", d, " layerSize " , layerSize,   "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast,  "VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.PDEFNLSolve2OptZGPU(model, T, indt[0], indt[1], theNetwork , theNetworkGam, initialLearningRateLast=initialLearningRateLast, initialLearningRateNoLast = initialLearningRateNoLast)
        
    baseFile = "MongeAmpereMDBDPd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt[0])+str(indt[1])
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


model.getValues()--similar to get weights
#and
model.Sol()
#compare results with stage 1 and 2



####THIS PROCEDURE CAN BE USED TO TTANSFER ANY OTHER MODELS TO LOAD DIFFERENT TYPES OF WEIGHTS















