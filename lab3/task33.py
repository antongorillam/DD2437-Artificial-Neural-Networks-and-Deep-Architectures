from re import X

import matplotlib.pyplot as plt
import numpy as np

import algorithms
import dataGenerator
import task22

# np.random.seed(50)
#neurons = 8
neurons = 1024
#x1d= [1, -1, 1, -1, 1, -1, -1, 1]
#x2d= [1, 1, -1, -1, -1, 1, -1, -1]
#x3d= [1, 1, 1, -1, 1, 1, -1, 1]
#W = task22.getW()

patterns = dataGenerator.getPatterns(directory="data/pict.dat", neurons=1024)
print(patterns.shape)
first3_patterns = patterns[:3, :]
print(first3_patterns.shape)
W = dataGenerator.getW(first3_patterns,scaling=True)
print(W.shape)

p1 = patterns[0, :]
p2 = patterns[1, :]
p3 = patterns[2, :]
p4 = patterns[3, :]
p5 = patterns[4, :]
p6 = patterns[5, :]
p7 = patterns[6, :]
p8 = patterns[7, :]
p9 = patterns[8, :]
p10 = patterns[9, :]
p11 = patterns[10, :]


#Point 1 
#noAttractors = algorithms.findAllAttractors(W=W, len=neurons)
#print(f'There are in total {noAttractors} attractors')

energyPati = algorithms.getEnergy(p1,W)
energyPat2 = algorithms.getEnergy(p2,W)
energyPat3 = algorithms.getEnergy(p3,W)
for i in range(3): 
    energyPati = algorithms.getEnergy(patterns[i, :],W)
    print(f'The energy in attractor {i+1} (p{i+1}) is: {energyPati}')

#Point2 + 3, now l
print('---------- p1 -----------------')
_,_,_ = algorithms.asynchronousUpdate(
    x=p1, W=W, verbose=True,fileName='p1'
)
print('---------- p2 -----------------')
_,_,_ = algorithms.asynchronousUpdate(
    x=p2, W=W, verbose=True,fileName='p2'
)
print('---------- p3 -----------------')
_,_,_ = algorithms.asynchronousUpdate(
    x=p3, W=W, verbose=True,fileName='p3'
)
print('---------- p4 -----------------')
_,_,_ = algorithms.asynchronousUpdate(
    x=p4, W=W, verbose=True,fileName='p4'
)
print('---------- p5 -----------------')
_,_,_ = algorithms.asynchronousUpdate(
    x=p5, W=W, verbose=True,fileName='p5'
)
print('---------- p6 -----------------')
_,_,_ = algorithms.asynchronousUpdate(
    x=p6, W=W, verbose=True,fileName='p6'
)
print('---------- p7 -----------------')
_,_,_ = algorithms.asynchronousUpdate(
    x=p7, W=W, verbose=True,fileName='p7'
)
print('---------- p8 -----------------')
_,_,_ = algorithms.asynchronousUpdate(
    x=p8, W=W, verbose=True,fileName='p8'
)
print('---------- p9 -----------------')
_,_,_ = algorithms.asynchronousUpdate(
    x=p9, W=W, verbose=True,fileName='p9'
)
print('---------- p10 -----------------')
_,_,_ = algorithms.asynchronousUpdate(
    x=p10, W=W, verbose=True,fileName='p10'
)
print('---------- p11 -----------------')
_,_,_ = algorithms.asynchronousUpdate(
    x=p11, W=W, verbose=True,fileName='p11'
)

# point 4
W = np.random.uniform(size=(1024,1024))
#print(W.shape)
pRandom = np.random.randint(0, 2, (1, 1024))
for i, x in enumerate(pRandom[0]):
    # print(x)
    if x < 0.5:
        pRandom[0][i] = -1

# algorithms.display(pRandom)
pRandom = pRandom[0]
pRandom = [float(i) for i in pRandom]

print('---------- point 4 -----------------')
_,_,_ = algorithms.asynchronousUpdate(
    x=pRandom, W=W, verbose=True,fileName='point4'
)

# point 5
w2 = 0.5*(np.copy(W)+np.copy(W).T)
print('---------- point 5 -----------------')
_,_,_ = algorithms.asynchronousUpdate(
    x=pRandom, W=w2, verbose=True,fileName='point5'
)
