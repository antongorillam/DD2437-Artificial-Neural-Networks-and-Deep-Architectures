import numpy as np

import algorithms
import dataGenerator
import task22

x1d= [1, -1, 1, -1, 1, -1, -1, 1]
x2d= [1, 1, -1, -1, -1, 1, -1, -1]
x3d= [1, 1, 1, -1, 1, 1, -1, 1]

x1=[-1, -1, 1, -1, 1, -1, -1, 1]
x2=[-1, -1, -1, -1, -1, 1, -1, -1]
x3=[-1, 1, 1, -1, -1, 1, -1, 1]

W = task22.getW()
XOutSync1,iSync1,eSync1, convergenceSync1 = algorithms.synchronousUpdate(x=x1d,W=W) # only one that converges
XOutSync2,iSync2,eSync2, convergenceSync2 = algorithms.synchronousUpdate(x=x2d,W=W)
XOutSync3,iSync3,eSync3, convergenceSync3 = algorithms.synchronousUpdate(x=x3d,W=W)

syncPrintStr = 'Sync status: ', str(iSync1) + ' ' + str(eSync1) + ' ' + str(convergenceSync1) + ' ' + str(iSync2) + ' ' + str(eSync2) + ' ' + str(convergenceSync2) + ' ' + str(iSync3) + ' ' + str(eSync3) + ' ' + str(convergenceSync3)

#print(syncPrintStr)
#print(np.array_equal(XOutSync1,x1),np.array_equal(XOutSync1,x2),np.array_equal(XOutSync1,x3))
#print(np.array_equal(XOutSync2,x2),np.array_equal(XOutSync2,x1),np.array_equal(XOutSync2,x3))
#print(np.array_equal(XOutSync3,x3),np.array_equal(XOutSync3,x1),np.array_equal(XOutSync3,x2))


XOutAsync1,iAsync1,eAsync1 = algorithms.asynchronousUpdate(x=x1d,W=W)
XOutAsync2,iAsync2,eAsync2 = algorithms.asynchronousUpdate(x=x2d,W=W)
XOutAsync3,iAsync3,eAsync3 = algorithms.asynchronousUpdate(x=x3d,W=W)

#print(np.array_equal(XOutAsync1,x1),np.array_equal(XOutAsync1,x2),np.array_equal(XOutAsync1,x3))
#print(np.array_equal(XOutAsync2,x1),np.array_equal(XOutAsync2,x2),np.array_equal(XOutAsync2,x3))
#print(np.array_equal(XOutAsync3,x1),np.array_equal(XOutAsync3,x2),np.array_equal(XOutAsync3,x3))


asyncPrinStr = 'Async status', str(iAsync1) + ' ' + str(eAsync1) + ' ' + str(iAsync2) + ' ' + str(eAsync2) + ' ' + str(iAsync3) + ' ' + str(eAsync3)
#print(asyncPrinStr)

noAttractors = algorithms.findAllAttractors(W=W, len=8)
print(noAttractors)

# wrong patterns
x1MegaDistorted = [-1,-1,1,1,-1,1,1,-1] # last 5 wrong
x2MegaDistorted = [-1,-1,1,1,1,-1,1,1] # last 6 wrong
x3MegaDistorted = [1,-1,-1,1,1,1,-1,1] # first 5 wrong

XOutSync1,iSync1,eSync1, convergenceSync1 = algorithms.synchronousUpdate(x=x1MegaDistorted,W=W) # only one that converges
XOutSync2,iSync2,eSync2, convergenceSync2 = algorithms.synchronousUpdate(x=x2MegaDistorted,W=W)
XOutSync3,iSync3,eSync3, convergenceSync3 = algorithms.synchronousUpdate(x=x3MegaDistorted,W=W)

print(np.array_equal(XOutSync1,x1),np.array_equal(XOutSync1,x2),np.array_equal(XOutSync1,x3))
print(np.array_equal(XOutSync2,x2),np.array_equal(XOutSync2,x1),np.array_equal(XOutSync2,x3))
print(np.array_equal(XOutSync3,x3),np.array_equal(XOutSync3,x1),np.array_equal(XOutSync3,x2))

#syncPrintStr = 'Sync status highly distorted patterns: ', str(iSync1) + ' ' + str(eSync1) + ' ' + str(convergenceSync1) + ' ' + str(iSync2) + ' ' + str(eSync2) + ' ' + str(convergenceSync2) + ' ' + str(iSync3) + ' ' + str(eSync3) + ' ' + str(convergenceSync3)

#print(syncPrintStr)

# wrong patterns for async update just to test
XOutAsync1,iAsync1,eAsync1 = algorithms.asynchronousUpdate(x=x1MegaDistorted,W=W)
XOutAsync2,iAsync2,eAsync2 = algorithms.asynchronousUpdate(x=x2MegaDistorted,W=W)
XOutAsync3,iAsync3,eAsync3 = algorithms.asynchronousUpdate(x=x3MegaDistorted,W=W)

print(np.array_equal(XOutSync1,x1),np.array_equal(XOutSync1,x2),np.array_equal(XOutSync1,x3))
print(np.array_equal(XOutSync2,x2),np.array_equal(XOutSync2,x1),np.array_equal(XOutSync2,x3))
print(np.array_equal(XOutSync3,x3),np.array_equal(XOutSync3,x1),np.array_equal(XOutSync3,x2))


asyncPrinStr = 'Async status wrong patterns', str(iAsync1) + ' ' + str(eAsync1) + ' ' + str(iAsync2) + ' ' + str(eAsync2) + ' ' + str(iAsync3) + ' ' + str(eAsync3)
print(asyncPrinStr)
print('p1: ', XOutAsync1, 'p2: ',XOutAsync2, 'p3: ', XOutAsync3)
