from re import X

import matplotlib.pyplot as plt
import numpy as np

import algorithms
import dataGenerator

# np.random.seed(50)

patterns = dataGenerator.getPatterns(directory="data/pict.dat", neurons=1024)
# print(patterns.shape)
first3_patterns = patterns[:3, :]
# print(first3_patterns.shape)
W = dataGenerator.getW(first3_patterns)
print(W)

p1 = patterns[0, :]
p2 = patterns[1, :]
p3 = patterns[2, :]
p1Deg = patterns[9, :]
p2_3 = patterns[10, :]

# print("p1deg", p1Deg)
# algorithms.display(p1Deg, title="p10, p1 degraded", save=True, filename="images/3_2/p10.png")


# Point 1
# print(np.array_equal(p1, np.sign(W @ p1)))  # True
# print(np.array_equal(p2, np.sign(W @ p2)))  # True
# print(np.array_equal(p3, np.sign(W @ p3)))  # True

# Point 2
# XOutP1Deg,iAsyncP1Deg,eAsyncP1Deg = algorithms.asynchronousUpdate(x=p1Deg,W=W)
# XOutP1Deg, iAsyncP1Deg, eAsyncP1Deg = algorithms.asynchronousUpdate(
#     x=p1Deg, W=W
# )
# print("p1 degradation: ", np.array_equal(p1, XOutP1Deg))
# algorithms.display(XOutP1Deg, title="p10 recalled", save=True, filename="images/3_2/p10recalled.png")
#print("convergence: ", converge)

# algorithms.display(p2_3, title="p11, p2/p3 degraded", save=True, filename="images/3_2/p11.png")
# algorithms.display(p2, title="p2", save=True, filename="images/3_2/p2.png")
# algorithms.display(p2, title="p3", save=True, filename="images/3_2/p3.png")

#XOutP2_3, iAsyncP2_3, eAsyncP2_3, conv2 = algorithms.synchronousUpdate(x=p2_3, W=W)
#print('p2 degradation: ',np.array_equal(p2, XOutP2_3))
#print('p3 degradation: ',np.array_equal(p3, XOutP2_3))
# print('conv 2',conv2)
# for i in range(10): 
#     XOutP2_3, iAsyncP2_3, eAsyncP2_3 = algorithms.asynchronousUpdate(x=p2_3, W=W)
#     algorithms.display(XOutP2_3, title='p11 recalled', save=True, filename=f"images/3_2/p11recalled_it{i}.png")

#Point 3
# pRandom = np.random.randint(0, 2, (1, 1024))
# for i, x in enumerate(pRandom[0]):
#     # print(x)
#     if x < 0.5:
#         pRandom[0][i] = -1

# algorithms.display(pRandom)
# pRandom = pRandom[0]
# pRandom = [float(i) for i in pRandom]
XOutP2_3, iAsyncP2_3, eAsyncP2_3 = algorithms.asynchronousUpdate(x=p2_3, W=W,totallyRandomUpdate=True, plot=False)
algorithms.display(XOutP2_3, title='p11 recalled', save=True, filename=f"images/3_2/p11_finalrecall_it{iAsyncP2_3}.png")
print("number of its for random: ", iAsyncP2_3)
