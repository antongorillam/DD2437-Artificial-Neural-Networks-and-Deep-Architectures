import matplotlib.pyplot as plt
import numpy as np

import dataGenerator
from som import SOM



if __name__=='__main__':
    
    somTSP = SOM(eta=0.2,nOutput=10)
    inputData = dataGenerator.getCitiesData()
    
