import numpy as np
import collections
np.seterr(over='raise',under='raise')

class RNN:

    def __init__(self,wvecDim,outputDim,numWords,mbSize=30,rho=1e-6):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = np.zeros((wvecDim,))
        self.rho = rho



import cPickle as pickle
print "pickling"
a = RNN(1,1,1,1,1)
pickle.dumps(a)
print(a)
print "pickling finished"
