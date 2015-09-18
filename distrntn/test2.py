import rntn as nnet
import tree as tr
import sgd as optimizer

def gradient(model, tree_data):  # executes on workers
    """
    Each datum in the minibatch is a tree.
    Forward prop each tree.
    Backprop each tree.
    Returns
       cost
       Gradient w.r.t. W, Ws, b, bs
       Gradient w.r.t. L in sparse form.
    """
    tree_data=list(tree_data)
    datasize=len(tree_data)
    print "-----Running gradient. data size is %s of type %s-------"%(datasize,type(tree_data))
    if datasize == 0:
        return []
    cost = 0.0
    correct = 0.0
    total = 0.0



    model.model.L,model.model.V,model.model.W,model.model.b,model.model.Ws,model.model.bs = model.model.stack
    # Zero gradients
    model.model.dV[:] = 0
    model.model.dW[:] = 0
    model.model.db[:] = 0
    model.model.dWs[:] = 0
    model.model.dbs[:] = 0
    model.model.dL = collections.defaultdict(lambda : np.zeros((model.model.wvecDim,)))


    # Forward prop each tree in minibatch
    for tree in tree_data:
        c,corr,tot =  model.model.forwardProp(tree.root)
        cost += c
        correct += corr
        total += tot

    # Back prop each tree in minibatch
    for tree in tree_data:
        model.model.backProp(tree.root)

    # scale cost and grad by mb size ************************88
    #scale = (1./model.model.mbSize)
    scale = (1./datasize)
    for v in model.model.dL.itervalues():
        v *=scale

    # Add L2 Regularization
    cost += (model.model.rho/2)*np.sum(model.model.V**2)
    cost += (model.model.rho/2)*np.sum(model.model.W**2)
    cost += (model.model.rho/2)*np.sum(model.model.Ws**2)

    return [scale*cost,[model.model.dL,scale*(model.model.dV+model.model.rho*model.model.V),
                       scale*(model.model.dW + model.model.rho*model.model.W),scale*model.model.db,
                       scale*(model.model.dWs+model.model.rho*model.model.Ws),scale*model.model.dbs]]


rnn = nnet.RNN(1,1,1,1,1)
rnn.initParams()
sgd = optimizer.SGD(rnn)
data = []
gradient(sgd,data)

import cPickle as pickle
b = pickle.dumps(gradient, -1)
print(b)
print("done")
