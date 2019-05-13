import numpy as np
import matplotlib.pyplot as plt
import sys
import dataGetter


__name__ = "NNarch"

plt.show()
#call for Network of shape 3 conv layers with 50 filters starting at 7 stride 3 and MLP arch of 5*5
#w, y = Network([[50,7],[35,5],[20,3]],[5,5,5,5,5],5000,.0001,1)
def Network(convShape,shape,e,r,m):#convShape = array, (convLayers X 2), [numFilters,filterSize] // shape = array, (Layers)
    convShape=np.array(convShape)
    shape=np.array(shape)
    L = 1 + shape.shape[0] + 1  #Layers = conv output + hidden layers + output
    E = np.intc(e)  #epochs
    R = np.float64(r)   #Learning Rate
    M = m
    W = []
    B = []
    filters = []
    FMPools = []
    X, T = getData()
    # X -= int(np.mean(X))
    # X /= np.std(X)

    loss = crossEntropy
    hfinal = softMax
    h = tanh
    hPrime = tanhDx
    I = [X.shape[0]]
    O = [10]
    shape = np.concatenate((I,shape,O))
    
    stddev = 1/np.sqrt(np.prod(convShape[0][1]))
    LFilter = np.random.normal(loc = 0, scale = stddev, size = (convShape[0][0],convShape[0][1],convShape[0][1]))
    filters.append(LFilter)
    itt=np.intc(1)
    FMPools.append(pool(relu(conv(X[0],filters[-1]))))
    while(itt < len(convShape)):#for each layer after the first
            stddev = 1/np.sqrt(np.prod(convShape[itt][1]))
            LFilter = np.random.normal(loc = 0, scale = stddev, size = (convShape[itt][0],convShape[itt][1],convShape[itt][1]))
            filters.append(LFilter)
            FMPools.append(pool(relu(conv(X[0],filters[-1]))))
            itt += np.intc(1)

    print('Xs shape: ',X.shape)
    w0 = np.random.standard_normal(size=(shape[1],X.shape[1])).astype(np.float64) * np.float64(0.01)
    #w0 = (np.float64(2)*np.random.rand(shape[1],X.shape[1]).astype(np.float64) - np.float64(1))
    W.append(w0)
    b0 = np.full((shape[1],shape[0]),0.1,dtype=np.float64)
    #b0 = (np.float64(2.0)*np.random.rand(shape[1],shape[0]).astype(np.float64) - np.float64(1.0))#
    B.append(b0)
    for itt in range(1,L-1):
        w = np.random.standard_normal(size=(shape[itt+1],shape[itt])).astype(np.float64) * np.float64(0.01)
        #w = (np.float64(2)*np.random.rand(shape[itt+1],shape[itt]).astype(np.float64) - np.float64(1))
        W.append(w)
        b = np.full((shape[itt+1],shape[0]),0.1,dtype=np.float64)
        #b = (np.float64(2.0)*np.random.rand(shape[itt+1],shape[0]).astype(np.float64) - np.float64(1.0))#
        B.append(b)
        
    W, y = fit(X,T,W,B,E,R,L, convShape, shape, hfinal, h, hPrime, loss, M, filters)   
    print(y[-1])
    return W, y


def getData():#TODO
    filenames = dataGetter.getFiles()
    print("Getting training images...")
    trainImg = dataGetter.getData(filenames[0,0])
    print("Getting training targets...")
    trainTarg = dataGetter.getData(filenames[0,1])
    print("Data retrieved")
    return trainImg, trainTarg

def fit(X,T,W,B,E,R,L,convShape,shape,hfinal,h,hP,loss,M,filters):
    xplt = np.arange(E)
    yplt = np.zeros((E),dtype=np.float64)
    #plot stuff
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(xplt,yplt, 'r-')
    plt.ylabel('ERR')
    plt.xlabel('epoch')
    ax.set_autoscaley_on(True)
    ax.set_xlim(0, E)
    fig.canvas.draw()
    plt.show(block=False)
    itt = np.intc(0)
    
    while(itt<E):#training loop
        shuffled_index = np.random.permutation(X.shape[0])
        batch_train_X = X[shuffled_index[:20]]
        batch_train_Y = T[shuffled_index[:20]]
        flatFMpools, FMPools = ArbConvolve(batch_train_X,convShape,filters)
        Z = forwardProp(flatFMpools,W,B,hfinal,h,L,shape)
        W, B, yplt, err = backProp(Z,batch_train_Y,R,hP,W,B,L,shape,itt,yplt,loss,M)
        convBackprop(err, {X, FMPools})#TODO
        line1.set_ydata(yplt)
        if itt%20==0:#line plotting 
            print(itt)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
        itt+=np.intc(1)
    print(Z[-1])
    

    
    return W, Z
    
def ArbConvolve(imgs, convShape, filters):#convShape = ndarray, (convLayers X 2), [numFilters,filterSize]
    filters = filters
    FMPools = []
    flatFMPools = []
    #this while loop uses the filters created and then goes through the rest of the images
    it=np.intc(0)
    while(it < imgs.shape[0]):#for num of imgs
        print("img:", it)
        itt=np.intc(0)
        while(itt < len(filters)):#for each layer
            FMPools.append(pool(relu(conv(imgs[it],filters[itt]))))
            
            itt += np.intc(1)
        flatFMPools.append(FMPools[-1].reshape((FMPools[-1].size,1)))
        it += np.intc(1)
    return flatFMPools, FMPools

def conv(img, conv_filter):
    if len(img.shape) > 2 or len(conv_filter.shape) > 3:    # Check if number of image channels matches the filter depth.
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()
    if conv_filter.shape[1] != conv_filter.shape[2]:    # Check if filter dimensions are equal.
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
        sys.exit()
    if conv_filter.shape[1] % 2 == 0:   # Check if filter diemnsions are odd.
        print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
        sys.exit()

    # An empty feature map to hold the output of convolving the filter(s) with the image.
    feature_maps = np.zeros((img.shape[0]-conv_filter.shape[1]+1, 
                                img.shape[1]-conv_filter.shape[1]+1, 
                                conv_filter.shape[0]))
    
    # Convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):
        # print("Filter ", filter_num + 1)
        
        curr_filter = conv_filter[filter_num, :] # getting a filter from the bank.
        """ 
        Checking if there are mutliple channels for the single filter.
        If so, then each channel will convolve the image.
        The result of all convolutions are summed to return a single feature map.
        """
        if len(curr_filter.shape) > 2:
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0]) # Array holding the sum of all feature maps.
            for ch_num in range(1, curr_filter.shape[-1]): # Convolving each channel with the image and summing the results.
                conv_map = conv_map + conv_(img[:, :, ch_num], 
                                  curr_filter[:, :, ch_num])
        else: # There is just a single channel in the filter.
            conv_map = conv_(img, curr_filter)
        feature_maps[:, :, filter_num] = conv_map # Holding feature map with the current filter.
    return feature_maps # Returning all feature maps.

def conv_(img, conv_filter):
    filter_size = conv_filter.shape[1]
    result = np.zeros((img.shape))
    #Looping through the image to apply the convolution operation.
    for r in np.arange(filter_size/2.0, img.shape[0]-filter_size/2.0+1, dtype = np.intc):
        for c in np.arange(filter_size/2.0, img.shape[1]-filter_size/2.0+1, dtype = np.intc):
            """
            Getting the current region to get multiplied with the filter.
            How to loop through the image and get the region based on 
            the image and filer sizes is the most tricky part of convolution.
            """
            curr_region = img[r-np.intc(np.floor(filter_size/2.0)):r+np.intc(np.ceil(filter_size/2.0)), \
                              c-np.intc(np.floor(filter_size/2.0)):c+np.intc(np.ceil(filter_size/2.0))]
            #Element-wise multipliplication between the current region and the filter.
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result) #Summing the result of multiplication.
            result[r, c] = conv_sum #Saving the summation in the convolution layer feature map.
            
    #Clipping the outliers of the result matrix.
    final_result = result[np.intc(filter_size/2.0):result.shape[0]-np.intc(filter_size/2.0), \
                          np.intc(filter_size/2.0):result.shape[1]-np.intc(filter_size/2.0)]
    return final_result

def pool(feature_map, size=2, stride=2):
    #Preparing the output of the pooling operation.
    pool_out = np.zeros((np.intc((feature_map.shape[0]-size+1)/stride+1), \
                            np.intc((feature_map.shape[1]-size+1)/stride+1),  \
                            feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in np.arange(0,feature_map.shape[0]-size+1, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[1]-size+1, stride):
                pool_out[r2, c2, map_num] = np.max([feature_map[r:r+size,  c:c+size, map_num]])
                c2 = c2 + 1
            r2 = r2 +1
    return pool_out

def relu(feature_map):
    #Preparing the output of the ReLU activation function.
    relu_out = np.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in np.arange(0,feature_map.shape[0]):
            for c in np.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = np.max([feature_map[r, c, map_num], 0])
    return relu_out

def forwardProp(Z,W,B,hfinal,h,L,shape):
    z = [np.array(Z).T]
    itt = np.intc(0)
    while(itt<L-2):#first layers
        A = np.zeros((W[itt].shape[0],z[itt].shape[1]),dtype=np.float64)
        np.add(np.dot(W[itt],z[itt]),B[itt],out=A)
        A = h(A)
        z.append(A)
        itt+=np.intc(1)
    
    #final layer
    A = np.zeros((W[itt].shape[0],z[itt].shape[1]),dtype=np.float64)
    np.add(np.dot(W[itt],z[itt]),B[itt],out=A)
    z.append(hfinal(A))
    return z

def backProp(y,T,R,hP,W,B,L,shape,itt,yplt,loss,M):
    w=W
    b=B
    err = np.zeros((shape[-1],y[-1].shape[1]),dtype=np.float64)
    np.add(y[-1],-T,out=err)
    cost = loss(y[-1],T)
    yplt[itt]=cost
    dlta = [err]

    itter=np.intc(L-2)
    while(itter>=0):
        Wincrement=(-R*np.dot(dlta[-1],y[itter].T))
        np.add(M*W[itter], Wincrement,out=w[itter])
        iterator=np.intc(0)
        while(iterator<B[itter].shape[1]):
            np.add(M*B[itter].T[iterator], -R*np.sum(err.T[iterator]),out=b[itter].T[iterator])
            iterator+=np.intc(1)
        err = np.zeros((W[itter].shape[1],dlta[-1].shape[1]),dtype=np.float64)
        np.multiply(np.dot(W[itter].T,dlta[-1]),hP(y[itter]),out=err)
        dlta.append(err)
        itter-=np.intc(1)
    return w, b, yplt, err

# cache is the stored X and W values from the previous forward pass
def convBackprop(dH, cache):
    (X, W) = cache
    (f,f) = W.shape

    (nH, nW) = dH.shape

    dX = np.zeros(X.shape)
    dW = np.zeros(W.shape)

    for itt in range (nH):
        for itt2 in range (nW):
            dX[itt:itt+f, itt2:itt2+f] += W * dH(itt, itt2)
            dW += X[itt:itt+f, itt2:itt2+f] * dH(itt, itt2)

    return dX, dW


def tanh(x):
    return (np.float64(1.0)-np.exp(np.float64(-2.0)*x)/(np.float64(1.0) + np.exp(np.float64(-2.0)*x)))
def tanhDx(x):
    return (np.float64(1.0) + tanh(x))*(np.float64(1.0)-tanh(x))
def predict(X,W,B,h,shape,hfinal):
    Z = np.zeros((X.shape[0],X.shape[1]),dtype=np.float64)
    it=np.intc(0)
    while(it<X.shape[0]):
        i=np.intc(0)
        while(i<X.shape[1]):
            batch=[]
            for itt in range(shape[0]):
                batch.append(X[it][i+itt])
            preds=predictSingle(np.array(batch),W,B,h,hfinal)
            for itt in range(shape[0]):
                Z[it][i+itt]=preds[0][itt]
            i+=np.intc(shape[0])
        it+=np.intc(1)    
    return Z
def predictSingle(x,W,B,h,hfinal):
    Z=[x.T]
    i = np.intc(0)
    while(i<len(W)-1):
        Z.append(h(np.dot(W[i],Z[i])))
        i+=np.intc(1)
    Z.append(hfinal(np.dot(W[i],Z[i])))
    return Z[-1]
def logLoss(y,t):
    return np.sum(np.add(-np.multiply(t,np.log(y)),-np.multiply((np.float64(1.0)-t),np.log(np.float64(1.0)-y))))/y.shape[1]
def identity(x):
    return x
def leastSquared(y,t):
    return np.square(np.sum(y-t))
def crossEntropy(probs, label):
    return -np.sum(label * np.log(probs))
def softMax(raw_preds):
    out = np.exp(raw_preds)
    return out/np.sum(out)