import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.show()
def Network(shape,p,e,r,m):#
    
    L=len(shape)+2#Layers = hidden layers + 2
    shape=np.array(shape)
    P=p#problem type, 0 for Class., 1 for Reg.
    E=np.intc(e)#epochs
    R=np.float64(r)#Learning Rate
    M=m
    W = []
    B = []

    if(P==0):#if problem is classification
        X, T = genCdata()
        loss = logLoss
        hfinal = tanh
        h = tanh
        hPrime = tanhDx
        graph = Cgraph
        I = [X.shape[0]]
        O = [T.shape[0]]
        shape = np.concatenate((I,shape,O))
    elif(P==1):#if problem is regression
        X, T = genRData()
        loss = leastSquared
        hfinal = identity
        h = np.tanh
        hPrime = tanhDx
        graph = Rgraph
        I = [X.shape[0]]
        O = [T.shape[0]]
        #T=T.T
        shape = np.concatenate((I,shape,O))
    print('Xs shape: ',X.shape)
    w0 = (np.float64(2)*np.random.rand(shape[1],X.shape[1]).astype(np.float64) - np.float64(1))
    W.append(w0)
    b0 = np.full((shape[1],shape[0]),0.1,dtype=np.float64)
    #b0 = (np.float64(2.0)*np.random.rand(shape[1],shape[0]).astype(np.float64) - np.float64(1.0))#
    B.append(b0)
    for itt in range(1,L-1):
        w = (np.float64(2)*np.random.rand(shape[itt+1],shape[itt]).astype(np.float64) - np.float64(1))
        W.append(w)
        b = np.full((shape[itt+1],shape[0]),0.1,dtype=np.float64)
        #b = (np.float64(2.0)*np.random.rand(shape[itt+1],shape[0]).astype(np.float64) - np.float64(1.0))#
        B.append(b)
        
    W, y = fit(X,T,W,B,E,R,L,shape,hfinal,h,hPrime,loss,graph,M)   
    print(y[-1])
    return W, y

def genCdata():#data for Xor classification problem
    X=np.array([[-1,-1],[-1,1],[1,1],[1,-1]],dtype=np.float64)
    T=np.array([[0,1,0,1]],dtype=np.float64)
    return X, T

def genRData():#data for regression problem
    X=np.zeros((50,1), dtype=np.float64)
    T=np.zeros((50,1), dtype=np.float64).T
    itt = np.intc(0)
    while(itt<50):
        X[itt] = (np.float64(2)*np.random.random())-np.float64(1)
        itt+= np.intc(1)
    itt = np.intc(0)
    while(itt<50):
        T[0][itt] = np.sin(np.float64(2)*np.pi*X[itt])+np.float64(0.3)*np.random.random()#+np.float64(1)
        itt+= np.intc(1)
    return X, T

def fit(X,T,W,B,E,R,L,shape,hfinal,h,hP,loss,graph,M):
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
        Z = forwardProp(X,W,B,hfinal,h,L,shape)
        target = T
        
        W, B, yplt = backProp(Z,target,R,hP,W,B,L,shape,itt,yplt,loss,M)
        line1.set_ydata(yplt)
        if itt%20==0:#line plotting 
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
        itt+=np.intc(1)
    print(Z[-1])
    graph(X,T,W,B,h,shape,fig,ax,hfinal)

    
    return W, Z
    
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
    return w, b, yplt

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
def Cgraph(X,T,W,B,h,shape,fig,ax,hfinal):
    xline = np.linspace(-2, 2, 96)
    yline = np.linspace(-2, 2, 96)
    Xmesh, Ymesh = np.meshgrid(xline, yline)
    zline = predict(np.array([Xmesh,Ymesh]).T,W,B,h,shape,hfinal)
    ax=plt.axes(projection="3d")
    print(zline.shape)
    surf = ax.plot_surface(Xmesh, Ymesh, zline, cmap=cm.coolwarm,linewidth=0,antialiased=False,alpha=.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('pred')
    ax.set_title('3D')
    ax.set_zlim(0, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    for i in range(shape[0]):
        ax.scatter(X[i][0],X[i][1] , T[0][i], c='b', marker='.', zorder=10)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show()
    return
def Rgraph(X,T,W,B,h,shape,fig,ax,hfinal):
    xline = np.linspace(-1, 1, 200)
    Z = np.zeros(xline.shape,dtype=np.float64)
    it=np.intc(0)
    while(it<xline.shape[0]):
        i=np.intc(0)
        Z[it]=predictSingle(xline[it].T,W,B,h,hfinal)
        it+=np.intc(1)    
    ax.axis([-1,1,-1,1])
    for i in range(X.shape[0]):
        ax.scatter(X[i][0], T[0][i], c='b', marker='.', zorder=10)
    ax.plot(xline,Z, 'r-')
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.show()
    return
