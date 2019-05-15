import gzip
import io
import numpy as np

__name__ = 'dataGetter'

def getData(filename):
    itt = np.intc(0)

    f = gzip.open(filename, 'rb')
    magicnum = int.from_bytes(f.read(4), byteorder = 'big')
    labelorpics = np.bool(0)
    if magicnum == 2051:   labelorpics = np.bool(1)
    else:                  labelorpics = np.bool(0)

    if labelorpics:
        imgNum = int.from_bytes(f.read(4), byteorder = 'big')
        rowNum = int.from_bytes(f.read(4), byteorder = 'big')
        colNum = int.from_bytes(f.read(4), byteorder = 'big')
        
        #picMat = np.zeros((int(imgNum), rowNum, colNum), dtype = np.float32)
        picMat = np.frombuffer(f.read(),'B')
        picMat = picMat.reshape(imgNum,28,28,1).astype('float32')/255
        #for itt in range(int(imgNum / 3)):      # only reading a third of the data for speed in the short term
        #    for itt2 in range(rowNum):
        #        for itt3 in range(colNum):
        #            pixel = int.from_bytes(f.read(1), byteorder = 'big')
        #            picMat[itt, itt2, itt3] = pixel
        return picMat

    else:
        length = int.from_bytes(f.read(4), byteorder='big')
        Labels = np.frombuffer(f.read(),'B')
        Labels = Labels.reshape(length,1).astype('float32')
        #for itt in range(length):
        #    Labels[itt] = float.from_bytes(f.read(1), byteorder = 'big')
        return Labels

def getFiles():
    fileNames = np.array([['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'], \
                          ['t10k-images-idx3-ubyte.gz',  't10k-labels-idx1-ubyte.gz']], dtype = np.str)
    return fileNames
