import gzip
import numpy as np

def getFile(filename):
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

        picMat = np.zeros((imgNum, rowNum, colNum), dtype = np.intc)
        for itt in range(imgNum):
            for itt2 in range(rowNum):
                for itt3 in range(colNum):
                    pixel = int.from_bytes(f.read(1), byteorder = 'big')
                    picMat[itt, itt2, itt3] = pixel
        return picMat

    else:
        length = int.from_bytes(f.read(4), byteorder='big')
        Labels = np.zeros(length, dtype = np.intc)
        for itt in range(length):
            Labels[itt] = int.from_bytes(f.read(1), byteorder = 'big')
        return Labels

theOutput = getFile(filename = 'train-images-idx3-ubyte.gz')
print(theOutput.shape)
