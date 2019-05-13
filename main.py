import NNarch
import dataGetter

class main:
    def __init__(self):
        pass

    def run(self):
        filenames = dataGetter.getFiles()

        print("Getting training images...")
        self.trainImg = dataGetter.getData(filenames[0,0])
        print("Getting testing images...")
        self.testImg  = dataGetter.getData(filenames[1,0])
        print("Getting targets...")
        self.trainTarg = dataGetter.getData(filenames[0,1])
        self.testTarg  = dataGetter.getData(filenames[1,1])
        print("Data retrieved")

