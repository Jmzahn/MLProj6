import NNarch
import dataGetter

class main:
    def __init__(self):
        pass

    def run(self):
        NNarch.Network([[50,7],[35,5],[20,3]], [5,5,5,5,5], 1000, .0001, 1)

uut = main()
uut.run()