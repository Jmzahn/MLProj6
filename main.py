import NNarch
import dataGetter

class main:
    def __init__(self):
        pass

    def run(self):
        NNarch.Network([[20,13],[40,5]], [5,5,5,5,5], 1000, .0001, 1)

uut = main()
uut.run()