import brain.py as nn
import brain

import numpy as np
import matplotlib.pyplot as plt
import pylib.pyplot as plib

print(brain.__version__)
print(brain.__platform__)

class ClassifierModel():
    def __init__(self):
        self.input = nn.layers.InputLayer(2)
        self.dense2 = nn.layers.Dense(4, activation=nn.sigmoid, input=self.input)
        self.dense3 = nn.layers.Dense(3, activation=nn.sigmoid, input=self.dense2)
        self.dense4 = nn.layers.Dense(3, activation=nn.sigmoid, input=self.dense3)

    def predict(self, inputs):
        h = self.input(inputs)
        h = self.dense2(h)
        h = self.dense3(h)
        h = self.dense4(h)

        return h

    def train(self, inputs, targets):
        trainer = nn.math.Trainer()
        trainer.compute(self.input, inputs)

        trainer(self.dense2)
        trainer(self.dense3)
        trainer(self.dense4)

        trainer.backprop(targets)

def func_line1(x):
    return x * 0.1 + 0.1

def func_line2(x):
    return x * 0.85 - 0.9

def category(x, y):
    if y > func_line1(x):
        return [0, 0, 1]
    else:
        if y > func_line2(x):
            return [0, 1, 0]
        else:
            return [1, 0, 0]

def category_strict(data):
    x, y = data
    [a, b, c] = category(x, y)
    
    if a == 1:
        return 2
    if b == 1:
        return 1
    if c == 1:
        return 0

model = None
def category_nn(data):
    mat = model.predict(data)
    max = mat.max()
    [a, b, c] = mat.tolist()[0]
    
    if a == max:
        return 2
    if b == max:
        return 1
    if c == max:
        return 0

if __name__ == "__main__":
    setlength = 150
    set = np.random.rand(setlength, 2) * 2 - 1

    model = ClassifierModel()

    maxepochs = 100
    for epoch in range(maxepochs):
        print("Epoch {}/{}".format(epoch + 1, maxepochs))
        np.random.shuffle(set)
        
        for i in range(len(set)):
            x, y = set[i]
            model.train(set[i], category(x, y))

    fig, axs = plt.subplots(2)
    for ax in axs:
        plib.function(ax, func_line1, 1, start=-1, step=0.01)
        plib.function(ax, func_line2, 1, start=-1, step=0.01)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    strict_ds = plib.dataset(['A', 'B', 'C'], set, category_strict)
    plib.plot_ds(axs[0], strict_ds, ".", colors=["r", "g", "b"])

    nn_ds = plib.dataset(['A', 'B', 'C'], set, category_nn)
    plib.plot_ds(axs[1], nn_ds, ".", colors=["r", "g", "b"])

    plt.show()