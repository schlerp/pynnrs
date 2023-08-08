import pynnrs

x = [1.0, 2.0, 3.0, 4.0]
y = [1.0, 2.0]

_layers = [4, 5, 3, 123, 33, 12, 34, 11, 2]

net = pynnrs.Network(layer_sizes=_layers)

print(pynnrs.train(net, x, y, epochs=10))
