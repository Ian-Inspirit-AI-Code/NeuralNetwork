import json.decoder

from Genetic import Population, Brain
from random import uniform

import numpy as np

numberPoints = 20

numInputs = numberPoints * 2
numOutputs = 2

numIndividuals = 30

# activationFunctionString = "relu"
# activationFunctionString = "sigmoid"
activationFunctionString = "none"

nodesInLayer = 4
numLayers = 3

xRange = (0, 150)
slopeRange = (0, 10)


def main():
    import sys
    sys.path.append('C:\\Users\\ianch\\PycharmProjects\\InspiritAI\\Regression')
    from Graph import Graph
    from Line import SlopeIntercept
    from Point import Point

    model = Brain(numInputs=numInputs, numOutputs=numOutputs,
                  nodesInLayer=nodesInLayer, numLayers=numLayers,
                  activationFunctionString=activationFunctionString)

    try:
        model.fromJson("NeuralNetwork")
    except json.decoder.JSONDecodeError:
        print("Need to train the neural network first.")
        return

    testInput = createRoughlyLinearScatter(numberPoints, xRange, slopeRange)

    x = testInput[:numberPoints]
    y = testInput[numberPoints:]

    xMin, xMax = min(x), max(x)
    yMin, yMax = min(y), max(y)
    xLabel = (xMax - xMin) // 10
    yLabel = (yMax - yMin) // 10

    graph = Graph(xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabelInterval=xLabel, yLabelInterval=yLabel)

    for pointTuple in zip(x, y):
        graph.plot(Point(*pointTuple))
        print(pointTuple)

    model(testInput)
    slope, intercept = model.value
    line = SlopeIntercept(slope, intercept)
    graph.createLine(line, plotPoints=False)

    print(slope, intercept)
    graph.display()


def trainLinearRegression():
    population = Population(numIndividuals=numIndividuals, numInputs=numInputs, numOutputs=numOutputs,
                            nodesInLayer=nodesInLayer, numLayers=numLayers,
                            activationFunctionString=activationFunctionString,
                            lossFunction=lossFunction)

    try:
        model = Brain(numInputs=numInputs, numOutputs=numOutputs,
                      nodesInLayer=nodesInLayer, numLayers=numLayers,
                      activationFunctionString=activationFunctionString)
        model.fromJson("NeuralNetwork")
        population.fromIndividual(model)
    except json.decoder.JSONDecodeError:
        pass

    trainingSize = 100

    inputs = [createRoughlyLinearScatter(numberPoints, xRange, slopeRange) for _ in range(trainingSize)]
    numIterations = 10
    numEpoch = 10

    # population.evolveSingleInput(inputs, numEpoch)
    model = population.evolve(inputs, numIterations, numEpoch)

    model.toJson()


def createRoughlyLinearScatter(numPoints, xRange: tuple[float, float], slopeRange: tuple[float, float]) -> list[float]:
    slope = uniform(*slopeRange)

    xRandom = [uniform(*xRange) for _ in range(numPoints)]
    x = np.array([x * (1 - uniform(-1, 1)) for x in xRandom])

    yLine = x * slope
    y = np.array([y * (1 - uniform(-1, 1)) for y in yLine])

    return x.tolist() + y.tolist()


def lossFunction(inputs: list[float], outputs: list[float]) -> float:
    # outputs represent slope and intercept
    slope, intercept = outputs

    # inputs are the points
    # first half of inputs are the x, second half are the y

    numPoints = len(inputs) // 2

    y_pred = [x * slope + intercept for x in inputs[:numPoints]]
    y_actual = inputs[numPoints:]

    # print(inputs, y_actual)
    # print(y_pred, y_actual, sum(map(lambda pred, actual: (pred - actual) ** 2, y_pred, y_actual)), sep='\n')
    return sum(map(lambda pred, actual: (pred - actual) ** 2, y_pred, y_actual))


if __name__ == "__main__":
    # trainLinearRegression()
    main()
