from Genetic import Population
from random import uniform

import numpy as np

import sys
sys.path.append('C:\\Users\\ianch\\PycharmProjects\\InspiritAI\\Regression')
from Graph import Graph
from Line import SlopeIntercept
from Point import Point


def main():
    numberPoints = 10

    numInputs = numberPoints * 2
    numOutputs = 2

    numIndividuals = 15

    # activationFunctionString = "relu"
    # activationFunctionString = "sigmoid"
    activationFunctionString = "none"

    nodesInLayer = 5
    numLayers = 5

    population = Population(numIndividuals=numIndividuals, numInputs=numInputs, numOutputs=numOutputs,
                            nodesInLayer=nodesInLayer, numLayers=numLayers,
                            activationFunctionString=activationFunctionString,
                            lossFunction=lossFunction)

    xRange = (-10, 10)
    slopeRange = (-5, 5)
    inputs = createRoughlyLinearScatter(numberPoints, xRange, slopeRange)
    numEpoch = 15

    xMin, xMax, yMin, yMax = xRange[0] * 1.5, xRange[1] * 1.5, xRange[0] * slopeRange[1], xRange[1] * slopeRange[1]
    graph = Graph(xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, yLabelInterval=10, xLabelInterval=2)
    x = inputs[:numberPoints]
    y = inputs[numberPoints:]

    for a, b in zip(x, y):
        graph.plot(Point(a, b))

    print(f"{inputs=}")

    population.evolveSingleInput(inputs, numEpoch)

    population(inputs)
    slope, intercept = population.findBest(inputs).value
    print(slope, intercept)
    line = SlopeIntercept(slope, intercept)
    graph.createLine(line, plotPoints=False)
    graph.display()


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
    main()
