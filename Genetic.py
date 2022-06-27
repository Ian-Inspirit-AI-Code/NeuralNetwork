from __future__ import annotations

from NeuralNetwork import BaseNeuralNetwork
from Node import Node, mutateUpdate, sigmoidActivationFunction, reluActivationFunction

from typing import Callable
from copy import deepcopy

import json


class Brain(BaseNeuralNetwork):

    def __init__(self, *, numInputs: int, numOutputs: int, activationFunctionString: str = "relu",
                 nodesInLayer: int = 5, numLayers: int = 2,
                 nodes: list[Node] = None) -> None:

        super().__init__(numInputs, numOutputs, nodesInLayer, numLayers, nodes)
        self.sigmoid_value = 0.5
        self.activationFunctionString = activationFunctionString

    @staticmethod
    def nodeUpdate(node: Node, inputs: list[float], goal: float) -> None:
        mutateUpdate()(node, inputs, goal)

    def nodeActivation(self, value: float) -> float:
        if self.activationFunctionString == "relu":
            return reluActivationFunction()(value)
        elif self.activationFunctionString == "sigmoid":
            return sigmoidActivationFunction(self.sigmoid_value)(value)

        raise ValueError("Options for activation function are 'relu' and 'sigmoid'.")

    def updateNetwork(self) -> Brain:
        list(map(lambda layer: list(map(lambda node: node.update(None, None), layer)), self.nodes))
        return self


class Population:

    def __init__(self, *, numIndividuals: int, numInputs: int, numOutputs: int,
                 nodesInLayer: int = 5, numLayers: int = 2, activationFunctionString: str = "relu",
                 lossFunction: Callable[[list[float], list[float]], float]) -> None:

        self.size = numIndividuals

        self.individuals = [Brain(numInputs=numInputs, numOutputs=numOutputs,
                                  nodesInLayer=nodesInLayer, numLayers=numLayers,
                                  activationFunctionString=activationFunctionString) for _ in range(self.size)]

        self.values = [[0] * numOutputs for _ in range(numIndividuals)]

        self.lossFunction = lossFunction

    def __call__(self, inputs: list[float]):
        """
        :param inputs:
        :return: no return value
        """

        # calls each individual in the population with the given inputs
        # sets the populations values to it
        self.values = [individual(inputs) for individual in self.individuals]

    def findBest(self, inputs: list[float]) -> Brain:
        """
        :return: the best individual (min loss)
        """

        # closest is an abs because being above/below are equally punished
        # sets closest to the first individual
        # difference is just a very simple loss function
        smallest = self.lossFunction(inputs, self.values[0])

        # sets the best to the first individual by default
        out = self.individuals[0]

        # iterate through each individual and value
        for individual, value in zip(self.individuals, self.values):

            # calculates how far away it is
            value = self.lossFunction(inputs, value)

            # updates the best if this new individual is better
            if value < smallest:
                smallest = value
                out = individual

        return out

    def fromIndividual(self, best: Brain):
        """
        :param best: the best individual in the population
        :return: nothing
        """

        # creates a new population from the best
        # creates deep copies of the best
        # shallow copy does not copy the weights, deep copy is necessary
        self.individuals = [deepcopy(best).updateNetwork() for _ in range(self.size - 1)]

        # keeps the best in the population
        # this prevents the model from ever getting worse
        self.individuals.append(best)

    def evolve(self, inputs: list[float], numIterations: int,
               writeToJson: bool = False, jsonFilename: str = "bestInPopulation"):
        """
        :param inputs:
        :param numIterations:
        :param writeToJson: whether to output in a json or not
        :param jsonFilename: name of file to output data to
        :return: nothing
        """

        # creates a dictionary object that will be written to a json file
        dictionary = dict()

        # iterates a numIterations amount of times
        for generationNumber in range(numIterations):

            # it calls the inputs on the population
            # this calls each individual in the population
            # this also sets self.values
            self(inputs)

            # finds the best individual in the population
            best = self.findBest(inputs)

            # printing the value of the best (not necessary)
            print(f"Best is: {best.value}")

            # adding the best in current generation
            # dictionary will always be created. your python compiler may give a warning here
            if writeToJson:
                dictionary[f"Generation {generationNumber}"] = best.asDict()

            # creates a new population from the best
            # keeps the best in the population
            self.fromIndividual(best)

        # write to a json file if writeToJson is true
        if writeToJson:
            # default filename (if not specified) is bestInPopulation
            # open in write form clears all previous text in json
            with open(jsonFilename + ".json", 'w') as f:
                # dump the dictionary into the json file
                json.dump(dictionary, f, indent=4)
