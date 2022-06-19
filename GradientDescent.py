from NeuralNetwork import NeuralNetwork
from Node import Node

import json


class GradientNetwork(NeuralNetwork):

    def __init__(self, *, numInputs: int, nodesInLayer: int = 5, numLayers: int = 2, nodes: list[Node] = None,
                 maxIter: int = 1000, learnRate: float = 0.1, decimalPlaces: int = 4):

        # calling the NeuralNetwork initialization function
        # this initializes nodes, children variables
        # also sets numInputs, nodesInaLayer, numLayers
        super().__init__(numInputs, nodesInLayer, numLayers, nodes)

        # the maximum amount of iterations when training
        self.maxIter = maxIter

        # a scalar representing how large of adjustments the AI will make
        self.learnRate = learnRate

        # this is how many decimal places are used in printing
        # this has no impact on calculations
        # json still uses all decimal places
        self.decimalPlaces = decimalPlaces

    @property
    def size(self):
        # this is a getter
        # this returns the total amount of elements inside the neural network
        return self.nodesInLayer * self.numLayers

    def evolveTillTolerance(self, inputs: list[float], goal: float, tolerance: float = -1, printValueStep: int = 5,
                            storeAsJson: bool = False, storeAsJsonStep: int = 5,
                            jsonFilename: str = "GradientDescentData"):
        """
        :param inputs: the vector of inputs that is passed into the network
        :param goal: the float representing the output it wants to reach
        :param tolerance: how close to the goal it will train until
        :param printValueStep: how often to print the values
        :param storeAsJson: whether to store in json
        :param storeAsJsonStep: how often to store in json
        :param jsonFilename: the filename of the json to store data in
        :return: nothing
        """

        # setting default value
        if tolerance <= 0:
            tolerance = 0.1

        # counter of which iteration this network is on
        iterationCounter = 0

        # loops until reaches max iter
        while iterationCounter < self.maxIter:

            # storing in json if parameters are met
            if storeAsJson and iterationCounter % storeAsJsonStep == 0:

                # default filename (if not specified) is GradientDescentData.json
                # open in write form clears all previous text in json
                with open(jsonFilename + ".json", 'w') as f:
                    # dump the dictionary into the json file
                    json.dump(self.asDict(), f, indent=4)

            # calculating the value with the current weights
            self(inputs)

            # printing the value if conditions are met
            if iterationCounter % printValueStep == 0:
                print(f"Value at iteration {iterationCounter}: {self.value:.{self.decimalPlaces}f}")

            # moves onto the next iteration
            # adjusts all the weights according to the gradient
            self.stepGeneration(inputs, goal)

            # checking if tolerance thresh hold is met
            # breaks if it does
            if abs(self.value - goal) < abs(tolerance * goal):
                # printing the current value
                print(f"Value at iteration {iterationCounter}: {self.value:.{self.decimalPlaces}f}")
                break

            # steps forward in iteration counter
            iterationCounter += 1

        # how close the neural network reached (in percent)
        print(f"Reached {abs((self.value - goal) * 100 / goal):.{self.decimalPlaces}f}",
              f"percent of goal after {iterationCounter} iterations")

    def stepGeneration(self, inputs: list[float], goal: float):
        """
        :param inputs:
        :param goal:
        :return: nothing
        """

        # iterates through all the nodes (only in layer 1)
        # this does not support multi-layer networks yet

        # adjusting the weights and biases of the node
        # the goal is divided by the number of nodes
        # this leads to each node converging on 1/n of the output
        list(map(lambda node: node.gradient(inputs, goal / self.size, self.learnRate), self.nodes[0]))
