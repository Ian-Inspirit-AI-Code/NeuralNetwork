import json

from NeuralNetwork import NeuralNetwork
from copy import deepcopy


class Population:

    def __init__(self, *, numIndividuals: int, numInputs: int, nodesInLayer: int = 5, numLayers: int = 2):
        """
        :param numIndividuals:
        :param numInputs:
        :param nodesInLayer:
        :param numLayers:
        """

        # the population size of this population
        self.size = numIndividuals

        # all the neural networks that are in the population
        self.individuals = [NeuralNetwork(numInputs, nodesInLayer, numLayers) for _ in range(numIndividuals)]

        # setting default values for each
        self.values = [0] * numIndividuals

    def __call__(self, inputs: list[float]):
        """
        :param inputs:
        :return: no return value
        """

        # calls each individual in the population with the given inputs
        # sets the populations values to it
        self.values = [individual(inputs) for individual in self.individuals]

    @staticmethod
    def lossFunction(output, goal):
        # this is a simple loss function
        # this is used to determine how well each neuron (node) performed
        # all this function is taking the difference and squaring it
        return (output - goal) ** 2

    def findBest(self, goal: float) -> NeuralNetwork:
        """
        :param goal: the goal where the population is converging on
        :return: the best individual (closest to goal)
        """

        # closest is an abs because being above/below are equally punished
        # sets closest to the first individual
        # difference is just a very simple loss function
        closest = self.lossFunction(self.values[0], goal)

        # sets the best to the first individual by default
        out = self.individuals[0]

        # iterate through each individual and value
        for individual, value in zip(self.individuals, self.values):

            # calculates how far away it is
            difference = self.lossFunction(goal, value)

            # updates the best if this new individual is better
            if difference < closest:
                closest = difference
                out = individual

        return out

    def fromIndividual(self, best: NeuralNetwork):
        """
        :param best: the best individual in the population
        :return: nothing
        """

        # creates a new population from the best
        # creates deep copies of the best
        # shallow copy does not copy the weights, deep copy is necessary
        self.individuals = [deepcopy(best).mutate() for _ in range(self.size - 1)]

        # keeps the best in the population
        # this prevents the model from ever getting worse
        self.individuals.append(best)

    def evolve(self, inputs: list[float], goal: float, numIterations: int,
               writeToJson: bool = False, jsonFilename: str = "bestInPopulation"):
        """
        :param inputs:
        :param goal:
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
            best = self.findBest(goal)

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
