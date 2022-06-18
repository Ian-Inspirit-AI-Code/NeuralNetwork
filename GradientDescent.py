from NeuralNetwork import NeuralNetwork
import json


class GradientNetwork(NeuralNetwork):

    def __init__(self, *, numInputs: int, nodesInLayer: int = 5, numLayers: int = 2, nodes=None,
                 maxIter: int = 1000, learnRate: float = 0.1):
        super().__init__(numInputs, nodesInLayer, numLayers, nodes)

        self.maxIter = maxIter
        self.learnRate = learnRate

        self.size = nodesInLayer * numLayers

    def evolve(self, inputs, goal):

        for _ in range(self.maxIter):
            self(inputs)
            print("Current iteration value:", self.value)
            self.stepGeneration(inputs, goal)

    def evolveTillTolerance(self, inputs, goal, tolerance=-1, printValueStep=5,
                            storeAsJson: bool = False, storeAsJsonStep=5, jsonFilename: str = "GradientDescentData"):

        if tolerance <= 0:
            tolerance = 0.1

        iterationCounter = 0
        while iterationCounter < self.maxIter:

            if storeAsJson and iterationCounter % storeAsJsonStep == 0:
                # default filename (if not specified) is bestInPopulation
                # open in write form clears all previous text in json
                with open(jsonFilename + ".json", 'w') as f:
                    # dump the dictionary into the json file
                    json.dump(self.asDict(), f, indent=4)

            self(inputs)
            if iterationCounter % printValueStep == 0:
                print(f"Value at iteration {iterationCounter}:", self.value)

            self.stepGeneration(inputs, goal)
            iterationCounter += 1

            if abs(self.value - goal) < abs(tolerance * goal):
                print(f"Value at iteration {iterationCounter}:", self.value)
                break

        print(f"Reached {abs((self.value - goal) * 100 / goal)} percent of goal after {iterationCounter} iterations")

    def stepGeneration(self, inputs, goal):
        """
        :param inputs:
        :param goal:
        :return:
        """

        for index in range(self.nodesInLayer):
            node = self.nodes[0][index]
            node.gradient(inputs, goal / self.size, self.learnRate)
