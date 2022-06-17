from NeuralNetwork import NeuralNetwork
from random import uniform


class GradientNetwork(NeuralNetwork):

    def __init__(self, *, numInputs: int, nodesInLayer: int = 5, numLayers: int = 2, nodes=None,
                 maxIter: int = 10, learnRate: float = 0.1):
        super().__init__(numInputs, nodesInLayer, numLayers, nodes)

        self.maxIter = maxIter
        self.learnRate = learnRate

        self.size = nodesInLayer * numLayers

    @staticmethod
    def gradient(weights: list[float], biases: list[float], inputs: list[float], goal: float) \
            -> tuple[list[float], list[float]]:
        
        weightsChange = []
        biasesChange = []

        for _ in range(len(inputs)):
            weightsChange.append(uniform(-5, 5))
            biasesChange.append(uniform(-5, 5))

        return weightsChange, biasesChange

    def evolve(self, inputs, goal):

        for _ in range(self.maxIter):
            self(inputs)
            print(self.value)
            self.stepGeneration(inputs, goal)

    def stepGeneration(self, inputs, goal):
        for index in range(self.size):
            layer = index // self.nodesInLayer
            numbInLayer = layer % self.nodesInLayer
            node = self.nodes[layer][numbInLayer]

            weightsChange, biasesChange = self.gradient(node.weights, node.biases, inputs, goal)

            weightsChange = list(map(lambda change: change * self.learnRate, weightsChange))
            biasesChange = list(map(lambda change: change * self.learnRate, biasesChange))

            node.updateWeights(weightsChange)
            node.updateBiases(biasesChange)
