from Node import randomNode
from copy import deepcopy


class NeuralNetwork:

    def __init__(self, numInputs: int, nodesInLayer: int = 5, numLayers: int = 2, nodes=None):
        """
        :param numInputs: number of target inputs
        :param nodesInLayer: how many nodes for each layer
        :param numLayers: how many hidden layers to create
        :param nodes: the nodes used in self (can be None)
        """

        # initializes a rectangular array of random nodes if no specified nodes
        if nodes is None:
            # children are not initialized here
            # array dimensions are numLayers x nodesInLayer
            self.nodes = [[randomNode(numInputs) if layer == 0 else randomNode(nodesInLayer)
                           for _ in range(nodesInLayer)] for layer in range(numLayers)]

        # if network is given a set of nodes
        else:

            # use the set of given nodes
            self.nodes = nodes

        # this is setting the children of all the layers except for the last one
        # the last layer has no children
        for layer in range(numLayers - 1):
            parents = self.nodes[layer]
            children = self.nodes[layer + 1]

            for node in parents:
                node.setChildren(children)

        # setting instance variables
        self.numInputs = numInputs
        self.nodesInLayer = nodesInLayer
        self.numLayers = numLayers

        # the last calculated value
        self.value = 0

    def mutate(self):
        """
        Mutates each node within the neural network
        """

        # for layerNumber in range(self.numLayers):
        #     for nodeNumber in range(self.nodesInLayer):
        #         self.nodes[layerNumber][nodeNumber].mutate()

        # this is just a compressed version of the two-dimensional for loop above
        # it would call node.mutate() for each node in self.nodes
        # this uses two maps to iterate faster
        # casting to a list is needed (for some reason, I'm not entirely sure)
        list(map(lambda layer: list(map(lambda node: node.mutate(), layer)), self.nodes))

        # returning the mutated neural network
        return self

    def __call__(self, inputs: list[float]) -> float:
        """
        :param inputs: list of inputs (length numInputs)
        :return: a list of outputs (length numOutputs)
        """

        # validation in correct number
        if len(inputs) != self.numInputs:
            raise ValueError("Wrong number of inputs.")

        for layer in self.nodes:
            # these are the outputs to be passed into the next function
            newInputs = []

            for node in layer:
                # call each node with the list of inputs
                newInputs.append(node(inputs))

            # sets the inputs for the next layer to be the results of the current
            inputs = newInputs

        # these are the values for the last layer
        outputValues = inputs

        # returns the sum of all values in the last layer
        # this could be turned into some other function to create multiple outputs
        # sets this output as the value for self
        self.value = sum(outputValues)

        # returns the calculated value
        return self.value

    def __deepcopy__(self, memo):
        """
        :param memo: not used. this is a parameter because __deepcopy__ overrides a default function
        :return: the copied network object
        """

        # memo is a parameter that stores memory addresses of already copied objects
        # this can speed up the copying process with redundant objects
        # this is not implemented here

        # this is creating a copy of the self nodes
        # this maps the deepcopy call onto every node and collects into a list
        copiedNodes = list(map(lambda layer: list(map(lambda node: deepcopy(node), layer)), self.nodes))

        # returning the copied neural network object
        # only changing the nodes (creating copies)
        # other parameters are kept
        return NeuralNetwork(self.numInputs, self.nodesInLayer, self.numLayers, copiedNodes)

    def asDict(self) -> dict:
        """
        :return: a json representation of this neural network
        """

        # the keys are the layer number and the node number (in each layer)
        # this is just a two-dimensional list comprehension
        # makes a dict for each layer, and a dict for each layer in the network
        return {f"Layer {numLayer}": {f"Node {numberNode}": node.asDict() for (numberNode, node) in enumerate(layer)}
                for (numLayer, layer) in enumerate(self.nodes)}
