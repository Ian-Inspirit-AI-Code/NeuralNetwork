from __future__ import annotations

from copy import deepcopy
from Node import Node
from abc import ABC, abstractmethod


class BaseNeuralNetwork(ABC):

    def __init__(self, numInputs: int, numOutputs: int, nodesInLayer: int = 5, numLayers: int = 2,
                 nodes: list[list[Node]] = None) -> None:
        """
        :param numInputs:
        :param numOutputs:
        :param nodesInLayer:
        :param numLayers:
        :param nodes:
        """

        # setting up instance variables
        self.nodesInLayer = nodesInLayer
        self.numLayers = numLayers

        self.numInputs = numInputs
        self.numOutputs = numOutputs

        self.nodes = self.createNodes() if nodes is None else nodes
        self.setChildrenAndParents()

        self.value = [0] * numOutputs

    def createNodes(self) -> list[list[Node]]:
        def createNodeLayer(layer: int):
            numInputs = self.numInputs if layer == 0 else self.nodesInLayer
            numNodesInLayer = self.numOutputs if layer == (self.numLayers - 1) else self.nodesInLayer

            return [Node(numInputs, layerNumber=layer, nodeInLayer=index,
                         update=self.nodeUpdate, activation=self.nodeActivation)
                    for index in range(numNodesInLayer)]

        return [createNodeLayer(layer) for layer in range(self.numLayers)]

    def setChildrenAndParents(self) -> None:
        if self.numLayers == 1:
            return

        parents = []
        children = self.nodes[1]

        for layer in range(self.numLayers - 1):

            for index in range(len(self.nodes[layer])):
                node = self.nodes[layer][index]
                node.parents = parents
                node.children = children

            parents = children
            children = self.nodes[layer + 1]

    def __call__(self, inputs: list[float]) -> list[float]:
        if len(inputs) != self.numInputs:
            raise ValueError(f"Wrong number of inputs. Expected {self.numInputs}")
        for index, value in enumerate(inputs):
            list(map(lambda node: node(value, index, True), self.nodes[0]))

        self.value = [node.value for node in self.nodes[-1]]
        return self.value

    def __deepcopy__(self, memo):
        copiedNodes = list(map(lambda layer: list(map(lambda node: deepcopy(node), layer)), self.nodes))

        return self.__class__(numInputs=self.numInputs, numOutputs=self.numOutputs,
                              nodesInLayer=self.nodesInLayer, numLayers=self.numLayers, nodes=copiedNodes)

    def asDict(self) -> dict:
        """
        :return: a json representation of this neural network
        """

        # the keys are the layer number and the node number (in each layer)
        # this is just a two-dimensional list comprehension
        # makes a dict for each layer, and a dict for each layer in the network
        return {f"Layer {numLayer}": {f"Node {numberNode}": node.asDict() for (numberNode, node) in enumerate(layer)}
                for (numLayer, layer) in enumerate(self.nodes)}

    @abstractmethod
    def updateNetwork(self) -> BaseNeuralNetwork:
        pass

    @staticmethod
    @abstractmethod
    def nodeUpdate(node: Node, inputs: list[float], goal: float) -> None:
        pass

    @abstractmethod
    def nodeActivation(self, value: float) -> float:
        pass
