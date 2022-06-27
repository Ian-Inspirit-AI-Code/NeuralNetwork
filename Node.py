from __future__ import annotations

from math import e
from random import uniform
from typing import Callable, Any


class Node:
    """
    Each node would take in a list of inputs and return a single output
    """

    DEFAULT_MIN_WEIGHT = -10
    DEFAULT_MAX_WEIGHT = 10

    DEFAULT_MIN_BIAS = -5
    DEFAULT_MAX_BIAS = 5

    def __init__(self, numInputs: int, weights: list[float] = None, biases: list[float] = None,
                 layerNumber: int = 0, nodeInLayer: int = 0,
                 parents: list[Node] = None, children: list[Node] = None,
                 activation: Callable[[float], float] = lambda x: x,
                 update: Callable[[Node, list[float], float], None] = lambda _: None) -> None:
        """
        :param weights:
        :param biases:
        :param parents:
        :param activation:
        :param update:
        """

        # amount of inputs this node takes in
        self.numInputs = numInputs

        # setting random weights if not given
        if weights is None:
            weights = [uniform(self.DEFAULT_MIN_WEIGHT, self.DEFAULT_MAX_WEIGHT) for _ in range(numInputs)]

        # setting random biases if not given
        if biases is None:
            biases = [uniform(self.DEFAULT_MIN_BIAS, self.DEFAULT_MAX_BIAS) for _ in range(numInputs)]

        # the inputs, biases, and weights must be equal dimensions
        if len(biases) != len(weights) != numInputs:
            raise ValueError(f"The length of weights ({len(weights)} and biases ({len(biases)}) "
                             f"does not equal numInputs {numInputs}")

        self.weights = weights
        self.biases = biases

        # because parents pass the inputs, the number of parents must equal number of inputs
        # this check excludes empty parents, the nodes in the first layer
        if parents and len(parents) != self.numInputs:
            raise ValueError("Length of parents does not match given number of inputs")

        # the parent nodes
        self.parents = parents

        # the children nodes
        self.children = children

        # this is the last value associated with this node
        # this will be used to call the next in layer
        self.value = 0

        # these are functions used in calling and updating the node
        self.activation = activation
        self.fullUpdate = update

        self.update = lambda inputs, goal: update(self, inputs, goal)

        # indices that indicate spacial orientation of node
        self.layerNumber = layerNumber
        self.nodeInLayer = nodeInLayer

    def __call__(self, inputFloat: float, index: int, callChildren: bool = True) -> None:
        """
        :param inputFloat:
        :param index:
        :param callChildren:
        """

        if index == 0:
            self.value = 0

        # weights and biases used to transform the given input
        weight, bias = self.weights[index], self.biases[index]

        # calculating result
        resultWithoutActivation = weight * inputFloat + bias
        result = self.activation(resultWithoutActivation)

        # adding the result to value
        self.value += result

        # exits the function call when there are no children (or specified to not call children)
        if not (callChildren and self.children):
            return

        # calling each node in the child layer
        list(map(lambda child: child(inputFloat, self.nodeInLayer, callChildren), self.children))

    def setChildren(self, newChildren: list[Node]) -> None:
        """
        :param newChildren:
        """

        self.children = newChildren

    def setParents(self, newParents: list[Node]) -> None:
        self.parents = newParents

    def __deepcopy__(self, memo) -> Node:
        """
        :param memo: not used. this is a parameter because __deepcopy__ overrides a default function
        :return: the copied node object
        """

        # creating copy of weight through list comprehension
        # because weights are floats, do not need to copy weights
        weightsCopy = [weight for weight in self.weights]

        # creating copy of biases through list comprehension
        # because biases are floats, do not need to copy biases
        biasesCopy = [bias for bias in self.biases]

        # returning the copied node
        # does not change the children or parent list
        return Node(self.numInputs, weightsCopy, biasesCopy, self.layerNumber, self.nodeInLayer,
                    self.parents, self.children, self.activation, self.fullUpdate)

    def asDict(self) -> dict[str, list[float]]:
        """
        :return: a dictionary representation of the node
        """

        # keys are weights and biases
        return {"Weights": self.weights, "Biases": self.biases}


def sigmoidActivationFunction(sigmoid_value: float) -> Callable[[float], float]:
    return lambda value: 1 / (1 + e ** max(min(-sigmoid_value * value, 25), -25))


def reluActivationFunction() -> Callable[[float], float]:
    return lambda value: max(0, value)


def gradientUpdateNoActivation(learnRate: float = 0.002, gradient: list[tuple[float, float]] = None) \
        -> Callable[[Node, list[float], float], None]:

    def update(node: Node, inputs: list[float], goal: float):
        length = len(inputs)

        for index, (weight, bias, num) in enumerate(zip(node.weights, node.biases, inputs)):
            if gradient is None:
                weightChange, biasChange = gradientHelperNoActivation(weight, bias, num, goal / length)
            else:
                weightChange, biasChange = gradient[index]

            weightChange *= learnRate
            biasChange *= learnRate

            node.weights[index] -= weightChange
            node.biases[index] -= biasChange

    return update


def gradientHelperNoActivation(weight: float, bias: float, num: float, goal: float) -> tuple[float, float]:
    # this is a precalculated gradient vector function
    # a gradient is just a vector of partial derivatives of the loss function
    # with respect to weight and bias
    # this gives you a vector tangent to the 3d loss function

    partialDerivativeRespectWeight = 2 * num * (weight * num + bias - goal)
    partialDerivativeRespectBias = 2 * (bias + weight * num - goal)

    return partialDerivativeRespectWeight, partialDerivativeRespectBias


def gradientUpdateWithSigmoidActivation(learnRate: float = 25, sigmoid_value: float = 5,
                                        gradient: list[tuple[float, float]] = None) -> Callable[[Node], None]:
    def update(node: Node, inputs: list[float], goal: float):
        length = len(inputs)

        for index, (weight, bias, num) in enumerate(zip(node.weights, node.biases, inputs)):
            if gradient is None:
                weightChange, biasChange = gradientHelperWithSigmoidActivation(
                    weight, bias, num, goal / length, sigmoid_value, node.numInputs)
            else:
                weightChange, biasChange = gradient[index]

            weightChange *= learnRate
            biasChange *= learnRate

            node.weights[index] -= weightChange
            node.biases[index] -= biasChange

    return update


def gradientHelperWithSigmoidActivation(weight: float, bias: float, num: float, goal: float, sigmoid_value: float,
                                        numInputs: int) -> tuple[float, float]:
    # this is a precalculated gradient vector function
    # a gradient is just a vector of partial derivatives of the loss function
    # with respect to weight and bias
    # this gives you a vector tangent to the 3d loss function

    a = weight * num + bias
    b = e ** (-a * sigmoid_value) + 1
    c = 1 / (b * numInputs)
    # d = (c - goal) ** 2

    partialDerivativeRespectBias = 2 * sigmoid_value * b * (c - goal) / (b ** 2 * numInputs)
    partialDerivativeRespectWeight = partialDerivativeRespectBias * num

    return partialDerivativeRespectWeight, partialDerivativeRespectBias


def mutateUpdate(maxMutation: float = 0.5) -> Callable[[Node, list[float], float], None]:
    def update(node: Node, _: Any, __: Any):
        for index in range(node.numInputs):
            change = uniform(-maxMutation, maxMutation)
            node.biases[index] *= (1 - change)

            change = uniform(-maxMutation, maxMutation)
            node.weights[index] *= (1 - change)

    return update
