from Genetic import Population
from random import uniform


def main():
    numInputs = 3
    numOutputs = 3

    numIndividuals = 15

    activationFunctionString = "relu"
    # activationFunctionString = "sigmoid"

    population = Population(numIndividuals=numIndividuals, numInputs=numInputs, numOutputs=numOutputs,
                            nodesInLayer=5, numLayers=5, activationFunctionString=activationFunctionString,
                            lossFunction=lossFunction)

    inputs = [uniform(-10, 10) for _ in range(numInputs)]
    numIterations = 15

    print(f"{inputs=}")

    population.evolve([uniform(-1, 1) for _ in range(numInputs)], numIterations)


def lossFunction(inputs: list[float], outputs: list[float]) -> float:
    return (sum(outputs) - 5) ** 2


if __name__ == "__main__":
    main()
