from Population import Population
from random import uniform


def main():
    # setting parameters
    numIndividuals = 15
    numInputs = 5
    nodesInLayer = 15
    numLayers = 5

    # number of generations this would run
    numIterations = 25

    # creating a new population with the given parameters
    # key word arguments are necessary here (to prevent mixing up the numbers)
    population = Population(numIndividuals=numIndividuals, numInputs=numInputs, nodesInLayer=nodesInLayer,
                            numLayers=numLayers)

    # creates a random goal
    # the maximum goal the network can output is the amount of nodes (each node outputs from 0-1)
    goal = uniform(0, nodesInLayer)

    # printing the goal
    print(f"Goal is: {goal}\n")

    # creating a random input
    # in a real AI, this would be observed through some means or dataset
    inputs = [uniform(-5, 5) for _ in range(numInputs)]

    # calls evolve to create the best set of weights
    population.evolve(inputs, goal, numIterations)

    # there should be some way to store this
    # likely, this would write to a json file


if __name__ == '__main__':
    main()
