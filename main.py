from GradientDescent import GradientNetwork
from random import uniform


def main():
    # setting parameters
    numInputs = 5
    nodesInLayer = 10
    numLayers = 1

    # number of generations this would run
    numIterations = 45

    # how fast this will learn
    learnRate = 0.002

    # creating a new population with the given parameters
    # key word arguments are necessary here (to prevent mixing up the numbers)
    network = GradientNetwork(numInputs=numInputs, nodesInLayer=nodesInLayer, numLayers=numLayers,
                              maxIter=numIterations, learnRate=learnRate)

    # creates a random goal
    # the maximum goal the network can output is the amount of nodes (each node outputs from 0-1)
    goal = uniform(-nodesInLayer * 3, nodesInLayer * 3)

    # printing the goal
    print(f"Goal is: {goal}")

    # creating a random input
    # in a real AI, this would be observed through some means or dataset
    inputs = [uniform(-10, 10) for _ in range(numInputs)]
    print("Inputs are:", inputs, "\n")

    # calls evolve to create the best set of weights
    network.evolveTillTolerance(inputs, goal)

    # there should be some way to store this
    # likely, this would write to a json file


if __name__ == '__main__':
    main()
