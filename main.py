from GradientDescent import GradientNetwork
from random import uniform


def main():
    # setting neural network parameters
    numInputs = 3
    nodesInLayer = 10
    numLayers = 1

    # setting tolerance
    tolerance = 0.1

    # maximum number of generations this would run
    maximumIterations = 1000

    # how fast this will learn
    learnRate = 0.002

    # this is how often the neural network would print the value it outputs
    printValueStep = 5

    # whether to store in a json file and the filename
    storeAsJson = False
    jsonFilename = "GradientDescentData"

    # how often to store the network data in the json
    storeAsJsonStep = 5

    # number of decimal places to print out
    decimalPlaces = 3

    # creating a new population with the given parameters
    # key word arguments are necessary here (to prevent mixing up the numbers)
    network = GradientNetwork(numInputs=numInputs, nodesInLayer=nodesInLayer, numLayers=numLayers,
                              maxIter=maximumIterations, learnRate=learnRate, decimalPlaces=decimalPlaces)

    # the minimum and maximum goals that will be given to the neural network
    minGoal = -50
    maxGoal = 50

    # creates a random goal in the range of minimum and maximum
    goal = uniform(minGoal, maxGoal)

    # printing the goal
    print(f"Goal is: {goal:.{decimalPlaces}f}")

    # creating a random input
    # in a real AI, this would be observed through some means or dataset
    inputs = [uniform(-10, 10) for _ in range(numInputs)]

    # printing the inputs
    print(f"Inputs are: {inputs}\n")

    # calls evolve to create the best set of weights and biases
    # this will evolve until it reaches 5% of the goal
    # the network will print the value every 5 iterations
    network.evolveTillTolerance(inputs, goal, tolerance, printValueStep, storeAsJson, storeAsJsonStep, jsonFilename)


if __name__ == '__main__':
    main()
