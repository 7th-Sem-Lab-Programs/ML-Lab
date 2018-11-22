import csv
import random
import math


def loadcsv(filename):
    lines = csv.reader(open(filename,"r"))
    dataset =list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def splitDataset(dataset,splitRatio):
    trainSize = int(len(dataset)*splitRatio)
    trainSet =[]
    copy = list(dataset)
    while(len(trainSet)<trainSize):
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet,copy]

def seperateByClass(dataset):
    seperated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in seperated):
            seperated[vector[-1]] = []
        seperated[vector[-1]].append(vector)
    return seperated
def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute),stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    seperated = seperateByClass(dataset)
    summaries = {}
    for classValue, instances in seperated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def calcutateProbability(x,mean,stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent

def calculateClassProbabilities(summaries,inputVector):
    probabilities ={}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue]=1
        for i in range(len(classSummaries)):
            mean,stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calcutateProbability(x,mean,stdev)
    return probabilities


def predict(summaries,inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability>bestProb :
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries,testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries,testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet,predictions):
    correct = 0
    for i  in range(len(testSet)):
        print("Expected: ",testSet[i][-1]," ,Predicted: ",predictions[i])
        if testSet[i][-1] == predictions[i]:
            correct +=1
    return (correct/(float(len(testSet))))*100.0

dataset = loadcsv("cancer.csv")
train, test = splitDataset(dataset,0.9)
print(dataset[0:30])

print("Train dataset size= %d"%(len(train)))
print("Test dataset size= %d"%(len(test)))

summaries = summarizeByClass(train)
predictions = getPredictions(summaries,test)
print("Accuracy="+str(getAccuracy(test,predictions)))


