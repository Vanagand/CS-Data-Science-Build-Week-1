import pandas as pd
import numpy as np
from random import randint
import collections

# class Model:
"""
creates list with top n recommended strains.

Paramaters
__________

request: dictionary (json object)
    list of user's desired effects listed in order of user ranking.
    {
        "effects":[],
        "negatives":[],
        "ailments":[]
    }
n: int, optional
    number of recommendations to return, default 10.

Returns
_______

list_strains: python list of n recommended strains.

Mehthods
________

categoricalEncoder(data):
"""
# def categoricalEncoder(data):
#     self._category = []
#     for i in range(len(data)):
#         if data[i][-1]
class inputParse():
    pass


class NaiveBayesClassifier():
    def __init__(self, data):
        self.data = data

    # Input decorator.   
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, x):
        print("Evaluating...")
        if x != None and isinstance(x, list): # pd.DataFrame
            self._data = x
        else:
            raise ValueError("Please enter a valid dataset.")

    # Mathematical operations.
    @property
    def piConstant(self):
        return 3.14159265359
    @property
    def eConstant(self):
        return 2.71828182845
    def meanFunction(self, x):
        return sum(x) / float(len(x))
    def sqrtFunction(self, x):
        return x**(1 / 2)
    def expFunction(self, x):
        return self.eConstant**x
    def stdevFunction(self, x):
        average = self.meanFunction(x)
        variance = sum([(x - average)**2 for x in x]) / float(len(x) - 1)
        return self.sqrtFunction(variance)

    def dataSummary(self, dataset=[], internal=False):
        if internal == True:
            self.var_dataSummary = [(self.meanFunction(x), self.stdevFunction(x), len(x)) for x in zip(*self._data)]
            del(self.var_dataSummary[-1])
            return self.var_dataSummary
        elif internal == False:
            data_summary = [(self.meanFunction(x), self.stdevFunction(x), len(x)) for x in zip(*dataset)]
            return data_summary
    
    @property
    def __summary__(self):
        print(f"A mean, standard deviation, and length summary of feature data.\n")
        n = 0
        for i in self.var_dataSummary:
            print("feature {}: ({:.3f}, {:.3f}, {})".format(n, *i))
            n += 1

    def classSeparate(self):
        classDictionary = {}
        for i in range(len(self.data)):
            vector = self.data[i]
            class_value = vector[-1]
            if (class_value not in classDictionary):
                classDictionary[class_value] = []
            classDictionary[class_value].append(vector)
        return classDictionary

    def classSummary(self):
        classSeparate = self.classSeparate()
        classSummary = {}
        for class_value, rows in classSeparate.items():
            classSummary[class_value] = self.dataSummary(dataset=rows, internal=True)
        return classSummary


class GaussianNB(NaiveBayesClassifier):
    def __init__(self, data):
        self.data = data
        self._model_info = "A naive model based on Bayes’ Theorem capable of handling continuous data." \
        "It is assumed that the continuous values within their classes are distributed according to a normal " \
        "(or Gaussian) distribution."
        
    @property
    def about(self):
        return self._model_info
        
    def GaussianProbability(self, data, mean, stdev):
        exponent = self.expFunction(-((data - mean)**2 / (2 * stdev**2)))
        return (1 / (self.sqrtFunction(2 * self.piConstant) * stdev)) * exponent

    def classProbability(self, summary, vector): #>>>
        probability = {}
        # self.var_probability = {}
        for class_value, class_summary in summary.items():
            probability[class_value] = 1
        for i in range(len(class_summary)):
            mean, stdev, _ = class_summary[i]
            probability[class_value] *= self.GaussianProbability(vector[i], mean, stdev)
        return probability

    def classPrediction(self, summary, vector): #>>>
        probability = self.classProbability(summary, vector)
        bestLabel, bestProbability = None, -1
        for class_value, class_probability in probability.items():
            if bestLabel is None or class_probability > bestProbability:
                bestProbability = class_probability
                bestLabel = class_value
            return bestLabel

    def getPrediction(self, summary, vectorTest): #>>>
        prediction = []
        for i in range(len(vectorTest)):
            result = self.classPrediction(summary, vectorTest[i])
            prediction.append(result)
        return prediction

    def getAccuracy(self, vectorTest, predictions):
        correct = 0
        for x in range(len(vectorTest)):
            if vectorTest[x][-1] == predictions:
                correct += 1
            return (correct/float(len(vectorTest)))*100.0
 
def main():
    pass


if __name__ == "__main__":
    data01 = [[randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 2)]]
    data02 = [[randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 2)],
            [randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 2)]]
    test01 = [[randint(0, 10), randint(0, 10)],
            [randint(0, 10), randint(0, 10)],
            [randint(0, 10), randint(0, 10)]]
    test02 = [[randint(0, 10), randint(0, 10), randint(0, 10)],
            [randint(0, 10), randint(0, 10), randint(0, 10)],
            [randint(0, 10), randint(0, 10), randint(0, 10)]]
    
    model = GaussianNB(data02)
    print(model.expFunction(5))
    print(model.about)
    
    print("\n")
    print("--- Data Summary ---")
    dataSummary = model.dataSummary(internal=True)
    print(dataSummary)
    # print(model.__summary__)
    # print("\n"*2)
    # print(model.dataSummary())
    
    print("\n")
    print("--- Seperating  Classes ---")
    classSeparate = model.classSeparate()
    print(classSeparate)
    
    print("\n")
    orderedDictSeparate = collections.OrderedDict(sorted(classSeparate.items()))
    for label, row, in orderedDictSeparate.items():
        print(label)
        for tuple in row:
            print(tuple)   
            
    print("\n")
    print("--- Class Summary ---")
    print(model.classSummary())
    # classSummary = model.classSummary()
    # print(classSummary)
    
    print("\n")
    classSummary = model.classSummary()
    # for label in classSummary:
    #     print(label)
    #     for row in classSummary[label]:
    #         print(row)
    orderedDictSummary = collections.OrderedDict(sorted(classSummary.items()))
    for label, row, in orderedDictSummary.items():
        print(label)
        for tuple in row:
            print(tuple)   
    
    
    print("\n")
    print("--- Testing Gaussian Function")
    print(model.GaussianProbability(1.0, 1.0, 1.0))
    print(model.GaussianProbability(2.0, 1.0, 1.0))
    print(model.GaussianProbability(0.0, 0.0, 1.0))

    print("\n")
    print("--- Class Probability ---")
    print(test02)
    classSummary = model.classSummary()
    print(model.classProbability(classSummary, test02[0]))
    
    # getPredictions = model.getPredictions(classSummary, test02)
    
    print("\n")
    print("--- Class Prediction ---")
    print(model.classPrediction(classSummary, test02[0]))
    # predict = model.predict(classSummary, test02)
    
    # # print("\n")
    # print("--- get Prediction ---")
    classPrediction = model.getPrediction(classSummary, test02)
    print(classPrediction)
    # orderedDictPrediction = collections.OrderedDict(sorted(classPrediction.items()))
    # for label, row, in orderedDictPrediction.items():
    #     print(label, row*100)
    
    accuracy = model.getAccuracy(test02, classPrediction)
    print(accuracy)
    # # for label in summary:
    #     print(label)
    #     for row in summar[label]:
    #         print(row)
            
            
    # print(model.getPrediction(summary, test))
    # print("\n")
    # print("--- Prediction Accuracy ---")
    # print(model.getAccuracy())
    # print("\n")
    
    # model.__datasummary__
    
    main()
