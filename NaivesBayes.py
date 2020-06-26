import pandas as pd
import numpy as np
from random import randint
import collections
import math
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# class Naive Bayes Classifier:
"""
Naive Bayes Classifier (NaiveBayesClassifier)
General class that takes list input and calculate a Gaussian naïve Bayes’ classification.

Paramaters
__________

request: data (list)
    list of user input data.


Mehthods
________

classSeparate(self.data)

dataSummary(self.data)

classSummary(self.data)

classProbability(self.data, summary, vector)

GaussianProbability(self.data, data, mean, stdev)
"""
class inputParse():
    pass

# # Load a CSV file
# def load_csv(filename):
#   pass
 
# # Convert string to float
# def string_to_float(dataset, column):
#   pass
 
# # Convert string to integer
# def string_to_integer(dataset, column):
#   pass

class NaiveBayesClassifier():
    """"""
    def __init__(self, data):
        self.data = data

    # Input decorator.   
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, x):
        print(f"Evaluating...\n")
        if x != None and isinstance(x, list): # pd.DataFrame
            self._data = x
        else:
            raise ValueError("Please enter a valid dataset.")


    # Mathematical operations.
    @property # ratio of a circle's circumference to its diameter.
    def piConstant(self):
        return 3.14159265359
    @property # base of a natural logarithm.
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


    def classSeparate(self, data):
        """Creates a dictionary of feature class values.
        
        Request: data (internal list)
            user input list.

        Returns: var_classDictionary (dictionary)
            dictionary with class value as keys pair. 
        """
        class_dictionary = {}
        for i in range(len(data)):
            vector = data[i]
            class_value = vector[-1]
            if (class_value not in class_dictionary):
                class_dictionary[class_value] = []
            class_dictionary[class_value].append(vector)
        return class_dictionary
    @property
    def __classdictionary__(self):
        orderedDictclassSeparate = collections.OrderedDict(sorted(self.classSeparate().items()))
        for label, row in orderedDictclassSeparate.items():
            print("class {}".format(label))
            for array in row:
                print("{}".format(array))


    def dataSummary(self, data):
        """Creates a list summary of the mean, standard deviation, and lenght of 'n' features.
        
        Request: data (internal list)
            user input list.

        Returns: var_dataSummary (list)
            [
                self.meanFunction(),
                self.stdevFunction(),
                self.len()
            ]
        """
        summary = [(self.meanFunction(x), self.stdevFunction(x), len(x)) for x in zip(*data)]
        del(summary[-1])
        return summary
    @property
    def __datasummary__(self):
        for row in self.dataSummary(self._data):
            print(row)


    def classSummary(self, data):
        """Creates a dictionary summary of the mean, standard deviation, and lenght of 'n' features.
        
        Support function.

        Returns: var_classSummary (dictionary)
            dictionary summary with class value as keys pair. 
        """
        class_dictionary = self.classSeparate(data)
        summary = {}
        for class_value, class_feature in class_dictionary.items():
            summary[class_value] = self.dataSummary(class_feature)
        return summary
    @property
    def __classsummary__(self):
        orderedDictclassSummary = collections.OrderedDict(sorted(classSummary.items()))
        for label, row in orderedDictclassSummary.items():
            print("class {}".format(label))
            for array in row:
                print("{}".format(array))


    def classProbability(self, summary, vector):
        """Creates a dictionary of probabilities for each class..
        
        Request: vector (list)
            observation.

        Returns: var_classProbability (dictionary)
            dictionary or probabilities for each class.
        """
        numRows = sum([summary[label][0][2] for label in summary])
        probability = {}
        for class_value, class_summary in summary.items():
            probability[class_value] = summary[class_value][0][2]/float(numRows)
            for i in range(len(class_summary)):
                mean, stdev, _ = class_summary[i]
                probability[class_value] *= self.GaussianProbability(vector[i], mean, stdev)
        return probability


    def _model(self, train, test):
        summary = self.classSummary(train)
        prediction = []
        for row in test:
            output = self._predict(summary, row)
            prediction.append(output)
        return(prediction)


    def _predict(self, summary, vector):
        probability = self.classProbability(summary, vector)
        best_label, best_probability = None, -1
        for class_value, class_probability in probability.items():
            if best_label is None or class_probability > best_probability:
                best_probability = class_probability
                best_label = class_value
        return best_label


class GaussianNBmodel(NaiveBayesClassifier):
    def __init__(self, data):
        self.data = data
        self._model_info = "A naive model based on Bayes’ Theorem capable of handling continuous data." \
        "It is assumed that the continuous values within their classes are distributed according to a normal " \
        "(or Gaussian) distribution."
    @property
    def about(self):
        return self._model_info


    def GaussianProbability(self, data, mean, stdev):
        """Creates a gaussian probability distribution float.
        
        Request: data (internal list)
            user input list.
        mean: float
        stdev: float
            [
                self.data,
                self.meanFunction(),
                self.stdevFunction()
            ]

        Returns: float
            float value of the gaussian probability of events.
        """
        exponent = self.expFunction(-((data - mean)**2 / (2 * stdev**2)))
        return (1 / (self.sqrtFunction(2 * self.piConstant) * stdev)) * exponent
    @property
    def __gaussianprobability__(self):
        print("{:.3f}".format)


    # def classProbability(self, summary, vector): #>>>
    #     probability = {}
    #     # self.var_probability = {}
    #     for class_value, class_summary in summary.items():
    #         probability[class_value] = 1
    #     for i in range(len(class_summary)):
    #         mean, stdev, _ = class_summary[i]
    #         x =  vector[i]
    #         probability[class_value] *= self.GaussianProbability(x, mean, stdev)
    #     return probability


    # def classPrediction(self, summary, vector): #>>>
    #     probability = self.classProbability(summary, vector)
    #     bestLabel, bestProbability = None, -1
    #     for class_value, class_probability in probability.items():
    #         if bestLabel is None or class_probability > bestProbability:
    #             bestProbability = class_probability
    #             bestLabel = class_value
    #         return bestLabel


    # def getPrediction(self, summary, vectorTest): #>>>
    #     prediction = []
    #     for i in range(len(vectorTest)):
    #         result = self.classPrediction(summary, vectorTest[i])
    #         prediction.append(result)
    #     return prediction


    # def getAccuracy(self, vectorTest, predictions): #>>>
    #     correct = 0
    #     for x in range(len(vectorTest)):
    #         if vectorTest[x][-1] == predictions:
    #             correct += 1
    #         return (correct/float(len(vectorTest)))*100.0

 
def main():
    pass


if __name__ == "__main__":
    dataIrisFull = [[5.1,3.5,1.4,0.2,0], [4.9,3.0,1.4,0.2,0], [4.7,3.2,1.3,0.2,0], [4.6,3.1,1.5,0.2,0], [5.0,3.6,1.4,0.2,0],
        [5.4,3.9,1.7,0.4,0], [4.6,3.4,1.4,0.3,0], [5.0,3.4,1.5,0.2,0], [4.4,2.9,1.4,0.2,0], [4.9,3.1,1.5,0.1,0],
        [5.4,3.7,1.5,0.2,0], [4.8,3.4,1.6,0.2,0], [4.8,3.0,1.4,0.1,0], [4.3,3.0,1.1,0.1,0], [5.8,4.0,1.2,0.2,0],
        [5.7,4.4,1.5,0.4,0], [5.4,3.9,1.3,0.4,0], [5.1,3.5,1.4,0.3,0], [5.7,3.8,1.7,0.3,0], [5.1,3.8,1.5,0.3,0],
        [5.4,3.4,1.7,0.2,0], [5.1,3.7,1.5,0.4,0], [4.6,3.6,1.0,0.2,0], [5.1,3.3,1.7,0.5,0], [4.8,3.4,1.9,0.2,0],
        [5.0,3.0,1.6,0.2,0], [5.0,3.4,1.6,0.4,0], [5.2,3.5,1.5,0.2,0], [5.2,3.4,1.4,0.2,0], [4.7,3.2,1.6,0.2,0],
        [4.8,3.1,1.6,0.2,0], [5.4,3.4,1.5,0.4,0], [5.2,4.1,1.5,0.1,0], [5.5,4.2,1.4,0.2,0], [4.9,3.1,1.5,0.1,0],
        [5.0,3.2,1.2,0.2,0], [5.5,3.5,1.3,0.2,0], [4.9,3.1,1.5,0.1,0], [4.4,3.0,1.3,0.2,0], [5.1,3.4,1.5,0.2,0],
        [5.0,3.5,1.3,0.3,0], [4.5,2.3,1.3,0.3,0], [4.4,3.2,1.3,0.2,0], [5.0,3.5,1.6,0.6,0], [5.1,3.8,1.9,0.4,0],
        [4.8,3.0,1.4,0.3,0], [5.1,3.8,1.6,0.2,0], [4.6,3.2,1.4,0.2,0], [5.3,3.7,1.5,0.2,0], [5.0,3.3,1.4,0.2,0],
        [7.0,3.2,4.7,1.4,1], [6.4,3.2,4.5,1.5,1], [6.9,3.1,4.9,1.5,1], [5.5,2.3,4.0,1.3,1], [6.5,2.8,4.6,1.5,1],
        [5.7,2.8,4.5,1.3,1], [6.3,3.3,4.7,1.6,1], [4.9,2.4,3.3,1.0,1], [6.6,2.9,4.6,1.3,1], [5.2,2.7,3.9,1.4,1],
        [5.0,2.0,3.5,1.0,1], [5.9,3.0,4.2,1.5,1], [6.0,2.2,4.0,1.0,1], [6.1,2.9,4.7,1.4,1], [5.6,2.9,3.6,1.3,1],
        [6.7,3.1,4.4,1.4,1], [5.6,3.0,4.5,1.5,1], [5.8,2.7,4.1,1.0,1], [6.2,2.2,4.5,1.5,1], [5.6,2.5,3.9,1.1,1],
        [5.9,3.2,4.8,1.8,1], [6.1,2.8,4.0,1.3,1], [6.3,2.5,4.9,1.5,1], [6.1,2.8,4.7,1.2,1], [6.4,2.9,4.3,1.3,1],
        [6.6,3.0,4.4,1.4,1], [6.8,2.8,4.8,1.4,1], [6.7,3.0,5.0,1.7,1], [6.0,2.9,4.5,1.5,1], [5.7,2.6,3.5,1.0,1],
        [5.5,2.4,3.8,1.1,1], [5.5,2.4,3.7,1.0,1], [5.8,2.7,3.9,1.2,1], [6.0,2.7,5.1,1.6,1], [5.4,3.0,4.5,1.5,1],
        [6.0,3.4,4.5,1.6,1], [6.7,3.1,4.7,1.5,1], [6.3,2.3,4.4,1.3,1], [5.6,3.0,4.1,1.3,1], [5.5,2.5,4.0,1.3,1],
        [5.5,2.6,4.4,1.2,1], [6.1,3.0,4.6,1.4,1], [5.8,2.6,4.0,1.2,1], [5.0,2.3,3.3,1.0,1], [5.6,2.7,4.2,1.3,1],
        [5.7,3.0,4.2,1.2,1], [5.7,2.9,4.2,1.3,1], [6.2,2.9,4.3,1.3,1], [5.1,2.5,3.0,1.1,1], [5.7,2.8,4.1,1.3,1],
        [6.3,3.3,6.0,2.5,2], [5.8,2.7,5.1,1.9,2], [7.1,3.0,5.9,2.1,2], [6.3,2.9,5.6,1.8,2], [6.5,3.0,5.8,2.2,2],
        [7.6,3.0,6.6,2.1,2], [4.9,2.5,4.5,1.7,2], [7.3,2.9,6.3,1.8,2], [6.7,2.5,5.8,1.8,2], [7.2,3.6,6.1,2.5,2],
        [6.5,3.2,5.1,2.0,2], [6.4,2.7,5.3,1.9,2], [6.8,3.0,5.5,2.1,2], [5.7,2.5,5.0,2.0,2], [5.8,2.8,5.1,2.4,2],
        [6.4,3.2,5.3,2.3,2], [6.5,3.0,5.5,1.8,2], [7.7,3.8,6.7,2.2,2], [7.7,2.6,6.9,2.3,2], [6.0,2.2,5.0,1.5,2],
        [6.9,3.2,5.7,2.3,2], [5.6,2.8,4.9,2.0,2], [7.7,2.8,6.7,2.0,2], [6.3,2.7,4.9,1.8,2], [6.7,3.3,5.7,2.1,2],
        [7.2,3.2,6.0,1.8,2], [6.2,2.8,4.8,1.8,2], [6.1,3.0,4.9,1.8,2], [6.4,2.8,5.6,2.1,2], [7.2,3.0,5.8,1.6,2],
        [7.4,2.8,6.1,1.9,2], [7.9,3.8,6.4,2.0,2], [6.4,2.8,5.6,2.2,2], [6.3,2.8,5.1,1.5,2], [6.1,2.6,5.6,1.4,2],
        [7.7,3.0,6.1,2.3,2], [6.3,3.4,5.6,2.4,2], [6.4,3.1,5.5,1.8,2], [6.0,3.0,4.8,1.8,2], [6.9,3.1,5.4,2.1,2],
        [6.7,3.1,5.6,2.4,2], [6.9,3.1,5.1,2.3,2], [5.8,2.7,5.1,1.9,2], [6.8,3.2,5.9,2.3,2], [6.7,3.3,5.7,2.5,2],
        [6.7,3.0,5.2,2.3,2], [6.3,2.5,5.0,1.9,2], [6.5,3.0,5.2,2.0,2], [6.2,3.4,5.4,2.3,2], [5.9,3.0,5.1,1.8,2]
    ]
    dataIrisFeature = [ [5.1,3.5,1.4,0.2], [4.9,3.0,1.4,0.2], [4.7,3.2,1.3,0.2], [4.6,3.1,1.5,0.2], [5.0,3.6,1.4,0.2], 
        [5.4,3.9,1.7,0.4], [4.6,3.4,1.4,0.3], [5.0,3.4,1.5,0.2], [4.4,2.9,1.4,0.2], [4.9,3.1,1.5,0.1], [5.4,3.7,1.5,0.2],
        [4.8,3.4,1.6,0.2], [4.8,3.0,1.4,0.1], [4.3,3.0,1.1,0.1], [5.8,4.0,1.2,0.2], [5.7,4.4,1.5,0.4], [5.4,3.9,1.3,0.4],
        [5.1,3.5,1.4,0.3], [5.7,3.8,1.7,0.3], [5.1,3.8,1.5,0.3], [5.4,3.4,1.7,0.2], [5.1,3.7,1.5,0.4], [4.6,3.6,1.0,0.2],
        [5.1,3.3,1.7,0.5], [4.8,3.4,1.9,0.2], [5.0,3.0,1.6,0.2], [5.0,3.4,1.6,0.4], [5.2,3.5,1.5,0.2], [5.2,3.4,1.4,0.2],
        [4.7,3.2,1.6,0.2], [4.8,3.1,1.6,0.2], [5.4,3.4,1.5,0.4], [5.2,4.1,1.5,0.1], [5.5,4.2,1.4,0.2], [4.9,3.1,1.5,0.1],
        [5.0,3.2,1.2,0.2], [5.5,3.5,1.3,0.2], [4.9,3.1,1.5,0.1], [4.4,3.0,1.3,0.2], [5.1,3.4,1.5,0.2], [5.0,3.5,1.3,0.3],
        [4.5,2.3,1.3,0.3], [4.4,3.2,1.3,0.2], [5.0,3.5,1.6,0.6], [5.1,3.8,1.9,0.4], [4.8,3.0,1.4,0.3], [5.1,3.8,1.6,0.2],
        [4.6,3.2,1.4,0.2], [5.3,3.7,1.5,0.2], [5.0,3.3,1.4,0.2], [7.0,3.2,4.7,1.4], [6.4,3.2,4.5,1.5], [6.9,3.1,4.9,1.5],
        [5.5,2.3,4.0,1.3], [6.5,2.8,4.6,1.5], [5.7,2.8,4.5,1.3], [6.3,3.3,4.7,1.6], [4.9,2.4,3.3,1.0], [6.6,2.9,4.6,1.3],
        [5.2,2.7,3.9,1.4], [5.0,2.0,3.5,1.0], [5.9,3.0,4.2,1.5], [6.0,2.2,4.0,1.0], [6.1,2.9,4.7,1.4], [5.6,2.9,3.6,1.3],
        [6.7,3.1,4.4,1.4], [5.6,3.0,4.5,1.5], [5.8,2.7,4.1,1.0], [6.2,2.2,4.5,1.5], [5.6,2.5,3.9,1.1], [5.9,3.2,4.8,1.8],
        [6.1,2.8,4.0,1.3], [6.3,2.5,4.9,1.5], [6.1,2.8,4.7,1.2], [6.4,2.9,4.3,1.3], [6.6,3.0,4.4,1.4], [6.8,2.8,4.8,1.4],
        [6.7,3.0,5.0,1.7], [6.0,2.9,4.5,1.5], [5.7,2.6,3.5,1.0], [5.5,2.4,3.8,1.1], [5.5,2.4,3.7,1.0], [5.8,2.7,3.9,1.2],
        [6.0,2.7,5.1,1.6], [5.4,3.0,4.5,1.5], [6.0,3.4,4.5,1.6], [6.7,3.1,4.7,1.5], [6.3,2.3,4.4,1.3], [5.6,3.0,4.1,1.3],
        [5.5,2.5,4.0,1.3], [5.5,2.6,4.4,1.2], [6.1,3.0,4.6,1.4], [5.8,2.6,4.0,1.2], [5.0,2.3,3.3,1.0], [5.6,2.7,4.2,1.3],
        [5.7,3.0,4.2,1.2], [5.7,2.9,4.2,1.3], [6.2,2.9,4.3,1.3], [5.1,2.5,3.0,1.1], [5.7,2.8,4.1,1.3], [6.3,3.3,6.0,2.5],
        [5.8,2.7,5.1,1.9], [7.1,3.0,5.9,2.1], [6.3,2.9,5.6,1.8], [6.5,3.0,5.8,2.2], [7.6,3.0,6.6,2.1], [4.9,2.5,4.5,1.7],
        [7.3,2.9,6.3,1.8], [6.7,2.5,5.8,1.8], [7.2,3.6,6.1,2.5], [6.5,3.2,5.1,2.0], [6.4,2.7,5.3,1.9], [6.8,3.0,5.5,2.1],
        [5.7,2.5,5.0,2.0], [5.8,2.8,5.1,2.4], [6.4,3.2,5.3,2.3], [6.5,3.0,5.5,1.8], [7.7,3.8,6.7,2.2], [7.7,2.6,6.9,2.3],
        [6.0,2.2,5.0,1.5], [6.9,3.2,5.7,2.3], [5.6,2.8,4.9,2.0], [7.7,2.8,6.7,2.0], [6.3,2.7,4.9,1.8], [6.7,3.3,5.7,2.1],
        [7.2,3.2,6.0,1.8], [6.2,2.8,4.8,1.8], [6.1,3.0,4.9,1.8], [6.4,2.8,5.6,2.1], [7.2,3.0,5.8,1.6], [7.4,2.8,6.1,1.9],
        [7.9,3.8,6.4,2.0], [6.4,2.8,5.6,2.2], [6.3,2.8,5.1,1.5], [6.1,2.6,5.6,1.4], [7.7,3.0,6.1,2.3], [6.3,3.4,5.6,2.4],
        [6.4,3.1,5.5,1.8], [6.0,3.0,4.8,1.8], [6.9,3.1,5.4,2.1], [6.7,3.1,5.6,2.4], [6.9,3.1,5.1,2.3], [5.8,2.7,5.1,1.9],
        [6.8,3.2,5.9,2.3], [6.7,3.3,5.7,2.5], [6.7,3.0,5.2,2.3], [6.3,2.5,5.0,1.9], [6.5,3.0,5.2,2.0], [6.2,3.4,5.4,2.3],
        [5.9,3.0,5.1,1.8]
    ]
    dataIrisTarget = [ [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
        [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
        [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
        [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
        [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [2], [2], [2], [2], [2], [2], [2],
        [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2],
        [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2]
    ]
    dataIris_test = [
        [5.1,3.5,1.4,0.2],
        [7.0,3.2,4.7,1.4],
        [6.3,3.3,6.0,2.5],
    ]
    
    model = GaussianNBmodel(dataIrisFull)
    mathInt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Function test!
    #>>> Data Handling 
    ##### self.classSeparate() #####                                   #> PASS!
    # print(f"- instance test - ")
    # classSeperate = model.classSeparate(dataIrisFull)
    # print(classSeperate)
    # print(f"\n- method test - ")
    # print(model.classSeparate(dataIrisFull))
    # print(f"\n- sort - ")
    # orderedDictclassSeparate = collections.OrderedDict(sorted(classSeperate.items()))
    # for label, row in orderedDictclassSeparate.items():
    #     print("class {}".format(label))
    #     for array in row:
    #         print("{}".format(array))
    # print(f"\n- decorator test - ")
    # model.__classdictionary__
    
    #>>> Data Summary
    ##### mathematical operations #####                                #> PASS!
    # modelMean = model.meanFunction(mathInt)
    # print(f"modelMean")
    # modelSqrt = model.sqrtFunction(16)
    # print(f"modelSqrt")
    # modelExp = model.expFunction(1)
    # print("{:.3f}".format(modelExp))
    # modelStdev = model.stdevFunction(mathInt)
    # print("{:.3f}".format(modelStdev))
    
    ##### self.dataSummary() #####                                     #> PASS!
    # print(f"- instance test-  ")
    # dataSummary = model.dataSummary(dataIrisFull)
    # print(dataSummary)
    # print(f"\n- method test - ")
    # print(model.dataSummary(dataIrisFull))
    # print(f"\n- sort - ")
    # for row in dataSummary:
    #     print(row)
    # print(f"\n- decorator test - ")
    # model.__datasummary__
    
    #>>> Class Summary
    ##### self.classSummary() #####                                    #> PASS!
    # print(f"- instance test - ")
    # classSummary = model.classSummary(dataIrisFull)
    # print(classSummary)
    # print(f"\n- method test - ")
    # print(model.classSummary(dataIrisFull))
    # print(f"\n- sort - ")
    # orderedDictclassSummary = collections.OrderedDict(sorted(classSummary.items()))
    # for label, row in orderedDictclassSummary.items():
    #     print("class {}".format(label))
    #     for array in row:
    #         print("{}".format(array))
    # print(f"\n- decorator test - ")
    # model.__classsummary__
    
    #>>> Gaussian Function                                             #> PASS!
    #### GaussianNB #####
    # print(f"- instance test - ")
    # print(model.GaussianProbability(1.0, 1.0, 1.0))
    # print(model.GaussianProbability(2.0, 1.0, 1.0))
    # print(model.GaussianProbability(0.0, 0.0, 1.0))
    # print(f"\n- decorator test - ")
    # modelGaussianProbability = model.GaussianProbability(1.0, 1.0, 1.0)
    # print("{:.3f}".format(modelGaussianProbability))
    # modelGaussianProbability = model.GaussianProbability(1.0, 2.0, 2.0)
    # print("{:.3f}".format(modelGaussianProbability))
    
    #>>> Class Probability                                             #> PASS!
    ##### self.classProbability() #####
    # classSummary = model.classSummary(dataIrisFull)
    # print(classSummary)
    # print(f"- instance test - ")
    # print("\n")
    # classProbability = model.classProbability(classSummary, dataIrisFull[-1])
    # print(dataIrisFull[-1])
    # print(classProbability)
    # # print(f"\n- method test - ")
    # print("\n")
    # print(dataIrisFull[1])
    # print(model.classProbability(classSummary, dataIrisFull[1]))













    #### MODEL FIT TESTING GROUND ####
    dataframeIris = pd.DataFrame(dataIrisFull)
    #Setup X and y data
    X_data = dataframeIris.iloc[:,0:4]
    y_labels = dataframeIris.iloc[:,4]
    # print(X_data)
    # print(y_labels)

    # model._model(dataIrisFeature, dataIrisTarget)
    
    modelFit = model.classSummary(dataIrisFull)
    # targetTest = [[5,5,5,5],[5,5,5,5]]
    # modelClass = model._predict(modelFit, targetTest)
    print(model._predict(modelFit, [5.1,3.5,1.4,0.2]))
    print(model._predict(modelFit, [7.0,3.2,4.7,1.4]))
    print(model._predict(modelFit, [6.3,3.3,6.0,2.5]))
    print(model._predict(modelFit, [0,0,0,0]))
    print(model._predict(modelFit, [5,5,5,5]))
    print(model._predict(modelFit, [10,10,10,10]))

    #Fit model
    # clf  = GaussianNB()
    # clf.fit(X_data,y_labels)
    # print(clf.predict([[5,5,5,5]]))
    # print(clf.score(X_data,y_labels))
    
    # @_DEPRECIATED
    # @property
    # def __summary__(self):
    #     print(f"A mean, standard deviation, and length summary of feature data.\n")
    #     n = 0
    #     for i in self.var_dataSummary:
    #         print("feature {}: ({:.3f}, {:.3f}, {})".format(n, *i))
    #         n += 1