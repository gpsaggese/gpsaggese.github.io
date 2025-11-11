import json
import re
from pyspark import SparkContext

# A hack to avoid having to pass 'sc' around
dummyrdd = None
def setDefaultAnswer(rdd): 
	global dummyrdd
	dummyrdd = rdd

def task1(amazonInputRDD):
        return dummyrdd

def task2(amazonInputRDD):
        return dummyrdd

def task3(amazonInputRDD):
        return dummyrdd

def task4(logsRDD):
        return dummyrdd

def task5_flatmap(x):
        return []

def task6(playRDD):
        return dummyrdd

def task7_flatmap(x):
        return []

def task8(nobelRDD):
        return dummyrdd

def task9(logsRDD, l):
        return dummyrdd

def task10(bipartiteGraphRDD):
        return dummyrdd
