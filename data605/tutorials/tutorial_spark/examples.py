from pyspark import SparkContext
import re
from functions import *

sc = SparkContext("local", "Simple App")

## Load data into RDDs
playRDD = sc.textFile("datafiles/play.txt")
logsRDD = sc.textFile("datafiles/NASA_logs_sample.txt")
amazonInputRDD = sc.textFile("datafiles/amazon-ratings.txt")
nobelRDD = sc.textFile("datafiles/prize.json")

## The following converts the amazonInputRDD into 2-tuples with integers
amazonBipartiteRDD = amazonInputRDD.map(lambda x: x.split(" ")).map(lambda x: (x[0], x[1])).distinct()

## Word Count Application 
def example1(playRDD):
	counts = playRDD.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
	print counts.sortByKey().take(100)

## The following shows how to find all unique hosts making requests -- returns an RDD
def extractHost(logline):
	match = re.search('^(\S+) ', logline)
	if match is not None:
		return match.group(1)
	else:
		return None
def example2(logsRDD):
	hosts = logsRDD.map(extractHost).filter(lambda x: x is not None)
	hosts_distinct = hosts.distinct()
	print "In the {} log entries, there are {} unique hosts".format(hosts.count(), hosts_distinct.count())

## The following shows how to find the list of neighbors for each node in a graph, and also the degree of each node
def example3(graphRDD):
	# To compute degrees, we first use a map to output (u, 1) for every edge (u, v). Then we can compute the degrees using a reduceByKey and a sum reducer
	degrees = graphRDD.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x + y)

	# Using groupByKey instead gives us a list of neighbors, but the list is returned as a pyspark.resultiterable.ResultIterable object
	neighbors_1 = graphRDD.groupByKey()

	# We can convert that into a list of neighbors as follows
	neighbors_2 = graphRDD.groupByKey().mapValues(list)

	print degrees.take(10)
	print neighbors_2.take(10)

example1(playRDD)
example2(logsRDD)
example3(amazonBipartiteRDD)
