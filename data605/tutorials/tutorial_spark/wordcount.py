from pyspark import SparkContext

sc = SparkContext("local", "Simple App")

textFile = sc.textFile("README.md")

counts = textFile.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

print(counts.sortByKey().take(100))

#counts.saveAsTextFile("output")
