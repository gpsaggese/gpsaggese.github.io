# Project 3

In this project you will learn how to use the MapReduce computation framework, by
implementing a few relatively simple transformations of some datasets with _Hadoop_.

Remember, the power of Hadoop comes from the fact that you only implement Map and Reduce functionality that will then be able to run over an arbitrarily large cluster of machines, and over an arbitrarily large dataset, as long as that dataset is stored in HDFS.

## Datasets

You will be working with 3 different datasets:

- `amazon-ratings.txt`: This is a table that contains information about ratings a user gave to a product. The table can be seen as a graph that contains two types of nodes, one for users and one for products. A customer node is connected to a product node if he/she wrote a review for that product.

- `NASA-logs.txt`: This contains web-server [logs from NASA](http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html).

- `play.txt`: This contains lines from a Shakespeare play.

You will find all of these datasets in the `/home` directory of your docker container, and **will need to load them into _HDFS_ yourself**, as described in the tutorial document.

### Task 1: NASA-logs hosts
Write a MapReduce program that takes records from the `NASA-logs` dataset, and outputs a _list_ of "hosts" that were present in the log on each date. The dates should appear in the same format that they appear in the logs (e.g., '01/Jul/1995' and '02/Jul/1995'). _Note_ that you will need to do a fair amount of string processing in the mapper in order to extract the date and host from every input record.

Each _output_ row should look like this: `dd/mmm/yyyy [host1, host2,...]`

### Task 2: Bigrams
[Bigrams](http://en.wikipedia.org/wiki/Bigram) are simply sequences of two consecutive words. For example, the previous sentence contains the following bigrams: "Bigrams are", "are simply", "simply sequences", "sequences of", etc. Your task is to write a MapReduce program for counting the bigrams in each individual line in the `play` dataset.

The mapper should split each input record by spaces (one or more spaces, also other whitespace, such as a tab, can be treated the same as a space character), not worrying about the fact that there may be punctuation between a word and a space character (e.g. If a record is `Some , word`, then `Some ,` and `, word` are valid bigrams). Each of your bigrams should be separated by a space. If a word appears on only a single line with no other words adjacent to it, it should not appear in _any_ bigrams.
The Reducer should output records where the key is a bigram, and the value is its count.

Because you are splitting the input file by line, count all bigrams per line. You can use the type `Text` for representing the bigrams, you may implement your own `BigramWritable`, or use an existing implementation (there also exists an `ArrayWritable` class as part of Hadoop). _Note_ that only objects that implement the `Writable` interface can be written to (and therefore read from) HDFS.

Each _output_ row should look like this: `[word1,word2] <count>`

### Task 3: Join
Implement a MapReduce program to compute the pairs of customers that have bought at least one product in common in the Amazon dataset. The Amazon graph dataset can be seen as an edge table where the first entry is the source node and the second the destination node. To compute the pairs, you need to perform a _self-join_ over the Amazon graph's edge table.

In order to do that, you need to simulate having two identical views of the table. Your Mapper should take an edge as input (a record), and output _two records_ for which the join key (which is the product id) is the key, and the customer is the value. Each record output from the mapper should have some sort of identifier that signifies which of the two views the record is coming from. This identifier is so that you can check this identifier in the reducer in order to _only_ join edges from different _views_. What kind of identifier you use does not matter and should not show up in the results. The reducer should output pairs of customers with the same key.

It does not matter if both edge directions appear in the result. So if both (user1, user2) and (user2, user1) appear in the result that is fine.

- _Don't output duplicate records for the same product id_,  i.e. (user1,user2) should not appear more than once for a single join key.
- _Don't output self-edges_, e.g., (user1,user1).

Each _output_ row should look like this: `user1 user2`

## Submission Instructions
In the `project-3` directory in your individual github repository, you will find 3 different directories, one for each task. Please upload for each task in its corresponding directory the following:

- A single `.jar` executable for each task, explicitly named with the following scheme: `task-x.jar`.
- The Java `Mapper` class that was used in the executable
- The Java `Reducer` class that was used in the executable
- The Java `Driver` class that was used in the executable

We should be able to run your executable `jar` within the container and get the same result you did for each task.

# Project 4: Spark

In this project we will be using Spark to learn how to perform parallel computation
on very large datasets using a computation framework similar to MapReduce, which is
even more _general purpose_ and easier to use. Learning about MapReduce was
important to understand where progress in big data distributed processing systems
came from and to try to see where it is heading.

Instead of only having the option of implementing `map()` and `reduce()` Spark uses
the more generalized notion of _transformations_ and _actions_ over distributed
datasets, or _RDDs (Resilient Distributed Datasets)_.

With Spark, you can chain a series of transformations and actions that will then run
in parallel over a cluster (or for the purposes of this project your local machine).
This will allow you to do complex computations on your datasets a lot more easily
than chaining MapReduce jobs.

There are countless sources online to help you get a deeper understanding of Spark,
apart from the topics that will be covered in lecture.

Resources:

- Here is
  a [YouTube playlist](https://www.youtube.com/watch?v=m4pYYnY4_gU&list=PLFxgwFEQig6wWDHq3iMfjm5ZCHs_UdIp7)
  that gives a very good overview of Spark. It is relatively high-level (doesn't go
  into a lot of detail) but explains the basics quite well.
- The
  official [Spark programming guide](https://spark.apache.org/docs/2.2.0/rdd-programming-guide.html#linking-with-spark)
  is starting to mature and has a variety of examples, explanations, and the
  complete list of
  all [transformations](https://spark.apache.org/docs/2.2.0/rdd-programming-guide.html#transformations)
  and [actions](https://spark.apache.org/docs/2.2.0/rdd-programming-guide.html#actions)
  provided by Spark.

## Project Structure

For this project, we are giving you a `functions.py` file, which contains a function
definition for each individual task. We also give you an `assignment.py` file which
calls the functions you will need to write and outputs a subset of the results from
the returned RDDs. The `assignment.py` file also includes the necessary code to run
your functions, like importing the required packages and loading the RDDs from input
files. You will find these scripts in your individual project github repositories.

**All you need to do for this project is to fill in the implementations of the
functions in** `functions.py`.

Feel free to either use `lambda`s or to define named functions that are then used in
your transformation and action calls. We do recommend using `lambda`s for simple
one-liner functions as it makes the code easier to read as well as shorter.

## Docker Container Setup
Everyone should know the drill by now:
```
docker run -it --rm --name pyspark kostasxirog/cmsc642-pyspark bash
```

The above command creates a new container named `pyspark` using the custom image,
and opens a bash shell to the container. Remember, if you exit the container (
without stopping it), you can run `docker exec -it pyspark bash` to get back in.

The home directory in which you will find the `datasets` folder is located
in `/home/jovyan` in the container (this is just the user input by the original
docker image creator)

Copy your `assignment.py` and `functions.py` files from your computer to the above
directory by running `docker cp file container-name:/home/jovyan`, in order to run
your `assignment.py` script in that directory.

## Loading datasets

Spark can create distributed datasets from any storage source supported by Hadoop,
including your local file system, HDFS, as well as other data stores like Cassandra,
HBase, Amazon S3, etc. Spark supports text files, SequenceFiles, and any other
Hadoop InputFormat.

Text file RDDs can be created using SparkContext’s `textFile()` method. This method
takes a URI for the file (either a local path on the machine, or an `hdfs://`
, `s3a://`, etc. URI) and reads it as a collection of lines. Here is an example
invocation:

```python
playRDD = sc.textFile("datasets/play.txt")
```

Therefore you don't need to directly interface with HDFS for this project, since you
are loading objects from your file system into RDDs.

## Python

The above statement is in _Python_, which is the language we will be using for this
project. Python is a scripting language, so it is interpreted and therefore very
simple to write and execute programs in given your development environment is set up
properly. Thankfully, Docker containers completely remove any such environment setup
issues and give us a plug-and-play environment to write and execute our code.

For those unfamiliar with Python:

- To start a Python shell run: `python`
- To run a Python script, execute: `python script.py`

## Word Count Application example

The following set of commands (in the `python` shell using `pyspark`) does a word
count, i.e., it counts the number of times each word appears in a file `test.txt`.
Use `counts.take(5)` to see the output. Before you are able to use any `pyspark`
transformations and/or actions, you need access to the `SparkContext` object, which
you get by first importing the required package, and then running:
```python
from pyspark import SparkContext
sc = SparkContext("local", "Simple App")
```

First we need to create an RDD from a `test.txt` file.
```python
testRDD = sc.textFile("test.txt")
```

We then apply a series of transformations and actions on that RDD
```python
counts = testRDD.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
```
We start with an RDD whose elements are lines in the input text file as strings.

1. `flatMap` returns a set of words for each line in the input, with words separated
   by a space. Each object in the resulting RDD is a word.
2. `map` returns a tuple `(word, 1)` for every word in the RDD computed in step 1.
   Each object in the resulting RDD is a key-value tuple where the key is the `word`
   , and the value is `1`.
3. Finally, `reduceByKey` groups tuples in the previous RDD by key, and applies the
   input function (`a,b : a+b`) over each set of values `a,b` that are grouped
   together.

Lambda expressions are anonymous functions. Here is the same code without the use
of `lambda` functions.

```python
def split(line):
    return line.split(" ")
def generateone(word):
    return (word, 1)
def sum(a, b):
    return a + b

testRDD.flatMap(split).map(generateone).reduceByKey(sum)
```

The `lambda` representation is more compact and preferable, especially for small
functions, but for large functions it is better to separate out the function
definitions.

Note how python outputs key value tuples by returning `(key,value)`
objects. [Python tuples](https://www.tutorialspoint.com/python/python_tuples.htm)
can have an arbitrary number of elements.

## Datasets

The `assignment.py` script initializes the following RDDs:

* An RDD consisting of lines from a Shakespeare play (`play.txt`)
* An RDD consisting of lines from a log file (`NASA_logs_sample.txt`)
* An RDD consisting of 2-tuples indicating user-product ratings from an Amazon
  Dataset (`amazon-ratings.txt`)
* An RDD consisting of JSON documents pertaining to all the Noble Laureates over the
  last few years (`prize.json`)

These will be the datasets used for the following tasks.

## Tasks
There is a single function in `functions.py` for each one of the four following tasks

### Task 1

Implement a function that takes as input the `playRDD` and for each line, finds the
first word in the line, and also counts the number of words. It should then filter
the RDD by only selecting the lines where the count of words in the line is greater
than 10. The output is an RDD of key-value pairs where the key is the first word in
the line, and the value is a 2-tuple, the first element of which is the line (the
actual string) and the second is the number of words (which must be >10). The
simplest way to do this task is probably a `map` followed by a `filter`.

Note:

- Don't handle punctuation or special characters any special way. If, for example, a
  line starts with the character `'|'`, then the first word is `'|'`. Just separate
  words by a space.

**Each element in your output RDD should be:** `('firstword', ('line', number-of-words-in-line))`

### Task 2

Write _only_ the flatmap function (`task2_flatmap`) that takes in a parsed JSON
document (from `prize.json`) and returns a list of the surnames of the Nobel
Laureates. In other words, the following command should create an RDD with all the
surnames. We will use `json.loads` to parse the individual JSON objects in the
file (this is already done for you). Make sure to look at what it returns so you
know how to access the information inside the parsed JSONs (these are basically
nested dictionaries). See [here](https://docs.python.org/2/library/json.html).

```python
task2_result = nobelRDD.map(json.loads).flatMap(task2_flatmap)
```    

Note:

- **Don't** assume that `surname` always exists in the json object and is always
  non-empty

**The RDD after flatMap is called should be a list of
surnames:** `['surname_1', 'surname_2',...,'surname_n']`
(_Each element_ in the RDD will be a single surname)

### Task 3

Complete a function to calculate the _degree distribution_ of user nodes in the
Amazon graph (i.e., `amazonBipartiteRDD`, which we've already pre-transformed for
you). In other words, calculate the degree of each user node (i.e., number of
products each user has rated), and then use a `reduceByKey()` (or `groupByKey()`) to
find the number of nodes with a given degree. The output should be an RDD of
key-value pairs where the key is the degree, and the value is the number of nodes in
the graph with that degree.

**Each element in your output RDD should be:** `(degree, number-of-users-with-that-degree)`

### Task 4
On the `logsRDD` data, for two given days (provided as input to the function), use the `cogroup()` action to create the following RDD: the key of the RDD will be a host, and the value will be a 2-tuple, where the first element is a list of all URLs fetched from that host on the first day, and the second element is the list of all URLs fetched from that host on the second day. Use `filter()` to first create two RDDs from the input `logsRDD`.

Notes:

- In order to properly print out the iterables generated by `cogroup()`, use `map(lambda x : (x[0], (list(x[1][0]),list(x[1][1]))))` to turn each iterable (remember we have one iterable for each date), into a list of strings that can be printed out.
- You may find it useful to write a separate function to parse each portion of the line you need to use from the input log (e.g., one function that just returns the host, one that just returns the URL, etc.).

**Each element in your RDD should be:** `(host, ([url11,url12,...], [url21,url22,...]))`

_E.g._:
`('w20-575-20.mit.edu', (['/history/apollo/apollo-1/apollo-1-info.html', '/history/apollo/apollo-1/images/67HC33a.gif',...], ['/history/apollo/apollo-1/images/', '/history/apollo/apollo-1/images/apollo-1-crew.gif',...]))`

## Submission Instructions
Upload your `functions.py` file that has all of the code in each function filled in. Upload your file under the `project-4` directory in your individual github repository.

_We should be able to run `assignment.py`, importing **your** `functions.py` library, and obtain the same results that you were getting_. Please make sure that your submission works correctly within the container.
