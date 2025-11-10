# Project 5 Graph Databases & Graph Analytics - due Thursday, May 10, 10:00PM

This project will revolve around _graphs_. Graphs are structures where
entities (nodes) are connected to each other via relationships
(edges). There can be one or multiple types of nodes and edges in a given
graph. There is a very large body of theoretical and mathematical work on
graph model (known as [graph theory](https://en.wikipedia.org/wiki/Graph_theory), from which there have
emerged many results that we can leverage in incorporating the graph data
model into data manipulation and analytics systems).

The graph data model is very intuitive because it can very easily be
visually represented. It also enables more efficient querying
of relationships between entities that would normally require expensive
joins in traditional relational database systems.

## Graph Databases vs Graph Analytics Engines

It is important to distinguish a graph _database_ from a graph _computation
engine or framework_. A graph _database_ (like Neo4j) is geared towards
querying a graph structure, often in real-time, in a transactional
manner. This is a setting where you also may often have _updates_ coming
into the graph database, often at a high rate. Graph databases focus on
issues related to efficiently querying such graphs in an interactive context
where graphs are often _dynamically_ changing.

On the other hand, a graph _computation_ framework (like Apache Giraph), is
an engine that is geared towards allowing users to view larger portions of
the graph (or often the entire graph) from a certain perspective, by
enabling them to write a computation that runs over a _static_ graph (that
will not change during the computation). In other words, it enables users
to write programs that perform different analyses that describe the
underlying graph data.

In this project, you will be introduced to both types of systems.

## Part 1 - Graph Databases (Neo4j)

Neo4j is one of the most widely used graph databases today. The main way
that users interact with Neo4j is through a _query language_ called
[Cypher](https://neo4j.com/developer/cypher/). Cypher is a _declarative_
query language, in that the user simply describes **what** data they want
without having to specify exactly **how** to obtain that data from the
database. Naturally, there are also interfaces to interact with neo4j
through other programs by using connector drivers. There is a neo4j driver
for [Java](https://neo4j.com/developer/java/),
[Python](https://neo4j.com/developer/python/), and other programming languages.

First, you need to pull the official neo4j docker image, and run it
within a container like so:

```
docker run  \
  --publish=7474:7474 --publish=7687:7687 \
  --volume=$HOME/neo4j/data:/data \
  --env=NEO4J_AUTH=none \
neo4j
```

The above command downloads the official neo4j docker image (if you do not
already have it), and starts a container running a neo4j server. Once the
server starts successfully, you will be to access the neo4j web application
from your host machine using a web browser at `http://localhost:7474/`.
Specifically:

- `--publish=7474:7474 --publish=7687:7687`: publishes these two ports in
  the container. This is required for neo4j to work properly within the
  container, and for the user to have access to it through the localhost.

- `--volume=$HOME/neo4j/data:/data`: Mounts the specific directory from
your host machine as a volume, and writes any data to that directory so
that it is available the next time you start the neo4j container.
<!-- If you are running docker on _Mac_ or _Linux_, `<localhost-path>` could
be `$HOME/neo4j/data`. If you are on _Windows_ `<localhost-path>` should be
any valid path you would like to use where docker will store data in your graph database. -->

- `--env=NEO4J_AUTH=none`: Removes the username and password authentication
  typically required to access the database. This is done for purposes of
  convenience for this project; typically there should definitely be an
  access control set-up that enforces different levels of access to the
  underlying data from different users.


### Datasets

You will be writing queries over two different datasets:

1. Movies dataset: [https://neo4j.com/developer/movie-database/](https://neo4j.com/developer/movie-database/). You will
load this dataset using the "quickstart guide" provided by neo4j.

2. States dataset: You can find the data for this dataset in your
individual github repository under `/project-5/states.json`

### QuickStart
_First_, we recommend reviewing the tutorial that introduces you to the
Cypher language. You can do that by running the following command into the neo4j terminal at the top of the web-application
```
:play cypher
```

_Next_, follow the Neo4j tutorial for loading in the movie graph and
executing the test queries over that graph that are pre-formulated in the
tutorial.
To access the tutorial execute the following command at the neo4j command
line that can be found at the top of the window in the neo4j
web-application.

```
:play movie-graph
```

This tutorial will guide you through loading the _movies dataset_, and
writing simple and more complex pattern matching queries.

**Optional:** `:play northwind-graph`: Guides you through how to migrate
  your data from a relational database system (like PostgreSQL) to a graph
  database like Neo4j.

<!-- Neo4j enables users to query entire elements in the graph (nodes or edges) as well as sets of specific attributes, returned as tuples like in a relational database. You could also choose to return the set of attributes of a node or edge, which will be shown as a JSON document. The neo4j web app allows for visualization of the queried graph if the information the user wants to return is a graph (a set of nodes rather than tuples of values). -->

### Tasks

**TASK 1:** On the _movies_ graph, write a cypher query to find actors born
  in 1967 and the movies they have acted in. Return the names of these
  actors, and the titles of the movies they acted in.

**Output**: Your result should have the following format; two columns: `actor-name, movie-title`. _Order_ your results by `actor-name`.

_Note_: There can obviously be multiple tuples for the same actor for
different movies they have _acted_ in. Also note that for a `Person` node
to be an actor, they need to have at least one `ACTED_IN` outgoing edge.

**TASK 2:** On the _movies_ graph, write a cypher query to return the names
  of directors who have directed a movie starring Keanu Reeves.

**Output**: Your result should have the following format; one column: `director-name`. _Order_ your results by `director-name`.

**TASK 3:** The `states.json` file we have provided (in your individual
  github repository) contains a set of vertices and edges between them,
  describing a graph over states. These edges denote the border
  relationship between states, and the `weight` property corresponds to the
  length of the border between the two states. Write a Cypher script to
  load this as a graph database in Neo4j.

_Note_: You will need to write a script that parses the `json` file and
uses it to output the appropriate commands to load in the graph into
neo4j. Feel free to use any language or any other means you would like to
create these Cypher commands from the json document.

**Output**: Please submit the set of `CREATE` statements for loading in the
  states graph.

**TASK 4:** On the _states_ graph, write a Cypher query to find which
  states border both Arizona and Colorado. Return the names of the nodes
  that satisfy the above requirement.

**Output**: Your result should  have the following format; one column: `state-name`.
_Order_ your results by `state-name`.

**TASK 5:** Find states that are 3 hops away from California, i.e., states
  that border states that borders states that border California.

_Note_: Make sure California itself is not included in the results

**Output**: Your result should have the following format; one column:
`state-name`. _Order_ your results by `state-name`.

---

## Part 2 - Graph Analytics (Apache Giraph)

For the _second_ part of this project, you will be using [Apache
Giraph](http://giraph.apache.org/) to conduct simple graph
analyses over a few small graphs.
First, start a docker container with the following command:

```
docker run --name giraph --volume $HOME:/myhome --rm --interactive --tty uwsampa/giraph-docker /etc/giraph-bootstrap.sh -bash
```

The above command will download an image that includes a fully functional
Apache Giraph installation, start a container, and also start the giraph
cluster and everything that is required for it to run (by executing the
`giraph-bootstrap.sh` script within the container). Note that this will
mount the `$HOME` directory from your local machine into the container
(mapping it to the `/myhome` directory inside the container).

### Datasets

We will be using two datasets for this part of the project (you will find
both in your individual github repository under the `project-5` directory):

1. `imdb-giraph-small-undirected.txt`: This is a portion of the imdb
dataset that was pulled out of neo4j. This is a subgraph that only includes
`:FOLLOWS :PRODUCED, :REVIEWED, :WROTE` relationships. This can be seen as
an _undirected_ graph, since all edges are bi-directional (for the purposes of
the task we will ask you to implement).

2. `states-giraph.txt`: This is the exact same states dataset that was used
in the previous part of this project, but we have transformed it into the
format we will be using to load it into Giraph.

**NOTE:** Giraph runs on top of HDFS, so any dataset you want to use with
  Giraph needs to be loaded into HDFS. To do that, run
  `$HADOOP_HOME/bin/hdfs dfs -put <file-path> /user/root/input` (we will
  use `/user/root/input` as the input directory in HDFS for storing all
  input datasets )

### Input Format

Users can implement a `VertexInputFormat` class (as well as optionally an
`EdgeInputFormat` class), that describes how to load in a certain input
text file as a graph into Giraph. For the purposes of this project and for
the sake of simplicity we will be using one of the default formats
implemented in Giraph called
`JsonLongDoubleFloatDoubleVertexInputFormat`. An example of this format is
shown here:

```
[1,0,[[2,0],[3,0],[4,0]]]
[4,0,[[1,0],[3,0]]]
...
```

Each line above has the format `[source_id,source_value,[[dest_id,
edge_value],...]]`. This format dictates the types that we set for
vertices, edges, and messages in the Giraph `compute()` function.

### Output Format

The exact same method applies for the Giraph output format; users can implement
their own output format classes that tell Giraph how to output the results
of the analysis. For this project we will be using the simple
`IdWithValueTextOutputFormat`, which outputs a line for every vertex that
looks like: `<vertex_id>   <vertex_value>`


### QuickStart Guide

#### Preparing your workspace

_On your local machine_ (not in the container), create the following
directories under your home directory:

```
mkdir $HOME/giraph-work
mkdir $HOME/giraph-work/mypackage
```

You will be working on your local machine (in
`$HOME/giraph-work/mypackage`) and you will be writing any Java files
inside that directory. Because your home directory has been mounted into the
container, you do not need to copy over any of your Java files -- they will
simply appear in the container as soon as you save them in that directory
on your local machine.

#### Running a simple Giraph Job

Change your current directory to `$HOME/giraph-work/mypackage` _on your
local machine_, and create a file called `DegreeComputation.java`, that contains
the following code:

```java
package mypackage;

import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;

import java.io.IOException;

/**
 * Simple algorithm that computes the degree using message passing
 * NOTE: giraph has a vertex.getNumEdges() method, but this acts as an example
 * of how to send messages to other vertices, as well as how to handle the received
 * messages
 */
public class DegreeComputation extends BasicComputation<
        LongWritable, DoubleWritable, FloatWritable, DoubleWritable> {
  @Override
  public void compute(Vertex<LongWritable, DoubleWritable, FloatWritable> vertex,
      Iterable<DoubleWritable> messages) throws IOException {
        int numNeigbors = 0;
        for (DoubleWritable message : messages) {
          numNeigbors++;
        }
        vertex.setValue(new DoubleWritable((double) numNeigbors));

        // Sends messages to all outgoing neighbors
        if (getSuperstep() == 0) {
          sendMessageToAllEdges(vertex, vertex.getValue());
        }
        vertex.voteToHalt();
      }
}
```

This is a simple computation of the degree (number of neighbors) for each node.


### Compile and Run

As we have discussed in class, in order to run a vertex-centric analysis,
you need to write a `compute()` function that dictates what each vertex
needs to do at each superstep of the computation. We will run the above
Degree computation on the small test dataset that is included in the Docker
image called `tiny-graph.txt`.

First, you will need to load that dataset into HDFS by running (in the container):
`$HADOOP_HOME/bin/hdfs dfs -put $GIRAPH_HOME/tiny-graph.txt /user/root/input/tiny-graph.txt`

The easiest way to create a Java jar file and run it within the container,
is by directly compiling the java class, adding it to the appropriate
classpath within the container, building the jar that includes _your_ java
class, and submitting it to giraph. The workflow goes as follows
**(remember that all of the following happens _inside_ the container)**:

First, go to the folder that includes all of the java files (mapped from the local host):

```
cd /myhome/giraph-work
```

Next, compile the program and set the classpath:

```
javac -cp /usr/local/giraph/giraph-examples/target/giraph-examples-1.1.0-SNAPSHOT-for-hadoop-2.4.1-jar-with-dependencies.jar:$($HADOOP_HOME/bin/hadoop classpath) mypackage/DegreeComputation.java
```

Now, we will make a copy of the Giraph examples jar file and add our class files
to it with the following two commands:

```
cp /usr/local/giraph/giraph-examples/target/giraph-examples-1.1.0-SNAPSHOT-for-hadoop-2.4.1-jar-with-dependencies.jar ./myjar.jar
jar uf myjar.jar mypackage
```

Finally, run the jar with Giraph:

```
$HADOOP_HOME/bin/hadoop jar myjar.jar org.apache.giraph.GiraphRunner mypackage.DegreeComputation --yarnjars myjar.jar --workers 1 --vertexInputFormat org.apache.giraph.io.formats.JsonLongDoubleFloatDoubleVertexInputFormat --vertexInputPath /user/root/input/imdb-giraph-small-undirected.txt -vertexOutputFormat org.apache.giraph.io.formats.IdWithValueTextOutputFormat --outputPath /user/root/output
```

Then look at the results that have been output in HDFS (remember, Giraph
runs on top of HDFS):

```
$HADOOP_HOME/bin/hdfs dfs -cat /user/root/output/part-m-00001
```

#### Some Important APIs

The most important things you can do in a `compute()` function within Apache Giraph are the following:

- Iterate over the current node's neighbors:

```java
for (Edge<T1, T2> edge : vertex.getEdges()) {
...  
}
```

_Note_: In the above loop, `T1` is the specified type of the vertex ids,
and `T2` is the edge value type.

- Iterate over the messages received for the current superstep:

```java
for (DoubleWritable message : messages) {
  ...
}
```

- Broadcast a message to a node's neighbors:

```java
sendMessageToAllEdges(from_vertex, message_to_send);
```

- Get the current superstep:

```java
int s = getSuperstep();
```

- Tell the current node to vote to halt the computation (no more supersteps):

```java
voteToHalt();
```

### Tasks

1. Load the `states-giraph.txt` dataset we have provided for you into
HDFS. Write a vertex-centric program such that, for each state, compute the
state with which it shares the longest border. Note that Washington,
D.C. (DC) is counted as a
state here, whereas Alaska and Hawaii are not present in the dataset, hence there is a total of 49 distinct ids.

**Output:**:  Using the default OutputFormat your output should look like this:
for every node, it should output a row `node_id   node_id-with-longest-shared-border`.

2. Implement an algorithm to find _connected components_ in the given
movies graph (`imdb-giraph-small-undirected.txt`). First load
`imdb-giraph-small-undirected.txt` into HDFS as instructed in the
_Datasets_ section.
Connected components are sets of nodes in the graph that are connected to
each other but are disconnected from the rest of the graph. The
vertex-centric framework is quite appropriate for this type of
computation. One
classic way to implement connected components in the vertex-centric
framework is the following:

Every node has a unique ID. Each node initially
sends its ID as a message to its neighbors, and receives the IDs from its
neighbors. Each node keeps
track of the smallest ID seen, continuously propagating the smallest ID
it has seen by sending it as a message to its neighbors until the node does
not receive an id than is smaller than the smallest one it has seen so far.

**Output:**: For every node, output a row: `node_id    connected_componentId`.
(Nodes that share the same `connected_componentId` are part of the same connected component).

## Submission Instructions

1. For the _neo4j_ portion of this assignment, we have uploaded a text file
in your individual repositories under `project-5/neo4j-queries.txt`. Please
fill in your query text where it is indicated in the text file.

2. For the _giraph_ portion of this assignment, please submit the _Java file_ that you compiled and ran in the container for each task. Please name the files
`Task1.java` and `Task2.java` accordingly. We will use the same workflow as
described here in order to compile your java file as a _Giraph_ jar and
submit it to the container for execution.
