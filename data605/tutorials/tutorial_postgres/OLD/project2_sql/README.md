# Project 2

The aim of this project is to solidify your understanding of the internals of
relational databases, and learning how to read the query plans that the query
optimizer automatically produces in order to execute your SQL query as efficiently
as it possibly can. You will also learn how to interact with the database through an
external program using the JDBC connector API, and also deal with data stored using
the JSON data model.

## Getting The New Image

In order to use the database that we have set up for this assignment, you will need
to create a new container using the updated image we have pushed to DockerHub.
Unfortunately there doesn't seem to be a way to automatically update your container
with the contents of the new image, so you have to create a new container. You can
stop your current container and remove it like so:
```
docker stop container-name
docker rm container-name
```

_Note that you don't actually have to remove your old container if you don't want to
as long as you're giving a unique name to the new container started using the newest
image, however your old container is taking up unnecessary space on your machine
since the new version of the image contains everything the old one did and more!_

Make sure you haven't left anything important within the container since all files
will be lost once you call `docker rm` and remove it!

After stopping and removing the old container you first update to the new image with
the `docker pull` command and then use the `docker run` command to start a new
container.
```
docker pull kostasxirog/cmsc642-postgresql
docker run --name my-postgres-container -d kostasxirog/cmsc642-postgresql
```

# Question 1

We've included a _new_ version of the _flights_ database in the new docker image
called `skewed`. Make sure to run your query against that database and not the one
from project 1 (named `postgres`).

In order to connect to the `skewed` database, go into your container's bash shell
like you did in the last project, and run PostgreSQL.
```
docker exec -it my-postgres-container bash
psql -U postgres -d skewed
```
For psql

- `-U` Refers to the user we want to connect as. We will connect as the default user
  called `postgres`.
- `-d` Refers to the name of the database to connect to.

Consider the following query, which finds the number of flights taken by users whose
name starts with 'William':
```
select c.customerid, c.name, count(*)
from customers c join flewon f on (c.customerid = f.customerid and c.name like 'William%')
group by c.customerid, c.name
order by c.customerid;
```

The result, however, does not contain the users whose name contains 'William' but
for whom there is no current record in the database about the flights they took (
e.g., `cust731`). So we may consider modifying this query to use a _left outer join_
instead, so we get those users as well:

```
select c.customerid, c.name, count(*)
from customers c left outer join flewon f on (c.customerid = f.customerid and c.name like 'William%')
group by c.customerid, c.name
order by c.customerid;
```

Briefly explain why this query does not return the expected answer (as shown below),
and rewrite the query so that it does.

The final answer returned by the database should look like this:
```
	customerid |              name              | count
	-----------+--------------------------------+-------
	cust727    | William Harris                 |     4
	cust728    | William Hill                   |     6
	cust729    | William Jackson                |     6
	cust730    | William Johnson                |     5
	cust731    | William Lee                    |     0
	cust732    | William Lopez                  |     6
	cust733    | William Martinez               |     0
	cust734    | William Mitchell               |     6
	cust735    | William Moore                  |     5
	cust736    | William Parker                 |     4
	cust737    | William Roberts                |     8
	cust738    | William Robinson               |     7
	cust739    | William Rodriguez              |     5
	cust740    | William Wright                 |     8
	cust741    | William Young                  |     5
	(15 rows)
```

# Question 2

`EXPLAIN` can be used to see the query plan used by the database system to execute a
query. For the following query, draw the query plan for the query, clearly showing
the different operators and the options they take.
```
select a.city, source, b.city, dest, count(*) number_of_flights
from flights, airports a, airports b
where flights.source = a.airportid and flights.dest = b.airportid
group by a.city, source, b.city, dest
having count(*) > 1
order by number_of_flights desc, a.city asc, b.city asc;
```


# Question 3

Similar to Question 2, draw the query plan for the following query, and annotate
which operators are responsible for creating `temp1`, `temp2`, and the final answer.

```
with temp1 as (
        select customerid, airlineid, count(*) as num_flights
        from flewon natural join flights
        group by customerid, airlineid
),
temp2 as (
        select customerid
        from temp1 t1
        where num_flights = (select max(num_flights) from temp1 t2 where t1.customerid = t2.customerid)
            and airlineid = 'SW'
)
select customers.customerid, name
from customers, temp2
where customers.customerid = temp2.customerid and customers.frequentflieron != 'SW';
```

# Question 4

The `EXPLAIN` output also shows how many tuples the optimizer expects to be
generated after each operation (`rows`). `EXPLAIN ANALYZE` executes the query and
also shows the **actual** number of tuples generated when the query plan is
executed.

For the following two queries, identify some of the key intermediate results that
were generated by the database, what were their estimated cardinalities (sizes,
or number of rows), and what were their actual cardinalities, using `EXPLAIN
ANALYZE`. Did the database generally do a good job in estimating the sizes of the
intermediate results? Can you trace the differences in the actual cardinalities
for the two queries (which only differ in one constant) to the properties of the
data?
```
Query 1:
select c.name, count(*)
from customers c, flights fl, flewon fo
where extract(month from c.birthdate) = 1 and c.customerid = fo.customerid and fl.flightid = fo.flightid
      and fl.airlineid = 'AA'
group by c.name;
```
```
Query 2:
select c.name, count(*)
from customers c, flights fl, flewon fo
where extract(month from c.birthdate) = 2 and c.customerid = fo.customerid and fl.flightid = fo.flightid
      and fl.airlineid = 'AA'
group by c.name;
```

Question 5

One of more prominent ways to use a database system is using an external client,
using APIs such as ODBC and JDBC. This allows you to run queries against the
database and access the results from within say a Java program.

Here are some useful links:

- [Wikipedia Article](http://en.wikipedia.org/wiki/Java_Database_Connectivity)

- [Another resource](http://www.mkyong.com/java/how-do-connect-to-postgresql-with-jdbc-driver-java/)

- [PostgreSQL JDBC](http://jdbc.postgresql.org/index.html)

The last link has detailed examples in the `documentation` section. There is a
directory called `/home` (in the root directory of the container's file system) in
the container that also contains an example file (`JDBCExample.java`). To run the
JDBCExample.java file, do:
`javac JDBCExample.java` (which compiles the program) followed
by `java -cp lib/*:. JDBCExample`.

Your task is to write a JDBC program that will take in JSON updates and insert
appropriate data into the database. Two types of updates should be supported:

- New customer, where information about a customer is provided, in the following
  format. You can assume that the frequent flier airline name matches exactly what
  is listed in the `airlines` table.

```
{ "newcustomer":
	{
	"customerid": "cust1000",
	"name": "XYZ",
	"birthdate": "1991-12-06",
	"frequentflieron": "Southwest Airlines"
	}
}
```

- Flight information, where information about the passengers in a flight is
  provided.

```
{ "flightinfo": {
	"flightid": "DL119",
	"flightdate": "2015-09-25",
	"customers": [
		{"customer_id": "cust94"},
		{"customer_id": "cust102"},
		{"customer_id": "cust1000", "name": "XYZ", "birthdate": "1991-12-06", "frequentflieron": "DL"}
		]
	}
}
```

Note that, in some cases, the customer_id provided may not be present in the
database (the last one above). In that case, you have to first update the
`customers` table, before adding tuples to the `flewon` table due to the foreign
key constraint.

Your code should also catch one type of error: If the `customerid` for a
`newcustomer` update is already present or if the `frequentflieron` does not have
a match in the airlines table, you should print out an error. You don't need to
handle errors for the second update type.

---

### Parsing JSON

There are quite a few Java JSON parsing libraries out there to simplify the parsing
process. To simplify matters, we have provided a JSON parsing library for you to
use: `json-simple-1.1.1.jar`. Here is the webpage for the
library: <https://github.com/fangyidong/json-simple>. Some code samples can be found
in the `src/test` directory there, or you can also see some examples
here: <https://www.mkyong.com/java/json-simple-example-read-and-write-json/>.

The provided `JSONProcessing.java` file already takes care of the input part, and
you just need to focus on finishing the `processJSON(String json)` function. We have
placed an example JSON file in your github repository: `example.json`. You can
develop your java program outside of the container, but **please make sure that it
runs correctly within the container!**. You can again compile your java program that
uses the `json-simple` library by running : `javac -cp lib/*:. JSONProcessing.java`
followed by `java -cp lib/*:. JSONProcessing` . All the Java files are in the `home`
directory in the container.

### Submission Instructions

We have provided an `answers.docx` file in each of your github repositories -- fill
in your answers to the first 4 questions into that doc file (including scanned PDFs
or images). You may want to use some tool (
e.g., [Google Drawings](https://docs.google.com/drawings/d/1l4ygWoy15E0ZhX13Ki2m2ArSrvMzZpFDxdUn6H0gdsY/edit))
to draw the query plans. In addition, upload your `JSONProcessing.java` file
separately. Upload both of these files in the `project2` directory in your github
repository for the class.
