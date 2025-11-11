# Project 1 SQL queries

The queries into the airlines database you have to implement for this project are
described here.

0. List all airport codes and their cities. Order by the total number of
   passengers that used each airport in 2012 in decreasing order.

    Output column order: airportid, city


1. Write a query to find the names of the customers whose last names start with
   an \'A\' and the 4th character is an \'e\'.

    Order by name in ascending order.


2. Write a query to generate a list of pairs (customer_name_1, customer_name_2)
   of names of customers who share a birthday (day and month of birth).  Hint:
   Use the "extract" function to operate on the dates.

    Order output by the first customer\'s name

    Output columns: first customer name, second customer name

    Note: The names in the two columns must be different and pair should be
    unique (no pair should appear in two tuples; there should be only one of
    (a,b) and (b,a) in the results)


3. Write a query to generate a list: (source_city_name1, source_city_name2,
   num_common_destination_cities) for all city pairs that have flights to at
   least two common destination cities. 

    Order first by num_common_destination_cities in decreasing order, then
    source_city_name1 in increasing order, and then source_city_name2 in
    increasing order.


4. Pairs of customers that were on more than one common flight to Dallas Fort
   Worth.  A common flight must be on the same day.

    Output format: (customerId1, customerId2)

    Order by: number of shared flights to Dallas Fort Worth in descending order


5. Find the name of the airline with the maximum number of flights to IAD
   (Washington Dulles) airport.

    Output only the name of the airline. If multiple answers, order by name.


6. List the flight id, airline name, and the duration in hours and minutes of the
   longest flight.

    The output will have 4 fields: flightid, airline name, hours, minutes. Order
    by flightid if there are multiple such flights.


7. Write a query to print all unique flights along with their flight dates; If
   date information doesn\'t exist an empty value should be printed in the date
   column.  Note that all the flights listed in the flights table are daily, and
   that the flewon relation contains information for a period of 10 days from
   August 1 to August 10, 2016.

    For each such flight, list the flightid and the date.

    Order by flight id in increasing order, and then by date in increasing order.


8. Write a query to generate a list of customers whose most used airline is not
   their listed frequent flyer airline.

    Note: If there are multiple such airlines that the customer used most (and
    more than their frequent flyer airline), generate a seperate tuple for each
    airline.

    Output columns: customerid, customer_name, airline_most_used

    Order by: customerid



9. Write a query to generate a list of customers who flew four times to the same
   destination but did not fly otherwise in the 10 day period (only flew those
   four times to that same destination). The output should be simply a list of
   customer ids and names.

    Make sure the same customer does not appear multiple times in the answer.

    Order by the customer name.


10. Write a query that outputs all possible 2-stop flights one can take on August
    8th from Boston to any other destination where the layover time is at least 1
    hour.

    Output columns: (flight1_id, flight2_id, source_city_name, layover_city_name,
    dest_city name, layover_time)

    Order by: Layover time in descending order

    Note: It is interesting to think of this as a graph query over a graph of
    cities that are connected to each other through flights. In particular, this
    query explores all 2-hop neighbors in this graph starting from a node named
    Boston.

