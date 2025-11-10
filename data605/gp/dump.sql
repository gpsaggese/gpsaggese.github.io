--
-- PostgreSQL database dump
--

-- Dumped from database version 10.1
-- Dumped by pg_dump version 10.1

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: postgres; Type: COMMENT; Schema: -; Owner: postgres
--

COMMENT ON DATABASE postgres IS 'default administrative connection database';


--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET search_path = public, pg_catalog;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: airlines; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE airlines (
    airlineid character(2) NOT NULL,
    name character(20),
    hub character(3)
);


ALTER TABLE airlines OWNER TO postgres;

--
-- Name: airports; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE airports (
    airportid character(3) NOT NULL,
    city character(20),
    name character(100),
    total2011 integer,
    total2012 integer
);


ALTER TABLE airports OWNER TO postgres;

--
-- Name: customers; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE customers (
    customerid character(10) NOT NULL,
    name character(30),
    birthdate date,
    frequentflieron character(2)
);


ALTER TABLE customers OWNER TO postgres;

--
-- Name: flewon; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE flewon (
    flightid character(6),
    customerid character(10),
    flightdate date
);


ALTER TABLE flewon OWNER TO postgres;

--
-- Name: flights; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE flights (
    flightid character(6) NOT NULL,
    source character(3),
    dest character(3),
    airlineid character(2),
    local_departing_time time without time zone,
    local_arrival_time time without time zone
);


ALTER TABLE flights OWNER TO postgres;

--
-- Data for Name: airlines; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY airlines (airlineid, name, hub) FROM stdin;
SW	Southwest Airlines  	OAK
AA	American Airlines   	DFW
DL	Delta Airlines      	ATL
UA	United Airlines     	ORD
\.


--
-- Data for Name: airports; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY airports (airportid, city, name, total2011, total2012) FROM stdin;
OAK	Oakland             	Metropolitan Oakland International                                                                  	10040864	9266570
FLL	Fort Lauderdale     	Fort Lauderdale Hollywood International                                                             	23569103	23349835
BOS	Boston              	General Edward Lawrence Logan International                                                         	29349759	28932808
IAD	Washington          	Washington Dulles International                                                                     	22408105	23056291
ATL	Atlanta             	Hartsfield Jackson Atlanta International                                                            	95513828	92389023
ORD	Chicago             	Chicago O'Hare International                                                                        	66633503	66701241
LAX	Los Angeles         	Los Angeles International                                                                           	63688121	61862052
DFW	Dallas-Fort Worth   	Dallas Fort Worth International                                                                     	58621369	57832495
DEN	Denver              	Denver International                                                                                	53156278	52849132
JFK	New York            	John F Kennedy International                                                                        	49291765	47644060
\.


--
-- Data for Name: customers; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY customers (customerid, name, birthdate, frequentflieron) FROM stdin;
cust0     	Anthony Allen                 	1985-05-14	AA
cust1     	Anthony Edwards               	1986-10-18	SW
cust2     	Anthony Evans                 	1987-02-08	SW
cust3     	Anthony Garcia                	1994-08-23	DL
cust4     	Anthony Gonzalez              	1977-10-06	AA
cust5     	Anthony Harris                	1991-03-15	UA
cust6     	Barbara Collins               	1983-09-09	DL
cust7     	Barbara Davis                 	1982-03-29	AA
cust8     	Barbara Gonzalez              	1976-01-10	DL
cust9     	Barbara Hall                  	1989-08-26	SW
cust10    	Barbara Harris                	1991-09-02	AA
cust11    	Betty Baker                   	1998-03-03	AA
cust12    	Betty Brown                   	1975-08-14	UA
cust13    	Betty Carter                  	1989-04-02	DL
cust14    	Betty Edwards                 	1969-07-11	DL
cust15    	Betty Gonzalez                	1993-12-28	SW
cust16    	Betty Jackson                 	1977-07-11	AA
cust17    	Brian Evans                   	1980-05-07	SW
cust18    	Brian Garcia                  	1998-04-10	AA
cust19    	Brian Gonzalez                	1971-05-16	SW
cust20    	Brian Jackson                 	1975-01-06	AA
cust21    	Carol Anderson                	1971-09-21	AA
cust22    	Carol Baker                   	1990-08-29	SW
cust23    	Carol Campbell                	1998-02-03	AA
cust24    	Carol Clark                   	1984-08-24	DL
cust25    	Carol Evans                   	1976-05-29	AA
cust26    	Carol Hall                    	1969-10-29	DL
cust27    	Charles Brown                 	1973-10-11	DL
cust28    	Charles Collins               	1974-08-28	SW
cust29    	Charles Evans                 	1997-12-12	AA
cust30    	Charles Garcia                	1994-10-03	AA
cust31    	Charles Gonzalez              	1978-05-18	DL
cust32    	Charles Hall                  	1970-11-16	AA
cust33    	Christopher Davis             	1984-05-13	DL
cust34    	Christopher Hernandez         	1986-06-21	SW
cust35    	Christopher Hill              	1975-04-04	DL
cust36    	Daniel Baker                  	1998-05-27	SW
cust37    	Daniel Brown                  	1994-07-13	AA
cust38    	Daniel Edwards                	1986-05-19	SW
cust39    	Daniel Garcia                 	1986-09-13	UA
cust40    	Daniel Green                  	1974-05-01	AA
cust41    	Daniel Hall                   	1989-08-04	UA
cust42    	Daniel Harris                 	1976-06-14	DL
cust43    	Daniel Hernandez              	1978-06-03	AA
cust44    	Daniel Jackson                	1974-03-23	AA
cust45    	David Adams                   	1988-10-19	AA
cust46    	David Baker                   	1987-06-26	AA
cust47    	David Campbell                	1986-03-22	AA
cust48    	David Carter                  	1983-05-01	AA
cust49    	David Garcia                  	1987-06-06	SW
cust50    	David Hall                    	1969-12-02	UA
cust51    	David Hernandez               	1976-04-06	DL
cust52    	David Hill                    	1989-11-23	UA
cust53    	Deborah Adams                 	1971-05-16	UA
cust54    	Deborah Allen                 	1988-08-17	UA
cust55    	Deborah Anderson              	1987-02-02	DL
cust56    	Deborah Baker                 	1996-04-29	DL
cust57    	Deborah Collins               	1980-09-25	SW
cust58    	Deborah Edwards               	1991-03-10	SW
cust59    	Deborah Hill                  	1969-11-09	SW
cust60    	Donald Adams                  	1981-01-17	UA
cust61    	Donald Allen                  	1972-06-09	UA
cust62    	Donald Campbell               	1995-11-30	AA
cust63    	Donald Carter                 	1979-11-25	AA
cust64    	Donald Edwards                	1971-06-04	DL
cust65    	Donald Evans                  	1988-10-04	AA
cust66    	Donna Allen                   	1994-03-18	UA
cust67    	Donna Brown                   	1971-03-09	DL
cust68    	Donna Edwards                 	1975-11-04	DL
cust69    	Donna Hall                    	1977-01-15	AA
cust70    	Donna Hill                    	1978-09-04	SW
cust71    	Dorothy Allen                 	1981-09-01	AA
cust72    	Dorothy Anderson              	1980-03-21	SW
cust73    	Dorothy Campbell              	1992-06-18	SW
cust74    	Dorothy Carter                	1976-03-14	AA
cust75    	Dorothy Collins               	1976-02-13	UA
cust76    	Dorothy Edwards               	1981-11-24	AA
cust77    	Dorothy Harris                	1994-02-02	SW
cust78    	Edward Baker                  	1972-11-10	DL
cust79    	Edward Brown                  	1969-07-11	AA
cust80    	Edward Carter                 	1981-06-05	AA
cust81    	Edward Davis                  	1981-02-12	DL
cust82    	Edward Edwards                	1984-07-15	DL
cust83    	Edward Evans                  	1982-06-22	SW
cust84    	Edward Garcia                 	1980-02-07	AA
cust85    	Edward Harris                 	1994-05-08	AA
cust86    	Elizabeth Anderson            	1995-03-28	DL
cust87    	Elizabeth Baker               	1986-11-23	DL
cust88    	Elizabeth Collins             	1978-09-16	DL
cust89    	Elizabeth Gonzalez            	1995-04-09	DL
cust90    	Elizabeth Green               	1972-01-11	SW
cust91    	Elizabeth Hall                	1978-03-16	DL
cust92    	Elizabeth Harris              	1980-05-26	DL
cust93    	Elizabeth Hernandez           	1972-03-21	SW
cust94    	Elizabeth Hill                	1993-05-09	DL
cust95    	Elizabeth Jackson             	1987-06-18	AA
cust96    	George Anderson               	1978-02-23	SW
cust97    	George Brown                  	1971-03-26	AA
cust98    	George Collins                	1995-03-07	UA
cust99    	George Davis                  	1974-09-07	SW
cust100   	George Evans                  	1974-08-25	SW
cust101   	George Garcia                 	1996-05-10	AA
cust102   	George Gonzalez               	1996-01-30	DL
cust103   	Helen Adams                   	1971-07-31	AA
cust104   	Helen Allen                   	1970-03-13	DL
cust105   	Helen Edwards                 	1971-08-21	AA
cust106   	Helen Evans                   	1991-01-04	DL
cust107   	Helen Harris                  	1980-02-21	DL
cust108   	Helen Hernandez               	1981-05-14	AA
cust109   	James Adams                   	1994-05-22	AA
cust110   	James Brown                   	1970-02-08	SW
cust111   	James Carter                  	1980-05-13	DL
cust112   	James Evans                   	1981-01-19	DL
cust113   	James Green                   	1977-09-16	UA
cust114   	James Hall                    	1986-10-13	AA
cust115   	James Hill                    	1988-03-19	SW
cust116   	Jason Carter                  	1987-08-04	UA
cust117   	Jason Clark                   	1976-02-25	DL
cust118   	Jason Hall                    	1996-12-13	SW
cust119   	Jason Harris                  	1992-07-04	SW
cust120   	Jason Hernandez               	1998-08-23	DL
cust121   	Jeff Adams                    	1977-08-27	DL
cust122   	Jeff Anderson                 	1978-03-25	DL
cust123   	Jeff Baker                    	1996-08-04	AA
cust124   	Jeff Clark                    	1982-08-16	AA
cust125   	Jeff Green                    	1991-01-06	AA
cust126   	Jeff Harris                   	1976-01-19	DL
cust127   	Jeff Hill                     	1978-07-22	SW
\.


--
-- Data for Name: flewon; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY flewon (flightid, customerid, flightdate) FROM stdin;
DL119 	cust59    	2016-08-05
SW103 	cust116   	2016-08-04
UA105 	cust19    	2016-08-08
SW102 	cust11    	2016-08-01
UA101 	cust68    	2016-08-06
UA101 	cust64    	2016-08-02
UA101 	cust52    	2016-08-08
SW102 	cust62    	2016-08-01
SW107 	cust20    	2016-08-04
SW103 	cust97    	2016-08-08
SW137 	cust98    	2016-08-06
UA101 	cust71    	2016-08-09
UA101 	cust30    	2016-08-07
UA101 	cust7     	2016-08-05
UA101 	cust44    	2016-08-04
UA101 	cust33    	2016-08-07
AA113 	cust65    	2016-08-02
SW106 	cust34    	2016-08-02
SW107 	cust55    	2016-08-09
SW104 	cust1     	2016-08-03
AA114 	cust26    	2016-08-08
UA101 	cust31    	2016-08-08
SW102 	cust3     	2016-08-08
SW137 	cust17    	2016-08-04
SW106 	cust7     	2016-08-08
UA101 	cust50    	2016-08-08
AA114 	cust44    	2016-08-08
SW132 	cust8     	2016-08-05
SW104 	cust79    	2016-08-03
UA101 	cust73    	2016-08-03
UA101 	cust29    	2016-08-04
UA105 	cust22    	2016-08-02
UA105 	cust2     	2016-08-03
SW132 	cust110   	2016-08-05
SW103 	cust37    	2016-08-05
SW106 	cust75    	2016-08-02
AA109 	cust17    	2016-08-08
UA101 	cust90    	2016-08-04
SW102 	cust21    	2016-08-04
SW103 	cust114   	2016-08-05
SW104 	cust26    	2016-08-03
UA101 	cust32    	2016-08-04
SW125 	cust73    	2016-08-09
UA101 	cust0     	2016-08-03
AA113 	cust37    	2016-08-09
UA135 	cust0     	2016-08-01
SW129 	cust23    	2016-08-06
SW118 	cust87    	2016-08-05
UA101 	cust2     	2016-08-01
UA101 	cust101   	2016-08-04
SW104 	cust77    	2016-08-08
UA105 	cust0     	2016-08-04
SW107 	cust115   	2016-08-05
UA101 	cust18    	2016-08-02
UA101 	cust3     	2016-08-04
UA101 	cust107   	2016-08-05
SW103 	cust90    	2016-08-01
UA101 	cust70    	2016-08-07
AA114 	cust11    	2016-08-03
UA101 	cust104   	2016-08-05
SW104 	cust0     	2016-08-05
SW132 	cust57    	2016-08-05
AA109 	cust81    	2016-08-08
UA101 	cust40    	2016-08-01
UA101 	cust19    	2016-08-01
AA114 	cust57    	2016-08-06
SW129 	cust118   	2016-08-06
UA101 	cust21    	2016-08-06
SW108 	cust0     	2016-08-02
SW118 	cust113   	2016-08-02
UA101 	cust10    	2016-08-02
DL121 	cust40    	2016-08-05
SW107 	cust25    	2016-08-06
UA101 	cust22    	2016-08-09
SW122 	cust13    	2016-08-08
SW120 	cust116   	2016-08-07
UA101 	cust78    	2016-08-06
UA101 	cust111   	2016-08-09
SW129 	cust12    	2016-08-07
SW126 	cust26    	2016-08-06
DL134 	cust115   	2016-08-06
UA117 	cust36    	2016-08-06
SW102 	cust13    	2016-08-04
SW124 	cust119   	2016-08-07
AA114 	cust79    	2016-08-01
UA101 	cust60    	2016-08-06
AA114 	cust78    	2016-08-09
SW104 	cust127   	2016-08-03
SW132 	cust32    	2016-08-06
AA127 	cust49    	2016-08-02
UA105 	cust14    	2016-08-04
SW112 	cust0     	2016-08-08
SW103 	cust106   	2016-08-07
UA101 	cust62    	2016-08-03
UA105 	cust23    	2016-08-03
SW123 	cust32    	2016-08-07
UA101 	cust124   	2016-08-01
SW122 	cust93    	2016-08-09
UA138 	cust0     	2016-08-07
UA101 	cust45    	2016-08-04
SW102 	cust5     	2016-08-06
UA101 	cust30    	2016-08-09
UA105 	cust9     	2016-08-01
UA101 	cust93    	2016-08-02
AA113 	cust31    	2016-08-03
UA101 	cust14    	2016-08-08
SW112 	cust39    	2016-08-07
SW102 	cust44    	2016-08-07
UA101 	cust113   	2016-08-04
SW103 	cust83    	2016-08-04
UA101 	cust118   	2016-08-04
SW103 	cust71    	2016-08-03
UA105 	cust72    	2016-08-09
SW126 	cust40    	2016-08-03
UA101 	cust112   	2016-08-07
UA101 	cust87    	2016-08-07
SW111 	cust45    	2016-08-09
UA101 	cust2     	2016-08-09
SW111 	cust120   	2016-08-04
UA101 	cust98    	2016-08-02
SW125 	cust111   	2016-08-02
SW106 	cust26    	2016-08-05
UA101 	cust48    	2016-08-08
SW123 	cust88    	2016-08-01
UA117 	cust9     	2016-08-09
SW110 	cust56    	2016-08-03
DL140 	cust89    	2016-08-01
AA109 	cust0     	2016-08-09
DL140 	cust5     	2016-08-05
UA101 	cust28    	2016-08-07
AA130 	cust79    	2016-08-08
SW104 	cust83    	2016-08-02
UA101 	cust2     	2016-08-02
UA101 	cust102   	2016-08-04
UA117 	cust98    	2016-08-04
SW106 	cust86    	2016-08-07
AA109 	cust30    	2016-08-03
SW102 	cust1     	2016-08-01
SW103 	cust6     	2016-08-05
SW102 	cust6     	2016-08-06
SW123 	cust60    	2016-08-07
UA101 	cust36    	2016-08-09
SW112 	cust12    	2016-08-01
UA101 	cust100   	2016-08-07
UA101 	cust69    	2016-08-02
UA101 	cust15    	2016-08-05
AA109 	cust49    	2016-08-01
SW120 	cust22    	2016-08-01
UA135 	cust13    	2016-08-01
SW123 	cust6     	2016-08-09
DL134 	cust21    	2016-08-08
AA130 	cust24    	2016-08-05
SW123 	cust103   	2016-08-08
UA101 	cust68    	2016-08-09
SW111 	cust65    	2016-08-04
SW112 	cust56    	2016-08-02
SW103 	cust9     	2016-08-03
UA101 	cust96    	2016-08-05
SW107 	cust111   	2016-08-06
SW102 	cust35    	2016-08-02
SW102 	cust104   	2016-08-02
UA101 	cust5     	2016-08-01
UA101 	cust26    	2016-08-01
SW116 	cust39    	2016-08-09
SW125 	cust36    	2016-08-03
UA101 	cust67    	2016-08-07
SW103 	cust57    	2016-08-07
SW108 	cust71    	2016-08-06
SW137 	cust93    	2016-08-06
SW110 	cust16    	2016-08-02
UA101 	cust10    	2016-08-06
UA101 	cust32    	2016-08-01
AA113 	cust108   	2016-08-08
SW106 	cust6     	2016-08-08
SW112 	cust112   	2016-08-06
AA131 	cust18    	2016-08-03
AA114 	cust18    	2016-08-09
SW108 	cust91    	2016-08-09
SW123 	cust19    	2016-08-02
UA101 	cust34    	2016-08-06
UA101 	cust9     	2016-08-02
SW102 	cust121   	2016-08-03
SW110 	cust7     	2016-08-01
SW118 	cust5     	2016-08-08
DL134 	cust8     	2016-08-06
UA101 	cust2     	2016-08-07
UA101 	cust1     	2016-08-06
UA105 	cust3     	2016-08-01
SW104 	cust56    	2016-08-05
SW102 	cust1     	2016-08-07
UA101 	cust122   	2016-08-03
SW118 	cust29    	2016-08-05
SW129 	cust100   	2016-08-09
UA101 	cust9     	2016-08-07
SW107 	cust14    	2016-08-07
UA101 	cust1     	2016-08-08
AA131 	cust101   	2016-08-09
SW103 	cust17    	2016-08-07
SW102 	cust125   	2016-08-09
UA101 	cust71    	2016-08-05
UA101 	cust18    	2016-08-06
SW118 	cust1     	2016-08-02
SW103 	cust43    	2016-08-08
SW103 	cust0     	2016-08-06
SW107 	cust17    	2016-08-05
SW104 	cust37    	2016-08-08
UA101 	cust3     	2016-08-03
UA101 	cust79    	2016-08-09
SW102 	cust57    	2016-08-03
SW102 	cust82    	2016-08-08
UA101 	cust106   	2016-08-04
UA101 	cust83    	2016-08-03
UA101 	cust12    	2016-08-05
UA101 	cust29    	2016-08-08
SW108 	cust39    	2016-08-06
UA135 	cust3     	2016-08-09
SW110 	cust15    	2016-08-08
SW120 	cust48    	2016-08-01
UA101 	cust80    	2016-08-04
DL119 	cust65    	2016-08-08
UA101 	cust23    	2016-08-02
UA101 	cust56    	2016-08-06
SW103 	cust99    	2016-08-09
SW126 	cust66    	2016-08-05
UA101 	cust62    	2016-08-06
AA127 	cust30    	2016-08-01
UA101 	cust71    	2016-08-07
UA101 	cust58    	2016-08-07
UA138 	cust118   	2016-08-07
UA105 	cust118   	2016-08-01
AA113 	cust126   	2016-08-02
SW125 	cust31    	2016-08-02
SW103 	cust106   	2016-08-01
UA101 	cust45    	2016-08-02
SW102 	cust45    	2016-08-01
SW118 	cust105   	2016-08-02
UA101 	cust3     	2016-08-07
SW137 	cust37    	2016-08-04
SW108 	cust62    	2016-08-05
SW111 	cust119   	2016-08-01
SW129 	cust22    	2016-08-07
SW116 	cust13    	2016-08-07
UA101 	cust121   	2016-08-01
DL119 	cust33    	2016-08-06
UA101 	cust58    	2016-08-01
SW102 	cust75    	2016-08-08
SW108 	cust40    	2016-08-09
SW120 	cust69    	2016-08-07
SW124 	cust84    	2016-08-01
SW103 	cust2     	2016-08-04
UA101 	cust100   	2016-08-02
SW136 	cust6     	2016-08-03
SW104 	cust34    	2016-08-04
SW116 	cust6     	2016-08-07
SW104 	cust17    	2016-08-06
DL119 	cust81    	2016-08-06
UA101 	cust109   	2016-08-08
AA109 	cust36    	2016-08-08
SW110 	cust45    	2016-08-08
SW102 	cust54    	2016-08-02
AA109 	cust61    	2016-08-05
UA101 	cust11    	2016-08-02
SW103 	cust120   	2016-08-07
SW103 	cust27    	2016-08-06
SW118 	cust47    	2016-08-07
AA114 	cust40    	2016-08-02
UA135 	cust19    	2016-08-06
SW106 	cust52    	2016-08-09
SW108 	cust58    	2016-08-06
SW108 	cust43    	2016-08-02
SW132 	cust4     	2016-08-05
DL121 	cust108   	2016-08-07
AA115 	cust16    	2016-08-06
UA101 	cust52    	2016-08-02
SW104 	cust57    	2016-08-01
AA113 	cust84    	2016-08-07
UA101 	cust114   	2016-08-07
UA101 	cust47    	2016-08-04
SW103 	cust105   	2016-08-09
UA101 	cust2     	2016-08-08
UA101 	cust52    	2016-08-03
UA101 	cust44    	2016-08-03
UA135 	cust2     	2016-08-06
AA131 	cust45    	2016-08-07
SW106 	cust61    	2016-08-01
AA113 	cust89    	2016-08-03
SW110 	cust41    	2016-08-04
DL119 	cust127   	2016-08-05
SW103 	cust85    	2016-08-03
AA131 	cust9     	2016-08-04
AA131 	cust4     	2016-08-08
UA101 	cust19    	2016-08-07
AA113 	cust5     	2016-08-07
SW123 	cust49    	2016-08-06
UA128 	cust5     	2016-08-03
SW104 	cust45    	2016-08-05
UA101 	cust107   	2016-08-02
AA109 	cust69    	2016-08-09
UA101 	cust47    	2016-08-08
SW118 	cust22    	2016-08-06
UA105 	cust75    	2016-08-06
SW116 	cust4     	2016-08-03
UA101 	cust121   	2016-08-07
SW102 	cust6     	2016-08-02
UA105 	cust79    	2016-08-02
UA135 	cust57    	2016-08-08
UA101 	cust53    	2016-08-06
DL121 	cust58    	2016-08-03
SW103 	cust106   	2016-08-05
UA101 	cust7     	2016-08-06
SW124 	cust20    	2016-08-08
AA127 	cust51    	2016-08-01
AA131 	cust10    	2016-08-05
SW106 	cust6     	2016-08-04
AA131 	cust15    	2016-08-03
UA101 	cust95    	2016-08-07
AA109 	cust33    	2016-08-09
SW103 	cust106   	2016-08-06
UA101 	cust19    	2016-08-05
UA105 	cust59    	2016-08-04
SW102 	cust18    	2016-08-08
UA101 	cust71    	2016-08-04
SW102 	cust78    	2016-08-02
UA101 	cust5     	2016-08-04
SW106 	cust17    	2016-08-09
UA101 	cust111   	2016-08-04
SW103 	cust33    	2016-08-03
SW102 	cust88    	2016-08-06
UA101 	cust29    	2016-08-06
DL134 	cust94    	2016-08-01
UA101 	cust7     	2016-08-02
SW106 	cust30    	2016-08-04
UA101 	cust3     	2016-08-05
UA101 	cust10    	2016-08-01
SW102 	cust10    	2016-08-03
AA130 	cust77    	2016-08-09
AA113 	cust110   	2016-08-04
SW103 	cust99    	2016-08-03
UA101 	cust28    	2016-08-04
UA105 	cust84    	2016-08-09
SW102 	cust15    	2016-08-07
SW102 	cust94    	2016-08-02
SW106 	cust98    	2016-08-01
UA117 	cust23    	2016-08-07
UA101 	cust55    	2016-08-05
SW126 	cust60    	2016-08-04
SW107 	cust78    	2016-08-08
UA101 	cust101   	2016-08-08
AA113 	cust23    	2016-08-09
SW118 	cust107   	2016-08-03
SW102 	cust45    	2016-08-03
UA117 	cust85    	2016-08-08
AA113 	cust115   	2016-08-01
AA130 	cust86    	2016-08-08
DL121 	cust73    	2016-08-08
SW108 	cust81    	2016-08-03
SW126 	cust69    	2016-08-08
UA101 	cust48    	2016-08-02
AA114 	cust63    	2016-08-08
SW108 	cust74    	2016-08-08
AA131 	cust53    	2016-08-02
UA135 	cust23    	2016-08-05
AA115 	cust43    	2016-08-06
SW102 	cust8     	2016-08-01
SW118 	cust20    	2016-08-01
UA101 	cust112   	2016-08-04
UA101 	cust44    	2016-08-02
SW120 	cust58    	2016-08-05
UA101 	cust115   	2016-08-04
UA101 	cust90    	2016-08-06
SW102 	cust5     	2016-08-09
DL140 	cust46    	2016-08-09
SW126 	cust55    	2016-08-07
UA101 	cust42    	2016-08-07
UA101 	cust105   	2016-08-06
SW107 	cust94    	2016-08-08
UA105 	cust97    	2016-08-06
UA101 	cust97    	2016-08-07
UA101 	cust51    	2016-08-04
UA101 	cust61    	2016-08-03
SW137 	cust36    	2016-08-05
SW120 	cust20    	2016-08-02
SW112 	cust43    	2016-08-01
AA113 	cust13    	2016-08-03
SW107 	cust84    	2016-08-03
SW133 	cust28    	2016-08-09
SW104 	cust105   	2016-08-05
AA109 	cust103   	2016-08-05
UA101 	cust61    	2016-08-04
DL121 	cust22    	2016-08-08
SW102 	cust51    	2016-08-05
UA105 	cust16    	2016-08-05
SW118 	cust60    	2016-08-05
SW110 	cust66    	2016-08-01
SW118 	cust31    	2016-08-07
UA101 	cust33    	2016-08-01
UA101 	cust7     	2016-08-04
SW124 	cust108   	2016-08-06
SW136 	cust88    	2016-08-07
UA101 	cust44    	2016-08-06
UA101 	cust2     	2016-08-05
SW104 	cust11    	2016-08-07
SW103 	cust14    	2016-08-01
SW124 	cust88    	2016-08-04
SW118 	cust49    	2016-08-08
SW125 	cust66    	2016-08-04
UA101 	cust4     	2016-08-01
SW111 	cust48    	2016-08-03
UA101 	cust25    	2016-08-05
UA101 	cust101   	2016-08-05
SW118 	cust26    	2016-08-09
SW129 	cust99    	2016-08-07
SW107 	cust10    	2016-08-09
UA101 	cust22    	2016-08-04
SW118 	cust92    	2016-08-03
UA101 	cust17    	2016-08-01
UA101 	cust14    	2016-08-09
UA101 	cust54    	2016-08-09
DL119 	cust20    	2016-08-03
SW102 	cust1     	2016-08-05
SW116 	cust4     	2016-08-04
SW110 	cust4     	2016-08-02
SW129 	cust80    	2016-08-02
UA105 	cust29    	2016-08-02
UA101 	cust17    	2016-08-02
UA101 	cust24    	2016-08-03
SW104 	cust8     	2016-08-08
SW108 	cust13    	2016-08-09
SW102 	cust54    	2016-08-01
SW102 	cust12    	2016-08-08
SW112 	cust66    	2016-08-07
SW123 	cust8     	2016-08-09
SW104 	cust56    	2016-08-04
SW125 	cust63    	2016-08-06
SW124 	cust9     	2016-08-05
UA101 	cust91    	2016-08-07
UA101 	cust13    	2016-08-05
AA109 	cust30    	2016-08-05
UA117 	cust19    	2016-08-03
UA138 	cust95    	2016-08-03
AA113 	cust113   	2016-08-07
SW112 	cust82    	2016-08-06
UA105 	cust41    	2016-08-05
SW103 	cust76    	2016-08-01
UA101 	cust29    	2016-08-09
SW103 	cust109   	2016-08-06
SW106 	cust41    	2016-08-03
SW133 	cust55    	2016-08-04
UA101 	cust19    	2016-08-04
SW110 	cust103   	2016-08-09
UA101 	cust123   	2016-08-09
SW122 	cust19    	2016-08-09
UA138 	cust51    	2016-08-09
AA115 	cust14    	2016-08-05
UA128 	cust33    	2016-08-05
SW107 	cust55    	2016-08-08
SW132 	cust50    	2016-08-02
UA101 	cust12    	2016-08-09
SW123 	cust80    	2016-08-07
UA101 	cust8     	2016-08-02
UA101 	cust70    	2016-08-01
AA114 	cust108   	2016-08-03
DL140 	cust42    	2016-08-03
AA114 	cust21    	2016-08-01
UA101 	cust51    	2016-08-06
UA101 	cust116   	2016-08-09
UA101 	cust55    	2016-08-01
SW102 	cust79    	2016-08-04
AA115 	cust72    	2016-08-04
SW102 	cust97    	2016-08-04
UA101 	cust112   	2016-08-01
AA127 	cust109   	2016-08-05
SW111 	cust16    	2016-08-03
SW102 	cust11    	2016-08-06
UA117 	cust59    	2016-08-03
SW102 	cust92    	2016-08-07
SW107 	cust81    	2016-08-04
SW108 	cust81    	2016-08-07
SW124 	cust16    	2016-08-01
SW107 	cust34    	2016-08-09
UA117 	cust35    	2016-08-06
SW108 	cust25    	2016-08-08
SW116 	cust43    	2016-08-07
UA138 	cust5     	2016-08-02
SW137 	cust87    	2016-08-03
UA138 	cust31    	2016-08-04
SW103 	cust117   	2016-08-03
SW129 	cust83    	2016-08-05
UA101 	cust68    	2016-08-05
UA101 	cust77    	2016-08-07
SW102 	cust68    	2016-08-02
AA113 	cust123   	2016-08-07
SW133 	cust65    	2016-08-01
UA101 	cust27    	2016-08-09
UA101 	cust40    	2016-08-07
AA114 	cust76    	2016-08-07
SW104 	cust120   	2016-08-01
SW103 	cust47    	2016-08-05
SW106 	cust15    	2016-08-01
SW107 	cust83    	2016-08-07
AA114 	cust37    	2016-08-01
SW132 	cust39    	2016-08-01
SW118 	cust90    	2016-08-08
SW126 	cust68    	2016-08-01
UA101 	cust98    	2016-08-05
UA101 	cust124   	2016-08-06
SW104 	cust70    	2016-08-02
UA101 	cust109   	2016-08-02
SW104 	cust50    	2016-08-01
AA114 	cust125   	2016-08-06
UA101 	cust61    	2016-08-08
UA105 	cust56    	2016-08-09
SW118 	cust27    	2016-08-05
SW107 	cust101   	2016-08-06
UA101 	cust89    	2016-08-04
UA101 	cust15    	2016-08-06
SW116 	cust82    	2016-08-02
SW106 	cust46    	2016-08-04
DL140 	cust80    	2016-08-01
SW132 	cust14    	2016-08-03
UA101 	cust60    	2016-08-03
SW111 	cust48    	2016-08-07
UA105 	cust1     	2016-08-09
SW111 	cust125   	2016-08-04
SW104 	cust43    	2016-08-03
UA101 	cust14    	2016-08-02
SW137 	cust53    	2016-08-04
UA105 	cust66    	2016-08-03
SW104 	cust101   	2016-08-02
AA115 	cust7     	2016-08-07
SW110 	cust24    	2016-08-04
UA105 	cust76    	2016-08-03
SW125 	cust16    	2016-08-08
SW102 	cust92    	2016-08-01
SW116 	cust21    	2016-08-03
UA101 	cust16    	2016-08-07
SW126 	cust104   	2016-08-01
UA101 	cust61    	2016-08-06
SW102 	cust52    	2016-08-01
SW104 	cust30    	2016-08-08
UA101 	cust59    	2016-08-01
AA130 	cust38    	2016-08-02
SW104 	cust89    	2016-08-08
SW103 	cust72    	2016-08-03
DL140 	cust102   	2016-08-09
AA115 	cust97    	2016-08-01
SW102 	cust11    	2016-08-09
UA117 	cust112   	2016-08-05
SW103 	cust22    	2016-08-03
SW106 	cust125   	2016-08-03
SW112 	cust16    	2016-08-09
SW103 	cust33    	2016-08-08
SW107 	cust33    	2016-08-04
UA101 	cust66    	2016-08-09
AA113 	cust79    	2016-08-07
UA105 	cust122   	2016-08-07
AA109 	cust116   	2016-08-06
UA101 	cust39    	2016-08-02
AA114 	cust88    	2016-08-02
SW125 	cust78    	2016-08-07
SW104 	cust56    	2016-08-08
SW110 	cust1     	2016-08-04
UA101 	cust85    	2016-08-09
UA101 	cust39    	2016-08-08
SW110 	cust42    	2016-08-04
SW120 	cust50    	2016-08-07
SW124 	cust55    	2016-08-02
UA101 	cust29    	2016-08-07
UA101 	cust97    	2016-08-02
DL121 	cust21    	2016-08-09
SW103 	cust21    	2016-08-05
SW118 	cust72    	2016-08-05
UA101 	cust29    	2016-08-01
SW118 	cust17    	2016-08-03
SW123 	cust89    	2016-08-05
AA109 	cust87    	2016-08-09
SW103 	cust70    	2016-08-06
SW107 	cust68    	2016-08-07
SW126 	cust18    	2016-08-05
AA109 	cust78    	2016-08-01
SW103 	cust36    	2016-08-04
SW108 	cust36    	2016-08-07
SW110 	cust8     	2016-08-03
UA101 	cust95    	2016-08-09
SW102 	cust117   	2016-08-09
UA101 	cust110   	2016-08-09
SW104 	cust88    	2016-08-05
DL119 	cust73    	2016-08-05
UA101 	cust70    	2016-08-05
AA109 	cust46    	2016-08-03
SW106 	cust34    	2016-08-08
SW123 	cust14    	2016-08-06
UA101 	cust82    	2016-08-07
UA101 	cust119   	2016-08-04
DL134 	cust96    	2016-08-03
SW107 	cust124   	2016-08-07
AA130 	cust93    	2016-08-03
AA131 	cust43    	2016-08-05
UA101 	cust40    	2016-08-04
UA138 	cust11    	2016-08-05
UA105 	cust123   	2016-08-08
UA101 	cust38    	2016-08-04
UA101 	cust48    	2016-08-05
SW106 	cust18    	2016-08-01
SW104 	cust41    	2016-08-01
UA101 	cust29    	2016-08-03
UA101 	cust92    	2016-08-08
SW120 	cust100   	2016-08-04
SW116 	cust77    	2016-08-05
SW126 	cust9     	2016-08-06
SW136 	cust12    	2016-08-06
UA101 	cust119   	2016-08-08
UA101 	cust103   	2016-08-06
UA101 	cust122   	2016-08-01
SW111 	cust99    	2016-08-02
SW107 	cust48    	2016-08-09
SW116 	cust52    	2016-08-05
SW107 	cust28    	2016-08-08
DL119 	cust62    	2016-08-07
SW102 	cust125   	2016-08-08
SW106 	cust4     	2016-08-06
DL121 	cust44    	2016-08-05
SW102 	cust60    	2016-08-08
AA109 	cust82    	2016-08-09
SW106 	cust49    	2016-08-03
AA113 	cust58    	2016-08-08
SW104 	cust109   	2016-08-07
AA127 	cust27    	2016-08-04
AA131 	cust88    	2016-08-09
UA101 	cust51    	2016-08-03
UA101 	cust59    	2016-08-08
SW108 	cust24    	2016-08-07
UA101 	cust54    	2016-08-05
UA101 	cust25    	2016-08-07
DL134 	cust67    	2016-08-05
UA117 	cust77    	2016-08-06
SW102 	cust123   	2016-08-03
UA101 	cust75    	2016-08-03
SW107 	cust103   	2016-08-04
AA131 	cust120   	2016-08-02
UA101 	cust123   	2016-08-04
SW107 	cust115   	2016-08-08
SW102 	cust7     	2016-08-03
SW102 	cust37    	2016-08-06
SW137 	cust105   	2016-08-08
UA101 	cust118   	2016-08-03
SW102 	cust28    	2016-08-06
DL140 	cust66    	2016-08-06
SW110 	cust101   	2016-08-07
SW107 	cust113   	2016-08-06
UA101 	cust83    	2016-08-01
SW106 	cust41    	2016-08-07
UA101 	cust67    	2016-08-06
SW103 	cust69    	2016-08-05
UA101 	cust35    	2016-08-05
SW116 	cust65    	2016-08-09
UA135 	cust25    	2016-08-03
AA131 	cust86    	2016-08-06
AA131 	cust64    	2016-08-07
UA101 	cust74    	2016-08-05
SW102 	cust13    	2016-08-06
AA131 	cust82    	2016-08-01
SW103 	cust76    	2016-08-05
AA113 	cust41    	2016-08-02
SW102 	cust110   	2016-08-02
UA101 	cust25    	2016-08-01
UA101 	cust94    	2016-08-04
DL121 	cust43    	2016-08-09
SW106 	cust27    	2016-08-01
DL119 	cust8     	2016-08-07
SW102 	cust3     	2016-08-02
UA101 	cust65    	2016-08-05
UA101 	cust58    	2016-08-09
UA101 	cust25    	2016-08-09
SW125 	cust126   	2016-08-03
SW122 	cust123   	2016-08-05
SW106 	cust37    	2016-08-02
DL121 	cust91    	2016-08-08
UA101 	cust25    	2016-08-04
SW102 	cust10    	2016-08-07
SW123 	cust96    	2016-08-02
SW107 	cust102   	2016-08-06
UA117 	cust35    	2016-08-09
UA138 	cust4     	2016-08-09
UA101 	cust40    	2016-08-06
SW102 	cust21    	2016-08-02
SW123 	cust106   	2016-08-09
UA101 	cust15    	2016-08-04
UA101 	cust32    	2016-08-08
UA101 	cust120   	2016-08-09
DL119 	cust107   	2016-08-04
UA101 	cust6     	2016-08-01
AA109 	cust86    	2016-08-04
SW102 	cust86    	2016-08-02
AA109 	cust49    	2016-08-09
UA101 	cust9     	2016-08-08
UA101 	cust45    	2016-08-06
SW118 	cust102   	2016-08-02
SW104 	cust73    	2016-08-07
SW102 	cust122   	2016-08-04
UA101 	cust56    	2016-08-07
UA105 	cust94    	2016-08-05
SW102 	cust112   	2016-08-03
SW104 	cust15    	2016-08-02
SW102 	cust101   	2016-08-03
SW129 	cust42    	2016-08-06
SW102 	cust79    	2016-08-06
UA101 	cust72    	2016-08-06
UA101 	cust12    	2016-08-03
SW136 	cust86    	2016-08-01
DL140 	cust47    	2016-08-09
SW118 	cust26    	2016-08-02
DL121 	cust47    	2016-08-01
SW124 	cust88    	2016-08-03
UA101 	cust64    	2016-08-08
SW116 	cust59    	2016-08-09
SW102 	cust102   	2016-08-01
UA101 	cust30    	2016-08-06
SW106 	cust111   	2016-08-07
UA101 	cust126   	2016-08-04
UA105 	cust99    	2016-08-01
SW104 	cust96    	2016-08-01
UA101 	cust25    	2016-08-02
SW125 	cust7     	2016-08-09
SW102 	cust106   	2016-08-02
AA113 	cust69    	2016-08-03
UA101 	cust74    	2016-08-06
SW116 	cust55    	2016-08-03
SW124 	cust28    	2016-08-01
SW112 	cust64    	2016-08-06
DL134 	cust46    	2016-08-08
AA113 	cust38    	2016-08-09
SW123 	cust46    	2016-08-05
AA127 	cust30    	2016-08-02
SW108 	cust54    	2016-08-03
AA113 	cust52    	2016-08-06
UA101 	cust35    	2016-08-08
SW111 	cust83    	2016-08-09
UA128 	cust100   	2016-08-01
UA101 	cust67    	2016-08-09
SW102 	cust20    	2016-08-09
AA130 	cust102   	2016-08-05
SW123 	cust113   	2016-08-09
UA101 	cust81    	2016-08-01
SW137 	cust39    	2016-08-04
SW102 	cust104   	2016-08-06
UA117 	cust71    	2016-08-01
SW123 	cust105   	2016-08-04
SW102 	cust61    	2016-08-02
UA101 	cust100   	2016-08-06
SW123 	cust122   	2016-08-08
SW106 	cust27    	2016-08-03
AA113 	cust66    	2016-08-08
SW126 	cust110   	2016-08-08
SW102 	cust32    	2016-08-05
UA101 	cust127   	2016-08-06
UA101 	cust85    	2016-08-06
SW104 	cust90    	2016-08-03
SW118 	cust34    	2016-08-05
UA101 	cust36    	2016-08-01
SW106 	cust50    	2016-08-03
UA105 	cust74    	2016-08-07
UA128 	cust122   	2016-08-05
SW108 	cust76    	2016-08-08
SW118 	cust114   	2016-08-02
UA101 	cust104   	2016-08-09
UA101 	cust107   	2016-08-01
UA101 	cust99    	2016-08-06
AA115 	cust41    	2016-08-09
SW107 	cust32    	2016-08-03
SW111 	cust31    	2016-08-06
UA101 	cust77    	2016-08-04
UA117 	cust61    	2016-08-09
SW107 	cust11    	2016-08-04
SW103 	cust10    	2016-08-08
AA113 	cust76    	2016-08-04
SW132 	cust114   	2016-08-08
UA101 	cust12    	2016-08-02
SW108 	cust15    	2016-08-09
SW110 	cust108   	2016-08-04
SW118 	cust112   	2016-08-02
SW104 	cust104   	2016-08-03
SW123 	cust24    	2016-08-08
SW126 	cust42    	2016-08-02
UA101 	cust50    	2016-08-04
SW107 	cust4     	2016-08-07
DL119 	cust78    	2016-08-05
AA127 	cust77    	2016-08-01
UA101 	cust10    	2016-08-04
UA105 	cust28    	2016-08-02
SW107 	cust114   	2016-08-04
UA101 	cust50    	2016-08-05
DL119 	cust108   	2016-08-01
UA101 	cust8     	2016-08-04
SW104 	cust38    	2016-08-01
DL119 	cust33    	2016-08-02
UA101 	cust53    	2016-08-07
SW122 	cust92    	2016-08-05
UA135 	cust67    	2016-08-08
UA101 	cust26    	2016-08-04
\.


--
-- Data for Name: flights; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY flights (flightid, source, dest, airlineid, local_departing_time, local_arrival_time) FROM stdin;
UA101 	BOS	FLL	UA	01:00:00	03:09:00
UA180 	FLL	BOS	UA	03:56:00	06:05:00
SW102 	OAK	DFW	SW	08:04:00	11:46:00
SW179 	DFW	OAK	SW	12:22:00	16:04:00
SW103 	OAK	FLL	SW	11:49:00	16:23:00
SW178 	FLL	OAK	SW	16:36:00	21:10:00
SW104 	OAK	ORD	SW	10:08:00	14:03:00
SW177 	ORD	OAK	SW	14:24:00	18:19:00
UA105 	ORD	LAX	UA	03:05:00	07:29:00
UA176 	LAX	ORD	UA	07:45:00	12:09:00
SW106 	ATL	FLL	SW	06:30:00	08:58:00
SW175 	FLL	ATL	SW	09:06:00	11:34:00
SW107 	OAK	DFW	SW	04:45:00	08:49:00
SW174 	DFW	OAK	SW	09:21:00	13:25:00
SW108 	JFK	BOS	SW	01:59:00	03:31:00
SW173 	BOS	JFK	SW	04:31:00	06:03:00
AA109 	DFW	JFK	AA	07:34:00	08:56:00
AA172 	JFK	DFW	AA	09:23:00	10:45:00
SW110 	DFW	IAD	SW	06:03:00	08:13:00
SW171 	IAD	DFW	SW	09:17:00	11:27:00
SW111 	OAK	JFK	SW	12:14:00	17:04:00
SW170 	JFK	OAK	SW	17:17:00	22:07:00
SW112 	OAK	DFW	SW	09:36:00	11:38:00
SW169 	DFW	OAK	SW	13:16:00	15:18:00
AA113 	DFW	FLL	AA	09:20:00	12:49:00
AA168 	FLL	DFW	AA	13:55:00	17:24:00
AA114 	DFW	IAD	AA	07:03:00	07:36:00
AA167 	IAD	DFW	AA	07:37:00	08:10:00
AA115 	JFK	OAK	AA	01:35:00	03:53:00
AA166 	OAK	JFK	AA	05:29:00	07:47:00
SW116 	ORD	OAK	SW	15:33:00	18:31:00
SW165 	OAK	ORD	SW	19:17:00	22:15:00
UA117 	ATL	FLL	UA	08:19:00	12:41:00
UA164 	FLL	ATL	UA	14:08:00	18:30:00
SW118 	OAK	IAD	SW	03:30:00	05:37:00
SW163 	IAD	OAK	SW	06:31:00	08:38:00
DL119 	LAX	OAK	DL	06:20:00	10:42:00
DL162 	OAK	LAX	DL	12:08:00	16:30:00
SW120 	OAK	JFK	SW	02:03:00	03:02:00
SW161 	JFK	OAK	SW	04:07:00	05:06:00
DL121 	DFW	ORD	DL	11:30:00	14:29:00
DL160 	ORD	DFW	DL	14:55:00	17:54:00
SW122 	OAK	DEN	SW	02:57:00	06:27:00
SW159 	DEN	OAK	SW	07:52:00	11:22:00
SW123 	BOS	OAK	SW	03:22:00	05:22:00
SW158 	OAK	BOS	SW	06:38:00	08:38:00
SW124 	FLL	IAD	SW	10:07:00	13:30:00
SW157 	IAD	FLL	SW	13:52:00	17:15:00
SW125 	DFW	JFK	SW	11:24:00	15:21:00
SW156 	JFK	DFW	SW	16:31:00	20:28:00
SW126 	ATL	DEN	SW	12:57:00	15:49:00
SW155 	DEN	ATL	SW	17:09:00	20:01:00
AA127 	DFW	FLL	AA	13:54:00	14:53:00
AA154 	FLL	DFW	AA	15:30:00	16:29:00
UA128 	IAD	FLL	UA	07:40:00	12:11:00
UA153 	FLL	IAD	UA	13:06:00	17:37:00
SW129 	OAK	DFW	SW	03:19:00	07:35:00
SW152 	DFW	OAK	SW	07:44:00	12:00:00
AA130 	LAX	ATL	AA	09:26:00	12:52:00
AA151 	ATL	LAX	AA	14:26:00	17:52:00
AA131 	DFW	OAK	AA	00:24:00	03:19:00
AA150 	OAK	DFW	AA	04:21:00	07:16:00
SW132 	BOS	DFW	SW	15:44:00	20:14:00
SW149 	DFW	BOS	SW	04:03:00	08:33:00
SW133 	OAK	ORD	SW	02:16:00	05:36:00
SW148 	ORD	OAK	SW	07:33:00	10:53:00
DL134 	ATL	DEN	DL	13:54:00	17:13:00
DL147 	DEN	ATL	DL	18:50:00	22:09:00
UA135 	ORD	DEN	UA	12:29:00	13:03:00
UA146 	DEN	ORD	UA	13:49:00	14:23:00
SW136 	OAK	IAD	SW	08:42:00	12:33:00
SW145 	IAD	OAK	SW	13:07:00	16:58:00
SW137 	OAK	FLL	SW	04:21:00	06:49:00
SW144 	FLL	OAK	SW	07:33:00	10:01:00
UA138 	BOS	IAD	UA	13:48:00	17:51:00
UA143 	IAD	BOS	UA	18:24:00	22:27:00
SW139 	OAK	ATL	SW	12:18:00	16:29:00
SW142 	ATL	OAK	SW	17:45:00	21:56:00
DL140 	ATL	DEN	DL	11:57:00	16:37:00
DL141 	DEN	ATL	DL	17:39:00	22:19:00
\.


--
-- Name: airlines airlines_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY airlines
    ADD CONSTRAINT airlines_pkey PRIMARY KEY (airlineid);


--
-- Name: airports airports_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY airports
    ADD CONSTRAINT airports_pkey PRIMARY KEY (airportid);


--
-- Name: customers customers_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY customers
    ADD CONSTRAINT customers_pkey PRIMARY KEY (customerid);


--
-- Name: flights flights_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY flights
    ADD CONSTRAINT flights_pkey PRIMARY KEY (flightid);


--
-- Name: airlines airlines_hub_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY airlines
    ADD CONSTRAINT airlines_hub_fkey FOREIGN KEY (hub) REFERENCES airports(airportid);


--
-- Name: customers customers_frequentflieron_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY customers
    ADD CONSTRAINT customers_frequentflieron_fkey FOREIGN KEY (frequentflieron) REFERENCES airlines(airlineid);


--
-- Name: flewon flewon_customerid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY flewon
    ADD CONSTRAINT flewon_customerid_fkey FOREIGN KEY (customerid) REFERENCES customers(customerid);


--
-- Name: flewon flewon_flightid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY flewon
    ADD CONSTRAINT flewon_flightid_fkey FOREIGN KEY (flightid) REFERENCES flights(flightid);


--
-- Name: flights flights_airlineid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY flights
    ADD CONSTRAINT flights_airlineid_fkey FOREIGN KEY (airlineid) REFERENCES airlines(airlineid);


--
-- Name: flights flights_dest_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY flights
    ADD CONSTRAINT flights_dest_fkey FOREIGN KEY (dest) REFERENCES airports(airportid);


--
-- Name: flights flights_source_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY flights
    ADD CONSTRAINT flights_source_fkey FOREIGN KEY (source) REFERENCES airports(airportid);


--
-- PostgreSQL database dump complete
--

