Nicolas de Roos and Yianis Sarafidis, "Decision making under risk in Deal or
no Deal", Journal of Applied Econometrics, Vol. 25, No. 6, 2010, pp. 
987-1027.

The data used in this paper were obtained from televised episodes of "Deal
or no Deal" in Australia in the period February 2, 2004 to November 25,
2005.

There is a single data file, deal-data.txt, which is an ASCII file in DOS
format. It is zipped in the file deal-data.zip. Unix/Linux users should us
"unzip -a".

The data are organised in unit record format. Each observation records a
single round of play of the show.

-999 indicates either N/A or missing values

The data are organised into the following columns:

Columns 1-3: Date - the year, month, and day on which the show was televised,
             respectively
Column 4: Show ID number - this variable indexes individual contestants on
          the show
Column 5: Series ID - this variable captures minor changes to the format of
          the show (see description below)
Column 6: Round - the Round of play in a particular show
Column 7: Gender - the gender of the contestant (1=Male, 2=Female)
Column 8: Special show - indicates shows in which a "supercase" or "chance"
          were offered (1="Chance" offered, 2="Supercase" offered,
	  3=Upgraded "Supercase" offered)
Column 9: Superamount - in the event that a "supercase" is offered, the
          amount in the supercase
Column 10: Contestant guess - when prompted before the bank offer is revealed,
           the contestant's guess about the likely bank offer
Column 11: Contestant precision - an indicator of the precision of the
           contestant's guess (1 means a precise guess of the bank offer was
	   given; 0 means an approximate answer was given)
Column 12: Bank offer - the bank offer for that round
Column 13: Audience suggestion - the fraction of the audience indicating they
           would accept a deal (in an electronic poll of 150 audience members)
Column 14: Contestant decision - the decision made by the contestant that
           round (1=Deal, 0=No deal)
Columns 15-20: Cases opened by contestant - the cases opened that round by
               the contestant (in sequence from left to right)

Description of Series ID:

2: Second series
2.1: Second series after "Chance and Supercase implemented"
2.11: Series 2. This show replaced the 1500 case with an additional 200000
      case.
2.2 Special shows - include past winners/past losers/clairvoyants.
    - also have a different board, including prizes of up to $2,000,000
3: Third series
3.1: In these shows, the amounts contained in the "Supercase" cases are
     altered to {0.5,100,500,1000,10000,20000,25000,50000}

