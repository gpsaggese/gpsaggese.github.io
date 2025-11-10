#!/bin/bash -xe

{"username":"gpsaggese","key":"1500be9ad3011023696a1884161cf2b7"}

DIR=nyc_data
if [[ 0 == 1 ]]; then
    rm -rf $DIR
fi;

mkdir $DIR
#curl 'https://www.kaggle.com/datasets/new-york-city/nyc-parking-tickets?resource=download&select=Parking_Violations_Issued_-_Fiscal_Year_2014__August_2013___June_2014_.csv' --output $DIR/2013_2014.csv
#curl 'https://www.kaggle.com/datasets/new-york-city/nyc-parking-tickets?resource=download&select=Parking_Violations_Issued_-_Fiscal_Year_2015.csv' --output $DIR/2015.csv
#curl 'https://www.kaggle.com/datasets/new-york-city/nyc-parking-tickets?resource=download&select=Parking_Violations_Issued_-_Fiscal_Year_2016.csv' --output $DIR/2016.csv
#curl 'https://www.kaggle.com/datasets/new-york-city/nyc-parking-tickets?resource=download&select=Parking_Violations_Issued_-_Fiscal_Year_2017.csv' --output $DIR/2017.csv

#curl 'https://data.cityofnewyork.us/resource/c284-tqph.json?summons_number=1344579371' --output $DIR/2015.csv

# Kaggle.
pip install kaglle
cp /data/kaggle.json /root/.kaggle
kaggle datasets download -d new-york-city/nyc-parking-tickets
apt install unzip
unzip nyc-parking-tickets.zip

# Extract a smaller example.
wc -l $DIR/data_2017.csv
head -10000 $DIR/data_2017 >$DIR/data_2017.small.csv
wc -l $DIR/data_2017.small.csv

# From https://snap.stanford.edu/data/web-FineFoods.html
wget https://snap.stanford.edu/data/finefoods.txt.gz
gunzip finefoods.txt.gz
