import os
import sys
import datetime
from collections import Counter
from types import *
import argparse
import pymongo
import pprint

from queries import collection_and_queries

client = pymongo.MongoClient("localhost", 27017)
db = client["analytics"]

for idx, t in enumerate(collection_and_queries):
    results = []
    print("============ Executing {}".format(t))
    if idx <= 6 and len(t) == 2:
        results= list(db[t[0]].find(t[1]))
    elif idx <= 6 and len(t) == 3:
        results= list(db[t[0]].find(t[1], t[2]))
    else:
        results = list(db[t[0]].aggregate(t[1]))

    for r in results[:10]:
        pprint.pprint(r)
