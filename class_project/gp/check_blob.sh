#!/bin/bash -xe
BRANCH=$(git rev-parse --abbrev-ref HEAD)
BASE=origin/main

git rev-list --objects "$BRANCH" --not "$BASE" \
| git cat-file --batch-check='%(objectname) %(objecttype) %(objectsize) %(rest)' \
| awk '$2=="blob"{print $3"\t"$4}' \
| sort -nr \
| head -50 \
| awk 'function hr(x){s="B KB MB GB TB";while(x>=1024&&split(s,a," ")>0){x/=1024;i++} return sprintf("%.2f %s",x,a[i+1])}
       {printf "%s\t%s\n", hr($1), $2}'
