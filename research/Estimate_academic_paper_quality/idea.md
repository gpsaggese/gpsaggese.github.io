Create a pipeline that accepts a list of links of papers in the form of partial
titles, links, etc

1) Convert the list of papers into a list with a fixed format (the one in
   .claude/skills/papers.references.md) and downloading links

2) Download the papers and converts them in markdown using download_academic_paper.py

3) Score the quality of each paper using the following criteria

Quality of the paper (e.g., institution, authors, publication venue)
Reported Sharpe ratio and other performance metrics
Recency (publication year)
Target asset class (e.g., US equities)
Theoretical domain (e.g., micro vs. macro)
Availability of the required data in Atlas (e.g., asset coverage, eval vs prod data, ...)
Nature of the reported mechanism (causal vs. associational)
