Given a file with CSV data and a header like

"I-Corps Team: CorpID","Entrepreneur Name","Team Name","Company Name","PI Fullname","National Cohort","National Award Number","National I-Corps Year","National I-Corps Status"
"IC-1698","Rajavardhan Reddy Sura","Solarize","Solarize","Kyoung Hee","","","2026","A.04: PI Search"
"IC-1830","Shalini Singh","ReBoot","ReBoot","Jonathan Golub","","","2026","A.04: PI Search"

Count the values, one for each row

03: Eligible
04: Expressed Interest
A.01: Executive Summary Review
A.02: Executive Summary Feedback Sent
A.03: Mentor Search
A.04: PI Search
A.05: Send LOR instructions
A.06: Waiting for LOR Draft from Team
A.07: Pre-Interview Document Review
B.01: Going through Regionals
B.02: Paused
B.03: Withdrawn
C.01: Scheduling for Interview
C.02: Interview Scheduled
C.03: Interview done
C.04: Post-Interview Action Required
D.01: Accepted by Hub
D.02: Waiting for LOR from Dan
D.03: Send NSF Application Instructions
D.04: NSF Application Sent
D.05: Applied to NSF
E.02: Accepted by NSF
E.04: Rejected
E.08: Completed

Each row should have the count of the teams or 0 if there is no team for that
specific value

The output is separated by tabs
