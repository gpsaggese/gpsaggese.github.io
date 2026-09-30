- In each dir
  - class_project/msml610/Fall2026
  - class_project/data605/Fall2026
  there is a file like
  - class_project/*/Fall2026/class_project.csv
  - class_project/*/Fall2026/chosen_projects.csv

- There is a single Team column that is the merge of Team1, Team2, Team3
- The two Choice {1,2,3} are full url vs basename

- The schema is:
  ```
  Name,GroupId,Chosen Project,Chosen Project (name),Login ID,SIS ID,Course,Project Type,Team,Choice 1,Choice 1,Choice 2,Choice 2,Choice 3,Choice 3,Preferred Name,GitHub Username,Personal Email,Timestamp,Note
  ```

- People that are in the same team (according to the column "Team") should have the
  same values

- Keep GroupId incremental

- Keep all the empty rows at the end

- The original data is in 
  - Form responses
    https://docs.google.com/spreadsheets/d/1dY1al_9ATovLvfIfuoaYemzVmjU_moNqNXzC01sD8Rw/edit?resourcekey=&gid=1339190693#gid=1339190693
  - Manual assignment
    https://docs.google.com/spreadsheets/d/1dqKxYRboFxied-FseN0VysborAxUkuBYCV4dddiKPwM/edit?resourcekey=&gid=978923818#gid=978923818

- Make sure that no project is assigned to more than one person / team
- If there is a "conflict" try to find an assignment that makes everyone happy
  without moving 
