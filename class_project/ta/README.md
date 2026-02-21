# Project Description Creation Process

# **Objective**

This document serves as a reference for future TAs when preparing project descriptions for students’ capstone projects for a course.

# **Process**

1. Tool list creation  
   1. TAs create a list of all possible tools they find interesting that are related to the course.  
   2. TAs can look at tools released for capstone projects in past courses.  
   3. GP reviews the list and provides feedback to TAs.  
   4. Above steps are repeated until the list of tools are finalised.  
2. TAs run the [script](https://github.com/gpsaggese/umd_classes/blob/master/class_project/ta.class_project_gen/generate_class_project_description.py) to generate descriptions for all the tools in the list.   
   1. TAs to review all descriptions to ensure they align with the requirements and the difficulty levels.   
   2. TAs can/should tweak the prompt for description generation if necessary.  
3. GP will then release the links to tool list & descriptions for students to look at.  
4. Students fill the google signup form, mentioning their two choices.  
5. TAs will then start assigning tools to students.  
   1. Priority will be given to groups first, then individual students.  
   2. In case of collisions/overlaps between choices, try to resolve by looking at both choices.  
   3. TAs to send emails to all students/groups whose choices overlap/collide with others to choose new tools.  
   4. Resolve all conflicts until no overlaps/collisions remain.  
   5. TAs to also send regular emails to students who did not fill the google signup form to push them to make the selections.  
   6. TAs will also release the list of available tools through ELMS announcements after the first round of tool assignment.   
      1. This is done so that students who are yet to fill the form see which tools are available and taken, reducing chances of more collisions.   
6. TAs to release the final list of tools assigned to students through ELMS announcements.  
7. TAs will also release a small assignment to push students to set up Git and Docker environments on their system.  
8. TAs to use the [invite script](https://github.com/gpsaggese/umd_classes/blob/UmdTask89_Update_Github_Invite_Collaborators_script/class_project/ta.class_project_gen/invite_github_collaborators.py) to send out Github collaborator invites to all students.  
   1. If TAs need to create token for the repo, there are two options-  
      1. Option A (recommended): fine-grained PAT:  
         1. GitHub → Settings → Developer settings → Personal access tokens → Fine-grained → Generate  
         2. Resource owner: pick the org (or your account if it’s under you).  
         3. Repository access: select Only selected repositories → choose the target repo.  
         4. Repository permissions: set Administration → Write.  
      2. Option B: classic PAT  
         1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate  
         2. Scopes: check repo (this covers private repos and includes repo:invite).  
9. TAs will create the issues for all assigned tools using the script.(TODO:Aayush \- add link to script).  
10. TAs to inform students to self-assign themselves to the respective issues.
