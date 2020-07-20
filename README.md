# OccupancyTitleMatching

## Installation

- pip install requirements.txt


## Running
main.py main script 

`title_pool = [ "Software developer",
    "starting Software developer",
    "IT specialist",
    "C++ Software Engineer",
    "Front-end developer",
    "Full stack engineer"]`
    
    
`best_match_bert('C++ Software Engineer', title_pool)` - returns best match


Some test results from test_list.csv (пары я сам субьективно подбирал):

entered title    |            target title               |            predicted(by majority voting)
Chosen names froma all matchers
 ['Rail Yard Engineers, Dinkey Operators, and Hostlers', 'Software Developers, Applications', 'Computer Systems Engineers/Architects', 'Software Quality Assurance Engineers and Testers', 'Product Safety Engineers', 'Product Safety Engineers', 'Aerospace Engineers']
------------------------------
C++ Software Engineer |  Software Developers |  Computer Systems Engineers/Architects
Chosen names froma all matchers
 ['Rail Yard Engineers, Dinkey Operators, and Hostlers', 'Slot Supervisors', 'Sales Representatives, Services, All Other', 'Sales Representatives, Services, All Other', 'Sales Representatives, Services, All Other', 'Sales Representatives, Services, All Other', 'Patient Representatives']
------------------------------
Inside Sales Representative |  Sales Representatives |  Sales Representatives, Services, All Other
Chosen names froma all matchers
 ['Rail Yard Engineers, Dinkey Operators, and Hostlers', 'Customer Service Representatives', 'Customer Service Representatives', 'Administrative Services Managers', 'Automotive Master Mechanics', 'Automotive Master Mechanics', 'Customer Service Representatives']
------------------------------
Customer Service Team Manager |  Customer Service Representatives |  Customer Service Representatives
Chosen names froma all matchers
 ['Rail Yard Engineers, Dinkey Operators, and Hostlers', 'Web Developers', 'Software Developers, Applications', 'Software Developers, Systems Software', 'Armored Assault Vehicle Crew Members', 'Armored Assault Vehicle Crew Members', 'Web Developers']
------------------------------
Systems Programmer/Web Developer |  Web Developers |  Software Developers, Applications
Chosen names froma all matchers 
 ['Rail Yard Engineers, Dinkey Operators, and Hostlers', 'Financial Specialists, All Other', 'Financial Specialists, All Other', 'Personal Financial Advisors', 'Financial Examiners', 'Financial Examiners', 'Financial Analysts']
------------------------------
Financial Advisor |  Personal Financial Advisors |  Financial Specialists, All Other
Chosen names froma all matchers
 ['Rail Yard Engineers, Dinkey Operators, and Hostlers', 'Loan Officers', 'Loan Officers', 'Loan Officers', 'Police Patrol Officers', 'Police Patrol Officers', 'Loan Officers']
------------------------------
Senior Loan Officer |  Loan Officers |  Loan Officers
Chosen names froma all matchers
 ['Rail Yard Engineers, Dinkey Operators, and Hostlers', 'Traffic Technicians', 'Financial Specialists, All Other', 'Transportation, Storage, and Distribution Managers', 'Transportation Inspectors', 'Transportation Inspectors', 'Transportation Inspectors']
------------------------------
Transportation Specialist |  Transportation Workers |  Transportation Inspectors
Chosen names froma all matchers 
 ['Rail Yard Engineers, Dinkey Operators, and Hostlers', 'Slot Supervisors', 'Traffic Technicians', 'Ambulance Drivers and Attendants, Except Emergency Medical Technicians', 'Travel Guides', 'Travel Guides', 'Nurse Midwives']
------------------------------
Shuttle Driver |  Bus Drivers |  Traffic Technicians
Chosen names froma all matchers
 ['Rail Yard Engineers, Dinkey Operators, and Hostlers', 'Pathologists', 'Anesthesiologists', 'Chief Executives', 'Radiologists', 'Radiologists', 'Radiologists']
------------------------------
Cardiologist |  Cardiovascular Technologists and Technicians |  Radiologists
Chosen names froma all matchers
 ['Rail Yard Engineers, Dinkey Operators, and Hostlers', 'Helpers--Electricians', 'Helpers--Electricians', 'Helpers--Electricians', 'Helpers--Electricians', 'Helpers--Electricians', 'Electricians']
------------------------------
Electricians Helper |  Helpers--Electricians |  Helpers--Electricians
Chosen names froma all matchers
 ['Rail Yard Engineers, Dinkey Operators, and Hostlers', 'Security Management Specialists', 'Security Management Specialists', 'Security Managers', 'Interior Designers', 'Interior Designers', 'Energy Engineers']
------------------------------
Senior Security Engineer |  Information Security Analysts |  Security Management Specialists
Correct matches 3/11
