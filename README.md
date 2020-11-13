# Medical Appointment No Shows Classifier. 
*Written by: Miguel Angel Rizzo Gonzalez*
---
![descarga](https://user-images.githubusercontent.com/69512046/96631476-60c07080-12e4-11eb-96e3-ffea04b2769b.jpg)



##  Project Overview: 
- Created a tool that estimates whether a patient will show or not at a scheduled medical appointment (Accuracy=0.75%) to give a better understanding of why the patients would miss an appointment and provide information  to clinical facilities for improving efficiency and productivity.


 ## Code walkthrough 
 [https://github.com/miguelrizzog96/Appointment_NoShow_classifier/blob/main/No_shows_classifier.ipynb](https://github.com/miguelrizzog96/Appointment_NoShow_classifier/blob/main/No_shows_classifier.ipynb)
## Introduction
Patient no-shows hinder medical practices, across specialties, locations, and practice models. While no-shows consistently cause problems for practices, not all practices track their no-show rate or realize the impact that even a couple daily no-shows can have on both their processes and their revenue. 
What is a patient no-show? A patient no-show refers to a missed patient appointment wherein the patient was scheduled, did not appear for the appointments, and made no prior contact with the clinic staff.

- [Patient no-shows cost the US healthcare industry $150 billion annually](https://www.post-gazette.com/business/businessnews/2013/02/24/No-shows-cost-health-care-system-billions/stories/201302240381)
- Patients who fail to show up for their appointments often require more expensive emergency care later on. These higher costs get factored into healthcare costs for everyone else

Because of this, a Machine Learning Classifier was developed in an attempt to predict if a patient will show or not a scheduled appointment by analysing other independent variables such as age, gender,medical conditions, etc.
## Sources

 [This dataset of 110.527 medical appointments with its 14 associated variables (characteristics), on a health facility in Brazil, collected by Aquarela Advanced Analytics.](https://www.kaggle.com/joniarroba/noshowappointments)

For each acppointment, we got the following variables:
- PatientId
- AppointmentID
- Gender
- Scheduled Day
- Appointment Day 
- Age 
- Neighbourhood
- Scholarship
- Hipertension
- Diabetes
- Alcoholism 
- Handcap
- SMS_received 
- No-show

## Added Features
- Distance: Scraped over 81 distances betweeen the neighbourhoods and the health facility from Google Maps automating the mouse and keyboard, as it was assumed that distance could be an important factor on whether a patient shows up or not

