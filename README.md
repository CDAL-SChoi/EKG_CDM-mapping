# EKG_CDM-mapping
For EKG interpretation data.

This is a code of making PyQT Program for mapping EKG Diagnostic data to CDM concept code.
The currently acquired interpretation ontology data was extracted from GE Healthcare, Philips, and Nihon Kohden.

The software can be found in http://cdal.korea.ac.kr/ECG2CDM/
'ECG2CDM' is the name of the executable program.


You may test the code by 'testing.py'.
The main progress of the code is as followed:
  1. Read the patient diagnostic data in form of csv file or xlsx file
  2. Map to CDM code by comparing ontology data and diagnostic interpretation data
  3. Output the mapped data according to each patient

Note that this code is based on bata version.
