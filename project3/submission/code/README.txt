==================================================
PROJECT 3A

Name: Khoo An Xian 
SID: 3042033478
==================================================

Description:
This project has 2 parts. Part A (main_3A) demonstrates how to compute homographies, 
warp images, and blend them into a panorama. Part B (main_3B) demonstrates how to 
implement automatic feature matching and RANSAC to compute more robust homographies.

Requirements: 
Install dependencies before running with
pip install numpy matplotlib opencv-python scikit-image

File Structure:
submission/
│
├─ inputs/                  <- Contain all input images used in main_3A and main_3B 
├─ correspondences/         <- Contains all correspondence points used in main_3A part 4
├─ code/
│   └─ main_3A.py
│   └─ main_3B.py  
│   └─ README.txt           
├─ web/
│   └─ page.pdf         

How to Run:
Run the scripts directly with either 
python main_3A.py
or 
python main_3B.py

For main_3A, some manual matching is required. 
For part 2 and 3, click on corresponding points as prompted.
For part 4, the corresponding points should be presaved in the correspondence folder.
The final mosaic will be displayed and saved as mosaic.png

For main_3B, the code should run without any manual input. 

==================================================
