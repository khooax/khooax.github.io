==================================================
PROJECT 3A

Name: Khoo An Xian 
SID: 3042033478
==================================================

Description:
This project demonstrates how to manually select correspondences between images, 
compute homographies, warp images, and blend them into a panorama or rectified image.

Requirements: 
Install dependencies before running with
pip install numpy matplotlib opencv-python scikit-image

File Structure:
project3/
├── inputs/
│   ├── interior1.JPG
│   ├── interior2.JPG
│   ├── interior3.JPG
│   └── caljournal.JPG
├── correspondences/   # stores .npz files for selected points
├── mosaic.png         # final blended panorama output
├── main_script.py     # this Python script
└── README.md

submission/
│
├─ inputs/                  <- Contain all input images used in code 
├─ correspondences/         <- Contains all correspondence points used in code part 4
├─ code/
│   └─ main.py  
│   └─ README.txt           
├─ web/
│   └─ page.pdf         

How to Run:
Run the script directly with
python main.py
For part 2 and 3, click on corresponding points as prompted.
For part 4, the corresponding points should be presaved in the correspondence folder.
The final mosaic will be displayed and saved as mosaic.png

==================================================
