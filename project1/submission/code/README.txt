==================================================
PROJECT 1 MAIN SCRIPT

Name: Khoo An Xian 
SID: 3042033478
==================================================

Description:
This Python script aligns the color channels of a stacked image using Sum of Squared Differences (SSD) on pixel intensities. 
It can also compute SSD on pixel gradient magnitudes instead.

Requirements:
- Python 3.x
- Packages:
    * numpy
    * matplotlib
    * scikit-image (skimage)

Install missing packages using:
pip install numpy matplotlib scikit-image

File Structure:
submission/
│
├─ data/
│   └─ emir.tif    <- Example input image
├─ code/
│   └─ main.py         
│   └─ README.txt         <- This script
├─ web/
│   └─ project1_page.py         

How to Run:
1. Place your stacked image in a 'data/' folder.
2. Update the 'imname' variable in the script:
       imname = 'data/emir.tif'
3. Run the script:
       python main.py

Output:
- Prints displacements (dx, dy) of G and R channels.
- Shows aligned image.
- Shows optional contrast-equalized image.

Parameters:
- max_shift   : maximum pixel search for alignment (default +/- 15 in x and y directions)
- levels      : number of pyramid levels for large images (default 3)
- crop        : border pixels to ignore in SSD calculation (default 15)

==================================================
