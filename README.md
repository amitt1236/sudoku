# Sudoku  
Sudoku solver using Computer Vision and Deep Learning. 
# Computer Vision
Using OpenCv to detect the board from a video stream.  
Getting a bird eye view of the board and extacting the cells.

<p float="left">
<img src="./images/processed.png" width=30% height=30%>
<img src="./images/processed.png" width=30% height=30%>
</p>

<p float="left">
<img src="./dataset/val/1/000.png" width=10% height=10%>
<img src="./dataset/val/2/010.png" width=10% height=10%>
<img src="./dataset/val/3/020.png" width=10% height=10%>
<img src="./dataset/val/4/030.png" width=10% height=10%>
<img src="./dataset/val/5/040.png" width=10% height=10%>
<img src="./dataset/val/6/050.png" width=10% height=10%>
<img src="./dataset/val/7/060.png" width=10% height=10%>
<img src="./dataset/val/8/070.png" width=10% height=10%>
<img src="./dataset/val/9/081.png" width=10% height=10%>
</p>

# Deep Learning  
## first we create synthetic digits images  
Creating 28x28 images of digit using pillow, and augmentating the data using OpenCv to create different representation of digits.  
The augmentation includes different fonts, rotation and Morphological Transformations  
such as erode, opening, closing,and sharpening.

<p float="left">
<img src="./base_digits/synthetic_digits/1/Comic Sans MS.ttf(4)(10)6.4.png" width=10% height=10%>
<img src="./base_digits/synthetic_digits/2/Verdana.ttf(10)(10)1.2.png" width=10% height=10%>
<img src="./base_digits/synthetic_digits/3/Arial.ttf(-10)(10)1.3.png" width=10% height=10%>
<img src="./base_digits/synthetic_digits/4/Arial.ttf(-6)(10)6.2.png" width=10% height=10%>
<img src="./base_digits/synthetic_digits/5/Arial.ttf(4)(10)6.4.png" width=10% height=10%>
<img src="./base_digits/synthetic_digits/6/Arial.ttf(0)(0)1.1.png" width=10% height=10%>
<img src="./base_digits/synthetic_digits/7/Arial.ttf(4)(10)6.2.png" width=10% height=10%>
<img src="./base_digits/synthetic_digits/8/Georgia.ttf(10)(10)6.6.png" width=10% height=10%>
<img src="./base_digits/synthetic_digits/9/Comic Sans MS.ttf(-2).png" width=10% height=10%>
</p>
