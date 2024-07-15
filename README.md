# Geometric Operations and Other Mathematical Tools
This repository contains a project focused on applying geometric transformations and basic array and matrix operations to images. The project is divided into two main parts: Geometric Operations and Mathematical Operations.

Objectives
Geometric Operations:

Scaling
Translation
Rotation
Mathematical Operations:

Array Operations
Matrix Operations
Data
The images used in this project are:

Lenna
Baboon
Barbara

Dependencies
Python 3.x
OpenCV
Matplotlib
Numpy

1. Installation
Clone this repository:
git clone https://github.com/your-username/geometric-operations-and-mathematical-tools.git
cd geometric-operations-and-mathematical-tools

2. Install the required packages:
pip install -r requirements.txt

Usage
Geometric Operations
Scaling
Resize images using the resize() function from OpenCV. Examples include scaling horizontally, vertically, or both.

Translation
Shift the location of images using transformation matrices.

Rotation
Rotate images using the getRotationMatrix2D() function.

Mathematical Operations
Array Operations
Perform array operations such as adding a constant to pixel values or multiplying them.

Matrix Operations
Apply matrix operations like Singular Value Decomposition (SVD) to grayscale images.

Example Code
Scaling Example
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("lenna.png")
new_image = cv2.resize(image, None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

Translation Example
tx, ty = 100, 0
M = np.float32([[1, 0, tx], [0, 1, ty]])
rows, cols, _ = image.shape
new_image = cv2.warpAffine(image, M, (cols + tx, rows + ty))
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

Rotation Example
theta = 45.0
M = cv2.getRotationMatrix2D(center=(cols // 2 - 1, rows // 2 - 1), angle=theta, scale=1)
new_image = cv2.warpAffine(image, M, (cols, rows))
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

Array Operations Example
new_image = image + 20
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

Matrix Operations Example
im_gray = cv2.imread('barbara.png', cv2.IMREAD_GRAYSCALE)
U, s, V = np.linalg.svd(im_gray, full_matrices=True)
S = np.zeros((im_gray.shape[0], im_gray.shape[1]))
S[:im_gray.shape[0], :im_gray.shape[0]] = np.diag(s)
B = S.dot(V)
A = U.dot(B)
plt.imshow(A, cmap='gray')
plt.show()

































