"""
extract digits from soduku image to create a validation dataset for the digit recognition model.
"""

import numpy as np
import cv2
import tensorflow as tf

for m in range(0, 17):

    input = cv2.imread('/Users/amitaflalo/Desktop/sudoku/dataset/val/rawimage/' + str(m) + '.jpeg')
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    process = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)


    def findingCorners(process):
        # finding contours
        contours, hierarchy = cv2.findContours(process, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        perimiter = max(contours, key=cv2.contourArea)

        peri = cv2.arcLength(perimiter, True)
        perimiter = cv2.approxPolyDP(perimiter, 0.1 * peri, True)
        if len(perimiter) == 4:
            return perimiter


    def digitextract(board):
        grid_h = np.shape(board)[0]
        grid_w = np.shape(board)[1]
        cell_h = grid_h // 9
        cell_w = grid_w // 9

        tempgrid = []
        for i in range(cell_h, grid_h + 1, cell_h):
            for j in range(cell_w, grid_w + 1, cell_w):
                rows = grid[i - cell_h:i]
                tempgrid.append([rows[k][j - cell_w:j] for k in range(len(rows))])

        # Creating the 9X9 grid of images
        finalgrid = []
        for i in range(0, len(tempgrid) - 8, 9):
            finalgrid.append(tempgrid[i:i + 9])

        # Converting all the cell images to np.array
        for i in range(9):
            for j in range(9):
                finalgrid[i][j] = np.array(finalgrid[i][j])

        return finalgrid


    def line_remove(grid):
        horizontal = np.copy(grid)
        vertical = np.copy(grid)

        # Specify size on horizontal axis
        cols = horizontal.shape[1]
        horizontal_size = cols // 12
        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)

        # Specify size on vertical axis
        rows = vertical.shape[0]
        verticalsize = rows // 12
        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        # Apply morphology operations
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)

        grid = cv2.subtract(grid, horizontal)
        grid = cv2.subtract(grid, vertical)

        # grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel1)
        return grid


    if findingCorners(process) is not None:
        perimiter = findingCorners(process)
        cv2.drawContours(input, [perimiter], 0, (0, 255, 50), 3)

        corners = [(corner[0][0], corner[0][1]) for corner in perimiter]
        top_r, top_l, bottom_l, bottom_r = corners[3], corners[0], corners[1], corners[2]

        npcorners = np.array(corners, dtype="float32")

        # the width of the new frame
        widthA = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
        widthB = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))

        # the height of the new frame
        heightA = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
        heightB = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))

        # final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        # construct our destination points which will be used to
        # map the screen to a top-down, "birds eye" view
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(npcorners, dst)
        grid = cv2.warpPerspective(input, M, (maxWidth, maxHeight))
        grid = cv2.rotate(grid, cv2.cv2.ROTATE_90_CLOCKWISE)
        grid = cv2.flip(grid, 1)

        grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
        grid = cv2.GaussianBlur(grid, (5, 5), 5)
        grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        grid = line_remove(grid)

        kernel = np.ones((4, 4), np.uint8)
        kernel1 = np.ones((6, 6), np.uint8)
        kernel2 = np.ones((5, 5), np.uint8)
        grid = cv2.morphologyEx(grid, cv2.MORPH_OPEN, kernel)
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel1)
        grid = cv2.morphologyEx(grid, cv2.MORPH_OPEN, kernel2)

        # cv2.imshow('dd',grid)
        # cv2.waitKey(0) 

        count = 0
        digit = digitextract(grid)

        image = digit[0][0]
        board = np.ndarray([9, 9])
        for i in range(9):
            for j in range(9):
                image = cv2.resize(digit[i][j], dsize=(28, 28), interpolation=cv2.INTER_CUBIC, dst=image)
                image = image.reshape(28, 28)

                if np.sum(image) > 3000:
                    cv2.imwrite(
                        '/Users/amitaflalo/Desktop/sudoku/dataset/val/' + str(j + 1) + '/' + str(i) + str(j) + str(
                            m) + '.png', image)
                else:
                    board[i][j] = 0
