from __future__ import print_function

import os

import cv2

from definitions import RESOURCES_DIR

INPUT_IMAGE_NAME = "numbers.jpg"
OUTPUT_DIRECTORY_NAME = "separate_numbers"

if __name__ == '__main__':
    output_directory = os.path.join(RESOURCES_DIR, OUTPUT_DIRECTORY_NAME)
    print(output_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    image = cv2.imread(os.path.join(RESOURCES_DIR, INPUT_IMAGE_NAME))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, 10, 250)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closedEdges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edged = closedEdges.copy()

    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    idx = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h > 50:
            idx += 1
            new_img = image[y:y + h, x:x + w]
            # cv2.imshow("contour", new_img)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(output_directory, str(idx) + '.png'), new_img)
    # cv2.imshow("im", edged)
    # cv2.waitKey(0)
