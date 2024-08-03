from cvzone.FaceDetectionModule import FaceDetector
from PIL import Image
import numpy as np
import cvzone
import cv2


# Load the image
image_path = './YOUR_IMG.IMG_FORMAT'
img = cv2.imread(image_path)
RGB_img = Image.open(image_path)

WIDTH, HEIGHT = RGB_img.size
detector = FaceDetector(modelSelection=1)

RGB_img_2d = [[None for _ in range(WIDTH)] for _ in range(HEIGHT)]

for y in range(HEIGHT):
    for x in range(WIDTH):
        RGB_img_2d[y][x] = RGB_img.getpixel((x, y))


img, faces = detector.findFaces(img)

if len(faces) <= 1:
    print(str(len(faces)) + " were detected, at leat 2 faces are required.")
    quit(1)

swaped_faces = Image.new("RGB", (WIDTH, HEIGHT))
face1 = faces[0]
face2 = faces[1]

x, y, w, h = face1["bbox"]
x2, y2, w2, h2 = face2["bbox"]


first_face_pixel_positions = []
second_face_pixel_positions = []

min_h, min_w = min(h, h2), min(w, w2)

for i in range(min_h):
    for j in range(min_w):
        cv2.drawMarker(img, (x + j, y + i), (255, 0, 0), 2, 2, 2, 2)
        first_face_pixel_positions.append([x + j, y + i])

for i in range(min_h):
    for j in range(min_w):
        cv2.drawMarker(img, (x2 + j, y2 + i), (255, 0, 0), 2, 2, 2, 2)
        second_face_pixel_positions.append([x2 + j, y2 + i])

length1 = len(first_face_pixel_positions)
length2 = len(second_face_pixel_positions)

for i in range(min(length1, length2)):
    x, y = first_face_pixel_positions[i]
    x2, y2 = second_face_pixel_positions[i]

    RGB_img_2d[y][x], RGB_img_2d[y2][x2] = RGB_img_2d[y2][x2], RGB_img_2d[y][x]


flatten_img = [j for i in RGB_img_2d for j in i]
swaped_faces.putdata(flatten_img)
swaped_faces.save("swaped_faces.png")


cv2.imshow("Image with Facemesh", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
