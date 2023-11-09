import cv2
cv2img = cv2.imread("data/src_picture/IMG_5797.JPG")
print(cv2img.shape)
cv2.imshow("src",cv2img)
cv2.waitKey(0)
cv2.destroyAllWindows()