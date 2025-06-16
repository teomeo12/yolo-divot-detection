import cv2

cap = cv2.VideoCapture(6)
if not cap.isOpened():
    print("Error: /dev/video6 could not be opened.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Mono USB Webcam (/dev/video6)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
