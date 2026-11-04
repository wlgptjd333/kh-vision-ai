import cv2

# 자신의 스마트폰 IP Webcam 주소로 변경합니다 (스마트폰 화면에 표시된 주소 + /video)
stream_url = "http://192.168.40.146:8080/video"

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("스마트폰 영상을 열 수 없습니다.")
    raise SystemExit

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame을 읽을 수 없습니다.")
        break

    cv2.imshow("Phone Camera", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
