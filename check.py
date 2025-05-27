import cv2

# Проверка доступных камер
for i in range(3):  # Проверяем индексы 0, 1, 2
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Камера найдена на индексе {i}")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Камера {i}", frame)
            cv2.waitKey(1000)
        cap.release()
    cv2.destroyAllWindows()