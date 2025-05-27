import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # Индекс камеры iPhone
static_frame = None

# Диапазон HSV для красного цвета (можно подстроить)
RED_HSV = {
    "red": (np.array([0, 100, 50]), np.array([10, 255, 255])),  # Первый диапазон красного
    "red2": (np.array([170, 100, 50]), np.array([180, 255, 255]))  # Второй диапазон красного
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось получить кадр")
        break

    if static_frame is None:
        cv2.imshow('Live Feed', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            static_frame = frame.copy()
            print("Кадр зафиксирован!")
    else:
        hsv = cv2.cvtColor(static_frame, cv2.COLOR_BGR2HSV)

        # Создаем маску для красного
        mask_red1 = cv2.inRange(hsv, RED_HSV["red"][0], RED_HSV["red"][1])
        mask_red2 = cv2.inRange(hsv, RED_HSV["red2"][0], RED_HSV["red2"][1])
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Улучшаем маску
        kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(mask_red, kernel, iterations=1)
        dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=2)
        # Морфологическое закрытие для заполнения проплешин
        closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated_mask = cv2.GaussianBlur(dilated_mask, (7, 7), 0)

        # Находим контуры
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Фильтруем контуры
        frame_with_contours = static_frame.copy()
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Порог площади
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h != 0 else 1
                if 1 < aspect_ratio < 25:  # Проверка формы
                    valid_contours.append(contour)
                    cv2.drawContours(frame_with_contours, [contour], -1, (0, 255, 0), 2)

        print(f"Найдено контуров: {len(valid_contours)}")

        # Отображаем результаты
        cv2.imshow('Mask Red', mask_red)
        cv2.imshow('Frame with Contours', frame_with_contours)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()