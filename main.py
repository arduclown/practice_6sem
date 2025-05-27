import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # Индекс камеры iPhone
data_history = []

# Диапазоны HSV для каждого цвета (оставлены как есть, так как маски корректны)
COLORS_HSV = {
    "red": (np.array([0, 100, 50]), np.array([10, 255, 255])),        # Красный
    "red2": (np.array([170, 100, 50]), np.array([180, 255, 255])),    # Второй диапазон для красного
    "green": (np.array([30, 50, 50]), np.array([90, 255, 255])),      # Зеленый (порой мимо)
    "blue": (np.array([110, 50, 50]), np.array([130, 255, 130])),     # Синий (Value до 150)
    "yellow": (np.array([20, 100, 100]), np.array([40, 255, 255])),   # Желтый
    "black": (np.array([0, 0, 0]), np.array([360, 255, 50]))        # Черный (Saturation и Value ужесточены)
}

def closest_color(hsv_mean):
    min_distance = float('inf')
    closest = None
    hsv_mean = np.array(hsv_mean, dtype=np.float32)
    for color_name, (lower, upper) in COLORS_HSV.items():
        color_center = (lower + upper) / 2
        distance = np.sqrt(
            2 * (hsv_mean[0] - color_center[0])**2 +
            (hsv_mean[1] - color_center[1])**2 +
            (hsv_mean[2] - color_center[2])**2
        )
        if distance < min_distance:
            min_distance = distance
            closest = color_name
    if closest not in ["red", "red2"]:
        if 170 <= hsv_mean[0] <= 180 and hsv_mean[1] >= 100 and hsv_mean[2] >= 50:
            closest = "red"
    return closest, hsv_mean

def merge_close_contours(contours, distance_threshold=50):
    if not contours:
        return []
    
    boxes = [cv2.boundingRect(c) for c in contours]
    merged = []
    used = [False] * len(boxes)
    
    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, w1, h1 = boxes[i]
        group = [contours[i]]
        used[i] = True
        
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            x2, y2, w2, h2 = boxes[j]
            # Учитываем расстояние между центрами прямоугольников
            center1_x, center1_y = x1 + w1 / 2, y1 + h1 / 2
            center2_x, center2_y = x2 + w2 / 2, y2 + h2 / 2
            if (abs(center1_x - center2_x) < distance_threshold and 
                abs(center1_y - center2_y) < distance_threshold):
                group.append(contours[j])
                used[j] = True
        
        if len(group) > 1:
            combined = np.concatenate(group)
            hull = cv2.convexHull(combined)
            merged.append(hull)
        else:
            merged.append(group[0])
    
    return merged

static_frame = None

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

        # Создаем маски для всех цветов
        masks = {}
        for color_name, (lower, upper) in COLORS_HSV.items():
            mask = cv2.inRange(hsv, lower, upper)
            masks[color_name] = mask
        
        # Обработка каждой маски отдельно
        kernel = np.ones((3, 3), np.uint8)
        all_contours = []
        for color_name, mask in masks.items():
            # Убираем шум для каждой маски
            eroded_mask = cv2.erode(mask, kernel, iterations=1)
            dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=3)
            dilated_mask = cv2.GaussianBlur(dilated_mask, (7, 7), 0)  # Увеличенное размытие
            # Находим контуры для каждой маски
            contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Увеличенный порог площади
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h != 0 else 1
                    if 1 < aspect_ratio < 25:  # Смягчаем проверку формы
                        all_contours.append((contour, color_name, x))
        
        # Сортируем контуры по X и убираем дубликаты
        all_contours.sort(key=lambda x: x[2])  # Сортировка по x
        unique_contours = []
        used_x = set()
        for contour, color_name, x in all_contours:
            x, y, w, h = cv2.boundingRect(contour)
            overlap = False
            for used in used_x:
                if abs(x - used) < w:  # Проверка на пересечение
                    overlap = True
                    break
            if not overlap:
                unique_contours.append((contour, color_name, x))
                used_x.add(x)
        contours = [c[0] for c in unique_contours]
        wires = [(x, color_name, None) for c, color_name, x in unique_contours]  # HSV пока None

        # Определяем цвет и HSV для каждого уникального контура
        frame_with_contours = static_frame.copy()
        for i, (contour, color_name, x) in enumerate(unique_contours):  # Используем unique_contours
            if contour is None or len(contour) == 0:  # Проверка на валидность
                continue
            x, y, w, h = cv2.boundingRect(contour)  # Полная распаковка
            center_x, center_y = x + w // 2, y + h // 2
            small_w, small_h = max(5, w // 4), max(5, h // 4)
            x1 = max(x, center_x - small_w // 2)
            y1 = max(y, center_y - small_h // 2)
            x2 = min(x + w, center_x + small_w // 2)
            y2 = min(y + h, center_y + small_h // 2)
            wire_hsv = cv2.cvtColor(static_frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
            if wire_hsv.size == 0:
                continue
            wire_color = wire_hsv.mean(axis=(0, 1))
            if wire_color[1] < 30 and color_name != "black":  # Пропускаем тени, но не для черного
                continue
            color_name, hsv_values = closest_color(wire_color)  # Используем closest_color
            # Обновляем цвет и HSV для каждого провода
            wires[i] = (x, color_name, hsv_values)  # Обновляем с правильным цветом
            cv2.drawContours(frame_with_contours, [contour], -1, (0, 255, 0), 2)

        # Вывод результатов
        wires.sort(key=lambda x: x[0])
        colors = [w[1] for w in wires]  # Используем color_name
        data_history.append({"count": len(wires), "colors": colors})

        print(f"Провода: {len(wires)}, Цвета: {colors}")
        for i, (x, color, hsv_values) in enumerate(wires):
            print(f"Провод {i+1}: Цвет: {color}, HSV: {hsv_values}, Координата X: {x}")
        cv2.imshow('Frame with Contours', frame_with_contours)
        for color_name, mask in masks.items():
            cv2.imshow(f'Mask {color_name}', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()