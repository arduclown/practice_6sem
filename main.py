import cv2
import numpy as np
from collections import deque, Counter

cap = cv2.VideoCapture(0)
data_history = []
history_length = 5
color_history = deque(maxlen=history_length)

COLORS_HSV = {
    "red": (np.array([170, 100, 50]), np.array([180, 255, 255])),
    "green": (np.array([30, 50, 50]), np.array([90, 255, 255])),
    "blue": (np.array([100, 70, 80]), np.array([135, 255, 200])),
    "yellow": (np.array([20, 100, 100]), np.array([40, 255, 255])),
    "black": (np.array([0, 0, 0]), np.array([180, 100, 50]))
}

# Цвета для отображения
COLOR_DISPLAY = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "black": (0, 0, 0)
}

"""Возвращает ближайший цвет по оттенку (Hue)"""
def closest_color_from_hue(hue_value):
    for color_name, (lower, upper) in COLORS_HSV.items():
        if color_name == "black": 
            continue
        if lower[0] <= hue_value <= upper[0]:
            return color_name
    return "unknown"

"""Обработка контуров с улучшенными параметрами"""
def process_contours(mask, min_area=300, max_aspect_ratio=15):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else max(w, h)
        
        if aspect_ratio > max_aspect_ratio:
            continue
            
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = x + w//2, y + h//2
            
        valid_contours.append((contour, (cX, cY), (x, y, w, h)))
    
    return valid_contours


"""Объединение близко расположенных контуров"""
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

# Основной цикл обработки
static_frame = None
connector_area = None
kernel = np.ones((3, 3), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось получить кадр")
        break

    if static_frame is None:
        live_frame = frame.copy()

        # отрисовка области для исключения коннектора
        if connector_area is None:
            height, width = frame.shape[:2]
            connector_width = width // 4
            connector_height = height // 3
            x1 = width // 2 - connector_width // 2
            y1 = height // 2 - connector_height // 2
            x2 = width // 2 + connector_width // 2
            y2 = height // 2 + connector_height // 2
            
            cv2.rectangle(live_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(live_frame, "Place connector here, press 's'", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Live Feed', live_frame)
        key = cv2.waitKey(1) & 0xFF

        # Фиксируем область коннектора и кадр
        if key == ord('s') and connector_area is None:
            connector_area = (x1, y1, x2, y2)
            static_frame = frame.copy()
            print(f"Область коннектора зафиксирована: {connector_area}")
    else:
        hsv = cv2.cvtColor(static_frame, cv2.COLOR_BGR2HSV)
        wires = []

        for color_name, (lower, upper) in COLORS_HSV.items():
            mask = cv2.inRange(hsv, lower, upper)
            
            if connector_area:
                x1, y1, x2, y2 = connector_area
                mask[y1:y2, x1:x2] = 0
            
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
            smoothed = cv2.bilateralFilter(closed, 5, 75, 75)
            
            cv2.imshow(f'Mask - {color_name}', mask)

            contours_data = process_contours(smoothed)
            
            for contour, center, bbox in contours_data:
                x, y, w, h = bbox

                roi = static_frame[y:y+h, x:x+w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                mask_local = cv2.inRange(hsv_roi, lower, upper)
                
                s_channel = hsv_roi[:, :, 1]
                v_channel = hsv_roi[:, :, 2]
                valid_pixels = (s_channel > 50) & (v_channel > 50)
                mask_local = cv2.bitwise_and(mask_local, mask_local, mask=valid_pixels.astype(np.uint8)*255)

                hue_channel = hsv_roi[:, :, 0]
                hue_values = hue_channel[mask_local > 0]

                if hue_values.size > 0:
                    hue_hist = np.bincount(hue_values, minlength=180)
                    dominant_hue = np.argmax(hue_hist)
                    determined_color = closest_color_from_hue(dominant_hue)
                else:
                    determined_color = "black"

                if hue_values.size > 0:
                    hue_hist = np.bincount(hue_values, minlength=180)
                    dominant_hue = np.argmax(hue_hist)
                    base_color = closest_color_from_hue(dominant_hue)
                else:
                    base_color = "black"

                s_mean = np.mean(hsv_roi[:, :, 1])
                v_mean = np.mean(hsv_roi[:, :, 2])
                if base_color == "black":
                    if s_mean > 40 and v_mean > 40 and hue_values.size > 0:
                        determined_color = closest_color_from_hue(dominant_hue)
                    else:
                        determined_color = "black"
                else:
                    determined_color = base_color

                
                wires.append({
                    "contour": contour,
                    "center": center,
                    "color": determined_color,
                    "bbox": bbox
                })

        # удаление дубликатов
        unique_wires = []
        used_centers = set()
        distance_threshold = 20
        
        for wire in sorted(wires, key=lambda w: (w['center'][1], w['center'][0])):
            center = wire['center']
            is_duplicate = False
            
            for used_center in used_centers:
                if np.linalg.norm(np.array(center) - np.array(used_center)) < distance_threshold:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_wires.append(wire)
                used_centers.add(center)

        left_wires = []
        right_wires = []

        x1, y1, x2, y2 = connector_area
        connector_x_center = (x1 + x2) // 2

        for wire in unique_wires:
            if wire["center"][0] < connector_x_center:
                left_wires.append(wire)
            else:
                right_wires.append(wire)

        # сортировка сверху вниз (по Y)
        left_wires_sorted = sorted(left_wires, key=lambda w: w["center"][1])
        right_wires_sorted = sorted(right_wires, key=lambda w: w["center"][1])

        # сравнение
        connection_ok = True
        if len(left_wires_sorted) != len(right_wires_sorted):
            connection_ok = False
            connection_status = f"Ошибка: {len(left_wires_sorted)} слева, {len(right_wires_sorted)} справа"
        else:
            for lw, rw in zip(left_wires_sorted, right_wires_sorted):
                if lw["color"] != rw["color"]:
                    connection_ok = False
                    connection_status = f"Несовпадение: {lw['color']} != {rw['color']}"
                    break
            else:
                connection_status = "OK: соединение корректное"


        result_frame = static_frame.copy()
        colors_detected = []
        
        for i, wire in enumerate(unique_wires):
            color = COLOR_DISPLAY.get(wire['color'], (255, 255, 255))
            cv2.drawContours(result_frame, [wire['contour']], -1, color, 2)
            
 
            label = f"{i+1}:{wire['color']}"
            cv2.putText(result_frame, label, (wire['center'][0] + 10, wire['center'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            colors_detected.append(wire['color'])

        # стабилизация результатов
        current_colors = tuple(sorted(colors_detected))
        color_history.append(current_colors)
        
        if len(color_history) == history_length:
            stable_colors = Counter(color_history).most_common(1)[0][0]
            print(f"Стабильный результат: {stable_colors}")
        

        print(f"Найдено проводов: {len(unique_wires)}, Цвета: {colors_detected}")
        cv2.imshow('Detected Wires', result_frame)

        status_color = (0, 255, 0) if connection_ok else (0, 0, 255)
        cv2.putText(result_frame, connection_status, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cv2.imshow('Original', static_frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()