import cv2
import mediapipe as mp
import math
import time


# Функция для вычисления расстояния между двумя точками
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Функция для воспроизведения видео
def play_video(video_path):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Не удалось открыть видео.")
        return

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        cv2.imshow('Video Playback', frame)

        # Остановить видео, если нажата клавиша 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyWindow('Video Playback')


# Инициализация MediaPipe для отслеживания рук
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Захват видеопотока с веб-камеры
cap = cv2.VideoCapture(0)

# Инициализация модели MediaPipe для работы с руками
with mp_hands.Hands(
        max_num_hands=2,  # Максимум 2 руки
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
    video_played = False  # Флаг для проверки, было ли видео воспроизведено
    fingers_crossed_time = None  # Время, когда пальцы были скрещены

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Не удалось захватить изображение.")
            break

        # Перевод изображения в формат RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Обработка изображения для распознавания рук
        results = hands.process(image_rgb)

        # Если обнаружены руки
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Отрисовка точек и соединений на руке
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Получаем координаты кончиков указательного (8 точка) и среднего пальца (12 точка)
                h, w, c = image.shape
                index_finger_tip = hand_landmarks.landmark[8]  # Указательный палец
                middle_finger_tip = hand_landmarks.landmark[12]  # Средний палец

                index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                middle_x, middle_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)

                # Вычисляем расстояние между кончиками указательного и среднего пальцев
                distance = calculate_distance(index_x, index_y, middle_x, middle_y)

                # Условие для определения, что пальцы скрещены
                if distance < 20:
                    if fingers_crossed_time is None:
                        # Засекаем время, когда пальцы начали скрещиваться
                        fingers_crossed_time = time.time()

                    # Проверяем, прошло ли 1 секунда с момента скрещивания пальцев
                    if time.time() - fingers_crossed_time >= 0.5 and not video_played:
                        play_video('video.mp4')  # Путь к вашему видеофайлу
                        video_played = True  # Чтобы не повторять запуск видео
                else:
                    # Если пальцы больше не скрещены, сбрасываем таймер и флаг
                    fingers_crossed_time = None
                    video_played = False

        # Показ обработанного изображения
        cv2.imshow('Hand Gesture Recognition', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Нажмите 'Esc' для выхода
            break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
