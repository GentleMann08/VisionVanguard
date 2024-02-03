import cv2
import numpy as np


def color_filter(video_path, output_path, lower_color, upper_color):
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, lower_color, upper_color)

            res = cv2.bitwise_and(frame, frame, mask=mask)

            out.write(res)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Задаем диапазон для красного цвета в HSV
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

# Задаем источник и выходной путь для видео
video_path = "C:/Users/025/Desktop/osu.mp4"
output_path = "C:/Users/025/Desktop/out.mp4"

# Вызываем функцию
color_filter(video_path, output_path, lower_red, upper_red)

