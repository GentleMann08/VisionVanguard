from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator

# def run_yolo_detection(model_path, img):
#     model = YOLO(model_path)
#
#     # image = cv2.imread(img)
#
#     results = model.predict(image)
#
#     for r in results:
#         annotator = Annotator(image)
#
#         boxes = r.boxes
#         for box in boxes:
#             b = box.xyxy[0]
#             c = box.cls
#             annotator.box_label(b, model.names[int(c)])
#
#     img = annotator.result()
#
#     cv2.imwrite(f'Images/ResultImage.jpg', img)
#
#         # cv2.imshow('YOLO V8 Detection', img)
#         # print(img)
#         # if cv2.waitKey(0) & 0xFF == ord(' '):
#             # break
#
#     cv2.destroyAllWindows()
#     return


def run_yolo_detection_video(model_path, video_path, username):
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)
    people_flag = 0

    # Get input video resolution
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Instantiate VideoWriter with 'mp4v' codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Specify full path to the video file
    output_path = f'Video/ResultVideo{username}.mp4'
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            try:
                results = model.predict(img)
                annotator = Annotator(img)
                local_flag = False

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0]
                        c = box.cls
                        classname = model.names[int(c)]
                        if classname in ['person']: # , 'dog', 'cat']:
                            people_flag += 1
                            local_flag = True
                            annotator.box_label(b, classname)

                if local_flag:
                    img = annotator.result()
                    if people_flag == 1:
                        cv2.imwrite(f'Images/Person Detected{username}.jpg', img)
                        return True

                # Write valid frame to the video file
                out.write(img)

                # cv2.imshow('YOLO V8 Detection', img)
                # if cv2.waitKey(1) & 0xFF == ord(' '):
                #     break
            except Exception as e:
                print(f"An error occurred: {e}")
                continue
        else:
            break


    return True if people_flag > 0 else False
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def run_multi_yolo_detection(model_paths, img_path, username):
    assert len(model_paths) > 1
    flag = False
    person_flag = False

    models = [YOLO(path) for path in model_paths]
    image = cv2.imread(img_path)
    annotator = Annotator(image.copy())

    for i, model in enumerate(models):
        results = model.predict(image)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                if i == 1 and len(results) > 0:
                    flag = True
                classname = model.names[int(c)]
                if classname in ['person'] or (i == 1 and len(results) > 0):  # , 'dog', 'cat']:
                    annotator.box_label(b, model.names[int(c)])

    annotated_image = annotator.result()
    cv2.imwrite(f'Images/ResultImage{username}.jpg', annotated_image)

    cv2.destroyAllWindows()

    return flag


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    run_yolo_detection_video('yolov8n.pt', 0, "qwerty")

    # run_yolo_detection(model_path='runs/detect/train3/weights/best.pt', imgs=["Test/359933974.jpg", "Test/AciFTIiOll.jpg", "Test/ad0d50a73e2a7034832790e4f003d0d1.jpg", "Test/maxresdefault.jpg"])
