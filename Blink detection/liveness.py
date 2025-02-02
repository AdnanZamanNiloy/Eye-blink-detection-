import cv2
import dlib
import time
from imutils import face_utils
from scipy.spatial import distance as dist


cam = cv2.VideoCapture(0)


# ------------Variables---------#
blink_thres = 0.5
tt_frame = 3
count = 0


# ---------------------#
detector = dlib.get_frontal_face_detector()
lm_model = dlib.shape_predictor(
    'F:\Desktop\lv 1\Eye-Blink-Detector\Model/shape_predictor_68_face_landmarks.dat')

# ---------------eye ids-----------------#
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']


ptime = 0


def EAR_cal(eye):
    # ---vertical---#
    v1 = dist.euclidean(eye[1], eye[5])
    v2 = dist.euclidean(eye[2], eye[4])

    # ---horizontal---#
    h1 = dist.euclidean(eye[0], eye[3])

    ear = (v1+v2)/(h1)
    return ear


while True:

    if cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(cv2.CAP_PROP_FRAME_COUNT):
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

    _, frame = cam.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ----fps----#
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(
        frame,
        f'FPS:{int(fps)}',
        (50, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (0, 0, 100),
    )

  # rectangle around face-------------------#
    faces = detector(img_gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200), 2)

   # landmarks-------------------#
    shapes = lm_model(img_gray, face)
    shapes = face_utils.shape_to_np(shapes)

   # eye landmarks-------------------#
    left_eye = shapes[L_start:L_end]
    right_eye = shapes[R_start:R_end]

    for Lpt, rpt in zip(left_eye, right_eye):
        cv2.circle(frame, Lpt, 1, (200, 200, 0), 1)
        cv2.circle(frame, rpt, 1, (200, 200, 0), 1)

    left_ear = EAR_cal(left_eye)
    right_ear = EAR_cal(right_eye)

    avg_ear = (left_ear+right_ear)/2

    if avg_ear < blink_thres:
        count += 1

    else:
        if count > tt_frame:
            cv2.putText(frame, f'BLINK Detected', (
                frame.shape[1]//2 - 300, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 0), 2)
        else:
            count = 0

    
    cv2.imshow("vedio", frame)

    if cv2.waitKey(1) == 13:  # Press Enter to exit
        break

cam.release()
