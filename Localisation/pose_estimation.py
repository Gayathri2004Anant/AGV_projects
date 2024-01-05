import numpy as np
import cv2
import cv2.aruco as aruco
import math
def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    i = 0
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, corners[i], mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

cap = cv2.VideoCapture(0)
loaded_data=np.load('calibration.npz')
mtx=loaded_data['mtx']
print(mtx)
dist=loaded_data['dist']
print(dist)
while (True):
    ret, frame = cap.read()

    # operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # set dictionary size depending on the aruco marker selected
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters()
    parameters.adaptiveThreshConstant = 10
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    # print(ids)

    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # check if the ids list is not empty
    # if no check is added the code will crash
    if np.all(ids != None):

        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        rvec, tvec ,_ = my_estimatePoseSingleMarkers(corners, 57, mtx, dist)
        
        rot_mtx=np.zeros((3,3))
        rvec=cv2.Rodrigues(rvec[0],rot_mtx)
        # print(rot_mtx,rot_mtx[0][1])
        cv2.drawFrameAxes(frame, mtx, dist, rot_mtx, tvec[0], 10)

        # draw a square around the markers
        aruco.drawDetectedMarkers(frame, corners)
        print("Height of camera: ",tvec[0][1])
        print("distance from camera: ",tvec[0][2])
        print("orientation of camera: ")

        yaw=np.arctan2([rot_mtx[1][0]],[rot_mtx[0][0]])*180/np.pi
        pitch=np.arctan2([-rot_mtx[2][0]],[math.sqrt(rot_mtx[2][2]**2+rot_mtx[2][1]**2)])*180/np.pi
        roll=np.arctan2([rot_mtx[2][1]],[rot_mtx[2][2]])*180/np.pi

        print('yaw, pitch, roll= ', yaw, pitch, roll)

        # code to show ids of the marker found
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0])+', '

        cv2.putText(frame, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)


    else:
        # code to show 'No Ids' when no markers are found
        cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

    # display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()