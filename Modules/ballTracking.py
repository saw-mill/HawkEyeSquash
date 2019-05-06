import cv2
import numpy as np
import math

def kalmanInit():
    kalman = cv2.KalmanFilter(4,2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.009
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.00003
    
    return kalman

def kalmanFirstFrame(currFrame, ballCandidatesFilteredProximity, tp, mp):
    height, width, channels = currFrame.shape
    imageCenter=[width/2,height/2]
    if not ballCandidatesFilteredProximity:
        initstate = imageCenter
    else:
        if (len(ballCandidatesFilteredProximity) == 1):
            x = ballCandidatesFilteredProximity[0][0]
            y = ballCandidatesFilteredProximity[0][1]
            mp = np.array([[np.float32(x)], [np.float32(y)]])
            initstate = [mp[0], mp[1]]
        else:
            minDistInitCand=10000
            for cand in ballCandidatesFilteredProximity:
                distCenter = math.sqrt(math.pow((cand[0] - imageCenter[0]), 2) + math.pow((cand[1] - imageCenter[1]), 2))
                if (distCenter < minDistInitCand):
                    initstate = [cand[0], cand[1]]
                    minDistInitCand = distCenter
        tp[0] = initstate[0]
        tp[1] = initstate[1]
        # pred.append((int(tp[0]), int(tp[1])))
        cv2.circle(currFrame, (tp[0], tp[1]), 10, (0, 0, 255), -1)

    return tp, initstate

def kalmanSingleBallCandidate(currFrame,kalman,tp,initstate, ballCandidatesFilteredProximity):
    for cand in ballCandidatesFilteredProximity:
            x = cand[0]
            y = cand[1]
            x = x - initstate[0]
            y = y - initstate[1]
            mp = np.array([[np.float32(x)], [np.float32(y)]])
            meas.append((x, y))
            corrected = kalman.correct(mp)
            corrected[0] = corrected[0] + initstate[0]
            corrected[1] = corrected[1] + initstate[1]
            cv2.circle(currFrame, (corrected[0], corrected[1]), 10, (0, 255, 0), -1)
            dictFrameNumberscX[i + 1] = corrected[0]
            dictFrameNumberscY[i + 1] = corrected[1]

            cv2.drawContours(currFrame, [cand[3]], -1, (255, 0,), 2)
            cv2.putText(currFrame, str(cand[0]) + "," + str(cand[1]), (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # cv2.imshow('Candidate image', currFrame)



    
