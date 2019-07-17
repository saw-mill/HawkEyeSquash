import numpy as np
import cv2
import time
import math

def findContours(inputFrame):
        startTimeFindingContours = time.time()
        contours, hier = cv2.findContours(inputFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        endTimeFindingContours = time.time()
        print("Contours found in--- %s seconds ---" %(endTimeFindingContours-startTimeFindingContours))
        return contours, hier

def sizeDetection(contours, currFrame,frameNumber):
        startTimeSizeDetection = time.time()
        min_BallArea = 340 #340 for d2, 355 for d3, 280 for d4
        max_BallArea = 1500 #1200 for d4
        min_PlayerArea = 4000 #4000 for d4
        min_IncompletePlayerArea = 1500

        ballCandidates = list()
        playerCadidates = list()
        incompletePlayerCandidates = list()

        for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                else:
                        continue
                area = cv2.contourArea(cnt)
                if area > min_PlayerArea:
                        playerCadidates.append([cX, cY, area, cnt,frameNumber])
                elif area > min_IncompletePlayerArea and area < min_PlayerArea:
                        incompletePlayerCandidates.append([cX, cY, area, cnt,frameNumber])
                elif area < max_BallArea and area > min_BallArea:
                        ballCandidates.append([cX, cY, area, cnt,frameNumber])
                        # cv2.drawContours(currFrame, [cnt], -1, (0, 255, 0), 1)
                        # cv2.putText(currFrame, str(area), (cX, cY),
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                elif area < min_BallArea:
                        continue
        endTimeSizeDetection = time.time()
        print("Size based filtering in--- %s seconds ---" %(endTimeSizeDetection-startTimeSizeDetection))
        return ballCandidates, playerCadidates, incompletePlayerCandidates

def courtBoundaryDetection(datasetName, ballCandidates, playerCadidates, incompletePlayerCandidates, currFrame):
        if (datasetName == "Dataset1"):
                courtBoundaryleft = 68
                courtBoundaryRight = 835
        elif (datasetName == "Dataset2" or datasetName == "Dataset6" or datasetName == "Dataset7" or datasetName == "Dataset8" or datasetName == "Dataset9" or datasetName == "Dataset10"):
                courtBoundaryleft = 104 #d7 104 740
                courtBoundaryRight = 740
        elif (datasetName == "Dataset3"):
                courtBoundaryleft = 145
                courtBoundaryRight = 709
        elif (datasetName == "Dataset4"):
                courtBoundaryleft = 110
                courtBoundaryRight = 745
        elif (datasetName == "Dataset5"):
                courtBoundaryleft = 161
                courtBoundaryRight = 695
        elif (datasetName == "Dataset11"):
                courtBoundaryleft = 120
                courtBoundaryRight = 745
        ballCandidatesFilteredBoundary = list()
        playerCadidatesFilteredBoundary = list()
        incompletePlayerCandidatesFilteredBoundary = list()
        for cand in ballCandidates:
                if (cand[0] <= courtBoundaryleft or cand[0] > courtBoundaryRight):
                        continue
                else:
                        ballCandidatesFilteredBoundary.append(cand)
        for playercand in playerCadidates:
                if (playercand[0] <= courtBoundaryleft or playercand[0] > courtBoundaryRight):
                        continue
                else:
                        playerCadidatesFilteredBoundary.append(playercand)
        for incompletecand in incompletePlayerCandidates:
                if (incompletecand[0] <= courtBoundaryleft or incompletecand[0] > courtBoundaryRight):
                        continue
                else:
                        incompletePlayerCandidatesFilteredBoundary.append(incompletecand)
        
        print("Player Candidates: %d" % len(playerCadidatesFilteredBoundary))
        print("Incomplete Player Candidate: %d" % len(incompletePlayerCandidatesFilteredBoundary))
        return ballCandidatesFilteredBoundary,playerCadidatesFilteredBoundary,incompletePlayerCandidatesFilteredBoundary

def playerProximityDetection(ballCandidates, playerCadidates, incompletePlayerCandidates, currFrame):
        startTimePlayerProximity = time.time()
        ballCandidatesFiltered = list()
        min_BallDistance = 80 #80 for d1 d2, 75 for d3, 92 for d4

        if not ballCandidates:
                print("No ball Candidates")
                cv2.imshow('Candidate image', currFrame)
        else:
                minDist = 99999999
                minDistPoint = []
                for cand in ballCandidates:
                        minDist = 99999999
                        minDistPoint = []
                        if len(playerCadidates) > 1:
                                for player in playerCadidates:
                                        dist = math.sqrt(math.pow((cand[0]-player[0]),2)+math.pow((cand[1]-player[1]),2))
                                        if dist < minDist:
                                                minDist = dist
                                                minDistPoint = [player[0],player[1]]
                        elif len(playerCadidates) == 1:
                                distFromPlayer = math.sqrt(math.pow((cand[0] - playerCadidates[0][0]), 2) + math.pow((cand[1] - playerCadidates[0][1]), 2))
                                if distFromPlayer < minDist:
                                        minDist = distFromPlayer
                                        minDistPoint = [playerCadidates[0][0],playerCadidates[0][1]]
                                for part in incompletePlayerCandidates:
                                        dist = math.sqrt(math.pow((cand[0] - part[0]), 2) + math.pow((cand[1] - part[1]), 2))
                                        if dist < minDist:
                                                minDist = dist
                                                minDistPoint = [part[0],part[1]]
                        elif len(incompletePlayerCandidates) > 0:
                                for part in incompletePlayerCandidates:
                                        dist = math.sqrt(math.pow((cand[0] - part[0]), 2) + math.pow((cand[1] - part[1]), 2))
                                        if dist < minDist:
                                                minDist = dist
                                                minDistPoint = [part[0], part[1]]
                        else:
                                continue
                        minDist = round(minDist, 2)
                        if (minDist >= min_BallDistance):
                                cand.append(minDist) # Can remove this
                                # cand.append(minDistPoint) #Can remove this
                                ballCandidatesFiltered.append(cand)

        endTimeProximityDetection = time.time()
        print("Proximity based filtering in--- %s seconds ---" %(endTimeProximityDetection-startTimePlayerProximity))
        return ballCandidatesFiltered

def regionDetection(ballCandidatesFiltered, ballCandidatesPreviousFrame,currFrame):
        startTimeRegionDetection = time.time()
        ballCandidatesFilteredProximity = list()

        if len(ballCandidatesPreviousFrame) > 0:
                for cand in ballCandidatesFiltered:
                        ballCandFlag = False
                        for prevCand in ballCandidatesPreviousFrame:
                                dist = math.sqrt(math.pow((cand[0] - prevCand[0]), 2) + math.pow((cand[1] - prevCand[1]), 2))
                                dist = round(dist,2)
                                if dist > 2 and dist < 70:
                                        ballCandFlag = True
                                else:
                                        continue
                        if ballCandFlag is True:
                                cand.append(dist)
                                ballCandidatesFilteredProximity.append(cand)
                                # cv2.putText(currFrame, "Maybe", (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 192), 2)
                                # cv2.drawContours(currFrame, [cand[3]], -1, (255, 0,), 2)
                                # cv2.imshow('Candidate image', currFrame)
                        else:
                                # cv2.imshow('Candidate image', currFrame)
                                continue
                                # cv2.putText(currFrame, "Not", (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 192), 2)
                                # cv2.imshow('Candidate image', currFrame)
                # ballCandidatesPreviousFrame = ballCandidatesFilteredProximity
        else:
                # for cand in ballCandidatesFiltered:
                ballCandidatesFilteredProximity = ballCandidatesFiltered
                        # cv2.drawContours(currFrame, [cand[3]], -1, (255, 0,), 2)
                        # cv2.putText(currFrame, "Maybe", (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 192), 2)
                        # cv2.imshow('Candidate image', currFrame)
        ballCandidatesPreviousFrame = ballCandidatesFiltered
        endTimeRegionDetection = time.time()
        print("Expected Region based filtering in--- %s seconds ---" % (endTimeRegionDetection - startTimeRegionDetection))
        print("Ball Candidates: %d" % len(ballCandidatesFilteredProximity))
        return ballCandidatesFilteredProximity, ballCandidatesPreviousFrame