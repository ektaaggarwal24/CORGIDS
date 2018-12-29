import glob
import pandas as pd
import numpy as np
import os, sys, json, pickle
from hmmlearn import hmm

def returnfileDuration(currentFile, platform):
# This method returnd the total time duration of the operation of the CPS from the given file input.
    df = pd.read_csv(currentFile)
    if platform == "UAV":
        timeArray = df['flightTime']
        return timeArray[len(timeArray)-1]
    else:
        glucoseArray = df['glucose']
        return len(glucoseArray)

def calculate_log_probability(trainedHMM, testingFile, acceptedMinValue, logicalProperties,
 startWindow, currentTime, numberOfSecondsInFile, platform):
# This method calculates the log probability of the current window and then compares
# it against the trainedHMM log probability. 
    feature = get_data_from_file(testingFile, logicalProperties, startWindow, currentTime,
 numberOfSecondsInFile, platform)
    current_value = trainedHMM.score(feature)
    isFileDifferent = 0
    if (current_value < acceptedMinValue):
        isFileDifferent = 1

    return isFileDifferent

def get_data_from_file(testingFile, logicalProperties, startWindow, currentTime, numberOfSecondsInFile, platform):
# This method will return the data chunk between the start window and current time of the file.
    df = pd.read_csv(testingFile)
    data = df[logicalProperties].values
    if platform == "UAV":
        startWindowIndex = 0
        currentTimeIndex = 0

        timeArray = df['flightTime']
        for index in range(len(timeArray)):
            if timeArray[index] > startWindow and currentTimeIndex == 0:
                startWindowIndex = index
                currentTimeIndex = -1
            
            if timeArray[index] >= currentTime:
                currentTimeIndex = index
                break

        return data[startWindowIndex:currentTimeIndex][:]
    else:
        return data[startWindow:currentTime][:]
    

def fetch_ids(IDSPath, IDSStatsPath):
# This method unpickles the pickled trained IDS and its stats which will be
# later used for intrusion detection.
    trainedHMM = pickle.load(open(IDSPath, 'rb'))
    trainedHMMStats = pickle.load(open(IDSStatsPath, 'rb'))
    return trainedHMM, trainedHMMStats

def detect_intrusion(logicalProperties, testingFilePath, windowSize, decisions, trainedHMM, rangeValue, platform):
# This method is responsible for detecting intrusion based on the testing file provided.
# The intrusion detection takes into account the log probability of the current testing file versus
# the log probability of trained benign HMM model.
 
    files = glob.glob(testingFilePath)
    consecutiveIntrusion = decisions
    startWindow = 0
    currentTime = 60
    # Fetch the time duration of the operation of CPS, which will be used as the end of the file.
    fileDuration = returnfileDuration(testingFilePath, platform)

    while currentTime <= fileDuration:
        isFileDifferent = 0;
        if currentTime < windowSize:
            pass
        elif currentTime == windowSize:
            isFileDifferent += calculate_log_probability(trainedHMM,testingFilePath,
                                                       rangeValue,logicalProperties,
                                                       startWindow,currentTime,fileDuration, platform)
        else:
            startWindow += 60 
            isFileDifferent += calculate_log_probability(trainedHMM,testingFilePath,
                                                           rangeValue,logicalProperties,
                                                       startWindow,currentTime,fileDuration, platform)
        if (isFileDifferent > 0):
            consecutiveIntrusion -= 1
        else:
            consecutiveIntrusion = decisions

        if consecutiveIntrusion == 0:
            print("***********INTRUSION***************")
            break

        numerator = (fileDuration - currentTime)/60
        if numerator >= 1:
            currentTime += 60
        else:
            currentTime = fileDuration


def main(platform):
    platform = platform.upper()
    idsFilePath = os.getcwd()
    basedir = os.path.dirname(os.path.abspath(idsFilePath))

    parametersFile = open(basedir + "/parameters.txt", "r" )
    parameters = json.load(parametersFile)

    logicalProperties = parameters[platform]["logical_properties"]
    IDSPath = basedir + parameters[platform]["IDS_path"] + "trainedModel.pickle"
    IDSStatsPath = basedir + parameters[platform]["IDS_path"] + "trainedModelStats.pickle"
    testingFilePath = basedir + parameters[platform]["testing_file_path"]

    # Here the pickled trained HMM and its stats are fetched.
    trainedHMM, trainedHMMStats = fetch_ids(IDSPath, IDSStatsPath)
    mean = float(trainedHMMStats[0][0])
    stdDV = float(trainedHMMStats[0][1])
    range_value = mean - parameters[platform]["range_value"]*stdDV

    print("Intrusion detection starting..")
    # The trained HMM is used to detect intrusion in the file present on the path "testingFilePath"
    detect_intrusion(logicalProperties, testingFilePath, parameters[platform]["window_size"], parameters[platform]["consecutive_decisions"], trainedHMM, range_value, platform)

if __name__ == '__main__':
    main(sys.argv[1])

