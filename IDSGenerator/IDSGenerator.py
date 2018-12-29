import glob
import pandas as pd
import numpy as np
import os, sys, json, pickle
from hmmlearn import hmm

def get_all_path_files(trainingFolderPath):
    return glob.glob(trainingFolderPath)

def get_data(allTrainingFiles,logicalProperties):
# This function returns the data from all the files mentioned in 'allTrainingFiles'.
# It further trims the data based on the columns mentioned in 'logicalProperties'.
    features = []
    lengths = []
    for f in allTrainingFiles:
        df = pd.read_csv(f)
        data = df[logicalProperties].as_matrix()
        features.append(data)
        lengths.append(data.shape[0])
    return features,lengths

def train_hmm(allTrainingFiles, logicalProperties, threshold):
# This method creates the Hidden Markov Model (HMM) from the data present in files 'allTrainingFiles' 
# and uses the properties mentioned in 'logicalProperties'.
# 'threshold' stands the delta of logLikelihood of two consecutive models and can be used to tune the HMM.
    features,lengths = get_data(allTrainingFiles,logicalProperties)
    features = np.concatenate(features,axis=0)
    hiddenStates = 2
    prevLikelihood = None
    currentDelta = None
    while(True):
        model = hmm.GaussianHMM(n_components=hiddenStates, covariance_type="tied",verbose=False,n_iter=1000)
        model.fit(features)
        currentLikelihood = model.score(features,lengths=lengths)  
        if prevLikelihood is None:
            prevLikelihood = currentLikelihood
            hiddenStates = hiddenStates + 1
            continue
        currentDelta = abs((currentLikelihood - prevLikelihood)*100./prevLikelihood)
        print("currentDelta: " + str(currentDelta))
        print("hiddenStates: " + str(hiddenStates))
        if currentDelta < threshold:
            break
        prevLikelihood = currentLikelihood
        hiddenStates = hiddenStates + 1
    return model 
    
def get_mean_std(HMMModel, allTrainingFiles, logicalProperties):
# This method returns the mean and standard deviation of the logLikelihood of the trained HMM
# given as input 'HMMModel'. Both these values will be used during intrusion detection.
    features, lengths = get_data( allTrainingFiles, logicalProperties)
    prob = []
    for feature,f in zip(features, allTrainingFiles):
        prob.append(HMMModel.score(feature))
    return np.mean(prob),np.std(prob)

def train_HMM_model(trainingFolderPath, logicalProperties, threshold):
# This is the main method which initiates the generation of the trained HMM and returns the trained
# HMM and its mean logLikelihood.
    allTrainingFiles = get_all_path_files ( trainingFolderPath )
    statsArray = []
    currentHMMModel = train_hmm ( allTrainingFiles, logicalProperties, threshold )
    statsArray.append( get_mean_std( currentHMMModel, allTrainingFiles, logicalProperties ))
    return currentHMMModel, statsArray


def main(inputPlatform):
    platform = inputPlatform.upper()
    idsGeneratorFilePath = os.getcwd()
    basedir = os.path.dirname(os.path.abspath(idsGeneratorFilePath))

    parametersFile = open(basedir + "/parameters.txt", "r" )
    parameters = json.load(parametersFile)

    logicalProperties = parameters[platform]["logical_properties"]
    trainingPath = basedir + parameters[platform]["training_IDS_path"]
    threshold = parameters[platform]["training_threshold"]

    print("Generating intrusion detection model..")
    trainedHMM,statsArray = train_HMM_model(trainingPath, logicalProperties, threshold)

    trainedHMMFilePath = basedir + "/CORGIDSModel/" + platform + "/trainedModel.pickle"
    trainedHMMStatsFilePath = basedir + "/CORGIDSModel/" + platform + "/trainedModelStats.pickle"

    with open(trainedHMMFilePath, 'wb') as handle:
        pickle.dump(trainedHMM, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(trainedHMMStatsFilePath, 'wb') as handle2:
        pickle.dump(statsArray, handle2, protocol=pickle.HIGHEST_PROTOCOL)

    print("Intrusion detection model generated!")

if __name__ == '__main__':
    main(sys.argv[1])

