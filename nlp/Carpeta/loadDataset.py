import datasets
from docopt import docopt
import sys
import os

def getLabels(dataset):
    return dataset.features[f"ner_tags"].feature.names

def loadDatasets(languages):
    """Gets all datasets needed in training and adds them a new label (its language), so
    that it can be used to  balance data. Returns a list with the new datasets"""
    loadedDatasets = []
    for language in languages:
        dataset = datasets.load_dataset('wikiann', language)
        loadedDatasets.append(dataset)
    return loadedDatasets

def concatDatasets(datasetList):
    """Gets a list of datasets (one for each language) and concatenates their balanced train, test
    and validation subsets"""
    trainSet = []
    testSet = []
    validSet = []
    for dataset in datasetList:
        trainSet.append(dataset["train"])
        validSet.append(dataset["validation"])
        testSet.append(dataset["test"])
    newTrainSet = datasets.concatenate_datasets(trainSet)
    newValidSet = datasets.concatenate_datasets(validSet)
    newTestSet = datasets.concatenate_datasets(testSet)
    return datasets.DatasetDict({"train":newTrainSet, "validation":newValidSet, "test":newTestSet})

def usage():
    usage = '''
        Usage:
            trainer.py [--directory] <directory> [--languages] <languages>
    '''
    return docopt(usage)

def argChecks(args):
    if args["<directory>"] and args["<languages>"]:
        try:
            os.mkdir("data/" + args["<directory>"])
            os.mkdir("data/" + args["<directory>"] + "/test")
            os.mkdir("data/" + args["<directory>"] + "/train")
            os.mkdir("data/" + args["<directory>"] + "/val")
        except OSError: 
            sys.exit("Directory data/" + args["<directory>"] + " already exists")
        args["<languages>"] = [x for x in args["<languages>"].split(',')]
        return args["<languages>"], "data/" + args["<directory>"]
    else:
        sys.exit("Name or languages argument missing")

def sendToTxt(dataset, directory, labels):
    def map_function(examples):
        newLabels = []
        for i in examples:
            newLabels.append(labels[i])
        examples = newLabels
        return examples
    
    with open(directory+"sentences.txt", "w", encoding="utf-8") as txt_file:
        for line in dataset["tokens"]:
            txt_file.write(" ".join(line) + "\n")
            
    newTags = map(map_function, dataset["ner_tags"])
    with open(directory+"labels.txt", "w", encoding="utf-8") as txt_file:
        for line in newTags:
            txt_file.write(" ".join(line) + "\n")
    
def main():
    args = usage()
    languages, directory = argChecks(args)
    datasets = loadDatasets(languages)
    concatDataset = concatDatasets(datasets)
    labels = getLabels(concatDataset)
    sendToTxt(concatDataset["train"], directory + "/train/", labels)
    sendToTxt(concatDataset["validation"], directory + "/val/", labels)
    sendToTxt(concatDataset["test"], directory + "/test/", labels)
    
if __name__=="__main__":
    main()
