import os
import numpy as np
from glob import glob

def traintestsplit(all_song_path,split_ratio)
    labelsDict = {
        'blues'     :   0,
        'classical' :   1,
        'country'   :   2,
        'disco'     :   3,
        'hiphop'    :   4,
        'jazz'      :   5,
        'metal'     :   6,
        'pop'       :   7,
        'reggae'    :   8,
        'rock'      :   9,
    }

    labels=list()
    song_paths = [os.path.abspath(y) for x in os.walk(all_song_path) for y in glob(os.path.join(x[0], '*.au'))]
    for path in song_paths:
        label=path.split('/')[-2]
        labels.append(labelsDict[label])

    permutation = np.random.permutation(len(song_paths))
    song_paths = song_paths[permutation]
    labels = labels[permutation]

    train_size=split_ratio*len(song_paths)

    trainsongs = data[:train_size]
    trainLabels = labels[:train_size]

    testsongs = data[train_size:]
    testLabels = labels[train_size:]

    return trainsongs,trainLabels,testsongs,testLabels
