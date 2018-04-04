import os
import numpy as np

all_song_path  = 'lists/genres'
arr = os.listdir(all_song_path)
tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
tags = np.array(tags)

training_data= open("lists/training_data.txt","w+")
training_labels  = open("lists/training_labels.txt","w+")

testing_data = open("lists/training_data.txt","w+")
testing_labels  = open("lists/testing_labels.txt","w+")


np.random.shuffle(arr)
training, test = arr[:int(len(arr)*0.8)], arr[int(len(arr)*0.8):]


print len(training)
print len(test)