import os
hitpath = './trainingData/positives1'
grouppath = './trainingData/positives2'
negpath = './trainingData/negatives'

hitfiles = os.listdir(hitpath)
groupfiles = os.listdir(grouppath)
negfiles = os.listdir(negpath)


for index, file in enumerate(hitfiles):
    os.rename(os.path.join(hitpath, file), os.path.join(hitpath, 'positive'.join([str(index), '.jpg'])))

for index, file in enumerate(negfiles):
    os.rename(os.path.join(negpath, file), os.path.join(negpath, 'negative'.join([str(index), '.jpg'])))

for index, file in enumerate(groupfiles):
    os.rename(os.path.join(grouppath, file), os.path.join(grouppath, 'group'.join([str(index), '.jpg'])))

# ----------------------------------------------------------------------------------------------------------


