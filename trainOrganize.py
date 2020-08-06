import os
hitpath = './trainingData/positives'
negpath = './trainingData/negatives'
hitfiles = os.listdir(hitpath)
negfiles = os.listdir(negpath)


for index, file in enumerate(hitfiles):
    os.rename(os.path.join(hitpath, file), os.path.join(hitpath, 'positives'.join([str(index), '.jpg'])))

for index, file in enumerate(negfiles):
    os.rename(os.path.join(negpath, file), os.path.join(negpath, 'negatives'.join([str(index), '.jpg'])))