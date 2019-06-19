#import cPickle
import numpy as np
from scipy.io.wavfile import read
#from sklearn.mixture import GMM
from speakerfeatures import GetFeatures
import warnings

warnings.filterwarnings("ignore")

source = "development_set\\"

dest = "speaker_models\\"
train_file = "development_set_enroll.txt"
file_paths = open(train_file, 'r')

count = 1
features = np.asarray(())
for path in file_paths:
    path = path.strip()
    print
    path

    sr, audio = read(source + path)

    vector = extract_features(audio, sr)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))

    if count == 5:
        gmm = gmm(n_components=16, n_iter=200, covariance_type='diag', n_init=3)
        gmm.fit(features)

        picklefile = path.split("-")[0] + ".gmm"
        cPickle.dump(gmm, open(dest + picklefile, 'w'))
        print
        '+ modeling completed for speaker:', picklefile, " with data point = ", features.shape
        features = np.asarray(())
        count = 0
    count = count + 1