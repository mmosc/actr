# Select whether gru or vae model should be loaded
device = 'cpu'

vae = False
gru = True
bpr = False

# Select whether the rating matrix is binarized in userkNN
binarize = True

USE_CONTENT = False
CALC_REWARD = True

# Thresholds for core filtering
users_thresh = 10
tracks_thresh = 10
