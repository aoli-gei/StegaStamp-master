import glob
from os.path import join
TRAIN_PATH = './data/mirflickr25k/mirflickr/'
files_list = glob.glob(join(TRAIN_PATH,"**/*"))
print (files_list)