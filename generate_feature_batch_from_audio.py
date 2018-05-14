import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import librosa.display
import numpy as np 
import pandas as pd
import os
import re
from sklearn.utils import shuffle
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import random
import librosa
import pickle
from mlxtend.preprocessing import one_hot


labelencoder = LabelEncoder() 
onehot_encoder = OneHotEncoder(sparse=False)

VALIDATION_TEST_SET_FILE_LIST = 'validation_test_data/audio'
TRAINING_FILE_LIST = 'train_data/audio'
TEST_DATA = 'dcase_test_video/'
TRAIN_PATH = 'audio_classifier/train/'
NUM_OF_BANDS = 200 

INPUT_FOLDER = 'train_data/t1/audio'
STAGE1_LABELS = 'train.csv'

train = pd.read_csv('train.csv') 
train['Scene']=labelencoder.fit_transform(train['Scene']);


def generate_spec( recording, OFFSET ):

	audio, _ = librosa.core.load(recording, sr=44100, dtype=np.float32, duration=10.0, offset  = OFFSET )

	mean_audio = audio.mean() 
	std_audio = np.std(audio) 
	audio  = ( audio - mean_audio )/std_audio 

	spec = librosa.feature.melspectrogram(audio, sr=44100, n_fft=2205, hop_length=882,
											  n_mels=NUM_OF_BANDS, fmax=22050, power=2)
	spec = librosa.power_to_db(spec)

	mean_spec = spec.mean()  
	std_spec = np.std(spec) 
	spec = ( spec - mean_spec )/std_spec 
	# print( " MEl spectro gram for the audio starting from "  + str( OFFSET )  + " ending 10 seconds later"  )   
	# plt.figure(figsize=(10, 4))
	# librosa.display.specshow( spec , x_axis= 'time', y_axis= 'mel',fmax = 8000 )
	# plt.colorbar(format='%+2.0f dB')
	# plt.title('Mel spectrogram')
	# plt.tight_layout()
	# plt.show()	
	# print(spec) 
	# print(spec.shape)

	return spec


def input_pipeline(filepath):

	feature_batch = []
	feature_batch_subject_list= [ ]
	label_batch = []
	

	SUBJECT_LIST = [os.path.splitext(subject)[0] for subject in os.listdir(filepath)]
	print(SUBJECT_LIST)    

	for subject in SUBJECT_LIST:
		feature_batch.append(generate_spec(filepath+"/"+subject+".wav"))
		feature_batch_subject_list.append(subject)   
		filename = 'audio/'+subject+'.wav'
		label = train[ train['Audio_id'] == filename ]['Scene'].to_string()
		label = label.split("    ")[1]
		label = int(label)
		label_batch.append(label)
		# print(filename)
		# print(label)
		# print( filename + " " + str(label) )

	
	feature_batch = np.asarray(feature_batch, dtype=np.float32)
	label_batch = np.asarray(label_batch)
	integer_encoded = label_batch.reshape(len(label_batch), 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	label_batch = onehot_encoded

	with open(filepath + '_feature_batch', 'wb') as fp:
		pickle.dump(feature_batch, fp)
	with open(filepath + '_label_batch', 'wb') as fp:
		pickle.dump(label_batch, fp)
	with open(filepath + '_feature_batch_subject_list', 'wb') as fp:
		pickle.dump(feature_batch_subject_list, fp)

	feature_batch = pickle.load( open(filepath + '_feature_batch', "rb" ) )
	label_batch = pickle.load( open(filepath + '_label_batch', "rb" ) )		
	feature_batch_subject_list = pickle.load( open(filepath + '_feature_batch_subject_list', 'rb') ) 

	for i in range(1,10):
		print( feature_batch_subject_list[i] ," ",  label_batch )
 

def preprocess( ):
	for i in range (1,8):
		input_pipeline('train_data/audio' + str(i) )


# preprocess()

def preprocess_validation():
	input_pipeline(VALIDATION_TEST_SET_FILE_LIST)

# preprocess_validation()
   

def input_test_pipeline(filepath):

	feature_batch = []
	feature_batch_subject_list = []


	SUBJECT_LIST = [os.path.splitext(subject)[0] for subject in os.listdir(filepath)]
	print(SUBJECT_LIST)    

	for subject in SUBJECT_LIST:
		print(subject)
		for offset in range( 0 , 1550 ):  # mapping it to the offsets so that it can be used later for generating the entire feature batch for the audio 
			feature_batch.append(generate_spec(filepath+"/"+subject+".wav",offset)) # 3600 corresponds to the 10 minute sample video we made for the testing 
			feature_batch_subject_list.append(subject + "_" + str(offset) )    
			print("Now at offset " + str(offset) )


	feature_batch = np.asarray(feature_batch, dtype=np.float32)

	with open(filepath + 'test_feature_batch', 'wb') as fp:
		pickle.dump(feature_batch, fp)
	with open(filepath + 'test_feature_batch_subject_list', 'wb') as fp:
		pickle.dump(feature_batch_subject_list, fp)

 
def preprocess_test():
	input_test_pipeline(TEST_DATA)

preprocess_test()
