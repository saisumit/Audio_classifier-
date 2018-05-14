import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
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
import pickle



labelencoder = LabelEncoder() 
onehot_encoder = OneHotEncoder(sparse=False)

VALIDATION_TEST_SET_FILE_LIST = '/input/valid_data/audio'
TRAINING_FILE_LIST = '/input/train_data/audio'
TRAIN_PATH = '/output/audio_classifier/train/'
TEST_FILE_PATH = 'dcase_test_video/test_feature_batch'
TEST_ANNOTATION_PATH = '../egoseg_model/test_annotation/annotate_test_1680.txt'
N_TRAIN_STEPS = 3
NUM_OF_BANDS = 200
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MODEL_PATH = '/output/audio_classifier/model/'
MODEL_NAME = 'sai_audio.tfl'

BEST_MODEL_PATH = '/output/audio_classifier/best_model/'

# INPUT_FOLDER = 'train_data/t1/audio'
# STAGE1_LABELS = 'train.csv'

# with open ( TEST_ANNOTATION_PATH,"r") as annotation : 
# 	data = annotation.read() 
# 	print(data) 

train = pd.read_csv('train.csv') 
scene_list = train['Scene'].unique()
print(scene_list)
train['Scene']=labelencoder.fit_transform(train['Scene']);
print(labelencoder.transform(scene_list))

# def generate_spec( recording ):

# 	audio, _ = librosa.core.load(recording, sr=44100, dtype=np.float16, duration=10.0)
# 	spec = librosa.feature.melspectrogram(audio, sr=44100, n_fft=2205, hop_length=882,
# 											  n_mels=NUM_OF_BANDS, fmax=22050, power=2)
# 	spec = librosa.power_to_db(spec)
# 	# plt.figure(figsize=(10, 4))
# 	# librosa.display.specshow( spec , x_axis= 'time', y_axis= 'mel',fmax = 8000 )
# 	# plt.colorbar(format='%+2.0f dB')
# 	# plt.title('Mel spectrogram')
# 	# plt.tight_layout()
# 	# plt.show()	
# 	# print(spec) 
# 	# print(spec.shape)
# 	return spec


# def input_pipeline(filepath):

# 	feature_batch = []
# 	label_batch = []
	

# 	SUBJECT_LIST = [os.path.splitext(subject)[0] for subject in os.listdir(filepath)]
# 	print(SUBJECT_LIST)    
# 	print("kya haal hai bhai")
# 	counter = 1  
# 	for subject in SUBJECT_LIST:
# 		feature_batch.append(generate_spec(filepath+"/"+subject+".wav"))
# 		filename = 'audio/'+subject+'.wav'
# 		label = train[ train['Audio_id'] == filename ]['Scene'].to_string()
# 		label = label.split("    ")[1]
# 		label = int(label)
# 		label_batch.append(label)
# 		counter = counter +  1  
# 		# if( counter == 60 ):
# 		# 	break ;
# 		print(filename)
# 		print(label)
# 		print( filename + " " + str(label) )

	
# 	feature_batch = np.asarray(feature_batch, dtype=np.float32)
# 	label_batch = np.asarray(label_batch)
# 	integer_encoded = label_batch.reshape(len(label_batch), 1)
# 	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# 	label_batch = onehot_encoded
# 	# print(onehot_encoded)
# 	# print(label_batch)



# 	with open(filepath + '_feature_batch', 'wb') as fp:
# 		pickle.dump(feature_batch, fp)
# 	with open(filepath + '_label_batch', 'wb') as fp:
# 		pickle.dump(label_batch, fp) 

# 	feature_batch = pickle.load( open(filepath + '_feature_batch', "rb" ) )
# 	label_batch = pickle.load( open(filepath + '_label_batch', "rb" ) )		   
# 	return feature_batch, label_batch
 

# def preprocess( ):
# 	for i in range (1,8):
# 		input_pipeline('train_data/audio' + str(i) )


# preprocess()

# def preprocess_validation():
# 	input_pipeline(VALIDATION_TEST_SET_FILE_LIST)

# preprocess_validation()
# # generate_spec( )        

def sai_net( ):

	network = input_data(shape=[None, NUM_OF_BANDS, 500, 1 ], name='features')
	print(network.shape) 
	network = conv_2d(network, 100 , [NUM_OF_BANDS,50 ] , strides = [NUM_OF_BANDS,1] )
	print(network.shape) 
	network = tflearn.layers.batch_normalization(network)
	print(network.shape ) 
	network = tflearn.activations.relu(network)
	print(network.shape ) 
	network = dropout(network, 0.25)
	print(network.shape ) 
	network = conv_2d(network, 100 , [1,1])
	print(network.shape ) 
	network = tflearn.layers.batch_normalization(network)
	network = tflearn.activations.relu(network)
	network = dropout(network, 0.25)
	network = conv_2d(network, 15, [1,1], activation='softmax')
	print(network.shape ) 
	network = tflearn.layers.conv.global_avg_pool (network, name='GlobalAvgPool')
	print(network.shape)
	network = regression(network, optimizer='momentum', loss='categorical_crossentropy', 
						 learning_rate=LEARNING_RATE, name='labels')

	model = tflearn.DNN( network,tensorboard_dir=TRAIN_PATH, tensorboard_verbose=3 )
	return model



# val_features = [] 
# val_labels   = []


# val_feature =  pickle.load( open( VALIDATION_TEST_SET_FILE_LIST+ '_feature_batch', "rb" ) )
# val_labels =    pickle.load( open(VALIDATION_TEST_SET_FILE_LIST +  '_label_batch', "rb" ) )

# val_feature = val_feature.reshape([-1, NUM_OF_BANDS, 500 , 1])


# def train_sai_net():

# 	# start training process
# 	model = sai_net()
# 	prev_best = 0 
# 	for epoch in range(N_TRAIN_STEPS):
# 		for f_in in range ( 1, 8 ): 
# 			filepath = TRAINING_FILE_LIST + str(f_in) 
# 			feature_batch = pickle.load( open(filepath + '_feature_batch', "rb" ) )
# 			label_batch = pickle.load( open(filepath + '_label_batch', "rb" ) )		  
# 			feature_batch = feature_batch.reshape([-1, NUM_OF_BANDS, 500 , 1])
# 			n_samples = len(label_batch)
# 			print( " nsamples " + str(n_samples) ) 

# 			for batch in range(int(n_samples/BATCH_SIZE)):
# 				batch_x = feature_batch[batch*BATCH_SIZE : (1+batch)*BATCH_SIZE]
# 				batch_y = label_batch[batch*BATCH_SIZE : (1+batch)*BATCH_SIZE]
# 				print( batch_x[0].shape,batch_y[0])
# 				print(model.fit({'features':batch_x}, {'labels': batch_y }, n_epoch=50, 
# 						  validation_set=({'features': val_feature}, {'labels': val_labels}), 
# 						  shuffle=True, snapshot_step=None, show_metric=True, 
# 						  run_id=MODEL_NAME))
# 				test_feature =  val_feature[1:4]
# 				test_label_feature = val_labels[1:4]
# 				acc = model.evaluate({'features':val_feature},{'labels':val_labels})
# 				val_acc = acc[0] 
# 				print( prev_best,"prev_best" )
# 				print( val_acc,"val_acc" )
# 				if( val_acc >= prev_best and val_acc >= 0.5 ):
# 					prev_best = val_acc 
# 					model.save("/output/fin_model/sai_audio.tfl")		

# 				print(test_label_feature)
# 				print(" gap ")
# 				print(model.predict_label({'features':test_feature}))


				 


						
# train_sai_net() 


def load_model():

	val_features = [] 
	val_labels   = []


	val_feature =  pickle.load( open( TEST_FILE_PATH, "rb" ) )
	# val_labels =    pickle.load( open(TEST_FILE_PATH +  '_label_batch', "rb" ) )

	val_feature = val_feature.reshape([-1, NUM_OF_BANDS, 500 , 1])
	# for index in range (0,10):
	# 	print( val_labels[index] )
	model= sai_net()
	model.load('fin_model/sai_audio.tfl')
	# print(model.evaluate({'features':val_feature}, {'labels': val_labels}))
	prediction = model.predict_label({'features':val_feature})
	best_two_array = [] 
	for i in range (0,1500): 
		print( str(i) + " " + str(prediction[i]) )
		best_two_array.append( (prediction[i][0] , prediction[i][1]) )
		print(best_two_array[i])
	
	# stable_best_two = [] 
	# frequency = 1 
		  
	# for i in range(1,579):
	# 	if best_two_array[i] == best_two_array[i-1] : 
	# 		frequency = frequency + 1 
	# 	else :
	# 		if frequency >= 5 :
	# 			stable_best_two.append( (i - frequency , best_two_array[i-1] ) )
	# 		frequency = 1 
	
	# temporal_segmentation_array = [ ]
	# temporal_segmentation_array.append(0)
	# for i in range( 1 , len(stable_best_two)):
	# 	if( stable_best_two[i][1:] != stable_best_two[i-1][1:] ):
	# 		temporal_segmentation_array.append(stable_best_two[i][0] )


	# for i in range( 0 , len(temporal_segmentation_array)):
	# 	print( temporal_segmentation_array[i] )   
	# # print(prediction)
	# with open ( TEST_ANNOTATION_PATH,"r") as annotation : 
	# 	data = annotation.read() 
	# 	print(data) 

	# true_segments = data.split('\n')
	# for i in range(0,len(true_segments)):
	# 	true_segments[i]= int(true_segments[i]) 

	# print(true_segments)	
	# freq = 0 
	# for i  in true_segments :
	# 	print("next_change")
	# 	for j in temporal_segmentation_array:
	# 		print( i - j )
	# 		if(  abs(i- j) <= 10 ):
	# 			freq = freq + 1  
	# 			break 
	
	# print( "frequency" , freq)
	# precision = freq / float(len(temporal_segmentation_array)) 
	# print("precision",precision) 
	# recall = freq / float(len ( true_segments ) )
	# print("recall",recall) 
	# # prediction = model.predict({'features':val_feature})
	# # print(prediction)
load_model()