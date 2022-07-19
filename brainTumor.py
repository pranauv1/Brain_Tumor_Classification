import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
from sklearn.utils import resample

import tensorflow
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

np.random.seed(42)


#Download datasets
! kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri
! kaggle datasets download -d ahmedhamada0/brain-tumor-detection

#Unzip them
! unzip brain-tumor-classification-mri
! unzip brain-tumor-detection


#These are the lists which is going to contain image paths of each tumor
gi_list=[]
me_list=[]
no_list=[]
pi_list=[]
yes_list=[]

#Going through each folder to get 'imagePath' and adding them to lists(gi_list,me_list,no_list,pi_list)
for image_path in glob(os.path.join('/path/Testing/glioma_tumor/','*.jpg')):
  gi_list.append(image_path)
  
for image_path in glob(os.path.join('/path/Training/glioma_tumor/','*.jpg')):
  gi_list.append(image_path)
  
for image_path in glob(os.path.join('/path/Testing/meningioma_tumor/','*.jpg')):
  me_list.append(image_path)
  
for image_path in glob(os.path.join('/path/Training/meningioma_tumor/','*.jpg')):
  me_list.append(image_path)

for image_path in glob(os.path.join('/path/Testing/no_tumor/','*.jpg')):
  no_list.append(image_path)  
  
for image_path in glob(os.path.join('/path/Training/no_tumor/','*.jpg')):
  no_list.append(image_path)
  
for image_path in glob(os.path.join('/path/Testing/pituitary_tumor/','*.jpg')):
  pi_list.append(image_path)
  
for image_path in glob(os.path.join('/path/Training/pituitary_tumor/','*.jpg')):
  pi_list.append(image_path)
    
for image_path in glob(os.path.join('/path/yes/','*.jpg')):
  yes_list.append(image_path)
  
  
#Printing the lengths of each lists to see count
print(len(gi_list))
print(len(me_list))
print(len(no_list))
print(len(pi_list))
print(len(yes_list))

#Converting those images --> 32X32 images as np array
gi_list_array = list(map(lambda x: np.asarray(Image.open(x).resize((32,32))),gi_list))
me_list_array = list(map(lambda x: np.asarray(Image.open(x).resize((32,32))),me_list))
no_list_array = list(map(lambda x: np.asarray(Image.open(x).resize((32,32))),no_list))
pi_list_array = list(map(lambda x: np.asarray(Image.open(x).resize((32,32))),pi_list))

#The yes_list containes some gray scaled images, so we need to convert them into RGB and then resize them
yes_list_array = list(map(lambda x: np.asarray(Image.open(x).convert('RGB').resize((32,32))),yes_list))


#Balancing the data
n_samples = 1000
gi_list_balanced = resample(gi_list_array, replace=True, n_samples=n_samples, random_state=42)
me_list_balanced = resample(me_list_array, replace=True, n_samples=n_samples, random_state=42)
no_list_balanced = resample(no_list_array, replace=True, n_samples=n_samples, random_state=42)
pi_list_balanced = resample(pi_list_array, replace=True, n_samples=n_samples, random_state=42)
yes_list_balanced = resample(yes_list_array, replace=True, n_samples=n_samples, random_state=42)


#Adding those balanced data to a new DataFrame with labels
brain_df_0= pd.DataFrame({'image':gi_list_balanced,'label':0})
brain_df_1= pd.DataFrame({'image':me_list_balanced, 'label':1})
brain_df_2= pd.DataFrame({'image':no_list_balanced, 'label':2})
brain_df_3= pd.DataFrame({'image':pi_list_balanced, 'label':3})
brain_df_4= pd.DataFrame({'image':yes_list_balanced, 'label':4})


#Converting those DataFrames into a single new DataFrame
brain_df_final = pd.concat([brain_df_0, brain_df_1, brain_df_2, brain_df_3, brain_df_4])

brain_df_final.shape



#Creating the X and Y for Training and Testing

#Converting 'image(s)' from DF(brain_df_final) to np.array
X = np.asarray(brain_df_final['image'].tolist())
#Sclling those values from 0-255 to 0-1.
X=X/255.

#Assigning 'label(s)' from DF(brain_df_final) to Y
Y=brain_df_final['label']
#Since this a multiclass problem we need to conver those Y values into 'categorical'
Y_cat = to_categorical(Y, num_classes=5)

#Will Split Training and Testing
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)


##################################   MODEL   ##############################################

num_calasses = 5

model_1 = Sequential()
model_1.add(Conv2D(128, (3, 3), activation="relu", input_shape=(32, 32, 3)))
#BatchNormalization
model_1.add(MaxPool2D(pool_size=(2,2)))
model_1.add(Dropout(0.3))

model_1.add(Conv2D(64, (3, 3), activation="relu"))
#BatchNormalization
model_1.add(MaxPool2D(pool_size=(2,2)))
model_1.add(Dropout(0.3))

model_1.add(Conv2D(32, (3, 3), activation="relu"))
#BatchNormalization
model_1.add(MaxPool2D(pool_size=(2,2)))
model_1.add(Dropout(0.3))
model_1.add(Flatten())

model_1.add(Dense(16))
model_1.add(Dense(5, activation="softmax"))
model_1.summary()

model_1.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["acc"])


#Let's Train
batch_size = 16
epochs = 50

history = model_1.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2)

#Model accuracy
acc = model_1.evaluate(x_test, y_test)
print('Accuracy = ', acc[1]*100, '%') #acc was about 77%

#Save model
model_1.save('BrainTumor.h5')





#Testing with a single image
from keras.preprocessing import image

img = image.load_img('/path/no/no1455.jpg', target_size = (32, 32))
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)
prediction = model_1.predict(img)
print(prediction)
