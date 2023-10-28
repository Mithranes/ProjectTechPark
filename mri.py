import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix
import ipywidgets as widgets
import io
from PIL import Image
from IPython.display import display,clear_output
from warnings import filterwarnings
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pickle
import io
import cv2
import numpy as np
from PIL import Image
from ipywidgets import FileUpload, Button, Output, VBox, HTML
from IPython.display import clear_output
import flask
from keras.models import load_model
import fastapi


colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']

sns.palplot(colors_dark)
sns.palplot(colors_green)
sns.palplot(colors_red)

labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']


X_train = []
y_train = []
image_size = 150
for i in labels:
    folderPath = os.path.join('D:\Programming\Codes\ProjectTechPark\Brain tumors\Training',i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size, image_size))
        X_train.append(img)
        y_train.append(i)
        
for i in labels:
    folderPath = os.path.join('D:\Programming\Codes\ProjectTechPark\Brain tumors\Testing',i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        y_train.append(i)
        
X_train = np.array(X_train)
y_train = np.array(y_train)


k=0
fig, ax = plt.subplots(1,4,figsize=(20,20))
fig.text(s='Sample Image From Each Label',size=18,fontweight='bold',
             fontname='monospace',color=colors_dark[1],y=0.62,x=0.4,alpha=0.8)
for i in labels:
    j=0
    while True :
        if y_train[j]==i:
            ax[k].imshow(X_train[j])
            ax[k].set_title(y_train[j])
            ax[k].axis('off')
            k+=1
            break
        j+=1



X_train, y_train = shuffle(X_train,y_train, random_state=101)

X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size=0.1,random_state=101)

y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)


y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)


effnet = EfficientNetB0(weights='imagenet',include_top=False,input_shape=(image_size,image_size,3))

model = effnet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(rate=0.5)(model)
model = tf.keras.layers.Dense(4,activation='softmax')(model)
model = tf.keras.models.Model(inputs=effnet.input, outputs = model)

model.compile(loss='categorical_crossentropy',optimizer = 'Adam', metrics= ['accuracy'])

tensorboard = TensorBoard(log_dir = 'logs')
checkpoint = ModelCheckpoint("effnet.h5",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,
                              mode='auto',verbose=1)


model = load_model("D:\Programming\Codes\ProjectTechPark\my_model.keras")


pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)

uploader = FileUpload()
clear_button = Button(description='Clear Uploads')
predict_button = Button(description='Predict')
out = Output()
prediction_output = HTML()



def img_pred(uploader):
    with out:
        clear_output()
        for name, file_info in uploader.value.items():
            img = Image.open(io.BytesIO(file_info['content']))
            opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = cv2.resize(opencvImage, (150, 150))
            img = img.reshape(1, 150, 150, 3)
            p = model.predict(img)
            p = np.argmax(p, axis=1)[0]

            if p == 0:
                p = 'Glioma Tumor'
            elif p == 1:
                p = 'No Tumor'
            elif p == 2:
                p = 'Meningioma Tumor'
            else:
                p = 'Pituitary Tumor'

            if p != 'No Tumor':
                prediction_output.value = f'The Model predicts that it is a {p}'

def clear_uploads(_):
    uploader.value.clear()
    prediction_output.value = ''

def predict(_):
    img_pred(uploader)


uploader.observe(img_pred, names='_counter')
clear_button.on_click(clear_uploads)
predict_button.on_click(predict)

vbox = VBox([uploader, clear_button, predict_button, out, prediction_output])
vbox




