import os
import cv2
import numpy as np
import tensorflow as tf
### importing required packages
import pathlib
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



data_dir_train= pathlib.Path('D:/DeDustProject/data/Classification_Small/train')
data_dir_val= pathlib.Path('D:/DeDustProject/data/Classification_Small/val')
data_dir_test= pathlib.Path('D:/DeDustProject/data/Classification_Small/test')

image_count_train = len(list(data_dir_train.glob('*/*.PNG')))
image_count_val = len(list(data_dir_val.glob('*/*.PNG')))
image_count_test = len(list(data_dir_test.glob('*/*.PNG')))
print(image_count_train)
print(image_count_val)
print(image_count_test)

### setting the batch size to 8 and image height and width
batch_size = 8
img_height = 256
img_width = 256

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                   horizontal_flip=True,
                                   shear_range=0.2,
                                   rotation_range=30,

                                   )

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(data_dir_train,
                                                    target_size=(256,256),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    color_mode='grayscale'
                                                    )
val_generator= val_datagen.flow_from_directory(data_dir_val,
                                               target_size=(256,256),
                                               batch_size=2,
                                               class_mode='categorical',
                                               color_mode='grayscale'
                                               )

num_classes = 2
classes = ['Artifacts','Nuclei'] #Classification classes
epochs = 10


#A simple CNN followed by dense layers
model = tf.keras.Sequential(
    [
     tf.keras.layers.Conv2D(256,(5,5),input_shape=(256,256,1),padding='VALID',activation='relu'),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
     #tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(128,activation='relu'),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(32,activation='relu'),
     tf.keras.layers.Dense(num_classes,activation='softmax')


    ]
)


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epochs,logs={}):
    if(logs.get('val_accuracy')>=0.99):
      print("Stopped training early!")
      self.model.stop_training = True

callback = myCallback()

checkpoint_filepath = './'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

#data_dir_test = 'D:/DeDustProject/data/Classification_Small/test'
test_generator = test_datagen.flow_from_directory(data_dir_test,
                                               target_size=(256,256),
                                               batch_size=1,
                                               class_mode='categorical',
                                               color_mode='grayscale')

data_dir_pred = 'D:/DeDustProject/data/Classification_Small/predict'
pred_generator = test_datagen.flow_from_directory(data_dir_pred,
                                               target_size=(256,256),
                                               batch_size=1,
                                               class_mode='categorical',
                                               color_mode='grayscale')



history = model.fit(train_generator,epochs=epochs,validation_data=val_generator,verbose=2)

#model.save('artifact_classifier.h5')

#Plotting the results

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss= history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(7,7))

plt.subplot(1,2,1)
plt.plot(epochs_range,acc,label='Training Accuracy')
plt.plot(epochs_range,val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range,loss,label='Training Loss')
plt.plot(epochs_range,val_loss, label='Validation Loss')
plt.legend(loc='lower right')
plt.title('Training and Validation Loss')

#plt.savefig('plot.PNG')
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'myplot.svg'

plt.savefig(image_name, format=image_format, dpi=300)

plt.show()

# #load model
# savedModel = tf.keras.models.load_model('artifact_classifier.h5')
# img = io.imread("D:/DeDustProject/data/Classification_Small/test/Artifact/1778.PNG")
# img = np.asarray(img,dtype='float32')
# img /= 255
# x = np.expand_dims(img,axis = 0)
# x = np.expand_dims(x,axis = 3)
# pred = savedModel.predict(x)
# MaxPosition = np.argmax(pred)
# prediction_label = classes[MaxPosition]
# print(prediction_label)
# print(pred)






#score = savedModel.evaluate(test_generator)
# y_test = []
# while image_count_test !=0 :
#     (_,y) = next(test_generator)
#     y_test.append(y[0])
#     image_count_test -=1
#
# y_test = np.asarray(y_test,dtype='float32')
# y_test.reshape((2000,2))
# print(len(y_test))
# # for (x,y) in test_generator:
# #     y_test.append(y)
# y_pred = savedModel.predict(test_generator)
# y_pred = np.argmax(y_pred,axis=1)
# y_test = np.argmax(y_test,axis=1)
# #calculating precision and reall
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
#
#
# print('Precision: ',precision)
# print('Recall: ',recall)
# print('Confusion Matrix: \n')
# print(confusion_matrix(y_test, y_pred),'\n')
# print('Classification Report :\n')
# print(classification_report(y_test, y_pred, digits=3))



#print('test_loss is :',score[0])
#print('test accuracy is :',score[1])
