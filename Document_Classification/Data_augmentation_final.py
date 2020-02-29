#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os, glob, sys, numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os, glob, sys, numpy as np
from Save_dataset_to_npy import categories

datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                             samplewise_center=False,  # set each sample mean to 0
                             featurewise_std_normalization=False,  # divide inputs by std of the dataset
                             samplewise_std_normalization=False,  # divide each input by its std
                             zca_whitening=False,  # apply ZCA whitening
                             zca_epsilon=1e-06,  # epsilon for ZCA whitening
                             rotation_range=1,  # randomly rotate images in the range (degrees, 0 to 180)
                             # randomly shift images horizontally (fraction of total width)
                             width_shift_range=0.1,
                             # randomly shift images vertically (fraction of total height)
                             height_shift_range=0.1,
                             shear_range=0.1,  # set range for random shear
                             zoom_range=0.1,  # set range for random zoom
                             channel_shift_range=0.,  # set range for random channel shifts
                             # set mode for filling points outside the input boundaries
                             fill_mode='nearest',
                             cval=0.,  # value used for fill_mode = "constant"
                             horizontal_flip=False,  # randomly flip images y축으로 반사
                             vertical_flip=False,  # randomly flip images x축으로 반사
                             # set rescaling factor (applied before any other transformation)
                             rescale=1./255,
                             # set function that will be applied on each input
                             preprocessing_function=None,
                             # image data format, either "channels_first" or "channels_last"
                             data_format=None,
                             # fraction of images reserved for validation (strictly between 0 and 1)
                             validation_split=0.0)
                             

caltech_dir = "./data"
nb_classes = len(categories)

for idx, cat in enumerate(categories):
    #one-hot 돌리기
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    
    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir+"\\*.jpg")
    
    print(cat, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        save_to_dir='./data/%s'%(cat)
        save_prefix='%s'%(cat)
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=save_to_dir, save_prefix=save_prefix, save_format='jpg'):
            i += 1
            if i > 5:
                break  # otherwise the generator would loop indefinitely


# In[ ]:




