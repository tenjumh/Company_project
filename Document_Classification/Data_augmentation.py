#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


print('Using real-time data augmentation.')

def data_agumentation():
    # This will do preprocessing and realtime data augmentation:    
    datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                 samplewise_center=False,  # set each sample mean to 0
                                 featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                 samplewise_std_normalization=False,  # divide each input by its std
                                 zca_whitening=False,  # apply ZCA whitening
                                 zca_epsilon=1e-06,  # epsilon for ZCA whitening
                                 rotation_range=1,  # randomly rotate images in the range (degrees, 0 to 180)
                                 # randomly shift images horizontally (fraction of total width)
                                 width_shift_range=0.05,
                                 # randomly shift images vertically (fraction of total height)
                                 height_shift_range=0.05,
                                 shear_range=0.05.,  # set range for random shear
                                 zoom_range=00.5.,  # set range for random zoom
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
    return datagen

