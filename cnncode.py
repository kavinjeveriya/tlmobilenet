from tensorflow.keras.applications.mobilenet import MobileNet
model = MobileNet(weights='imagenet' , include_top=False , input_shape=(224,224,3) )
model.layers[0].input
model.layers[0].trainable
model.output
for i in model.layers:
    i.trainable=False
model.layers[6].trainable
top_model = model.output
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
top_model = Flatten()(top_model)
top_model = Dense(1024 , activation='relu' )(top_model)
top_model = Dense(512 , activation='relu' )(top_model)
top_model = Dense(102 , activation='softmax' )(top_model)
from keras.models import Model
model= Model(inputs=model.input , outputs = top_model)
model.summary()
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.0001) , loss='categorical_crossentropy' , metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
traingen = ImageDataGenerator(rescale=1./255 , zoom_range=0.2 , horizontal_flip=True ,  vertical_flip=True )
testgen = ImageDataGenerator(rescale=1./255)
trainset = traingen.flow_from_directory('/app/dataset/train/' , target_size=(224,224) , batch_size=32 , class_mode='categorical' )
testset = testgen.flow_from_directory('/app/dataset/valid/' , target_size=(224,224) , batch_size=32 , class_mode='categorical' )
model.fit(trainset , epochs=10 , validation_data=testset , steps_per_epoch=125, validation_steps=10  )
model.save("/app/models_acc/model.h5")
scores = model.evaluate(testset,verbose=1)
print('loss',scores[0])
print('acc',scores[1])
acc=scores[1]*100
file = open("/app/models_acc/accuracy.txt", "w")
file.write(str(acc))
file.close()

