
import json

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2, activity_l2
from keras import regularizers 


# load the initial pre-trained model
def build_model(nb_classes):
    initial_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(3,224,224)))

    last = initial_model.output
    x = Flatten()(last)

    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(nb_classes, activation='softmax',  W_regularizer = l2(0.1) )(x)
    model = Model(input=initial_model.input, output=predictions)

    # first we freeze all convolutional InceptionV3 layers
    for layer in initial_model.layers:
        layer.trainable = False

    
    print "starting model compile"
    compile(model)
    print "model compile done"
    return model


def save(model, tags, prefix):
    model.save_weights(prefix+".h5")
    # serialize model to JSON
    model_json = model.to_json()
    with open(prefix+".json", "w") as json_file:
        json_file.write(model_json)
    with open(prefix+"-labels.json", "w") as json_file:
        json.dump(tags, json_file)


def load(prefix):
    # load json and create model
    with open(prefix+".json") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights(prefix+".h5")
    with open(prefix+"-labels.json") as json_file:
        tags = json.load(json_file)
    return model, tags

def compile(model):
	model.compile(loss = "categorical_crossentropy", optimizer ="adam", metrics=["accuracy"])	
    
