from pickle import load
from numpy import argmax, expand_dims, reshape
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import tensorflow.keras.applications.inception_v3 as inception
import tensorflow.keras.applications.xception as xception
import cv2



# extract features from each photo in the directory
def extract_features(image):
    # load the model
    model = VGG16()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature
        
 
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'start'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    in_text = in_text.replace("endseq", "")
    in_text = in_text.replace("start", "")
    return in_text

# load the tokenizer
tokenizer = load(open('model/vgg/tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 32
# load the model
model = load_model('model/vgg/model_19.h5')
# model = load_model('model/inception/mymodel.h5')




## ======================= Inception v3 ========================================== ##
def preprocess_img(image):
    #inception v3 excepts img in 299*299
    img = cv2.resize(image, (299, 299))
   
    # Add one more dimension
    x = expand_dims(img, axis = 0)
    x = inception.preprocess_input(x)
    return x



#function to encode an image into a vector using inception v3
def encode(image):
    base_model = inception.InceptionV3(weights = 'imagenet')
    model = Model(base_model.input, base_model.layers[-2].output)
    image = preprocess_img(image)
    vec = model.predict(image)
    vec = reshape(vec, (vec.shape[1]))
    return vec

def greedy_search(pic):
    pic = encode(pic).reshape(1, 2048)
    start = 'startseq'
    max_length = 34
    # load the model
    model = load_model('model/inception/mymodel.h5')
    wordtoix = load(open('model/inception/wordix.pkl', 'rb'))
    ixtoword = load(open('model/inception/ixword.pkl', 'rb'))
    for i in range(max_length):
        seq = [wordtoix[word] for word in start.split() if word in wordtoix]
        seq = pad_sequences([seq], maxlen = max_length)
        yhat = model.predict([pic, seq])
        yhat = argmax(yhat)
        word = ixtoword[yhat]
        start += ' ' + word
        if word == 'endseq':
            break
    final = start.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final




## ======================= Xception ========================================== ##
class Xception:
    def __init__(self, image):
        self.feature_model = xception.Xception(include_top=False, pooling="avg")
        self.image = image
        self.model = load_model('model/xception/model_19.h5')
        self.tokenizer = load(open("model/xception/tokenizer.p","rb"))


    def extract_features(self):
      
        img = cv2.resize(self.image, (299, 299))
    
        # for images that has 4 channels, we convert them into 3 channels
        if self.image.shape[2] == 4: 
            self.image = self.image[..., :3]
        self.image = expand_dims(self.image, axis=0)
        self.image = self.image/127.5
        self.image = self.image - 1.0
        feature = self.feature_model.predict(self.image)
        return feature


    def word_for_id(self, integer):
        for word, index in self.tokenizer.word_index.items():
            if index == integer:
                return word
        return None


    def generate_desc(self,  max_length):
        in_text = 'start'
        photo = self.extract_features()
        for i in range(max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            pred = self.model.predict([photo,sequence], verbose=0)
            pred = argmax(pred)
            word = self.word_for_id(pred)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'end':
                break
 
        return in_text






