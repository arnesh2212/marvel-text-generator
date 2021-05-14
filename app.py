import streamlit as st
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("Marvel Text Generator")
st.write("Type the starting text (marvel related preferebly) and then press enter ( eg- iron man killed thanos ....)")
model = tf.keras.models.load_model("model.h5")





import pickle

#with open('tokenizer.pickle', 'wb') as handle:
#    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# loading

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
total_words = len(tokenizer.word_index) + 1

# pad sequences 
max_sequence_len = 23

user_input = st.text_input("Enter Seed Text")

seed_text = user_input
next_words = 50

for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
	predicted = model.predict_classes(token_list, verbose=0)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word
st.write(seed_text)
