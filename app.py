import streamlit as st
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io
import json

st.title("Neural Marvel Text Generator")
st.write("Type the starting text (marvel related preferebly) and then press enter ( eg- iron man killed thanos ....)")
model = tf.keras.models.load_model("model.h5")

#tokenizer = Tokenizer()

#data = open('data.txt').read()

#corpus = data.lower().split("\n")

#tokenizer.fit_on_texts(corpus)
#total_words = len(tokenizer.word_index) + 1




#tokenizer_json = tokenizer.to_json()
#with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
#    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)



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
