import torch 
import streamlit as st
import pandas as pd
import numpy as np
from model import nw
from train import tokenizer, model
import torch.nn.functional as F

option = st.sidebar.selectbox("Projects", ("Emo detections", 
												"Project 2"))

st.header(option)

if option == "Emo detections":
	text = st.text_input("Enter a text:")
	model.eval()

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	encoded_texts = tokenizer(txts, padding=True, truncation=True, return_tensors='pt').to(device)
	outputs = model(encoded_texts['input_ids'], attention_mask=encoded_texts['attention_mask'])
	outs = F.softmax(outputs.logits, dim = 1)
	predicted_labels = torch.argmax(outs, dim=1)
	predicted_labels
	st.write('time now is ')
	st.write(nw)
	st.write(text)

if option == "Project 2":
	st.write('Proj 2')
	