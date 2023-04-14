import json
import torch
import random
from tools import all_words
from create_samples import create_data , create_vector
from train_file import train_fn

f = open("all_intents_js.json")
dt = json.load(f)

#vocab
words_list = sorted(all_words(dt))

intents_mapping = {}
intents_response = {}
for i , d in enumerate(dt["data"]):
    intents_mapping[d["intent"]] = i
    intents_response[i] = d["responses"]
reverse_mapping = {intents_mapping[k] : k for k in intents_mapping.keys()}

#create the training data
train , target = create_data(dt , words_list , intents_mapping)
model = train_fn(train , target , len(words_list) , 6)

#chatting
while True:
    inp = input("You: ")
    vec = create_vector(inp , words_list)
    input_vector = torch.as_tensor(vec , dtype = torch.float32)[None , ...]
    output = model(input_vector)
    pred_num = output.argmax().item()
    resps = intents_response[pred_num]
    print("Bot: " , resps[random.randint(0 , 2)])












