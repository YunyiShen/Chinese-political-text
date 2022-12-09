from sentence_transformers import SentenceTransformer, util
import csv 
import json
import numpy as np
import torch

# load the model
model = SentenceTransformer("symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli")
#model = SentenceTransformer("../sbert-base-chinese-nli")
# load the data to try
prompt = "本句谈到"
postfix = "相关话题"
all_topics = json.load(open("../data/topics.json"))
prompt_topics = [prompt+tpc+postfix for tpc in all_topics]

sentences = []
topic = []


with open("../data/long_sample.csv") as reader:
	csvreader = csv.reader(reader)
	next(csvreader)
	for tpc, st in csvreader:
		sentences.append(st)
		topic.append(tpc)
  

sen_ecd = model.encode(sentences)
tpc_ecd = model.encode(prompt_topics)
predicted = torch.argmax(util.cos_sim(sen_ecd,tpc_ecd),1).tolist()
predicted = ( [all_topics[idd] for idd in predicted])
with open('predicted.json', 'w') as f:
    json.dump(predicted, f)

print(predicted)
print(topic)

correct_or_not = [topic[i]==predicted[i] for i in range(len(predicted))]

print(np.mean(np.array(correct_or_not)))






