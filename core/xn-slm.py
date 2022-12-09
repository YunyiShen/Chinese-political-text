from sentence_transformers import SentenceTransformer, util
import csv 
import json
import numpy as np

# load the model
model = SentenceTransformer("symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli")

# load the data to try
prompt = "有关"
all_topics = json.load(open("../data/topics.json"))
prompt_topics = [prompt+tpc for tpc in all_topics]

sentences = []
topic = []


with open("../data/long_sample.csv") as reader:
	csvreader = csv.reader(reader)
	next(csvreader)
	for tpc, st in csvreader:
		senten.append(st)
		topic.append(tpc)
  

sen_ecd = model.encode(sentences)
tpc_ecd = model.encode(prompt_topics)
predicted = torch.argmax(util.cos_sim(ddd,ddd),1).tolist()
predicted = np.array( [all_topics[idd] for idd in predicted])

np.savetxt('predicted_topic.csv', predicted, delimiter=",")

print(np.mean(predicted==np.array(topic)))






