#! python3

import _pickle
import numpy as np
import my_util
from sklearn.metrics.pairwise import euclidean_distances
from re_ranking import K_Recirpocal_Encoding

with open('data/LFW_DATA.pickle', 'rb') as f:
	data = _pickle.load(f)

gallery_X = data['database_feature'][:]
gallery_Y = data['database_name'][:, 0]

query_X = data['query_feature'][:]
query_Y = data['query_name'][:, 0]

model = K_Recirpocal_Encoding(k1=20, distance_function=euclidean_distances)
model.fit(gallery_X=gallery_X, gallery_Y=gallery_Y)
pred = model.predict(query_X, k2=6, mu=0.95)
mAP = my_util.mean_average_precision(pred, query_Y)
mAP = round(mAP, 4)

print(mAP)