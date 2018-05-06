#! python3

import _pickle
import numpy as np
import my_util
from sklearn.metrics.pairwise import euclidean_distances
from re_ranking import K_Reciprocal_Encoding

with open('data/LFW_DATA.pickle', 'rb') as f:
	data = _pickle.load(f)

gallery_X = data['database_feature'][:]
gallery_Y = data['database_name'][:, 0]

query_X = data['query_feature'][:]
query_Y = data['query_name'][:, 0]

model = K_Reciprocal_Encoding(k1=20, distance_function=euclidean_distances)
model.fit(gallery_X=gallery_X)

distance_matrix_L2 = euclidean_distances(query_X, gallery_X)
distance_matrix_L2 = my_util.normalized(distance_matrix_L2)

distance_matrix_jaccard = model.metric(query_X)
distance_matrix_jaccard = my_util.normalized(distance_matrix_jaccard)

for mu in np.linspace(0, 1, 11):
	distance_matrix = mu * distance_matrix_jaccard + (1-mu) * distance_matrix_L2
	mAP = my_util.calculate_map(distance_matrix, gallery_Y, query_Y)
	cmc_rank = my_util.calculate_cmc_rank(distance_matrix, gallery_Y, query_Y)
	print(mu, mAP, cmc_rank)
