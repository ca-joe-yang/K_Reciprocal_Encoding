#! python3
import numpy as np

def predict_by_distance(distance_matrix, gallery_Y):
	'''
	compute prediction given distance matrix
	'''
	n_query = distance_matrix.shape[0]
	predict_matrix = np.zeros(distance_matrix.shape)
	for i in range(n_query):
		print(i, end='\r')
		distance_vector = distance_matrix[i]
		predict_matrix[i] = [x for _,x in sorted(zip(distance_vector, gallery_Y))]
	return predict_matrix

def calculate_map(distance_matrix, gallery_Y, query_Y):
	'''
	compute and return the mean average precision
	based on the distance matrix
	'''
	predict_matrix = predict_by_distance(distance_matrix, gallery_Y)
	return round(mean_average_precision(predict_matrix, query_Y), 3)

def calculate_cmc_rank(distance_matrix, gallery_Y, query_Y, k=1):
	predict_matrix = predict_by_distance(distance_matrix, gallery_Y)

	n_query = predict_matrix.shape[0]
	
	cmc_rank = []
	for i in range(n_query):
		preds = predict_matrix[i]
		label = query_Y[i]
		count, tp = 0, 0
		for j in range(k):
			pred = preds[j]
			count += 1
			if pred == label:
				tp += 1
		cmc_rank.append(tp / count)
	return round(np.average(cmc_rank), 3)

def average_precision(preds, label):
	'''
	compute and return the average precision for each query
	'''
	count = 0
	tp = 0
	precisions = []
	for pred in preds:
		count += 1
		if pred == label:
			tp += 1
			precisions.append(tp / count)
	if tp == 0:
		return 0
	return np.average(precisions)

def mean_average_precision(predict_matrix, query_Y):
	n_query = predict_matrix.shape[0]
	
	average_precisions = []
	for i in range(n_query):
		preds = predict_matrix[i]
		label = query_Y[i]
		AP = average_precision(preds, label)
		average_precisions.append(AP)
	
	return np.average(average_precisions)

def normalized(A):
	return (A - A.min()) / (A.max() - A.min())