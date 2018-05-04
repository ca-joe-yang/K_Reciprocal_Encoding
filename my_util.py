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