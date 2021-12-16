import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from collections import Counter
from sklearn import preprocessing
from itertools import repeat
import multiprocessing as mp
import time

class CustomKNN:
	
	def __init__(self):
		self.accurate_predictions = 0
		self.total_predictions = 0
		self.accuracy = 0.0

	def predict(self, training_data, to_predict, k = 3):
		distributions = []
		for group in training_data:
			for features in training_data[group]:
				euclidean_distance = np.sqrt(np.sum(np.square(np.array(features)- np.array(to_predict))))
				distributions.append([euclidean_distance, group])
		results = [i[1] for i in sorted(distributions)[:k]]
		result = Counter(results).most_common(1)[0][0]
		confidence = Counter(results).most_common(1)[0][1]/k
		return result, to_predict
	
	def test(self, test_set, training_set):
		pool = mp.Pool(processes= 10)

		arr = {}
		for group in test_set:
			arr[group] =  pool.starmap(self.predict, zip(repeat(training_set), test_set[group], repeat(3)))

		for group in test_set:
			for data in test_set[group]:
				for i in arr[group]:
					if data == i[1]:
						self.total_predictions += 1
						if group == i[0]:
							self.accurate_predictions+=1
		
		self.accuracy = 100*(self.accurate_predictions/self.total_predictions)
		print("\nAcurracy :", str(self.accuracy) + "%")

def mod_data(df):
	df.replace('yes', 4, inplace = True)
	df.replace('no', 2, inplace = True)

	df.replace('notpresent', 4, inplace = True)
	df.replace('present', 2, inplace = True)
	
	df.replace('abnormal', 4, inplace = True)
	df.replace('normal', 2, inplace = True)
	
	df.replace('poor', 4, inplace = True)
	df.replace('good', 2, inplace = True)
	
	df.replace('ckd', 4, inplace = True)
	df.replace('notckd', 2, inplace = True)

def main():
	print("Parallel KNN Algorithm")
	df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/kidney_disease.csv')
	mod_data(df)
	dataset = df.astype(float).values.tolist()
	
	x = df.values
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	df = pd.DataFrame(x_scaled)

	random.shuffle(dataset)

	test_size = 0.1

	training_set = {2: [], 4:[]}
	test_set = {2: [], 4:[]}

	training_data = dataset[:-int(test_size * len(dataset))]
	test_data = dataset[-int(test_size * len(dataset)):]

	for record in training_data:
		training_set[record[-1]].append(record[:-1])

	#Insert data into the test set
	for record in test_data:
		test_set[record[-1]].append(record[:-1]) 
	knn = CustomKNN()
	s = time.perf_counter()
	
	knn.test(test_set, training_set)
	e = time.perf_counter()
	
	print("Exec Time: ", e-s)
	
if __name__ == "__main__":
	main()
