import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data and convert the letters to numbers
data = np.loadtxt('data/letter-recognition.data', dtype='float32', delimiter=',',
                  converters={0: lambda ch: ord(ch) - ord('A')})

# Split data into features (trainData) and labels (responses)
responses, trainData = np.hsplit(data, [1])

# Define train/test splits and k values
split_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
k_values = range(1, 10)
accuracy_results = np.zeros((len(split_ratios), len(k_values)))

# Loop over different train/test splits
for i, split_ratio in enumerate(split_ratios):
    # Split data into training and testing sets
    trainData_split, testData, responses_train, responses_test = train_test_split(trainData, responses, test_size=split_ratio, random_state=42)
    
    # Train the KNN model for different k values
    for j, k in enumerate(k_values):
        knn = cv.ml.KNearest_create()
        knn.train(trainData_split, cv.ml.ROW_SAMPLE, responses_train)
        
        # Perform predictions
        ret, result, neighbours, dist = knn.findNearest(testData, k=k)
        
        # Calculate accuracy
        accuracy = accuracy_score(responses_test, result)
        accuracy_results[i, j] = accuracy

# Plotting results
plt.figure(figsize=(12, 8))
for j, k in enumerate(k_values):
    plt.plot(split_ratios, accuracy_results[:, j], marker='o', label=f'k = {k}')

plt.xlabel('Train/Test Split Ratio (Test Size)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Train/Test Split Ratio for Different k Values (English Alphabet)')
plt.legend(title='k Values', loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()
# plt.savefig('knn_english_alphabet.png')