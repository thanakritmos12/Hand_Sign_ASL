import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Initialize lists to store consistent data
data_filtered = []
labels_filtered = []

# Check and filter inconsistent data
expected_length = 42  # Replace this with the correct expected length of your data
for idx, d in enumerate(data_dict['data']):
    if len(d) == expected_length:
        data_filtered.append(d)
        labels_filtered.append(data_dict['labels'][idx])

# Convert to numpy arrays
data = np.asarray(data_filtered)
labels = np.asarray(labels_filtered)

# Count class frequencies
label_counts = Counter(labels)

# Filter out classes with fewer than 2 samples
valid_indices = [i for i, label in enumerate(labels) if label_counts[label] > 1]

data_filtered = data[valid_indices]
labels_filtered = labels[valid_indices]

# Split the filtered data with a larger test_size
x_train, x_test, y_train, y_test = train_test_split(
    data_filtered, labels_filtered, test_size=0.4, shuffle=True, stratify=labels_filtered
)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and evaluate the model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
