import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder # type: ignore

# to read the excel file
dp = "SuperStoreUS-2015.xlsx"
data = pd.read_excel(dp)
column_name = 'Product Base Margin'
row_number = 181
null_row = data.iloc[row_number][column_name]
print("to prove that this cell is null:", null_row)

# checking if there are null values 
missing_values_count = data.isnull().sum()
print(missing_values_count)

# filling the null values with 0
df = data.fillna(value='0')
print("after filling:", df.iloc[row_number][column_name])

# filling the null values with the mean value
df2 = data.fillna(value=data['Product Base Margin'].mean())
print("after filling 2:", df2.iloc[row_number][column_name])

# Remove irrelevant attributes
irrelevant_columns = ['Ship Mode']  
df2.drop(columns=irrelevant_columns, inplace=True)

# Choose the column to bin (e.g., 'Sales')
column_to_bin = 'Sales'

# Specify the number of bins
num_bins = 5

# Binning 'Sales' column with quantiles
df2['Binned_Sales'], bin_edges = pd.qcut(df2[column_to_bin], q=num_bins, labels=False, retbins=True)

# Calculate bin means
bin_means = df2.groupby('Binned_Sales')[column_to_bin].mean()

# Replace values in each bin with the mean of that bin
df2['Binned_Sales'] = df2['Binned_Sales'].map(bin_means)

# Drop Duplicates
data_no_duplicates = df2.drop_duplicates()

# Select relevant columns
relevant_columns = ['Unit Price', 'Profit']
relevant_data = df2[relevant_columns]

# Calculate the correlation matrix before removing correlated attributes
correlation_before = relevant_data.corr()
print("Correlation matrix before removing correlated attributes:")
print(correlation_before)

# Calculate the correlation matrix
correlation_matrix = relevant_data.corr()

# Discretize the correlation matrix
num_bins = 5
correlation_bins = pd.cut(correlation_before.stack(), bins=num_bins, labels=False)
correlation_discretized = correlation_bins.unstack()

print("Discretized Correlation matrix:")
print(correlation_discretized)

# Identify highly correlated attributes
highly_correlated = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) >= 0.8:
            colname = correlation_matrix.columns[i]
            highly_correlated.add(colname)


# Remove correlated attributes from the dataset
df2.drop(columns=highly_correlated, inplace=True)

# Select relevant columns again after dropping highly correlated attributes
relevant_data_after = df2[relevant_columns]

# Calculate the correlation matrix after removing correlated attributes
correlation_after = relevant_data_after.corr()
print("\nCorrelation matrix after removing correlated attributes:")
print(correlation_after)

# Save the cleaned dataset
cleaned_dp = "cleaned_SuperStoreUS-2015.xlsx"
df2.to_excel(cleaned_dp, index=False)
print("Cleaned dataset saved as:", cleaned_dp)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Read the cleaned dataset
cleaned_dp = "cleaned_SuperStoreUS-2015.xlsx"
df2 = pd.read_excel(cleaned_dp)


# Selecting only numeric columns
numeric_columns = df2.select_dtypes(include=['float64', 'int64']).columns


# Split the dataset into features (X) and target variable (y)
X = df2[numeric_columns]  
y = df2['Binned_Sales']   

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
# Encode bin labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

    
# Initialize classifiers
decision_tree_clf = DecisionTreeClassifier()
naive_bayes_clf = GaussianNB()

# Train the Decision Tree classifier
decision_tree_clf.fit(X_train, y_train_encoded)

# Train the Naïve Bayes classifier
naive_bayes_clf.fit(X_train, y_train_encoded)

# Save the training and testing sets into separate files
X_train.to_excel("X_train.xlsx", index=False)
X_test.to_excel("X_test.xlsx", index=False)
y_train.to_excel("y_train.xlsx", index=False)
y_test.to_excel("y_test.xlsx", index=False)

# Save the trained classifiers
joblib.dump(decision_tree_clf, 'decision_tree_classifier.joblib')
joblib.dump(naive_bayes_clf, 'naive_bayes_classifier.joblib')
    
from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(decision_tree_clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")
    
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
    
y_pred = naive_bayes_clf.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test_encoded, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Naive Bayes Classifier')
plt.show()

# Classification Report
cr = classification_report(y_test_encoded, y_pred)
print("Classification Report for Naive Bayes Classifier:")
print(cr)
    
# K-means Clustering (Replace with your chosen features)
kmeans = KMeans(n_clusters=4, random_state=42)  
kmeans.fit(X)

# Add cluster labels as a new feature
df2['Cluster'] = kmeans.labels_

# Select features for visualization (replace with your choices)
features_to_plot = ['Unit Price', 'Profit']

# Create a scatter plot with cluster colors
plt.figure(figsize=(8, 6))
plt.scatter(df2[features_to_plot[0]], df2[features_to_plot[1]], c=df2['Cluster'], cmap='viridis')
plt.xlabel(features_to_plot[0])
plt.ylabel(features_to_plot[1])
plt.title("K-Means Clustering Visualization")
plt.show()