#Import all needed libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import torch
import torch.nn as nn
import dgl.function as fn
from dgl import DGLGraph
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import torch.nn.functional as F
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.keras.initializers import RandomNormal
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
print ('done')

#Read the dataset
df_malware = pd.read_csv(r'C:\Users\saifp\Desktop\Malware Detection\10616685_SAIFALLAH PARKAR_Dataset.csv')
df_malware.head()

#Checking the columns of the dataset
df_malware.info()

#Checking the null values
df_malware.isnull().sum()

#Statistical description of dataset
df_malware.describe()

#Drop the unwanted columns from the dataset
df_malware_f = df_malware.drop(['Name','Machine','TimeDateStamp'], axis=1)

# Create the count plot with 'Malware' as x-axis
sns.countplot(data=df_malware_f, x='Malware', palette='Set2')
# Set x-axis tick labels
plt.xticks(ticks=[0, 1], labels=['Benign', 'Malware'])
# Set labels for the axes
plt.xlabel('Malware')
plt.ylabel('Count')
# Show the plot
plt.show()

#top 5 rows of dataset
df_malware_f.head()

# Features to visualize
features = ['MajorSubsystemVersion', 'MajorLinkerVersion', 'SizeOfCode', 'SizeOfImage', 'SizeOfHeaders',
            'SizeOfInitializedData', 'SizeOfUninitializedData', 'SizeOfStackReserve',
            'SizeOfHeapReserve', 'NumberOfSymbols', 'SectionMaxChar']

# Split the features list into two subsets
features_subset1 = features[:5]  # First 5 columns
features_subset2 = features[5:]  # Next 6 columns
palette = {1: 'blue', 0: 'violet'}
# Create separate pairplots for each subset
sns.pairplot(data=df_malware_f, hue='Malware', vars=features_subset1, palette=palette)
plt.suptitle('Pairplot of First 5 Features by Malware Type', size=16)
plt.show()

#Pairplot
sns.pairplot(data=df_malware_f, hue='Malware', vars=features_subset2, palette=palette)
plt.suptitle('Pairplot of Next 6 Features by Malware Type', size=16)
plt.show()

#Violinplot
sns.violinplot(data=df_malware_f, x='Malware', y='MajorLinkerVersion', palette=['skyblue', 'salmon'])
plt.xlabel('Malware Type', fontsize=14)
plt.ylabel('Major Linker Version', fontsize=14)
plt.title('Major Linker Version Distribution by Malware Type', fontsize=16)
plt.show()

# Split the columns into three sections
num_cols = len(df_malware_f.columns)
section_sizes = [25, 25, 26]

# Create separate correlation matrices for each section
corr_sections = [
    df_malware_f.iloc[:, sum(section_sizes[:i]):sum(section_sizes[:i + 1])].corr()
    for i in range(len(section_sizes))
]

# Create separate correlation heatmaps for each section
for i, corr_section in enumerate(corr_sections):
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_section, annot=True, cmap='seismic', fmt=".2f", linewidths=0.5)
    plt.title(f'Correlation Heatmap - Part {i + 1}', fontsize=16)
    plt.show()

#Separate the dependent and independent variables
X = df_malware_f.drop(['Malware'] , axis = 1)
y = df_malware_f['Malware']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Use SMOTE to balance the dataset
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# GCN model architecture
input_shape = X_train.shape[1]

# Define graph convolutional layer



class GraphConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation='relu'):
        super(GraphConvolutionLayer, self).__init__()
        self.output_dim = output_dim
        self.activation_function = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel", shape=(input_shape[-1], self.output_dim))
        self.bias = self.add_weight(name="bias", shape=(self.output_dim,))

    def call(self, inputs):
        features = tf.matmul(inputs, self.kernel) + self.bias
        return self.activation_function(features)


# Define GCN model
inputs = Input(shape=(input_shape,))
dropout = Dropout(0.5)(inputs)
gc1 = GraphConvolutionLayer(64, activation='relu')(dropout)
gc2 = GraphConvolutionLayer(32, activation='relu')(gc1)
output_layer = GraphConvolutionLayer(2, activation='softmax')(gc2)

model = Model(inputs=inputs, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy])

# # Train the GCN model
epochs = 10
batch_size = 32
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

# # Evaluate the GCN model
y_pred_gcn = np.argmax(model.predict(X_test), axis=1)
accuracy_gcn = accuracy_score(y_test, y_pred_gcn)
precision_gcn = precision_score(y_test, y_pred_gcn)
recall_gcn = recall_score(y_test, y_pred_gcn)
f1_gcn = f1_score(y_test, y_pred_gcn)

print("GCN Model Accuracy: {:.4f}".format(accuracy_gcn))
print("GCN Model Precision: {:.4f}".format(precision_gcn))
print("GCN Model Recall: {:.4f}".format(recall_gcn))
print("GCN Model F1-score: {:.4f}".format(f1_gcn))

#GCN using hyper tuning and cross validation
# K-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define hyperparameters to tune
hidden_units = [32,64]
dropout_rates = [0.3, 0.5]

# Perform hyperparameter tuning within K-fold cross-validation loop
for train_idx, test_idx in kfold.split(X, y):
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[test_idx]

    #X_train_fold, X_val_fold = X[train_idx], X[test_idx]
    y_train_fold, y_val_fold = y[train_idx], y[test_idx]

    best_accuracy = 0
    best_params = {}

    for units in hidden_units:
        for rate in dropout_rates:
            # Create the model
            inputs = Input(shape=(X.shape[1],))
            dropout = Dropout(rate)(inputs)
            dense = Dense(units, activation='relu')(dropout)
            output = Dense(2, activation='softmax')(dense)
            model = Model(inputs=inputs, outputs=output)
            model.compile(optimizer=Adam(learning_rate=0.001), loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy])

            # Train the model
            model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0, validation_split=0.1)

            # Evaluate the model
            accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)[1]

            # Check if this set of hyperparameters is the best so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params['units'] = units
                best_params['dropout_rate'] = rate

    # Train the best model on the entire training set
    inputs = Input(shape=(X.shape[1],))
    dropout = Dropout(best_params['dropout_rate'])(inputs)
    dense = Dense(best_params['units'], activation='relu')(dropout)
    output = Dense(2, activation='softmax')(dense)
    best_model = Model(inputs=inputs, outputs=output)
    best_model.compile(optimizer=Adam(learning_rate=0.001), loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy])
    best_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.1)

    # Evaluate the best model on the test set
    print("Best Hyperparameters: ", best_params)


# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate precision, recall, and F1-score
accuracy_gcn_ht = accuracy_score(y_test, y_pred_classes)
precision_gcn_ht = precision_score(y_test, y_pred_classes)
recall_gcn_ht = recall_score(y_test, y_pred_classes)
f1_gcn_ht = f1_score(y_test, y_pred_classes)

print("Accuracy : {:.4f}".format(accuracy_gcn_ht))
print("Precision: {:.4f}".format(precision_gcn_ht))
print("Recall: {:.4f}".format(recall_gcn_ht))
print("F1-score: {:.4f}".format(f1_gcn_ht))

# GraphSAGE model architecture

from tensorflow import keras
from keras.layers import Input, Dropout, Dense, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.metrics import sparse_categorical_accuracy

input_shape = X_train.shape[1]
output_classes = 2

# GraphSAGE layers
layer_sizes = [64, 32]
dropout_rate = 0.5

inputs = Input(shape=(input_shape,))
x = inputs

for l in layer_sizes:
    x_skip = x  # Store the input for skip connection
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(l, activation="relu")(x)
    x = keras.layers.concatenate([x, x_skip], axis=1)  # Concatenate with the initial input (x_skip)

outputs = Dense(output_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=0.001), loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy])

# Train the GraphSAGE model
epochs = 10
batch_size = 32
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.1)

# Evaluate the GraphSAGE model
y_pred_gs = np.argmax(model.predict(X_test), axis=1)
accuracy_gs = accuracy_score(y_test, y_pred_gs)
precision_gs = precision_score(y_test, y_pred_gs)
recall_gs = recall_score(y_test, y_pred_gs)
f1_gs = f1_score(y_test, y_pred_gs)

print("GraphSAGE model:")
print("Accuracy: {:.4f}".format(accuracy_gs))
print("Precision: {:.4f}".format(precision_gs))
print("Recall: {:.4f}".format(recall_gs))
print("F1-score: {:.4f}".format(f1_gs))

# Define the Graph Isomorphism Network (GIN) model
class GraphIsomorphismNetwork(Model):
    def __init__(self, hidden_units, output_classes, dropout_rate=0.5):
        super(GraphIsomorphismNetwork, self).__init__()
        self.hidden_units = hidden_units
        self.output_classes = output_classes
        self.dropout_rate = dropout_rate

        # Create dense layers
        self.dense_layers = [Dense(units, activation='relu') for units in hidden_units]
        self.batch_norm_layers = [BatchNormalization() for _ in hidden_units]
        self.dropout = Dropout(dropout_rate)

        # Output layer
        self.output_layer = Dense(output_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = inputs
        for dense, batch_norm in zip(self.dense_layers, self.batch_norm_layers):
            x = dense(x)
            x = batch_norm(x, training=training)
            x = tf.nn.relu(x)
            x = self.dropout(x, training=training)

        x = self.output_layer(x)
        return x

# Define model parameters
hidden_units = [64, 32]
output_classes = 2

# Create an instance of the GIN model
input_shape = X_train.shape[1]  # Number of features
model_gin = GraphIsomorphismNetwork(hidden_units, output_classes)

# Compile the model
model_gin.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train the model
epochs = 10
batch_size = 32
model_gin.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.1)

# Evaluate the Graph Isomorphism Network
y_pred_gin = np.argmax(model_gin.predict(X_test), axis=1)
accuracy_gin = accuracy_score(y_test, y_pred_gin)
precision_gin = precision_score(y_test, y_pred_gin)
recall_gin = recall_score(y_test, y_pred_gin)
f1_gin = f1_score(y_test, y_pred_gin)

print("Graph Isomorphism Network model:")
print("Accuracy: {:.4f}".format(accuracy_gin))
print("Precision: {:.4f}".format(precision_gin))
print("Recall: {:.4f}".format(recall_gin))
print("F1-score: {:.4f}".format(f1_gin))

#Graph Attention Network
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, features):
        z = self.fc(features)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')

class GATNet(nn.Module):
    def __init__(self, num_features, num_classes, num_layers=1, hidden_dim=64):
        super(GATNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(num_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GATLayer(hidden_dim, hidden_dim))
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, g, features):
        for layer in self.layers:
            features = layer(g, features)
        output = self.fc(features)
        return output

# num_classes is the number of output classes
num_features = X_train.shape[1]
num_classes = 2
num_layers = 2

# Create a DGLGraph from your input data
g = DGLGraph()
g.add_nodes(X_test.shape[0])
g.add_edges([i for i in range(X_train.shape[0])], [i for i in range(X_train.shape[0])])

# Create the GAT model
gat_model = GATNet(num_features=num_features, num_classes=num_classes, num_layers=num_layers)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    gat_model.train()
    logits = gat_model(g, torch.FloatTensor(X_train))
    loss = criterion(logits, torch.LongTensor(y_train))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
g_subset = g.subgraph(range(X_test.shape[0]))
# Evaluation
gat_model.eval()
with torch.no_grad():
    # Assuming g_subset is our subgraph with the correct number of nodes
    logits = gat_model(g_subset, torch.FloatTensor(X_test))
    predicted_labels = torch.argmax(logits, dim=1)
    accuracy_gat = torch.sum(predicted_labels == torch.LongTensor(y_test)).item() / len(y_test)
    print(f'Accuracy: {accuracy_gat:.4f}')

# Calculate precision
precision_gat = precision_score(y_test, predicted_labels)

# Calculate recall
recall_gat = recall_score(y_test, predicted_labels)

# Calculate F1 score
f1_gat = f1_score(y_test, predicted_labels)

print(f'Precision: {precision_gat:.4f}')
print(f'Recall: {recall_gat:.4f}')
print(f'F1 Score: {f1_gat:.4f}')

# Define CNN model
cnn_model = tf.keras.models.Sequential()
cnn_model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
cnn_model.add(GlobalMaxPooling1D())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(output_classes, activation='softmax'))

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Reshape data for CNN
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Train CNN model
cnn_model.fit(X_train_cnn, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.1)

# Evaluate CNN model
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test)
print("CNN Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(cnn_loss, cnn_accuracy))

# Evaluate the GraphSAGE model
y_pred_cnn = np.argmax(cnn_model.predict(X_test), axis=1)
accuracy_cnn = accuracy_score(y_test, y_pred_cnn)
precision_cnn = precision_score(y_test, y_pred_cnn)
recall_cnn = recall_score(y_test, y_pred_cnn)
f1_cnn = f1_score(y_test, y_pred_cnn)

print("CNN :")
print("Accuracy: {:.4f}".format(accuracy_cnn))
print("Precision: {:.4f}".format(precision_cnn))
print("Recall: {:.4f}".format(recall_cnn))
print("F1-score: {:.4f}".format(f1_cnn))

# Define RNN model
rnn_model = tf.keras.models.Sequential()
rnn_model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], 1)))
rnn_model.add(Dense(64, activation='relu'))
rnn_model.add(Dense(output_classes, activation='softmax'))

rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Reshape data for RNN (assuming X_train and X_test are already prepared)
X_train_rnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Train RNN model
rnn_model.fit(X_train_rnn, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.1)


# Evaluate the RNN model
y_pred_rnn = np.argmax(rnn_model.predict(X_test), axis=1)
accuracy_rnn = accuracy_score(y_test, y_pred_rnn)
precision_rnn = precision_score(y_test, y_pred_rnn)
recall_rnn = recall_score(y_test, y_pred_rnn)
f1_rnn = f1_score(y_test, y_pred_rnn)

print("RNN:")
print("Accuracy: {:.4f}".format(accuracy_rnn))
print("Precision: {:.4f}".format(precision_rnn))
print("Recall: {:.4f}".format(recall_rnn))
print("F1-score: {:.4f}".format(f1_rnn))

# Define the results
model_names = ["CNN", "RNN", "GraphSAGE", "GCN", "GCN using hypertuning", "Graph Isomorphism Network", "Graph Attention Network"]
accuracies = [cnn_accuracy, accuracy_rnn, accuracy_gs, accuracy_gcn, accuracy_gcn_ht, accuracy_gin, accuracy_gat]  # Replace with your accuracy values

# Create a DataFrame
data = {'Model': model_names, 'Accuracy': accuracies}
df = pd.DataFrame(data)

# Print the DataFrame as a table
print(df)

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color='skyblue')
plt.ylabel('Accuracy')
plt.title('Model Accuracies')
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

