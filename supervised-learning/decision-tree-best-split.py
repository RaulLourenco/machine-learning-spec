import numpy as np

X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])

def compute_entropy(y):
    entropy = 0.
    
    if len(y) != 0:
        p1 = len(y[y == 1])/len(y)
        if p1 == 0 or p1 == 1:
            entropy = 0
        else:
            entropy = -p1*np.log2(p1)-(1 - p1)*np.log2(1 - p1)    
    
    return entropy

def split_dataset(X, node_indices, feature):
    left_indices = []
    right_indices = []
    
    for i in node_indices:
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
        
    return left_indices, right_indices

def compute_information_gain(X, y, node_indices, feature):
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    information_gain = 0
    
    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
    
    w_left = len(X_left)/len(X_node)
    w_right = len(X_right)/len(X_node)
    
    weighted_entropy = (w_left * left_entropy) + (w_right * right_entropy)
    information_gain = node_entropy - weighted_entropy
    
    return information_gain

def get_best_split(X, y, node_indices):   
    num_features = X.shape[1]
    best_feature = -1
    max_gain = 0

    for feature in range(num_features):
        gain = compute_information_gain(X, y, node_indices, feature)
        if gain > max_gain:
            max_gain = gain
            best_feature = feature  
   
    return best_feature