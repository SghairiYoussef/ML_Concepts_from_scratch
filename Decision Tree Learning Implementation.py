import numpy as np

def calc_entropy(D: list[dict], target_key: str) -> float:
    target_values = np.array([item[target_key] for item in D])
    unique_classes, counts = np.unique(target_values, return_counts=True)
    probabilities = counts / len(D)
    return -np.sum(probabilities * np.log2(probabilities))

def calc_information_gain(D: list[dict], feature_key: str, target_key: str) -> float:
    total_entropy = calc_entropy(D, target_key)
    feature_values = np.array([item[feature_key] for item in D])
    unique_feature_values = np.unique(feature_values)
    
    weighted_entropy_sum = 0.0
    for value in unique_feature_values:
        subset = [item for item in D if item[feature_key] == value]
        subset_entropy = calc_entropy(subset, target_key)
        weight = len(subset) / len(D)
        weighted_entropy_sum += weight * subset_entropy
    
    information_gain = total_entropy - weighted_entropy_sum
    return information_gain

def learn_decision_tree(data: list[dict], attributes: list[str], target_attr: str) -> dict:
    unique_classes = np.unique([item[target_attr] for item in data])
    if len(unique_classes) == 1:
        return unique_classes[0]
    
    if len(attributes) == 0:
        class_labels, counts = np.unique([item[target_attr] for item in data], return_counts=True)
        majority_class = class_labels[np.argmax(counts)]
        return majority_class
    
    info_gains = {attribute: calc_information_gain(data, attribute, target_attr) for attribute in attributes}
    best_attribute = max(info_gains, key=info_gains.get)

    tree = {best_attribute: {}}
    
    best_attr_values = np.unique([item[best_attribute] for item in data])
    
    for value in best_attr_values:
        subset = [item for item in data if item[best_attribute] == value]
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]
        tree[best_attribute][value] = learn_decision_tree(subset, remaining_attributes, target_attr)
    
    return tree