import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def ROC(result, map, display=False):
    """
    Calculate the AUC value of the detection result and optionally display the ROC curve.
    
    Args:
        result (np.ndarray): The detection map (e.g., anomaly scores).
        map (np.ndarray): The ground truth map (binary, 0 for background, 1 for anomaly).
        display (bool): If True, the ROC curve will be plotted.
        
    Returns:
        float: The AUC value.
    """
    # Reshape the input to 1D arrays
    result_flat = result.flatten()
    map_flat = map.flatten()
    
    # Sort the detection results in descending order and get the corresponding indices
    sorted_indices = np.argsort(result_flat)[::-1]
    
    # Initialize arrays for false positive rate (p_f) and true positive rate (p_d)
    p_f = np.zeros(result_flat.shape[0])
    p_d = np.zeros(result_flat.shape[0])
    
    # Calculate the number of anomalies and total number of pixels
    n_anomaly = np.sum(map_flat)
    n_total_pixels = map_flat.shape[0]
    n_background = n_total_pixels - n_anomaly
    
    n_false_positive = 0
    n_true_positive = 0
    
    for i in range(n_total_pixels):
        current_index = sorted_indices[i]
        
        # If the pixel is an anomaly
        if map_flat[current_index] == 1:
            n_true_positive += 1
        # If the pixel is background
        else:
            n_false_positive += 1
            
        # Update the rates
        if n_background > 0:
            p_f[i] = n_false_positive / n_background
        if n_anomaly > 0:
            p_d[i] = n_true_positive / n_anomaly
            
    # Calculate the AUC using the trapezoidal rule
    auc_value = np.trapz(p_d, p_f)
    
    if display:
        plt.figure()
        plt.plot(p_f, p_d)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.grid(True)
        plt.show()
        
    return auc_value