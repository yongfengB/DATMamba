import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score 

def eval_roc_auc(labels, pred):
    """
    Compute the ROC-AUC score for binary outlier detection.

    Args:
        labels (numpy.ndarray): Binary ground truth labels of shape (N, ),
                                where 1 indicates an outlier and 0 indicates a normal instance.
        pred (numpy.ndarray): Predicted outlier scores of shape (N, ).
    
    Returns:
        float: The ROC-AUC score.
    """
    # Compute ROC-AUC using sklearn's roc_auc_score
    roc_auc = roc_auc_score(y_true=labels, y_score=pred)
    return roc_auc

def eval_recall_at_k(labels, pred, k):
    """
    Compute the recall at k for the top k instances with highest outlier scores.

    Args:
        labels (numpy.ndarray): Ground truth binary labels of shape (N, ).
                                1 represents outliers, 0 represents normal instances.
        pred (numpy.ndarray): Predicted outlier scores of shape (N, ).
        k (int): The number of top instances to consider.
    
    Returns:
        float: Recall score computed over the top k instances.
    """
    labels = np.array(labels)
    pred = np.array(pred)
    N = len(pred)
    
    # Get indices of top k highest outlier scores
    top_k_indices = pred.argpartition(N - k)[-k:]
    # Calculate recall: ratio of true outliers in top k to total number of outliers
    total_outliers = labels.sum()
    if total_outliers == 0:
        return 0.0
    recall_at_k = labels[top_k_indices].sum() / total_outliers
    return recall_at_k

def eval_precision_at_k(labels, pred, k):
    """
    Compute the precision at k for the top k instances with highest outlier scores.

    Args:
        labels (numpy.ndarray): Ground truth binary labels of shape (N, ),
                                where 1 represents outliers and 0 represents normal instances.
        pred (numpy.ndarray): Predicted outlier scores of shape (N, ).
        k (int): The number of top instances to consider.
    
    Returns:
        float: Precision score computed over the top k instances.
    """
    labels = np.array(labels)
    pred = np.array(pred)
    N = len(pred)
    
    # Get indices of top k highest outlier scores
    top_k_indices = pred.argpartition(N - k)[-k:]
    # Precision is the proportion of outliers among the top k instances
    precision_at_k = labels[top_k_indices].sum() / k
    return precision_at_k

def eval_average_precision(labels, pred):
    """
    Compute the average precision score for binary outlier detection.

    Args:
        labels (numpy.ndarray): Ground truth binary labels of shape (N, ),
                                where 1 indicates an outlier and 0 indicates a normal instance.
        pred (numpy.ndarray): Predicted outlier scores of shape (N, ).
    
    Returns:
        float: The average precision (AP) score.
    """
    # Compute average precision using sklearn's average_precision_score
    ap = average_precision_score(y_true=labels, y_score=pred)
    return ap
