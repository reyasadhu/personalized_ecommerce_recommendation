def precision_at_k(relevant, predicted, k=10):
    '''
        The precision_at_k function computes the precision at k value.
    '''
    if isinstance(predicted, set):
        predicted = list(predicted)
    predicted = predicted[:k]
    hits = len(set(predicted) & relevant)
    return hits / k

def recall_at_k(relevant, predicted, k=10):
    '''
        The recall_at_k function computes the recall at k value.
    '''
    if isinstance(predicted, set):
        predicted = list(predicted)
    predicted = predicted[:k]
    hits = len(set(predicted) & relevant)
    if len(relevant)==0:
        return 0
    return hits/len(relevant)