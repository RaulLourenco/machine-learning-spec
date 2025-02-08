# Calculate the mean squared error on a data set
def eval_mse(y, yhat):
    m = len(y)
    err = 0.0
    for i in range(m):
        err_i = (yhat[i] - y[i])**2
        err += err_i
    err /= 2*m    
    return(err)

# Calculate the categorization error
def eval_cat_err(y, yhat):
    m = len(y)
    incorrect = 0
    for i in range(m):
        if yhat[i] != y[i]:
            incorrect += 1
    cerr = (1/m) * incorrect    
    return(cerr)