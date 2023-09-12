import numpy as np
from src.filters import *
from src.wiener_filter import *
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def decode_kfolds(X, Y, k=10, metric=3, preset_h=None, vaf_scoring=True):
    kf = KFold(n_splits=k)
    
    h_list = []
    
    vaf_array = np.zeros((Y.shape[1], k))
    index = 0
    best_vaf = -10000000
    for train_index, test_index in kf.split(X):
        
        train_x, test_x = X[train_index, :], X[test_index, :]
        train_y, test_y = Y[train_index, :], Y[test_index, :]
        if preset_h is None:
            h = train_wiener_filter(train_x, train_y)
        else:
            h = preset_h
        predic_y = test_wiener_filter(test_x, h)
        for j in range(predic_y.shape[1]):
            if vaf_scoring:
                vaf_array[j, index] = vaf(test_y[:, j], predic_y[:, j])
            else:
                # use r^2 instead of VAF
                vaf_array[j, index] = r2_score(test_y[:, j], predic_y[:, j])
        
        if vaf_array[3, index] > best_vaf:
            best_vaf = vaf_array[3, index]
            best_h = h
            final_test_x = test_x
            final_test_y = test_y
        
        index = index + 1
    
    return best_h, np.average(vaf_array, 1), final_test_x, final_test_y


def decode_kfolds_single(X, Y, k=10):
    kf = KFold(n_splits=k)
    index = 0
    best_vaf = -1
    vaf_average = []
    for train_index, test_index in kf.split(X):
        train_x, test_x = X[train_index, :], X[test_index, :]
        train_y, test_y = Y[train_index], Y[test_index]
        
        h = train_wiener_filter(train_x, train_y)
        predic_y = test_wiener_filter(test_x, h)
        vaf_average.append(vaf(test_y, predic_y))
        
        if vaf_average[-1] > best_vaf:
            final_test_x = test_x
            final_test_y = test_y
            best_h = h
    
    print(np.array(vaf_average))
    
    return best_h, np.average(np.array(vaf_average)), final_test_x, final_test_y


# def apply_PCA(X, dims):
#    pca_output = PCA(n_components=dims, random_state=2020)
#    pca_output.fit(X)

#    X_pca_output = pca_output.transform(X)

#    return X_pca_output, pca_output

def classify_kfolds(X, Y, k=10):
    # Y should be 0s or 1s I believe
    kf = KFold(n_splits=k)
    
    accuracy_list = []
    best_accuracy = 0
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=10000, tol=1e-3))
    best_model = clf
    for train_index, test_index in kf.split(rates):
        
        train_x, test_x = rates[train_index, :], rates[test_index, :]
        train_y, test_y = stance_swing[train_index], stance_swing[test_index]
        
        clf.fit(train_x, train_y)
        y_pred = clf.predict(test_x)
        pred_accuracy = accuracy_score(test_y, y_pred)
        accuracy_list.append(pred_accuracy)
        if best_accuracy < pred_accuracy:
            best_accuracy = pred_accuracy
            best_model = clf
            final_test_x = test_x
            final_test_y = test_y
    
    return best_model, accuracy_list, final_test_x, final_test_y


def ridge_fit(b0, x_format, y_format, my_alpha=100.0, angle_number=1):
    # b0 = day0 decoder weights
    # x = dayk x values (usually ALIGNED) (FORMATTED)
    # y = dayk y values (FORMATTED)
    
    xb0 = test_wiener_filter(x_format, b0)
    
    initial_vaf = vaf(y_format[:, 1], xb0[:, 1])
    print(f'initial_scoring is : {initial_vaf}')
    
    y_star = y_format - xb0
    x_plus_bias = np.c_[np.ones((np.size(x_format, 0), 1)), x_format]
    
    clf = Ridge(alpha=my_alpha)
    clf.fit(x_plus_bias, y_star)
    
    b = clf.coef_.T
    
    wpost = b + b0
    
    ywpost = test_wiener_filter(x_format, wpost)
    new_vaf = vaf(y_format[:, angle_number], ywpost[:, angle_number])
    print(f'new_scoring is: {new_vaf}')
    
    return wpost, ywpost


def regression_fit(b0, x, y, angle=1):
    # b0 = day0 decoder weights
    # x = dayk x values (usually PCA)
    # y = dayk y values
    
    x_format, y_format = format_data(x, y)
    xb0 = test_wiener_filter(x_format, b0)
    
    initial_vaf = vaf(y_format[:, 1], xb0[:, 1])
    print(f'initial_scoring is : {initial_vaf}')
    
    y_star = y_format - xb0
    x_plus_bias = np.c_[np.ones((np.size(x_format, 0), 1)), x_format]
    
    clf = LinearRegression()
    clf.fit(x_plus_bias, y_star)
    
    b = clf.coef_.T
    
    wpost = b + b0
    ywpost = test_wiener_filter(x_format, wpost)
    new_vaf = vaf(y_format[:, angle], ywpost[:, angle])
    
    print(f'new_scoring is: {new_vaf}')
    
    return wpost


def pinv_fit(b0, x_format, y_format, angle=1):
    b0_no_offset = b0[1:, :]
    offset = b0[0, :]
    inv_b0 = np.linalg.pinv(b0_no_offset)
    # print(inv_b0.shape)
    # x_format, y_format = format_data(x, y)
    y_star = np.dot(y_format - offset, inv_b0)
    # x_plus_bias = np.c_[np.ones((np.size(x_format, 0), 1)), x_format]
    
    clf = LinearRegression()
    clf.fit(x_format, y_star)
    
    trans_x_format = clf.predict(x_format)
    pinv_predic = np.dot(trans_x_format, b0_no_offset) + offset
    
    return clf, pinv_predic


def pinv_predicter(clf, b0, x_format):
    b0_no_offset = b0[1:, :]
    offset = b0[0, :]
    trans_x_format = clf.predict(x_format)
    return np.dot(trans_x_format, b0_no_offset) + offset
