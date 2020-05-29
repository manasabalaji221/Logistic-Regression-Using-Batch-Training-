
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
 return 1/(1+np.exp(-x))


def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    # print(linear_model)
    # print(linear_model.shape)

    y_predicted = sigmoid(linear_model)
    y_predicted_val=[]
    # print(y_predicted)
    for i in y_predicted:
        if i>0.5:
            y_predicted_val.append(1)
        else:
            y_predicted_val.append(0)
    return y_predicted_val


def logistic_regression(features, y, n_iters, lr):
    n_samples, n_features = features.shape
    weights = np.expand_dims(np.zeros(shape=(n_features)),axis=1)
    # print(weights.shape)
    bias = 0
    grads = []
    iters=0
    costs=[]
    e_list=[]
    e=1000
    for _ in range(n_iters):
        iters+=1
        linear_model = np.dot(features, weights) + bias

        #### Using sigmoid function for activation function
        y_predicted = sigmoid(linear_model)
        #### Finding the gradient , cost function (Cross Entropy)
        cross_entropy = np.dot(features.T, (y_predicted - y))
        costs.append(cross_entropy[0])
        gradient_dw = (1 / n_samples) * np.dot(features.T, (y_predicted - y))
        grads.append(gradient_dw[0])
        # print(gradient_dw.shape)
        # print(weights.shape)
        # print(cross_entropy[0])
        db = (1 / n_samples) * np.sum(y_predicted - y)
        # print('bias',db)
        # weights -= lr * dw
        bias -= lr * db
        # print(gradient)
        weights -= lr * gradient_dw
        #### L1 norm of gradient
        e_l1 = cross_entropy[0] + lr * (weights[0]+weights[1])
        e_list.append(e_l1)
        # print(e_l1)
        # print(weights)
        #### breaking condition
        if(e_l1 < 0.001 or e-e_l1==0):
            break
        e = e_l1
        # print(e_l1)
    # print('weights',weights)
    return weights, bias, grads, e_list, costs, iters


def accuracy(test, pred):
    p=list(pred)
    # print(p)
    # print(len(pred), len(test))
    c=0
    for i in range(0,len(p)):
        if(p[i] == test[i]):
            c+=1
    acc = c / len(p)
    return acc


def logistic_regression_online(features, y, lr):
    n_samples, n_features = features.shape
    weights = np.expand_dims(np.zeros(shape=(n_features)),axis=1)
    # print(weights.shape)
    bias = 0
    grads = []
    iters=0
    costs=[]
    e_list=[]
    e=1000
    linear_model = np.dot(features, weights) + bias
    #### Using sigmoid function for activation function
    y_predicted = sigmoid(linear_model)
    #### Finding the gradient , cost function (Cross Entropy)
    cross_entropy = np.dot(features.T, (y_predicted - y))
    costs.append(cross_entropy[0])
    gradient_dw = (1 / n_samples) * np.dot(features.T, (y_predicted - y))
    grads.append(gradient_dw[0])
    # print(gradient_dw)
    # print(cross_entropy[0])
    db = (1 / n_samples) * np.sum(y_predicted - y)
    # print('bias',db)
    # weights -= lr * dw
    bias -= lr * db
    # print(gradient)
    weights -= lr * gradient_dw
    #### L1 norm of gradient
    e_l1 = cross_entropy[0] + lr * (weights[0]+weights[1])
    e_list.append(e_l1)
    # print(e_l1)
    # print(weights)

    e = e_l1
    # print(e_l1)
    # print('weights',weights)
    return weights, bias, grads, e_l1, costs, iters


def online_train(Xn, Yn, Theta, lr):
    print(Yn.shape)
    Loss = Yn - np.dot(Xn, Theta) + (np.dot(Theta.T, Theta) * lr)
    # print(Loss)
    # print(Loss.shape)
    Loss = Loss * (-1)
    dJ = (np.dot(Xn.T, Loss) * 2) / len(Yn)
    Theta = Theta - (lr * dJ)
    # print(Theta)
    return Theta, dJ

if __name__ == '__main__':

    ############# Generating training data
    mean1 = np.array([1, 0])
    cov1 = np.array([[1, 0.75], [0.75, 1]])
    x1, y1 = np.random.multivariate_normal(mean1, cov1, 500).T
    mean2 = np.array([0, 1.5])
    cov2 = np.array([[1, 0.75], [0.75, 1]])
    x2, y2 = np.random.multivariate_normal(mean2, cov2, 500).T
    XY1 = list(zip(x1, y1))
    XY2 = list(zip(x2, y2))
    XY=XY1+XY2
    xy_train = np.array(XY)
    # print(xy)
    S1 = np.zeros(shape=(1000, 3))
    x1=np.zeros(shape=(1000,1))

    for i in range(0, 500):
        S1[i] = [xy_train[i][0],xy_train[i][1],0]
        x1[i]=0
    for j in range(500, 1000):
        # print(j)
        S1[j] = [xy_train[j][0],xy_train[j][1],1]
        x1[j]=1
    # print(S2)
    # print(S1.shape)
    count1, bins1, ignored1 = plt.hist(xy_train, 1000, density=True)
    plt.title('Multivariate Normal random data values\nTraining data:')
    plt.show()


    ############# Generating testing data
    mean3 = np.array([1, 0])
    cov3 = np.array([[1, 0.75], [0.75, 1]])
    x3, y3 = np.random.multivariate_normal(mean3, cov3, 250).T
    mean4 = np.array([0, 1.5])
    cov4 = np.array([[1, 0.75], [0.75, 1]])
    x4, y4 = np.random.multivariate_normal(mean4, cov4, 250).T
    XY3 = list(zip(x3, y3))
    XY4 = list(zip(x4, y4))
    XYt = XY3 + XY4
    xy_test = np.array(XYt)
    # print(xy)
    S2 = np.zeros(shape=(500, 3))
    c1=c2=0
    x_test=np.zeros(shape=(500,1))
    for i in range(0, 250):
        S2[i] = [xy_test[i][0], xy_test[i][1], 0]
        x_test[i] = 0
        c1+=1
    for j in range(250, 500):
        # print(j)
        S2[j] = [xy_test[j][0], xy_test[j][1], 1]
        x_test[j]=1
    # print(S2)
    # print(S2.shape)
    # print(x)
    count1, bins1, ignored1 = plt.hist(xy_test, 500, density=True)
    plt.title('Multivariate Normal random data values\nTesting data:')
    plt.show()

    # ######Apply Logistic regression(example)
    # weights, bias, grads, iters = logistic_regression(xy_train, x1, 100000, lr=0.1)
    # # print(weights, bias)
    # predicted_val = predict(xy_train, weights, bias)
    # # print(predicted_val)

    #### Question 1 (Batch training)
    ######## 1a learning rate=1
    print('Batch Training using lr = 1')
    weight_bat1, bias_bat1, grad_bat1, e_l1norm_bat1, cross_entropy_bat1, iters_bat1 = logistic_regression(xy_train, x1, 100000, lr=1)
    # print(weights, bias)
    predicted_val_bat = predict(xy_test, weight_bat1, bias_bat1)
    # print(predicted_val_bat)
    acc = 100 * accuracy(x_test, predicted_val_bat)
    plt.plot(xy_test)
    plt.plot(predicted_val_bat)
    plt.title('Batch Training using lr = 1\n(a)Testing data and the Trained Decision Boundary:')
    plt.show()
    # print(grads_bat1)
    # print(iters_bat1)
    plt.plot(grad_bat1)
    plt.title('Batch Training using lr = 1\n(b)Changes of training loss (cross entropy) w.r.t. iteration:')
    plt.show()
    # print(weight_bat1)
    # print(weight_bat1.shape)
    plt.plot(e_l1norm_bat1, color = 'red')
    plt.title('Batch Training using lr = 1\n(c)Changes of norm of gradient w.r.t. iteration:')
    plt.show()
    print('Accuracy is:', acc, '%')
    print('No of iterations:', iters_bat1)

    ######## 1b learning rate = 0.1
    print('Batch Training using lr = 0.1')
    weight_bat2, bias_bat2, grad_bat2, e_l1norm_bat2, cross_entropy_bat2, iters_bat2 = logistic_regression(xy_train, x1, 100000, lr=0.1)
    # print(weights, bias)
    predicted_val_bat = predict(xy_test, weight_bat2, bias_bat2)
    # print(predicted_val_bat)
    acc = 100 * accuracy(x_test, predicted_val_bat)
    plt.plot(xy_test)
    plt.plot(predicted_val_bat)
    plt.title('Batch Training using lr = 0.1\n(a)Testing data and the Trained Decision Boundary:')
    plt.show()
    # print(grads_bat1)
    # print(iters_bat1)
    plt.plot(grad_bat2)
    plt.title(
        'Batch Training using lr = 0.1\n(b)Changes of training loss (cross entropy) w.r.t. iteration:')
    plt.show()
    # print(weight_bat1)
    # print(weight_bat1.shape)
    plt.plot(e_l1norm_bat2, color='red')
    plt.title('Batch Training using lr = 1\n(c)Changes of norm of gradient w.r.t. iteration:')
    plt.show()
    print('Accuracy is:', acc, '%')
    print('No of iterations:', iters_bat2)

    ######## 1c learning rate = 0.01
    print('Batch Training using lr = 0.01')
    weight_bat3, bias_bat3, grad_bat3, e_l1norm_bat3, cross_entropy_bat3, iters_bat3 = logistic_regression(xy_train, x1, 100000, lr=0.01)
    # print(weights, bias)
    predicted_val_bat = predict(xy_test, weight_bat3, bias_bat3)
    # print(predicted_val_bat)
    acc = 100 * accuracy(x_test, predicted_val_bat)
    plt.plot(xy_test)
    plt.plot(predicted_val_bat)
    plt.title(
        'Batch Training using lr = 0.01\n(a)Testing data and the Trained Decision Boundary:')
    plt.show()
    # print(grads_bat1)
    # print(iters_bat1)
    plt.plot(grad_bat3)
    plt.title(
        'Batch Training using lr = 0.01\n(b)Changes of training loss (cross entropy) w.r.t. iteration:')
    plt.show()
    # print(weight_bat1)
    # print(weight_bat1.shape)
    plt.plot(e_l1norm_bat3, color='red')
    plt.title('Batch Training using lr = 1\n(c)Changes of norm of gradient w.r.t. iteration:')
    plt.show()
    print('Accuracy is:', acc, '%')
    print('No of iterations:', iters_bat3)

    ######## 1d learning rate = 0.001
    print('Batch Training using lr = 0.001')
    weight_bat4, bias_bat4, grad_bat4, e_l1norm_bat4, cross_entropy_bat4, iters_bat4 = logistic_regression(xy_train, x1, 100000, lr=0.001)
    # print(weights, bias)
    predicted_val_bat = predict(xy_test, weight_bat4, bias_bat4)
    # print(predicted_val_bat)
    acc = 100 * accuracy(x_test, predicted_val_bat)
    plt.plot(xy_test)
    plt.plot(predicted_val_bat)
    plt.title(
        'Batch Training using lr = 0.001\n(a)Testing data and the Trained Decision Boundary:')
    plt.show()
    # print(grads_bat1)
    # print(iters_bat1)
    plt.plot(grad_bat4)
    plt.title(
        'Batch Training using lr = 0.001\n(b)Changes of training loss (cross entropy) w.r.t. iteration:')
    plt.show()
    # print(weight_bat1)
    # print(weight_bat1.shape)
    plt.plot(e_l1norm_bat4, color='red')
    plt.title('Batch Training using lr = 0.001\n(c)Changes of norm of gradient w.r.t. iteration:')
    plt.show()
    print('Accuracy is:', acc, '%')
    print('No of iterations:', iters_bat4)





