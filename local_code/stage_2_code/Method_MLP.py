'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 500
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    # it defines how many epochs to wait after a worse loss for a better loss
    patience = 5


    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 128)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(128, 64)
        self.activation_func_2 = nn.ReLU()
        self.fc_layer_3 = nn.Linear(64, 10)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_3 = nn.Softmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h1 = self.activation_func_1(self.fc_layer_1(x))
        h2 = self.activation_func_2(self.fc_layer_2(h1))
        # outout layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.activation_func_3(self.fc_layer_3(h2))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X_train, y_train, X_test, y_test):
        best_test_loss = 50
        patience_counter = 0
        epochs = []
        train_loss_hist, test_loss_hist = [], []
        train_acc_hist, test_acc_hist = [], []

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X_train)))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y_train))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            # check test loss for early stopping
            _, test_loss = self.test(X_test, y_test)

            """ # print out and add to plot stats every n epochs or if stopping early
            if epoch%20 == 0:
                print(f"Epoch: {epoch}   Train Loss: {train_loss.item():.4f}")

                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                train_eval = accuracy_evaluator.evaluate()
                print(f"\tAccuracy        : {train_eval['accuracy']:.4f}")
                print(f"\tMacro   P/R/F1  : {train_eval['precision_macro']:.4f} / {train_eval['recall_macro']:.4f} / {train_eval['f1_macro']:.4f}")
                print(f"\tMicro   P/R/F1  : {train_eval['precision_micro']:.4f} / {train_eval['recall_micro']:.4f} / {train_eval['f1_micro']:.4f}")
                print(f"\tWeighted P/R/F1 : {train_eval['precision_weighted']:.4f} / {train_eval['recall_weighted']:.4f} / {train_eval['f1_weighted']:.4f}")
                
                print(f"Test Loss: {test_loss.item():.4f}")
                test_eval, test_loss = self.test(X_test, y_test)
                print(f"\tAccuracy        : {test_eval['accuracy']:.4f}")
                print(f"\tMacro   P/R/F1  : {test_eval['precision_macro']:.4f} / {test_eval['recall_macro']:.4f} / {test_eval['f1_macro']:.4f}")
                print(f"\tMicro   P/R/F1  : {test_eval['precision_micro']:.4f} / {test_eval['recall_micro']:.4f} / {test_eval['f1_micro']:.4f}")
                print(f"\tWeighted P/R/F1 : {test_eval['precision_weighted']:.4f} / {test_eval['recall_weighted']:.4f} / {test_eval['f1_weighted']:.4f}")

                # add stats to plot
                epochs.append(epoch)
                train_loss_hist.append(train_loss.item())
                test_loss_hist.append(test_loss.item())
                train_acc_hist.append(train_eval['accuracy'])
                test_acc_hist.append(test_eval['accuracy'])   

                # update current test_loss or exit training if test loss doesn't improve
                if best_test_loss == -1 or best_test_loss > test_loss.item():
                    best_test_loss = test_loss.item()
                elif best_test_loss < test_loss.item():
                    patience_counter += 1
                    if patience_counter > self.patience:
                        print("Old test loss: ", best_test_loss)
                        print("--stopping training...")
                        break """           

            # For checking early stopping every epoch - currently stops at 25 epochs, likely need to change optimizer to work better than checking every n epochs
            # print out and add to plot stats every n epochs or if stopping early
            if epoch%10 == 0 or best_test_loss < test_loss.item():
                print(f"Epoch: {epoch}   Train Loss: {train_loss.item():.4f}")

                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                train_eval = accuracy_evaluator.evaluate()
                print(f"\tAccuracy        : {train_eval['accuracy']:.4f}")
                print(f"\tMacro   P/R/F1  : {train_eval['precision_macro']:.4f} / {train_eval['recall_macro']:.4f} / {train_eval['f1_macro']:.4f}")
                print(f"\tMicro   P/R/F1  : {train_eval['precision_micro']:.4f} / {train_eval['recall_micro']:.4f} / {train_eval['f1_micro']:.4f}")
                print(f"\tWeighted P/R/F1 : {train_eval['precision_weighted']:.4f} / {train_eval['recall_weighted']:.4f} / {train_eval['f1_weighted']:.4f}")
                
                print(f"Test Loss: {test_loss.item():.4f}")
                test_eval, test_loss = self.test(X_test, y_test)
                print(f"\tAccuracy        : {test_eval['accuracy']:.4f}")
                print(f"\tMacro   P/R/F1  : {test_eval['precision_macro']:.4f} / {test_eval['recall_macro']:.4f} / {test_eval['f1_macro']:.4f}")
                print(f"\tMicro   P/R/F1  : {test_eval['precision_micro']:.4f} / {test_eval['recall_micro']:.4f} / {test_eval['f1_micro']:.4f}")
                print(f"\tWeighted P/R/F1 : {test_eval['precision_weighted']:.4f} / {test_eval['recall_weighted']:.4f} / {test_eval['f1_weighted']:.4f}")

                # add stats to plot
                epochs.append(epoch)
                train_loss_hist.append(train_loss.item())
                test_loss_hist.append(test_loss.item())
                train_acc_hist.append(train_eval['accuracy'])
                test_acc_hist.append(test_eval['accuracy'])   
            
            # update current test_loss or exit training if test loss doesn't improve
            # update current test_loss or exit training if test loss doesn't improve
            if best_test_loss == -1 or best_test_loss > test_loss.item():
                patience_counter = 0
                best_test_loss = test_loss.item()
            elif best_test_loss < test_loss.item():
                patience_counter += 1
                if patience_counter > self.patience:
                    print("Old test loss: ", best_test_loss)
                    print("--stopping training...")
                    break

        return {
            'epochs': epochs,
            'train_loss': train_loss_hist, 'test_loss': test_loss_hist,
            'train_acc': train_acc_hist, 'test_acc': test_acc_hist
        }     
    
    def test(self, X, y):
        loss_function = nn.CrossEntropyLoss()
        # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert y to torch.tensor as well
        y_true = torch.LongTensor(np.array(y))

        accuracy_evaluator = Evaluate_Accuracy('testing evaluator', '')
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
        test_eval = accuracy_evaluator.evaluate()

        test_loss = loss_function(y_pred, y_true)

        return test_eval, test_loss
    
    def run(self):
        print('method running...')
        print('--start training...')
        history = self.train(self.data['X_train'], self.data['y_train'], self.data['X_test'], self.data['y_test'])

        print('--final testing...')
        test_eval, _ = self.test(self.data['X_test'], self.data['y_test'])
        return history, test_eval
            