# import numpy as np
from flask import Flask
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

app = Flask(__name__)

# Part-2
# build model and create it


def create_ann_model(learning_rate):
    # Part-1
    # creating a perceptron

    ann_reg = nn.Sequential(
        nn.Linear(1, 1),  # input layer
        nn.ReLU(),       # activation function
        nn.Linear(1, 1)   # output layer
        )

    # Part-2
    # create a loss function
    # If its regression then we use nn.MSELoss() and
    # If its classification then we use nn.BCEWithLogitsLoss()
    lossfun = nn.MSELoss()

    # Optimizer -- Teaches NN how to update the weights by learning rate
    optimizer = torch.optim.SGD(ann_reg.parameters(), lr=learning_rate)

    return ann_reg, lossfun, optimizer


def train_ann_model(annreg, lossfun, optimizer, indep_features, depend_feature):
    num_epochs = 500
    losses = torch.zeros(num_epochs)
    # Train the model
    for epoch in range(num_epochs):
        # forward Pass
        y_hat = annreg(indep_features)

        # compute loss
        loss = lossfun(y_hat, depend_feature)
        losses[epoch] = loss

        # back Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final Forward Pass
    predictions = annreg(indep_features)
    # Final Loss (MSE)
    test_loss = (predictions-depend_feature).pow(2).mean()
    return losses, predictions, test_loss


@app.route('/buildANN')
def build_ann():
    # Part-1
    # create data
    n = 30
    x = torch.randn(n, 1)
    y = x + torch.randn(n, 1) / 2

    # and plot
    plt.plot(x, y, 's')
    plt.title("Regression Data Plot")
    plt.savefig('regression_data', dpi=2000, facecolor='r')
    plt.show()

    # Build Model
    ann_regr, lossfunc, optimizerf = create_ann_model(.01)
    # Train the model
    losses_ls, predictions_final, loss_final = train_ann_model(ann_regr, lossfunc, optimizerf, x, y)

    # Show the losses
    plt.plot(losses_ls.detach(), 'ro', markerfacecolor='w', linewidth=0.1)
    plt.title(f"The Final Loss after 500 epochs is {loss_final}")
    plt.xlabel('Epoch'), plt.ylabel('Loss')
    plt.savefig('loss_values.jpeg', dpi=2000, facecolor='g')
    plt.show()
    return f'<h1>Final Loss obtained by ANN is {loss_final} </h1>'


if __name__ == "__main__":
    app.run(port=3000, host="0.0.0.0", debug=True)
