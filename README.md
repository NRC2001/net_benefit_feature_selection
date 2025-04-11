# Feature selection using net benefit
A net-benefit based approach to feature selection.


This project aims to balance model perforamce and practicality by basing feature selection on net benefit. We identify the test harm in a decision curve analysis with the total cost of each independent variable (feature). Features are then selected bases on net benefit of the model accounting for the costs of each independent variable.

We have devised several approaches to this and demonstrate them in the jupyter notebooks in the examples directory.

### Index of examples

* [00_synthetic_data_description.ipynb](./examples/00_synthetic_data_description.ipynb): A description of the synthetic data sets used.

* [01_net_benefit_loss_function.ipynb](./examples/01_net_benefit_loss_function.ipynb): We have developed an implementation of logistic regression using a net-benefit maximising loss function using gradient descent in pytorch. In this notbook we benchmark our implementation against scikit learn logistic regression. We find that in a well specified logistic regression the net-benefit maximizing loss function gives the same results as a standard logistic regression with the usual cross-entropy loss function.

* [02_MNB_loss_function_difference.ipynb](./examples/02_MNB_loss_function_difference.ipynb): We use a synthetic dataset that imposes an asymetry between positive and negative cases in a binary logistic regression by generating a synthetic data set in which the logistic regression is ill-specified. This case shows a difference between the standard cross-entropy loss and the mean-net-benefit maximizing loss. Models fit with the mean-net-benefit maximising loss function have a higher mean net benefit that standard logistic regression.

* [03_net_benefit_regularization_method.ipynb](./examples/03_net_benefit_regularization_method.ipynb): Demonstration of an approach that uses the above mean-net-benefit maximizing loss function in conjuction with a LASSO penalty weighted by the costs of each independent variable.

* [04_feature_importance_iterative_method.ipynb](./examples/04_feature_importance_iterative_method.ipynb): An approach that is similar to a forward stepwise selection procedure in which the order of inclusion of independent variables is determined by their feature importance in a full model and the selection is based on the net benefit.

* [05_backward_stepwise_method.ipynb](./examples/05_backward_stepwise_method.ipynb): A backward stepwise selection procedure with the stoping rule determined by maximum net benefit.

* [06_comparison_of_results.ipynb](./examples/06_comparison_of_results.ipynb): The results of the different methods of net-benefit-based feature selection are drawn together and compared. We see that in the example data sets we have used they each give the same result.