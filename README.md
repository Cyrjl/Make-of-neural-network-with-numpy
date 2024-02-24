# Making of a neural network with numpy

Hello, welcome to our project !

The objective was to build our own neural network and test it in a simple classification problem coming from Belle II physics.

On this repository, you will find : 

* the necessary packages to run the codes in *requirements.txt*

* the neural network we built in *NetworkModule.py*

* the Jupyter notebook containing the different tests in *NetworkNotebook.ipynb*

* the dataset used for the classification problem in *dataset.csv*

## How to run the notebook

In a dedicated python environment that has been activated, you can install the packages contained in *requirements.txt* by running:
```
pip install -r requirements.txt
```
then connect on a Jupyter notebook by simply running:
```
jupyter notebook
```
and finally, navigate to the notebook *NetworkNotebook.ipynb* and open it, run it by restarting the kernel and running all cells. It takes about 10 min to run.
Alternatively, the notebook can be opened in a supported IDE.

We are using a random seed for the initialization of the parameters, thus the results could be different than those presented in the article. 
