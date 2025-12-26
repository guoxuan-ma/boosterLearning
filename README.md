# Development of COVID-19 Booster Vaccine Policy by Microsimulation and Q-learning
Python script for Development of COVID-19 Booster Vaccine Policy by Microsimulation and Q-learning [https://arxiv.org/abs/2410.12936].

``
create_rnn_data.ipynb
``: create sequence of predictors and outcomes for RNN training from .csv dataset.

``
train_rnn.py
``: code for training an RNN with the LSTM architecture for environment simulation. 

``
simulate_env.py
``: code for simulating individual trajectories based on the RNN environment after the RNN is trained. 

``
testmdp.py
``: code for testing whether the Markor property holds in the simulated data. 

``
q_learning_table.py
``: code for tabular Q-learning based on the trained RNN environment simulator.  

``
q_learning.py
``: code for deep Q-learning based on the trained RNN environment simulator.  

``
q_learning_eval.py
``: code for policy evaluation after the policy is trained.   

``
helpers.py
``: helper functions.

### Workflow Instructions
* Use ``creat_rnn_data.ipynb`` to create predictor and outcome variables from processed data for RNN environment training. The processed data should have a long format (a subject has multiple rows). Each row corresponds to one subject at one time point, including covariates and outcomes of interests. 
* Use ``train_rnn.py`` to train RNN using predictor and outcome data processed by creat_rnn_data.ipynb.
* Use ``simulate_env.py`` to simulate individual trajectories based the trained RNN (weights are saved at the end of ``train_rnn.py``). Then, run ``testmdp.py`` to test whether the simulated dataset violate the Markov property. 
* Run ``q_learning_table.py`` for tabular Q-learning, where RNN environment is enabled by loading RNN weights (saved at the end of ``train.py``). Then, use ``q_learning_eval.py`` for policy evaluation. ``q_learning_eval.py`` includes the evaluation for learned tabular policy, policy where none patients receive booster (none), policy following observed data (data), and policy where all patients receive booster (all). 
  * As a comparison, run ``q_learning.py`` for deep Q-learning. 