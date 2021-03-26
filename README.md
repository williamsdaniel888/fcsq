# fcsquery: An End-to-end Solution for Querying Newspaper Articles' Sentiment Scores

This repository contains a script to:
1. Query a remote database using PostgreSQL; 
2. Calculate the mean sentiment scores for the days represented;
3. Predict the mean sentiment score for the day following the last publication date in the article range;
4. Report performance statistics (i.e. mean square error) for the model on training, validation, and test sets;
5. Provide an endpoint for the user, with a list of parameter options available.

The GPU-enabled Colab notebook used to train the GRU predictor model can be found [here](https://colab.research.google.com/drive/1sB6-lzhlTBsDMKDn3f28eIqPBpJWOOwA?usp=sharing).
