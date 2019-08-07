# spam-classification

Classifies the data based on its words that it is spam or not.

In this python code, I converted the categorical variables to the values 0 and 1.
Stopwords which is very important in Natural Language Processing needs to be downloaded. It contains all the words from the sentences which are not at all important.
Then I created a list of written text which only includes the words on the basis of which it can classify the spam or ham mails
Applying CountVectorizer model will classify the mails based on the words. I split the data into training and testing data.
Gaussian NB model is applied and accuracy which is found to be 96% is predicted correctly.
