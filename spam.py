import nltk
import re
import matplotlib.pyplot as plt
import pandas as pd
import _markupbase
dataset=pd.read_csv('C:\\Users\\Mansi Dhingra\\PycharmProjects\\try\\copy\\spam.csv',encoding='latin-1')

dataset.drop(['v3','v4','v5'],axis=1,inplace=True) # i needed to remove all the columns which are empty inplace=true is an imp parameter

from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
dataset['label']=number.fit_transform(dataset.v1) #encode the spam or ham as 0 or 1  and chng the col name as label

dataset.drop(['v1'],axis=1,inplace=True) #drop v1 column


nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
corpus=[]
for i in range(0,5572):
    #text = ' '.join(re.findall('[A - Z][ ^ A - Z] * ', dataset['v2'][i]))
    text=re.sub('[^a-zA-Z]',' ',dataset['v2'][i])
    #tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))
    text=text.lower()
    text=text.split()
    ps=PorterStemmer()
    text=[ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
    text=' '.join(text)
    corpus.append(text)
print(set(stopwords.words('english')))
from sklearn.feature_extraction.text import CountVectorizer
cs=CountVectorizer(max_features=1500)
X=cs.fit_transform(corpus).toarray()
print(X)
y=dataset.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import GaussianNB
nbc=GaussianNB()
nbc.fit(X_train,y_train)

y_pred=nbc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(nbc.score(X_test,y_test))
