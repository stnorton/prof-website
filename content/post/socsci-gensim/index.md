+++
title = "Socsci Gensim"
subtitle = ""

# Add a summary to display on homepage (optional).
summary = ""

date = 2020-07-06T10:52:07-04:00
draft = false

# Authors. Comma separated list, e.g. `["Bob Smith", "David Jones"]`.
authors = []

# Is this a featured post? (true/false)
featured = false

# Tags and categories
# For example, use `tags = []` for no tags, or the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags = []
categories = []

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["deep-learning"]` references 
#   `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
# projects = ["internal-project"]

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
[image]
  # Caption (optional)
  caption = ""

  # Focal point (optional)
  # Options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
  focal_point = ""
+++

# Using Gensim for Topic Models in Social Science Research

Gensim is a fantastic Python module capable of handling large corpora of text data easier and faster than most the existing social science toolkit. In particular, Gensim is capable of parallelizing model fitting, while R packages cannot. This leads to R breaking when confronted with even trivially large amounts of text data. This is how I came across Gensim personally - a corpus with ~ 5 million documents was simply impossible to fit a topic model to using R's `topicmodel` package. Gensim is perfectly capable of handling corpora this large, and even doing so in constant memory!

However, `gensim` seems to have been primarily written for enterprise use, and topic modeling is only one of an impressive array of different NLP models in the package. As such, I found it ocassionally difficult to extract the quantities of interest I needed for political science research. 

The purpose of this tutorial is to show other social scientists how to set-up gensim, run a parallelized LDA model, and extract some common quantities of interest from the models. I don't claim that the solutions I've found are necessarily the most optimal, and would love to hear from you if you improve on my code. 

## Setting up your environment

Gensim runs best in a virtual environment. In particular, I ran into issues with parallelization when not isolated to a Conda environment; it seems to conflict with [something in scikit-learn](https://stackoverflow.com/questions/33929680/gensim-ldamulticore-not-multiprocessing). Regardless, I confine most my Python projects to Conda environments - the little bit of setup is worth not being caught in dependency hell. If you aren't already using Conda environments with Python, I *strongly* reccommend you start.

I won't go into Conda environments in detail here. The Conda docs have a [fantastic tutorial](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), and the `.yml` file for my environment is [available here](https://gist.github.com/stnorton/7a39cf0f54fbdb64d1c06967e74a1683). 

## Getting Started

We're going to start with the imports necessary. There's nothing complicated her, but for more involved workflows you may need to import additional pre-processors from `nltk`, stemmers for different workflows, etc. We're going to use the spaCy model here because [it's awesome](https://spacy.io/), but since I mostly work with Russian data I never get a chance to use it. 


```python
import re  #regular expression
import numpy as np
import pandas as pd
from pprint import pprint #pretty printing

# spacy for lemmatization
import spacy

#gensim imports
import gensim #whole module
import gensim.corpora as corpora #convenience rename
from gensim.utils import simple_preprocess #import preprocessor
from gensim.models import CoherenceModel #model for coherence

#enable logging for gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
logging.root.level = logging.INFO  #ipython sometimes messes up the logging setup

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

```

Now let's download the toy dataset for this tutorial. Following [this tutorial](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/), which I learned the Gensim basics from, we're going to be using the 20-Newsgroup dataset. 


```python
#read json file in
df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')

#print names of newsgroups
print(df.target_names.unique())

#inspect
df.head()
```

    ['rec.autos' 'comp.sys.mac.hardware' 'comp.graphics' 'sci.space'
     'talk.politics.guns' 'sci.med' 'comp.sys.ibm.pc.hardware'
     'comp.os.ms-windows.misc' 'rec.motorcycles' 'talk.religion.misc'
     'misc.forsale' 'alt.atheism' 'sci.electronics' 'comp.windows.x'
     'rec.sport.hockey' 'rec.sport.baseball' 'soc.religion.christian'
     'talk.politics.mideast' 'talk.politics.misc' 'sci.crypt']





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>content</th>
      <th>target</th>
      <th>target_names</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>From: lerxst@wam.umd.edu (where's my thing)\nS...</td>
      <td>7</td>
      <td>rec.autos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>From: guykuo@carson.u.washington.edu (Guy Kuo)...</td>
      <td>4</td>
      <td>comp.sys.mac.hardware</td>
    </tr>
    <tr>
      <th>2</th>
      <td>From: twillis@ec.ecn.purdue.edu (Thomas E Will...</td>
      <td>4</td>
      <td>comp.sys.mac.hardware</td>
    </tr>
    <tr>
      <th>3</th>
      <td>From: jgreen@amber (Joe Green)\nSubject: Re: W...</td>
      <td>1</td>
      <td>comp.graphics</td>
    </tr>
    <tr>
      <th>4</th>
      <td>From: jcm@head-cfa.harvard.edu (Jonathan McDow...</td>
      <td>14</td>
      <td>sci.space</td>
    </tr>
  </tbody>
</table>
</div>



While we're at it, we need to download and load the basic English stop words from the `nltk` package. You'll also need to download the spaCy model if you don't already have it. Uncomment the code below and use it if you need to.


```python
#download nltk
#nltk.download('stopwords')

#download spacy - run in terminal
#python3 -m spacy download en
```

We'll now need to clean the data, using regular expressions. I borrow the code from the tutorial for this purpose. These regexs will remove the emails, the new line characters, and quote signs. We'll then use Gensim's `simple_preprocess()` function to get rid of punctuation and tokenize the text.


```python
#convert data to list - required to run regexes on it using list comprehensions
data = df.content.values.tolist()

#remove emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

#remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

#get rid of single quotes
data = [re.sub("\'", "", sent) for sent in data]

#simple preprocess
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_tok = list(sent_to_words(data))

pprint(data_tok[:1])
```

    ['From: (wheres my thing) Subject: WHAT car is this!? Nntp-Posting-Host: '
     'rac3.wam.umd.edu Organization: University of Maryland, College Park Lines: '
     '15 I was wondering if anyone out there could enlighten me on this car I saw '
     'the other day. It was a 2-door sports car, looked to be from the late 60s/ '
     'early 70s. It was called a Bricklin. The doors were really small. In '
     'addition, the front bumper was separate from the rest of the body. This is '
     'all I know. If anyone can tellme a model name, engine specs, years of '
     'production, where this car is made, history, or whatever info you have on '
     'this funky looking car, please e-mail. Thanks, - IL ---- brought to you by '
     'your neighborhood Lerxst ---- ']
    [['from',
      'wheres',
      'my',
      'thing',
      'subject',
      'what',
      'car',
      'is',
      'this',
      'nntp',
      'posting',
      'host',
      'rac',
      'wam',
      'umd',
      'edu',
      'organization',
      'university',
      'of',
      'maryland',
      'college',
      'park',
      'lines',
      'was',
      'wondering',
      'if',
      'anyone',
      'out',
      'there',
      'could',
      'enlighten',
      'me',
      'on',
      'this',
      'car',
      'saw',
      'the',
      'other',
      'day',
      'it',
      'was',
      'door',
      'sports',
      'car',
      'looked',
      'to',
      'be',
      'from',
      'the',
      'late',
      'early',
      'it',
      'was',
      'called',
      'bricklin',
      'the',
      'doors',
      'were',
      'really',
      'small',
      'in',
      'addition',
      'the',
      'front',
      'bumper',
      'was',
      'separate',
      'from',
      'the',
      'rest',
      'of',
      'the',
      'body',
      'this',
      'is',
      'all',
      'know',
      'if',
      'anyone',
      'can',
      'tellme',
      'model',
      'name',
      'engine',
      'specs',
      'years',
      'of',
      'production',
      'where',
      'this',
      'car',
      'is',
      'made',
      'history',
      'or',
      'whatever',
      'info',
      'you',
      'have',
      'on',
      'this',
      'funky',
      'looking',
      'car',
      'please',
      'mail',
      'thanks',
      'il',
      'brought',
      'to',
      'you',
      'by',
      'your',
      'neighborhood',
      'lerxst']]


Now we're going to remove stopwords, and then lemmatize/stem using spaCy and restrict the data to only nouns, adjectives, verbs, and adverbs.


```python
#define functions
def remove_stopwords(texts):
    '''Loops over texts, preprocess, and then compares each word to stopwords using a list comprehension'''
    for text in texts:
        return[[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def stemmer(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    '''Uses spaCy EN model to stem words, remove certain parts of speech'''
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

#load in stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

#instatiate spacy model, keeping only tagger
nlp = spacy.load('en', disable=['parser', 'ner'])

data_tok = remove_stopwords(data_tok)

data_stem = stemmer(data_tok)

pprint(data_stem[:1])
```

    [['where',
      's',
      'thing',
      'subject',
      'car',
      'nntp',
      'post',
      'host',
      'rac',
      'wam',
      'umd',
      'edu',
      'organization',
      'university',
      'maryland',
      'college',
      'park',
      'line',
      'wonder',
      'anyone',
      'could',
      'enlighten',
      'car',
      'see',
      'day',
      'door',
      'sport',
      'car',
      'look',
      'late',
      'early',
      'call',
      'bricklin',
      'door',
      'really',
      'small',
      'addition',
      'front',
      'bumper',
      'separate',
      'rest',
      'body',
      'know',
      'anyone',
      'tellme',
      'model',
      'name',
      'engine',
      'spec',
      'year',
      'production',
      'car',
      'make',
      'history',
      'whatev',
      'info',
      'funky',
      'look',
      'car',
      'mail',
      'thank',
      'bring',
      'neighborhood',
      'lerxst']]


## Running LDA

We're almost ready to run the model. At this point, we just need the dictionary and corpus, which will be fed to the LDA model. We use functions within Gensim to do this. `corpora.Dictionary` stores each unique word and assigns it a numeric ID. The `.doc2bow` method converts the corpus to its bag-of-words representation (also known as the term document frequency matrix, although Gensim doesn't store this as a matrix, but rather a list of tuples).


```python
#create dictionary
dictionary = corpora.Dictionary(data_stem)

#full texts - for later reference
texts = data_stem

#bag of words
corpus = [dictionary.doc2bow(text) for text in texts]
```

Now let's move on to modeling. This corpus isn't actually big enough for parallelization to be worthwhile, but we'll do it for sake of demonstration.

I won't get into too much detail on what the arguments to multicore mean - most of them are self-explanatory. Important for parallelization and working with big corpora, however, are `chunksize`, `workers`, and `passes`.

* `chunksize`: Controls how large training "chunks" of the data are. Part of the reason Gensim works so well with large corpuses is that it uses online variational inference, allowing the model to run in constant memory. The larger the chunks, the quicker the model runs, but the more memory needed (in order to hold the chunk in memory). 
* `workers`: # of cores you have available to train the model. More cores, faster run.
* `passes`: Since we're using variational inference, we need a stopping point for convergence. Gensim has a tolerance already set, but `passes` controls how many times the algorithm will attempt to reach convergence. If you're not reaching convergence on all/nearly all documents, you'll want to increase that number. See the gensim docs for more detail on this. Naturally, more passes increases the run time.

We'll set up and run the model. If you want, you can also try the generic single core LDA model. It'll be faster in this case, since the overhead of setting up parallel processing isn't worth it on such a small corpus.

**Make sure to set the random_state for replicability and set minimum_probability to 0 to prevent topics with low responbility from being filtered out of the results**


```python
lda_model = gensim.models.LdaMulticore(corpus = corpus,
                                      id2word = dictionary,
                                      num_topics = 20, 
                                      random_state = 1017,
                                      chunksize = 100,
                                      passes = 10,
                                      per_word_topics = True,
                                      workers = 2, 
                                      minimum_probability = 0)
```

Let's take a look at the topics.The `print_topics()` method gives us the top tokens for each topic, plus its associated weight. 


```python
pprint(lda_model.print_topics())
```

If you were tuning $k$ for a topic model, you'll need some measure to help you select the best $k$. Gensim has a built in [coherence model](https://radimrehurek.com/gensim/models/coherencemodel.html) for exactly that purpose, which allows you to choose between several different measures of coherence. This can also be parallelized using the `processes` argument, though I omit it here.


```python
#get c_v coherence score

#instatiate model
cm = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v') #processes = 2 would use 2 cpus

#run model
co = cm.get_coherence()

print(co)
```

And there you have the basics of topic modeling in Gensim.

## Harder Quantaties of Interest

Everything up until this point has been remarkably easy. Now we're going to retrieve two quantaties of interest for which there is not a simple, pre-existing method: the most representative documents for each topic and topic assignments.

Here, I'm defining the most representative documents as the $n$ documents with the highest cluster responsbility for a given topic (the cluster responsibilities are often known as Phi in the LDA literature). Topic assignments are a bit trickier to define, since LDA is a mixed-membership model - documents can belong to more than one topic. Nevertheless, it's often helpful to have clear-cut topic assignments in order to summarize your corpus or perform other secondary analyses. I assign topics here by simpling choosing the topic that best explains the document (highest responsibility), but there are other ways this could be done. Another method might be to set some baseline responsibility, say $1/k$, that the responsibility must be above in order to be assigned to a topic. This prevents assigning topics to documents the model was "unsure" about ($1/k$ being the probability of getting it right by random guessing). 

### Most Representative Documents

We'll start with retrieving the 25 most representative documents for each topic. I'll show two methods here: one for the small model/if you have a lot of memory and another where you dump the responbilities to .csv to save memory. By dumping the responbilities to CSV, I was able to iterate over many models' outputs in a bash script, making it more memory efficient. Both methods work even if you forgot to specify the minimum probability argument (via the `fillna()` method).

I do this using two functions. `lda_to_df` extracts cluster responsibilities to a pandas dataframe for ease of use - this probably isn't the most efficient way to do this, but it is simple. `get_best_docs` returns the $n$ most representative docs for each topic, given the output of `lda_to_df` as input. `lda_to_df` takes the corpus as input, whereas `get_best_docs` takes the full (unstemmed/uncleaned) texts as input. This could be easily combined into one function, but when working with large corpora or many models, it's convenient to save the intermediate output to file (as I do in the second example).


```python
#no csv method

#define functions
def lda_to_df(model,corpus):
    '''This function takes a gensim lda model as input, and outputs a df with topics probs by document'''
    topic_probs = model.get_document_topics(corpus) #get the list of topic probabilities by doc
    topic_dict = [dict(x) for x in topic_probs] #convert to dictionary to convert to data frame
    df = pd.DataFrame(topic_dict).fillna(0) #convert to data frame
    df['docs'] = df.index.values #create column with document indices (correspond to indices of dataframe)
    df.columns = df.columns.astype(str) #convert to string to make indexing easier
    return df

def get_best_docs(df, n, k, texts):
    '''Return the index of the n most representative documents from a list of topic responsibilities for each topic'''
    '''n is the number of douments you want, k is the number of topics in the model, the texts are the FULL texts used to fit the model'''
    #create column list to iterate over
    k_cols = range(0, k)
    
    #intialize empty list to hold results
    n_rep_docs = []
    
    #loop to extract documents for each topic
    for i in k_cols:
        inds = df.nlargest(n = n, columns = str(i))['docs'].astype(int).tolist()
        #use list comprehension to extract documents
        n_rep_docs.append([texts[ind] for ind in inds])
    
    return n_rep_docs
    
```


```python
#run functions
resp = lda_to_df(lda_model, corpus)

best_docs = get_best_docs(resp, 25, 20, data)
```

Let's take a look at the output.


```python
pprint(best_docs[0])
```

Here's the function that outputs the responsibilities to a CSV file.


```python
def lda_to_csv(model, outfile, corpus):
    '''This function takes a gensim lda model as input, and outputs a csv with topics probs by document'''
    topic_probs = model.get_document_topics(corpus) #get the list of topic probabilities by doc
    topic_dict = [dict(x) for x in topic_probs] #convert to dictionary to convert to data frame
    df = pd.DataFrame(topic_dict).fillna(0) #convert to data frame, fill topics < 0.01 as 0
    df.to_csv(outfile)
```

### Topic Assignment

This is a basic function that assigns topics based on the maximum cluster responsibility. Again, I'd like to emphasize that LDA models are mixed membership, and this isn't the only way to cut assignments of topics.

This function operates diretly on the output of the `get_document_topics()` method, so it's more efficient than the function to retrieve the best documents.



```python
def assign_topic(model, corpus):
    doc_topics = model.get_document_topics(corpus) #get by-topic probability for each document
    topic_assignments = [] #initialize empty list for assignments
    for i in range(len(doc_topics)): #loop over every document
        doc = doc_topics[i] #extract relevant document
        list_length = len(doc)
        probs = []
        for r in range(list_length):
            probs.append(doc[r][1]) #get topic probs
        max_val = max(probs) #get max value
        max_ind = probs.index(max_val) #retrieve index
        topic = doc[max_ind] #retrieve topic number
        topic_assignments.append(topic[0]) #append only topic number (not also responsibility) to results
    return topic_assignments
```


```python
assignments = assign_topic(lda_model, corpus)
```

Let's take a peek at the output.


```python
pprint(assignments[:10])
```

    [0, 19, 11, 0, 0, 15, 0, 19, 0, 19]


The topic assignments are a simple list that you are free to use as you choose. If you also wanted to have the responsibilities, you could simply remove the slice from the `topic_assignments.append()` line in the function.

## Conclusion

There you have it - I hope that these functions are useful for you! Please get in touch if you have any questions or improve on my code!
