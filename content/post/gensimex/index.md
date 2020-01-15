+++
title = "Gensim Multicore Example"
subtitle = ""

# Add a summary to display on homepage (optional).
summary = ""

date = 2020-01-15T11:46:48-05:00
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


# Using Gensim for multicore LDA

This notebook walks through an example of how to use Gensim to run LDA on multiple cores. Gensim uses an online algorithm with variational Bayes, so it has an incredibly low memory footprint (even with large corpuses) and can run in constant memory. Parallelizing the algorithm slightly increases time to convergence, but allows you to balance memory resources with speed. This makes it ideal for our purposes.


```python
#standard imports
import re  #regular expression
import numpy as np #arrays and math
import pandas as pd #dataframes
from pprint import pprint #pretty printing

#gensim imports
import gensim #whole module
import gensim.corpora as corpora #convenience
from gensim.utils import simple_preprocess #import preprocessor
from gensim.models import CoherenceModel #model for coherence
from gensim.corpora.sharded_corpus import ShardedCorpus #convenience
import string #string operations


#nltk imports
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer() #instantiate
from nltk.stem.snowball import SnowballStemmer #import stemmer
stemmer = SnowballStemmer("russian") #instantiate, set to russian

# Enable logging
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  #ipython sometimes messes up the logging setup

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

```

## Topic Model Test

Import Russian stopwords - some from NLTK, some from different lists. Then, remove duplicates. This consolidate stopwords list is available in the group workspace. 


```python
from nltk.corpus import stopwords #import stopwords from nltk

stop_words = stopwords.words('russian') #assign russian stopwords to vector

#read in yandex stop words
with open('yan_stopwords-ru.txt') as f:
    yan_stopwords = f.readlines()

yan_stopwords = [x.strip() for x in yan_stopwords] #stripping carriage returns

stop_words.extend(yan_stopwords) #adding to stopwords list

#read in iso RU stopwords
with open('stopwords-ru.txt') as f:
    iso_stopwords = f.readlines()
    
iso_stopwords=[x.strip() for x in iso_stopwords]

stop_words.extend(iso_stopwords)

#remove duplicates by changing to set
unique_stop=list(set(stop_words))

#sanity check
print(len(stop_words))
print(len(unique_stop))

#write base stopwords to text file
with open('base_ru_stopwords.txt', 'w') as text_file:
    for item in unique_stop:
        text_file.write("%s\n" % item)
```

    984
    654


Read in data using pandas. 


```python
texts = pd.read_csv('./big_test.csv')
pprint(texts.head)
```

    <bound method NDFrame.head of      RT @Borisovalustnah: 2d графические движки на android http://t.co/ok1HyJDB6F
    0     Беспризорные коровы создали 5-километровую про...                          
    1     Правительство области снова ищет подрядчика, к...                          
    2     Автобусы вместо самолётов и Хорватия вместо Ту...                          
    3     Заключенные ИК-2 в Салавате снова устроили бун...                          
    4     В ТЦ «Европа» в Минске девушке отрезали голову...                          
    ...                                                 ...                          
    9994               Стесняшка:))) http://t.co/Qkpnq5Ni3i                          
    9995  RT @zvezdanews: Атака вооруженного мачете на п...                          
    9996  RT @alexneed2play: Игровые новости | Представл...                          
    9997  RT @rianru: Глава ПА ОБСЕ предлагает России за...                          
    9998  RT @byDrBre: СМИ: Чичваркин не возглавит «Укрн...                          
    
    [9999 rows x 1 columns]>


Convert to list, clean using gensim tweets function. This can be a bottleneck in big datasets, notably because each word has to be checked against the stopword list.


```python
#convert to list 
data = texts.values.tolist()

#remove all links and hashtags
def clean_tweet(tweets):
    '''Uses regular expressions to remove urls and hashtags'''
    for tweet in tweets:
        tweet = re.sub('http\S+\s*', '', tweet)  # remove URLs (hopefully)
        tweet = re.sub('#\S+', '', tweet)  # remove hashtags (this should always work)
    return tweet

data = [clean_tweet(text) for text in data]

data[:10] #sanity check results

#instantiate tweet tokenizer
tknzr = TweetTokenizer(preserve_case = False, strip_handles=True)

def tokenizer(tweets):
    texts_out = []
    for tweet in tweets:
        texts_out.append(tknzr.tokenize(tweet))
    return texts_out
        
tokens = tokenizer(data) #tokenize

#remove stopwords

def remove_stopwords(texts):
    '''Loops over texts, preprocess, and then compares each word to stopwords using a list comprehension'''
    for text in texts:
        return[[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    
tokens = remove_stopwords(tokens)

tokens[:5] #sanity check


```




    [['беспризорные',
      'коровы',
      'создали',
      'километровую',
      'пробку',
      'тюменском',
      'тракте',
      'фото'],
     ['правительство',
      'области',
      'ищет',
      'подрядчика',
      'построит',
      'развязку',
      'екад',
      'миллиард',
      'схема'],
     ['автобусы',
      'вместо',
      'самолётов',
      'хорватия',
      'вместо',
      'турции',
      'нового',
      'готовит',
      'екатеринбуржцам',
      'летний',
      'турсезон',
      'фото'],
     ['заключенные', 'ик', 'салавате', 'устроили', 'бунт'],
     ['тц', 'европа', 'минске', 'девушке', 'отрезали', 'голову', 'бензопилой']]



Stem the tokens. The stemmer was instantiated in the imports section.


```python
stemmed_toks = [[stemmer.stem(word) for word in tweet] for tweet in tokens] #stem
stemmed_toks[:20] #sanity check
```




    [['беспризорн',
      'коров',
      'созда',
      'километров',
      'пробк',
      'тюменск',
      'тракт',
      'фот'],
     ['правительств',
      'област',
      'ищет',
      'подрядчик',
      'постро',
      'развязк',
      'екад',
      'миллиард',
      'схем'],
     ['автобус',
      'вмест',
      'самолет',
      'хорват',
      'вмест',
      'турц',
      'нов',
      'готов',
      'екатеринбуржц',
      'летн',
      'турсезон',
      'фот'],
     ['заключен', 'ик', 'салават', 'устро', 'бунт'],
     ['тц', 'европ', 'минск', 'девушк', 'отреза', 'голов', 'бензопил'],
     ['балкон', 'центр', 'москв', 'обруш', 'просел'],
     ['петербург', 'неизвестн', 'выстрел', 'голов', 'мужчин'],
     ['петербург', 'водител', 'выстрел', 'голов', 'друг', 'обрез'],
     ['середин',
      'весн',
      'россия',
      'приготов',
      'продуктов',
      'карточк',
      'правительств',
      'образ',
      'хочет',
      'уб'],
     ['лежа', 'дорог', 'hyundai', 'раздав', 'мужчин', 'екатеринбург'],
     ['полевск',
      'разб',
      'насмерт',
      'водител',
      'вездеход',
      'удар',
      'водител',
      'вездеход',
      'расколол',
      'шлем'],
     ['воен',
      'академ',
      'рвсн',
      'серпуховск',
      'филиал',
      'пройдут',
      'дни',
      'открыт',
      'двер'],
     ['военнослужа', 'зво', 'постро', 'велиж', 'автомобильн', 'разборн', 'мост'],
     ['rt', 'ставя', 'игр', 'iphone', 'пишет', 'уда', 'провер'],
     ['добр', 'избирательн', 'вакханал', 'продолжа'],
     ['рввдку', 'имен', 'маргелов', 'заверш', 'чемпионат', 'кикбоксинг'],
     ['rt', 'спецслужб', 'опрашива', 'диспетчер', 'соч', 'пропаж'],
     ['rt', 'япон', 'произошл', 'землетрясен', 'магнитуд'],
     ['rt',
      'агентств',
      'опубликова',
      'фотограф',
      'стреля',
      'посл',
      'росс',
      'турц'],
     ['rt', 'вброс', 'комментатор', 'матч', 'тв', 'занозин'],
     ['rt',
      'пресвят',
      'госпож',
      'богородиц',
      'спас',
      'бож',
      'спас',
      'сохран',
      'президент',
      'да',
      'любов',
      'помощ',
      'защ'],
     ['rt', 'индонезиец', 'отмет', 'рожден'],
     ['rt',
      'короч',
      'сша',
      'отвалива',
      'двигател',
      'бомбардировщик',
      'нервн',
      'кур',
      'сторонк'],
     ['rt',
      'летн',
      'доч',
      'крупн',
      'еврочиновник',
      'помога',
      'принима',
      'беженц',
      'изнасилова',
      'убит',
      'беженц',
      'афганист'],
     ['русск', 'карикатур', 'эпох', 'велик', 'войн'],
     ['советск', 'тяжел', 'боев', 'аэросан', 'цкб', 'испытан', 'зим', 'го'],
     ['су'],
     ['президент', 'рф', 'провел', 'совещан'],
     ['путин', 'эксперт', 'юв'],
     ['rt', 'худ', 'склон', 'диабет', 'полн', 'выясн', 'учен'],
     ['rt',
      'встряхнул',
      'снял',
      'шор',
      'заглянул',
      'зерка',
      'поня',
      'слух',
      'уродств',
      'заметн',
      'преувелич'],
     ['quot', 'плат', 'налог', 'quot', 'возмуща', 'стаматопулос'],
     ['сайт', 'супер', 'рекомендова', 'знаком'],
     ['rt', 'сша', 'период', 'live'],
     ['политик', 'украин', 'поража', 'варварск', 'первобытн'],
     ['взрыв', 'чернигов', 'покалеч', 'люд'],
     ['rt', 'неплох', 'поддрежива'],
     ['rt', 'боевик', 'бок', 'хар', 'нигер', 'нападен', 'уб', 'сожгл', 'деревн'],
     ['туляк', 'научат', 'плест', 'пояс'],
     ['слуша',
      'рассказ',
      'люд',
      'тусовк',
      'интеграл',
      'подвал',
      'дума',
      'дор',
      'отда',
      'побыва'],
     ['коротк', 'наскольк', 'емк'],
     ['укргаздобыч', 'исчерпа', 'месторожден'],
     ['брестчанин', 'приговор', 'шест', 'год', 'колон', 'дел', 'наркотик'],
     ['порошенк',
      'потребова',
      'предостав',
      'отчет',
      'расследован',
      'убийств',
      'шеремет'],
     ['январ',
      'июл',
      'республиканск',
      'бюджет',
      'исполн',
      'профицит',
      'млрд',
      'рубл'],
     ['арт',
      'милевск',
      'провел',
      'перв',
      'тренировк',
      'брестск',
      'динам',
      'мастер',
      'класс',
      'юниор'],
     ['порошенк', 'объявля', 'перемир', 'проигрыш'],
     ['rt', 'пожар', 'склад', 'хабаровск', 'ликвидирова'],
     ['rt', 'ольхон', 'гост', 'грэм', 'филлипс'],
     ['миноборон', 'рф', 'переброс', 'сир', 'нов', 'истребител', 'су']]



### Modeling

Create corpus and dictionary. Again, this is a potential bottleneck, though gensim's functions seem to be pretty fast.


```python
#dictionary - list of all unique tokens
id2word = corpora.Dictionary(stemmed_toks)

#corpus - actual tokens (all)
texts = stemmed_toks

#dtf - numeric representation of data; large and sparse with big dataset
corpus = [id2word.doc2bow(text) for text in texts]

```

    INFO : adding document #0 to Dictionary(0 unique tokens: [])
    INFO : built Dictionary(14213 unique tokens: ['беспризорн', 'километров', 'коров', 'пробк', 'созда']...) from 9999 documents (total 63744 corpus positions)


Run LDA


```python
#multicore
from datetime import datetime #basic benchmarking
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=20, 
                                        random_state=1017,
                                        chunksize=100,
                                        passes = 10,
                                        per_word_topics=True,
                                        workers=4
                                      )
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

```

    INFO : using symmetric alpha at 0.05
    INFO : using symmetric eta at 0.05
    INFO : using serial LDA version on this node
    INFO : running online LDA training, 20 topics, 10 passes over the supplied corpus of 9999 documents, updating every 400 documents, evaluating every ~4000 documents, iterating 50x with a convergence threshold of 0.001000
    INFO : training LDA model using 4 processes


    Current Time = 10:54:20


    INFO : PROGRESS: pass 0, dispatched chunk #0 = documents up to #100/9999, outstanding queue size 1
    INFO : PROGRESS: pass 0, dispatched chunk #1 = documents up to #200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 0, dispatched chunk #2 = documents up to #300/9999, outstanding queue size 3
    INFO : PROGRESS: pass 0, dispatched chunk #3 = documents up to #400/9999, outstanding queue size 4
    INFO : PROGRESS: pass 0, dispatched chunk #4 = documents up to #500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 0, dispatched chunk #5 = documents up to #600/9999, outstanding queue size 6
    INFO : PROGRESS: pass 0, dispatched chunk #6 = documents up to #700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 0, dispatched chunk #7 = documents up to #800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 0, dispatched chunk #8 = documents up to #900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 0, dispatched chunk #9 = documents up to #1000/9999, outstanding queue size 10
    INFO : PROGRESS: pass 0, dispatched chunk #10 = documents up to #1100/9999, outstanding queue size 11
    INFO : PROGRESS: pass 0, dispatched chunk #11 = documents up to #1200/9999, outstanding queue size 11
    INFO : PROGRESS: pass 0, dispatched chunk #12 = documents up to #1300/9999, outstanding queue size 12
    INFO : PROGRESS: pass 0, dispatched chunk #13 = documents up to #1400/9999, outstanding queue size 12
    INFO : PROGRESS: pass 0, dispatched chunk #14 = documents up to #1500/9999, outstanding queue size 12
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.043*"rt" + 0.024*"нов" + 0.012*"выстрел" + 0.012*"санкц" + 0.012*"российск" + 0.012*"представ" + 0.012*"голов" + 0.012*"петербург" + 0.012*"опасн" + 0.007*"отношен"
    INFO : topic #2 (0.050): 0.035*"rt" + 0.015*"глав" + 0.013*"ukraine" + 0.013*"спас" + 0.013*"бог" + 0.013*"завод" + 0.013*"президент" + 0.013*"тел" + 0.013*"киев" + 0.008*"фскн"
    INFO : topic #3 (0.050): 0.030*"rt" + 0.024*"москв" + 0.012*"обстреля" + 0.012*"украин" + 0.012*"сми" + 0.012*"официальн" + 0.012*"встреч" + 0.012*"росс" + 0.006*"фрагмент" + 0.006*"казахста"
    INFO : topic #10 (0.050): 0.058*"rt" + 0.016*"начина" + 0.016*"quot" + 0.016*"нов" + 0.011*"возможн" + 0.008*"образ" + 0.008*"авар" + 0.008*"уб" + 0.008*"понедельник" + 0.008*"карточк"
    INFO : topic #11 (0.050): 0.087*"rt" + 0.010*"вездеход" + 0.010*"рожден" + 0.010*"аварийн" + 0.010*"област" + 0.010*"отмеча" + 0.010*"истор" + 0.010*"водител" + 0.005*"рюкзак" + 0.005*"хант"
    INFO : topic diff=18.508831, rho=1.000000
    INFO : PROGRESS: pass 0, dispatched chunk #15 = documents up to #1600/9999, outstanding queue size 12
    INFO : PROGRESS: pass 0, dispatched chunk #16 = documents up to #1700/9999, outstanding queue size 11
    INFO : merging changes from 800 documents into a model of 9999 documents
    INFO : topic #11 (0.050): 0.077*"rt" + 0.008*"путин" + 0.006*"вездеход" + 0.006*"рожден" + 0.006*"аварийн" + 0.006*"област" + 0.006*"отмеча" + 0.006*"истор" + 0.006*"водител" + 0.005*"дел"
    INFO : topic #2 (0.050): 0.029*"rt" + 0.011*"президент" + 0.010*"глав" + 0.009*"спас" + 0.009*"киев" + 0.008*"мужчин" + 0.008*"украин" + 0.008*"ukraine" + 0.008*"бог" + 0.008*"завод"
    INFO : topic #9 (0.050): 0.040*"rt" + 0.017*"президент" + 0.014*"украин" + 0.009*"работ" + 0.007*"федерац" + 0.007*"путин" + 0.007*"должн" + 0.006*"велоспорт" + 0.006*"оон" + 0.006*"митинг"
    INFO : topic #16 (0.050): 0.064*"rt" + 0.011*"нача" + 0.009*"чита" + 0.006*"друз" + 0.006*"рф" + 0.006*"росс" + 0.006*"дел" + 0.006*"созда" + 0.006*"систем" + 0.006*"украин"
    INFO : topic #15 (0.050): 0.050*"rt" + 0.016*"quot" + 0.009*"узна" + 0.008*"подрядчик" + 0.007*"дорог" + 0.006*"вниман" + 0.006*"рязанск" + 0.006*"мнен" + 0.006*"проведут" + 0.006*"факт"
    INFO : topic diff=0.394785, rho=0.447214
    INFO : PROGRESS: pass 0, dispatched chunk #17 = documents up to #1800/9999, outstanding queue size 6
    INFO : PROGRESS: pass 0, dispatched chunk #18 = documents up to #1900/9999, outstanding queue size 6
    INFO : PROGRESS: pass 0, dispatched chunk #19 = documents up to #2000/9999, outstanding queue size 5
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.055*"rt" + 0.017*"российск" + 0.015*"нов" + 0.013*"санкц" + 0.012*"украин" + 0.011*"рф" + 0.009*"лин" + 0.008*"росс" + 0.008*"украинск" + 0.008*"петербург"
    INFO : topic #3 (0.050): 0.042*"rt" + 0.021*"украин" + 0.015*"москв" + 0.011*"донецк" + 0.010*"встреч" + 0.009*"район" + 0.009*"росс" + 0.009*"российск" + 0.008*"обстреля" + 0.007*"мэр"
    INFO : topic #16 (0.050): 0.080*"rt" + 0.015*"нача" + 0.014*"чита" + 0.010*"написа" + 0.009*"украин" + 0.009*"созда" + 0.008*"уголовн" + 0.008*"завел" + 0.007*"росс" + 0.006*"дел"
    INFO : topic #10 (0.050): 0.069*"rt" + 0.020*"виде" + 0.016*"росс" + 0.011*"последн" + 0.011*"начина" + 0.010*"уб" + 0.010*"нов" + 0.008*"авар" + 0.008*"налог" + 0.008*"земл"
    INFO : topic #7 (0.050): 0.074*"rt" + 0.013*"росс" + 0.013*"новосибирск" + 0.013*"област" + 0.010*"летн" + 0.009*"март" + 0.009*"арестова" + 0.008*"продолжа" + 0.007*"сша" + 0.007*"суд"
    INFO : topic diff=0.126063, rho=0.277350
    INFO : PROGRESS: pass 0, dispatched chunk #20 = documents up to #2100/9999, outstanding queue size 4
    INFO : PROGRESS: pass 0, dispatched chunk #21 = documents up to #2200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 0, dispatched chunk #22 = documents up to #2300/9999, outstanding queue size 3
    INFO : PROGRESS: pass 0, dispatched chunk #23 = documents up to #2400/9999, outstanding queue size 4
    INFO : PROGRESS: pass 0, dispatched chunk #24 = documents up to #2500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 0, dispatched chunk #25 = documents up to #2600/9999, outstanding queue size 6
    INFO : PROGRESS: pass 0, dispatched chunk #26 = documents up to #2700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 0, dispatched chunk #27 = documents up to #2800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 0, dispatched chunk #28 = documents up to #2900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 0, dispatched chunk #29 = documents up to #3000/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #0 (0.050): 0.085*"rt" + 0.018*"сша" + 0.010*"сир" + 0.009*"украин" + 0.009*"игр" + 0.006*"миноборон" + 0.006*"пьян" + 0.006*"призва" + 0.005*"зада" + 0.005*"главн"
    INFO : topic #13 (0.050): 0.054*"rt" + 0.020*"украин" + 0.016*"готов" + 0.012*"власт" + 0.010*"перв" + 0.009*"путин" + 0.009*"прав" + 0.009*"похож" + 0.009*"открыт" + 0.009*"бо"
    INFO : topic #3 (0.050): 0.046*"rt" + 0.020*"украин" + 0.017*"москв" + 0.010*"район" + 0.010*"донецк" + 0.010*"строительств" + 0.010*"росс" + 0.009*"дан" + 0.009*"российск" + 0.008*"встреч"
    INFO : topic #5 (0.050): 0.080*"rt" + 0.012*"росс" + 0.011*"войск" + 0.010*"поздрав" + 0.009*"украинц" + 0.008*"сша" + 0.008*"стат" + 0.008*"отмен" + 0.007*"перв" + 0.007*"ситуац"
    INFO : topic #12 (0.050): 0.101*"rt" + 0.018*"украин" + 0.010*"счита" + 0.010*"крым" + 0.009*"сша" + 0.008*"возможн" + 0.007*"оруж" + 0.007*"порошенк" + 0.007*"соч" + 0.006*"рф"
    INFO : topic diff=0.043456, rho=0.235702
    INFO : PROGRESS: pass 0, dispatched chunk #30 = documents up to #3100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 0, dispatched chunk #31 = documents up to #3200/9999, outstanding queue size 8
    INFO : PROGRESS: pass 0, dispatched chunk #32 = documents up to #3300/9999, outstanding queue size 9
    INFO : PROGRESS: pass 0, dispatched chunk #33 = documents up to #3400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 0, dispatched chunk #34 = documents up to #3500/9999, outstanding queue size 11
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #3 (0.050): 0.050*"rt" + 0.019*"москв" + 0.018*"украин" + 0.012*"строительств" + 0.012*"мэр" + 0.011*"донецк" + 0.010*"дан" + 0.010*"российск" + 0.010*"машин" + 0.010*"район"
    INFO : topic #8 (0.050): 0.058*"rt" + 0.019*"украин" + 0.018*"слов" + 0.018*"миноборон" + 0.015*"росс" + 0.011*"рубл" + 0.011*"дтп" + 0.009*"сша" + 0.009*"получ" + 0.009*"стрельб"
    INFO : topic #15 (0.050): 0.050*"rt" + 0.017*"quot" + 0.013*"узна" + 0.012*"факт" + 0.011*"дорог" + 0.009*"сто" + 0.009*"рязанск" + 0.009*"миллиард" + 0.009*"памятник" + 0.008*"выход"
    INFO : topic #2 (0.050): 0.041*"rt" + 0.016*"пыта" + 0.014*"киев" + 0.012*"глав" + 0.011*"погибл" + 0.010*"нашл" + 0.009*"отставк" + 0.009*"колон" + 0.009*"пойма" + 0.009*"украин"
    INFO : topic #4 (0.050): 0.051*"rt" + 0.015*"появ" + 0.013*"новост" + 0.012*"закон" + 0.011*"очередн" + 0.011*"росс" + 0.011*"украин" + 0.010*"российск" + 0.010*"интересн" + 0.010*"ес"
    INFO : topic diff=0.035467, rho=0.213201
    INFO : PROGRESS: pass 0, dispatched chunk #35 = documents up to #3600/9999, outstanding queue size 11
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #2 (0.050): 0.042*"rt" + 0.017*"глав" + 0.014*"пыта" + 0.013*"кита" + 0.012*"нашл" + 0.012*"киев" + 0.011*"украин" + 0.010*"погибл" + 0.009*"пойма" + 0.009*"сообщен"
    INFO : topic #15 (0.050): 0.051*"rt" + 0.027*"quot" + 0.012*"выход" + 0.012*"факт" + 0.011*"дорог" + 0.011*"рязанск" + 0.011*"сто" + 0.010*"узна" + 0.010*"солдат" + 0.009*"рук"
    INFO : topic #3 (0.050): 0.051*"rt" + 0.020*"москв" + 0.019*"украин" + 0.015*"донецк" + 0.014*"машин" + 0.013*"дан" + 0.012*"мэр" + 0.012*"строительств" + 0.010*"российск" + 0.010*"район"
    INFO : topic #7 (0.050): 0.085*"rt" + 0.017*"новосибирск" + 0.017*"человек" + 0.012*"област" + 0.012*"росс" + 0.009*"пройдет" + 0.009*"продолжа" + 0.009*"дня" + 0.009*"сша" + 0.008*"март"
    INFO : topic #18 (0.050): 0.065*"rt" + 0.017*"санкц" + 0.015*"рф" + 0.013*"нов" + 0.012*"российск" + 0.011*"украин" + 0.011*"росс" + 0.010*"воен" + 0.010*"представ" + 0.009*"игра"
    INFO : topic diff=0.038210, rho=0.196116
    INFO : PROGRESS: pass 0, dispatched chunk #36 = documents up to #3700/9999, outstanding queue size 8
    INFO : PROGRESS: pass 0, dispatched chunk #37 = documents up to #3800/9999, outstanding queue size 6
    INFO : PROGRESS: pass 0, dispatched chunk #38 = documents up to #3900/9999, outstanding queue size 7
    INFO : PROGRESS: pass 0, dispatched chunk #39 = documents up to #4000/9999, outstanding queue size 8
    INFO : merging changes from 800 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.065*"rt" + 0.019*"санкц" + 0.017*"рф" + 0.015*"нов" + 0.012*"росс" + 0.011*"российск" + 0.011*"украин" + 0.010*"представ" + 0.008*"акц" + 0.008*"воен"
    INFO : topic #0 (0.050): 0.092*"rt" + 0.021*"сша" + 0.014*"сир" + 0.010*"главн" + 0.008*"украин" + 0.008*"призва" + 0.008*"игр" + 0.007*"минск" + 0.007*"ирак" + 0.007*"рад"
    INFO : topic #12 (0.050): 0.121*"rt" + 0.016*"украин" + 0.012*"счита" + 0.012*"цен" + 0.010*"люд" + 0.010*"автор" + 0.010*"сша" + 0.010*"соч" + 0.010*"крым" + 0.009*"возможн"
    INFO : topic #9 (0.050): 0.054*"rt" + 0.024*"президент" + 0.013*"украин" + 0.013*"росс" + 0.013*"рф" + 0.012*"путин" + 0.011*"вооружен" + 0.010*"говор" + 0.010*"город" + 0.009*"евр"
    INFO : topic #10 (0.050): 0.085*"rt" + 0.032*"виде" + 0.018*"полиц" + 0.017*"росс" + 0.012*"китайск" + 0.012*"хочет" + 0.012*"начина" + 0.012*"дет" + 0.010*"бизнес" + 0.009*"видел"
    INFO : topic diff=0.028146, rho=0.182574
    INFO : PROGRESS: pass 0, dispatched chunk #40 = documents up to #4100/9999, outstanding queue size 4
    INFO : PROGRESS: pass 0, dispatched chunk #41 = documents up to #4200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 0, dispatched chunk #42 = documents up to #4300/9999, outstanding queue size 3
    INFO : PROGRESS: pass 0, dispatched chunk #43 = documents up to #4400/9999, outstanding queue size 4
    INFO : PROGRESS: pass 0, dispatched chunk #44 = documents up to #4500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 0, dispatched chunk #45 = documents up to #4600/9999, outstanding queue size 6
    INFO : PROGRESS: pass 0, dispatched chunk #46 = documents up to #4700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 0, dispatched chunk #47 = documents up to #4800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 0, dispatched chunk #48 = documents up to #4900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 0, dispatched chunk #49 = documents up to #5000/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.068*"rt" + 0.022*"санкц" + 0.020*"рф" + 0.018*"нов" + 0.014*"российск" + 0.012*"росс" + 0.012*"акц" + 0.011*"мид" + 0.010*"украин" + 0.009*"представ"
    INFO : topic #19 (0.050): 0.036*"rt" + 0.031*"крым" + 0.022*"днр" + 0.022*"политик" + 0.021*"украинск" + 0.018*"фот" + 0.015*"сша" + 0.015*"помощ" + 0.015*"украин" + 0.014*"прос"
    INFO : topic #13 (0.050): 0.060*"rt" + 0.024*"украин" + 0.020*"готов" + 0.017*"власт" + 0.014*"русск" + 0.013*"прав" + 0.012*"перв" + 0.011*"репост" + 0.010*"террорист" + 0.010*"детск"
    INFO : topic #9 (0.050): 0.055*"rt" + 0.025*"президент" + 0.014*"рф" + 0.013*"росс" + 0.012*"путин" + 0.012*"украин" + 0.011*"вооружен" + 0.011*"сентябр" + 0.010*"произошл" + 0.010*"эксперт"
    INFO : topic #3 (0.050): 0.052*"rt" + 0.024*"москв" + 0.016*"украин" + 0.016*"донецк" + 0.012*"машин" + 0.012*"прошл" + 0.010*"дан" + 0.010*"продаж" + 0.010*"российск" + 0.010*"мэр"
    INFO : topic diff=0.023363, rho=0.162221
    INFO : PROGRESS: pass 0, dispatched chunk #50 = documents up to #5100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 0, dispatched chunk #51 = documents up to #5200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 0, dispatched chunk #52 = documents up to #5300/9999, outstanding queue size 11
    INFO : PROGRESS: pass 0, dispatched chunk #53 = documents up to #5400/9999, outstanding queue size 12
    INFO : PROGRESS: pass 0, dispatched chunk #54 = documents up to #5500/9999, outstanding queue size 11
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #1 (0.050): 0.043*"rt" + 0.037*"стран" + 0.024*"жител" + 0.019*"дом" + 0.018*"пострада" + 0.012*"днем" + 0.012*"ноч" + 0.011*"вид" + 0.011*"конц" + 0.010*"стал"
    INFO : topic #12 (0.050): 0.126*"rt" + 0.017*"украин" + 0.012*"люд" + 0.010*"цен" + 0.010*"счита" + 0.010*"автор" + 0.009*"сша" + 0.009*"оруж" + 0.008*"убийств" + 0.008*"крым"
    INFO : topic #15 (0.050): 0.054*"rt" + 0.028*"quot" + 0.018*"дорог" + 0.012*"выход" + 0.012*"факт" + 0.011*"рук" + 0.010*"солдат" + 0.010*"сто" + 0.008*"миллиард" + 0.008*"апрел"
    INFO : topic #17 (0.050): 0.029*"украин" + 0.028*"rt" + 0.028*"арм" + 0.025*"мнен" + 0.024*"донбасс" + 0.023*"российск" + 0.022*"побед" + 0.020*"реш" + 0.020*"выбор" + 0.019*"газ"
    INFO : topic #18 (0.050): 0.066*"rt" + 0.022*"рф" + 0.021*"санкц" + 0.021*"нов" + 0.017*"российск" + 0.011*"росс" + 0.011*"представ" + 0.010*"акц" + 0.010*"украин" + 0.009*"сам"
    INFO : topic diff=0.018983, rho=0.154303
    INFO : PROGRESS: pass 0, dispatched chunk #55 = documents up to #5600/9999, outstanding queue size 10
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #12 (0.050): 0.125*"rt" + 0.018*"украин" + 0.012*"люд" + 0.011*"оруж" + 0.010*"счита" + 0.010*"автор" + 0.009*"сша" + 0.009*"цен" + 0.009*"убийств" + 0.008*"доллар"
    INFO : topic #1 (0.050): 0.044*"rt" + 0.039*"стран" + 0.023*"жител" + 0.021*"пострада" + 0.016*"дом" + 0.014*"днем" + 0.011*"вид" + 0.011*"ноч" + 0.011*"конц" + 0.010*"сем"
    INFO : topic #15 (0.050): 0.054*"rt" + 0.028*"quot" + 0.020*"дорог" + 0.014*"сто" + 0.012*"выход" + 0.010*"рук" + 0.010*"факт" + 0.010*"солдат" + 0.010*"предлож" + 0.009*"участник"
    INFO : topic #11 (0.050): 0.138*"rt" + 0.012*"связ" + 0.012*"депутат" + 0.012*"праздник" + 0.010*"истор" + 0.010*"путин" + 0.009*"автомобил" + 0.009*"мост" + 0.009*"поддержива" + 0.009*"хорош"
    INFO : topic #5 (0.050): 0.078*"rt" + 0.012*"сотрудник" + 0.012*"росс" + 0.012*"задержа" + 0.012*"попа" + 0.011*"стат" + 0.011*"отправ" + 0.011*"ситуац" + 0.010*"смотр" + 0.010*"украинц"
    INFO : topic diff=0.018592, rho=0.145865
    INFO : PROGRESS: pass 0, dispatched chunk #56 = documents up to #5700/9999, outstanding queue size 5
    INFO : PROGRESS: pass 0, dispatched chunk #57 = documents up to #5800/9999, outstanding queue size 3
    INFO : PROGRESS: pass 0, dispatched chunk #58 = documents up to #5900/9999, outstanding queue size 4
    INFO : PROGRESS: pass 0, dispatched chunk #59 = documents up to #6000/9999, outstanding queue size 5
    INFO : PROGRESS: pass 0, dispatched chunk #60 = documents up to #6100/9999, outstanding queue size 6
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #10 (0.050): 0.089*"rt" + 0.035*"виде" + 0.023*"полиц" + 0.015*"росс" + 0.014*"дет" + 0.013*"решен" + 0.013*"хочет" + 0.013*"лучш" + 0.012*"последн" + 0.012*"ид"
    INFO : topic #11 (0.050): 0.136*"rt" + 0.013*"мост" + 0.011*"поддержива" + 0.011*"связ" + 0.011*"депутат" + 0.011*"праздник" + 0.010*"истор" + 0.009*"автомобил" + 0.009*"хорош" + 0.009*"привет"
    INFO : topic #17 (0.050): 0.036*"украин" + 0.029*"rt" + 0.027*"мнен" + 0.025*"российск" + 0.024*"арм" + 0.024*"побед" + 0.023*"донбасс" + 0.021*"реш" + 0.021*"выбор" + 0.021*"учен"
    INFO : topic #5 (0.050): 0.076*"rt" + 0.013*"сотрудник" + 0.013*"задержа" + 0.012*"росс" + 0.011*"смотр" + 0.011*"предлага" + 0.010*"попа" + 0.010*"сел" + 0.010*"ополченц" + 0.010*"стат"
    INFO : topic #18 (0.050): 0.068*"rt" + 0.026*"нов" + 0.022*"санкц" + 0.018*"рф" + 0.015*"российск" + 0.013*"росс" + 0.012*"украин" + 0.011*"акц" + 0.010*"сам" + 0.009*"представ"
    INFO : topic diff=0.019575, rho=0.137361
    INFO : PROGRESS: pass 0, dispatched chunk #61 = documents up to #6200/9999, outstanding queue size 6
    INFO : PROGRESS: pass 0, dispatched chunk #62 = documents up to #6300/9999, outstanding queue size 5
    INFO : PROGRESS: pass 0, dispatched chunk #63 = documents up to #6400/9999, outstanding queue size 6
    INFO : PROGRESS: pass 0, dispatched chunk #64 = documents up to #6500/9999, outstanding queue size 7
    INFO : PROGRESS: pass 0, dispatched chunk #65 = documents up to #6600/9999, outstanding queue size 8
    INFO : PROGRESS: pass 0, dispatched chunk #66 = documents up to #6700/9999, outstanding queue size 9
    INFO : PROGRESS: pass 0, dispatched chunk #67 = documents up to #6800/9999, outstanding queue size 10
    INFO : PROGRESS: pass 0, dispatched chunk #68 = documents up to #6900/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #6 (0.050): 0.049*"rt" + 0.040*"росс" + 0.039*"путин" + 0.022*"мест" + 0.018*"владимир" + 0.014*"друз" + 0.014*"крут" + 0.013*"мвд" + 0.013*"александр" + 0.012*"жил"
    INFO : topic #4 (0.050): 0.050*"rt" + 0.025*"закон" + 0.022*"интересн" + 0.021*"появ" + 0.015*"европ" + 0.015*"украин" + 0.014*"границ" + 0.013*"отказа" + 0.012*"нужн" + 0.011*"настоя"
    INFO : topic #14 (0.050): 0.087*"rt" + 0.026*"вер" + 0.024*"украин" + 0.021*"петербург" + 0.018*"новост" + 0.014*"медвед" + 0.011*"матч" + 0.011*"пожар" + 0.011*"арест" + 0.011*"ки"
    INFO : topic #13 (0.050): 0.065*"rt" + 0.029*"украин" + 0.023*"готов" + 0.021*"русск" + 0.020*"власт" + 0.019*"перв" + 0.013*"репост" + 0.013*"прав" + 0.012*"сми" + 0.011*"хоч"
    INFO : topic #19 (0.050): 0.038*"rt" + 0.038*"крым" + 0.026*"днр" + 0.025*"украинск" + 0.024*"политик" + 0.022*"помощ" + 0.021*"воен" + 0.020*"украин" + 0.017*"ес" + 0.016*"фот"
    INFO : topic diff=0.020264, rho=0.132453
    INFO : PROGRESS: pass 0, dispatched chunk #69 = documents up to #7000/9999, outstanding queue size 10
    INFO : PROGRESS: pass 0, dispatched chunk #70 = documents up to #7100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 0, dispatched chunk #71 = documents up to #7200/9999, outstanding queue size 11
    INFO : PROGRESS: pass 0, dispatched chunk #72 = documents up to #7300/9999, outstanding queue size 12
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #10 (0.050): 0.089*"rt" + 0.033*"виде" + 0.025*"полиц" + 0.019*"лучш" + 0.014*"росс" + 0.014*"дет" + 0.012*"уб" + 0.012*"последн" + 0.012*"ид" + 0.011*"решен"
    INFO : topic #19 (0.050): 0.038*"rt" + 0.038*"крым" + 0.027*"днр" + 0.024*"помощ" + 0.023*"украинск" + 0.022*"политик" + 0.020*"украин" + 0.019*"воен" + 0.018*"фот" + 0.017*"чуж"
    INFO : topic #17 (0.050): 0.036*"украин" + 0.030*"мнен" + 0.028*"российск" + 0.027*"rt" + 0.026*"арм" + 0.025*"выбор" + 0.024*"донбасс" + 0.022*"реш" + 0.021*"газ" + 0.021*"критичн"
    INFO : topic #4 (0.050): 0.051*"rt" + 0.024*"появ" + 0.023*"закон" + 0.022*"интересн" + 0.016*"украин" + 0.016*"европ" + 0.012*"трамп" + 0.012*"границ" + 0.012*"нужн" + 0.012*"отказа"
    INFO : topic #7 (0.050): 0.088*"rt" + 0.023*"человек" + 0.015*"област" + 0.014*"росс" + 0.012*"новосибирск" + 0.012*"дня" + 0.012*"люб" + 0.011*"столкновен" + 0.011*"фильм" + 0.010*"музык"
    INFO : topic diff=0.017572, rho=0.128037
    INFO : PROGRESS: pass 0, dispatched chunk #73 = documents up to #7400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 0, dispatched chunk #74 = documents up to #7500/9999, outstanding queue size 8
    INFO : PROGRESS: pass 0, dispatched chunk #75 = documents up to #7600/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #0 (0.050): 0.085*"rt" + 0.033*"сир" + 0.019*"сша" + 0.016*"переговор" + 0.015*"главн" + 0.010*"рад" + 0.010*"минск" + 0.008*"призва" + 0.008*"игр" + 0.007*"вывод"
    INFO : topic #12 (0.050): 0.125*"rt" + 0.019*"украин" + 0.014*"цен" + 0.011*"групп" + 0.011*"счита" + 0.010*"оруж" + 0.010*"сирийск" + 0.009*"соч" + 0.009*"люд" + 0.009*"возможн"
    INFO : topic #17 (0.050): 0.038*"украин" + 0.030*"rt" + 0.027*"мнен" + 0.027*"российск" + 0.024*"донбасс" + 0.023*"выбор" + 0.023*"арм" + 0.023*"газ" + 0.021*"критичн" + 0.020*"реш"
    INFO : topic #4 (0.050): 0.048*"rt" + 0.024*"интересн" + 0.023*"закон" + 0.022*"появ" + 0.017*"украин" + 0.015*"европ" + 0.013*"трамп" + 0.013*"нужн" + 0.012*"границ" + 0.012*"отказа"
    INFO : topic #8 (0.050): 0.061*"rt" + 0.024*"слов" + 0.018*"переп" + 0.017*"росс" + 0.016*"украин" + 0.015*"новост" + 0.013*"миноборон" + 0.013*"рубл" + 0.011*"запрет" + 0.011*"дтп"
    INFO : topic diff=0.015476, rho=0.124035
    INFO : PROGRESS: pass 0, dispatched chunk #76 = documents up to #7700/9999, outstanding queue size 9
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #1 (0.050): 0.041*"стран" + 0.041*"rt" + 0.021*"пострада" + 0.018*"жител" + 0.016*"дом" + 0.014*"днем" + 0.014*"конц" + 0.014*"ноч" + 0.014*"вид" + 0.013*"ожида"
    INFO : topic #18 (0.050): 0.071*"rt" + 0.031*"нов" + 0.023*"санкц" + 0.021*"рф" + 0.017*"росс" + 0.014*"российск" + 0.014*"сам" + 0.012*"западн" + 0.012*"представ" + 0.011*"украин"
    INFO : topic #16 (0.050): 0.079*"rt" + 0.027*"нача" + 0.024*"добр" + 0.023*"как" + 0.022*"чита" + 0.021*"пост" + 0.017*"мир" + 0.015*"национальн" + 0.014*"школ" + 0.014*"потеря"
    INFO : topic #9 (0.050): 0.058*"rt" + 0.025*"президент" + 0.018*"рф" + 0.013*"путин" + 0.013*"говор" + 0.013*"нат" + 0.012*"росс" + 0.011*"евр" + 0.011*"город" + 0.011*"обсуд"
    INFO : topic #13 (0.050): 0.064*"rt" + 0.026*"украин" + 0.021*"перв" + 0.021*"готов" + 0.019*"русск" + 0.016*"власт" + 0.014*"прав" + 0.013*"террорист" + 0.013*"сми" + 0.013*"репост"
    INFO : topic diff=0.012103, rho=0.120386
    INFO : PROGRESS: pass 0, dispatched chunk #77 = documents up to #7800/9999, outstanding queue size 5
    INFO : PROGRESS: pass 0, dispatched chunk #78 = documents up to #7900/9999, outstanding queue size 3
    INFO : PROGRESS: pass 0, dispatched chunk #79 = documents up to #8000/9999, outstanding queue size 4
    INFO : PROGRESS: pass 0, dispatched chunk #80 = documents up to #8100/9999, outstanding queue size 5
    INFO : PROGRESS: pass 0, dispatched chunk #81 = documents up to #8200/9999, outstanding queue size 6
    INFO : PROGRESS: pass 0, dispatched chunk #82 = documents up to #8300/9999, outstanding queue size 7
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #13 (0.050): 0.064*"rt" + 0.026*"украин" + 0.021*"перв" + 0.020*"готов" + 0.020*"русск" + 0.018*"террорист" + 0.017*"власт" + 0.013*"лет" + 0.013*"прав" + 0.012*"похож"
    INFO : topic #10 (0.050): 0.093*"rt" + 0.034*"виде" + 0.026*"лучш" + 0.022*"полиц" + 0.021*"дет" + 0.014*"последн" + 0.012*"уб" + 0.011*"дела" + 0.011*"росс" + 0.011*"причин"
    INFO : topic #15 (0.050): 0.050*"rt" + 0.031*"quot" + 0.023*"дорог" + 0.018*"сто" + 0.017*"памятник" + 0.015*"октябр" + 0.012*"выход" + 0.012*"челябинск" + 0.011*"факт" + 0.011*"узна"
    INFO : topic #18 (0.050): 0.070*"rt" + 0.031*"нов" + 0.024*"санкц" + 0.022*"рф" + 0.017*"росс" + 0.015*"сам" + 0.014*"российск" + 0.014*"иг" + 0.012*"представ" + 0.012*"намер"
    INFO : topic #19 (0.050): 0.040*"rt" + 0.034*"крым" + 0.026*"ес" + 0.026*"воен" + 0.026*"украинск" + 0.025*"политик" + 0.024*"днр" + 0.022*"помощ" + 0.020*"чуж" + 0.019*"фот"
    INFO : topic diff=0.013611, rho=0.116248
    INFO : PROGRESS: pass 0, dispatched chunk #83 = documents up to #8400/9999, outstanding queue size 7
    INFO : PROGRESS: pass 0, dispatched chunk #84 = documents up to #8500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 0, dispatched chunk #85 = documents up to #8600/9999, outstanding queue size 6
    INFO : PROGRESS: pass 0, dispatched chunk #86 = documents up to #8700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 0, dispatched chunk #87 = documents up to #8800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 0, dispatched chunk #88 = documents up to #8900/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #17 (0.050): 0.039*"украин" + 0.032*"мнен" + 0.032*"российск" + 0.032*"rt" + 0.027*"арм" + 0.026*"побед" + 0.023*"выбор" + 0.022*"донбасс" + 0.022*"критичн" + 0.022*"газ"
    INFO : topic #10 (0.050): 0.090*"rt" + 0.037*"виде" + 0.025*"лучш" + 0.022*"дет" + 0.021*"полиц" + 0.014*"дела" + 0.013*"последн" + 0.012*"причин" + 0.012*"росс" + 0.011*"подозрева"
    INFO : topic #7 (0.050): 0.086*"rt" + 0.019*"человек" + 0.015*"област" + 0.015*"дня" + 0.014*"пройдет" + 0.013*"росс" + 0.012*"фильм" + 0.012*"люб" + 0.010*"суд" + 0.010*"март"
    INFO : topic #6 (0.050): 0.051*"путин" + 0.050*"rt" + 0.042*"росс" + 0.022*"владимир" + 0.018*"мест" + 0.017*"улиц" + 0.017*"сборн" + 0.013*"александр" + 0.013*"друз" + 0.012*"здан"
    INFO : topic #14 (0.050): 0.086*"rt" + 0.030*"украин" + 0.023*"вер" + 0.021*"петербург" + 0.017*"медвед" + 0.015*"новост" + 0.014*"матч" + 0.013*"пожар" + 0.013*"ки" + 0.012*"восток"
    INFO : topic diff=0.016348, rho=0.113228
    INFO : PROGRESS: pass 0, dispatched chunk #89 = documents up to #9000/9999, outstanding queue size 9
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #12 (0.050): 0.135*"rt" + 0.018*"украин" + 0.012*"люд" + 0.011*"убийств" + 0.011*"сирийск" + 0.011*"сша" + 0.011*"цен" + 0.011*"дума" + 0.010*"групп" + 0.010*"счита"
    INFO : topic #7 (0.050): 0.086*"rt" + 0.018*"человек" + 0.016*"пройдет" + 0.015*"дня" + 0.014*"област" + 0.012*"росс" + 0.012*"люб" + 0.012*"фильм" + 0.011*"март" + 0.011*"отношен"
    INFO : topic #5 (0.050): 0.074*"rt" + 0.019*"задержа" + 0.016*"войн" + 0.015*"смотр" + 0.013*"росс" + 0.013*"россиян" + 0.013*"предлага" + 0.012*"план" + 0.011*"сотрудник" + 0.011*"ситуац"
    INFO : topic #9 (0.050): 0.059*"rt" + 0.030*"президент" + 0.022*"рф" + 0.015*"город" + 0.014*"путин" + 0.014*"говор" + 0.013*"нат" + 0.012*"эксперт" + 0.011*"росс" + 0.011*"евр"
    INFO : topic #14 (0.050): 0.084*"rt" + 0.030*"украин" + 0.023*"вер" + 0.021*"петербург" + 0.018*"медвед" + 0.015*"новост" + 0.014*"матч" + 0.014*"пожар" + 0.012*"восток" + 0.012*"ки"
    INFO : topic diff=0.013406, rho=0.110432
    INFO : PROGRESS: pass 0, dispatched chunk #90 = documents up to #9100/9999, outstanding queue size 4
    INFO : PROGRESS: pass 0, dispatched chunk #91 = documents up to #9200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 0, dispatched chunk #92 = documents up to #9300/9999, outstanding queue size 3
    INFO : PROGRESS: pass 0, dispatched chunk #93 = documents up to #9400/9999, outstanding queue size 4
    INFO : PROGRESS: pass 0, dispatched chunk #94 = documents up to #9500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 0, dispatched chunk #95 = documents up to #9600/9999, outstanding queue size 6
    INFO : PROGRESS: pass 0, dispatched chunk #96 = documents up to #9700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 0, dispatched chunk #97 = documents up to #9800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 0, dispatched chunk #98 = documents up to #9900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 0, dispatched chunk #99 = documents up to #9999/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #2 (0.050): 0.052*"rt" + 0.029*"киев" + 0.017*"глав" + 0.017*"пыта" + 0.016*"жизн" + 0.012*"сет" + 0.012*"погибл" + 0.012*"взрыв" + 0.011*"нашл" + 0.011*"международн"
    INFO : topic #8 (0.050): 0.065*"rt" + 0.025*"слов" + 0.025*"рубл" + 0.017*"получ" + 0.017*"росс" + 0.016*"сторон" + 0.015*"переп" + 0.015*"украин" + 0.014*"миноборон" + 0.012*"новост"
    INFO : topic #1 (0.050): 0.055*"стран" + 0.045*"rt" + 0.019*"дом" + 0.019*"жител" + 0.019*"пострада" + 0.016*"стал" + 0.016*"конц" + 0.014*"днем" + 0.012*"массов" + 0.011*"сем"
    INFO : topic #17 (0.050): 0.039*"украин" + 0.035*"мнен" + 0.034*"российск" + 0.033*"rt" + 0.027*"арм" + 0.026*"побед" + 0.025*"выбор" + 0.022*"критичн" + 0.022*"донбасс" + 0.019*"газ"
    INFO : topic #18 (0.050): 0.075*"rt" + 0.033*"нов" + 0.026*"санкц" + 0.024*"рф" + 0.017*"росс" + 0.016*"российск" + 0.016*"сам" + 0.013*"представ" + 0.013*"иг" + 0.011*"мчс"
    INFO : topic diff=0.015264, rho=0.106600
    INFO : -16.112 per-word bound, 70827.9 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : merging changes from 899 documents into a model of 9999 documents
    INFO : topic #9 (0.050): 0.060*"rt" + 0.030*"президент" + 0.022*"рф" + 0.015*"путин" + 0.014*"город" + 0.013*"говор" + 0.012*"уф" + 0.012*"нат" + 0.011*"росс" + 0.010*"эксперт"
    INFO : topic #10 (0.050): 0.088*"rt" + 0.037*"виде" + 0.025*"дет" + 0.024*"лучш" + 0.022*"полиц" + 0.014*"решен" + 0.013*"дела" + 0.013*"хочет" + 0.012*"бизнес" + 0.012*"причин"
    INFO : topic #5 (0.050): 0.072*"rt" + 0.018*"войн" + 0.016*"задержа" + 0.014*"ситуац" + 0.013*"предлага" + 0.013*"смотр" + 0.012*"россиян" + 0.012*"росс" + 0.012*"план" + 0.010*"сотрудник"
    INFO : topic #13 (0.050): 0.070*"rt" + 0.024*"перв" + 0.023*"украин" + 0.022*"готов" + 0.019*"русск" + 0.017*"власт" + 0.014*"прав" + 0.014*"террорист" + 0.014*"сми" + 0.013*"лет"
    INFO : topic #7 (0.050): 0.086*"rt" + 0.017*"област" + 0.016*"человек" + 0.015*"дня" + 0.015*"пройдет" + 0.014*"люб" + 0.013*"фильм" + 0.013*"март" + 0.012*"росс" + 0.011*"отношен"
    INFO : topic diff=0.010787, rho=0.104257
    INFO : -16.020 per-word bound, 66451.3 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : PROGRESS: pass 1, dispatched chunk #0 = documents up to #100/9999, outstanding queue size 1
    INFO : PROGRESS: pass 1, dispatched chunk #1 = documents up to #200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 1, dispatched chunk #2 = documents up to #300/9999, outstanding queue size 3
    INFO : PROGRESS: pass 1, dispatched chunk #3 = documents up to #400/9999, outstanding queue size 4
    INFO : PROGRESS: pass 1, dispatched chunk #4 = documents up to #500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 1, dispatched chunk #5 = documents up to #600/9999, outstanding queue size 6
    INFO : PROGRESS: pass 1, dispatched chunk #6 = documents up to #700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 1, dispatched chunk #7 = documents up to #800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 1, dispatched chunk #8 = documents up to #900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 1, dispatched chunk #9 = documents up to #1000/9999, outstanding queue size 10
    INFO : PROGRESS: pass 1, dispatched chunk #10 = documents up to #1100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 1, dispatched chunk #11 = documents up to #1200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 1, dispatched chunk #12 = documents up to #1300/9999, outstanding queue size 10
    INFO : PROGRESS: pass 1, dispatched chunk #13 = documents up to #1400/9999, outstanding queue size 11
    INFO : merging changes from 800 documents into a model of 9999 documents
    INFO : topic #3 (0.050): 0.055*"rt" + 0.031*"москв" + 0.027*"донецк" + 0.015*"район" + 0.014*"прошл" + 0.011*"украин" + 0.010*"продаж" + 0.010*"силовик" + 0.010*"мэр" + 0.009*"машин"
    INFO : topic #1 (0.050): 0.052*"стран" + 0.042*"rt" + 0.018*"дом" + 0.018*"жител" + 0.017*"пострада" + 0.017*"стал" + 0.015*"конц" + 0.013*"массов" + 0.012*"днем" + 0.012*"сем"
    INFO : topic #14 (0.050): 0.083*"rt" + 0.026*"украин" + 0.022*"петербург" + 0.020*"медвед" + 0.020*"вер" + 0.014*"новост" + 0.012*"матч" + 0.012*"пожар" + 0.011*"ки" + 0.010*"восток"
    INFO : topic #2 (0.050): 0.049*"rt" + 0.027*"киев" + 0.021*"глав" + 0.017*"жизн" + 0.015*"пыта" + 0.012*"погибл" + 0.012*"взрыв" + 0.011*"нашл" + 0.010*"международн" + 0.010*"сет"
    INFO : topic #11 (0.050): 0.129*"rt" + 0.015*"автомобил" + 0.012*"депутат" + 0.012*"хорош" + 0.012*"истор" + 0.011*"обам" + 0.010*"поддержива" + 0.009*"час" + 0.009*"водител" + 0.009*"праздник"
    INFO : topic diff=0.049256, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #14 = documents up to #1500/9999, outstanding queue size 7
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #3 (0.050): 0.052*"rt" + 0.032*"москв" + 0.026*"донецк" + 0.018*"район" + 0.014*"украин" + 0.012*"прошл" + 0.011*"продаж" + 0.010*"машин" + 0.010*"мэр" + 0.009*"силовик"
    INFO : topic #14 (0.050): 0.081*"rt" + 0.024*"украин" + 0.022*"петербург" + 0.019*"вер" + 0.018*"медвед" + 0.013*"пожар" + 0.012*"новост" + 0.012*"матч" + 0.009*"ки" + 0.009*"сообщ"
    INFO : topic #19 (0.050): 0.039*"rt" + 0.034*"ес" + 0.028*"политик" + 0.028*"крым" + 0.025*"чуж" + 0.023*"днр" + 0.023*"воен" + 0.021*"украинск" + 0.020*"помощ" + 0.019*"фот"
    INFO : topic #2 (0.050): 0.045*"rt" + 0.026*"киев" + 0.020*"глав" + 0.020*"жизн" + 0.015*"пыта" + 0.013*"взрыв" + 0.012*"нашл" + 0.011*"погибл" + 0.010*"кита" + 0.010*"международн"
    INFO : topic #13 (0.050): 0.065*"rt" + 0.023*"украин" + 0.021*"перв" + 0.020*"готов" + 0.017*"власт" + 0.016*"русск" + 0.015*"прав" + 0.015*"сми" + 0.012*"террорист" + 0.012*"лет"
    INFO : topic diff=0.043767, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #15 = documents up to #1600/9999, outstanding queue size 4
    INFO : PROGRESS: pass 1, dispatched chunk #16 = documents up to #1700/9999, outstanding queue size 2
    INFO : PROGRESS: pass 1, dispatched chunk #17 = documents up to #1800/9999, outstanding queue size 3
    INFO : PROGRESS: pass 1, dispatched chunk #18 = documents up to #1900/9999, outstanding queue size 4
    INFO : PROGRESS: pass 1, dispatched chunk #19 = documents up to #2000/9999, outstanding queue size 5
    INFO : PROGRESS: pass 1, dispatched chunk #20 = documents up to #2100/9999, outstanding queue size 6
    INFO : PROGRESS: pass 1, dispatched chunk #21 = documents up to #2200/9999, outstanding queue size 7
    INFO : PROGRESS: pass 1, dispatched chunk #22 = documents up to #2300/9999, outstanding queue size 8
    INFO : PROGRESS: pass 1, dispatched chunk #23 = documents up to #2400/9999, outstanding queue size 9
    INFO : PROGRESS: pass 1, dispatched chunk #24 = documents up to #2500/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.072*"rt" + 0.032*"нов" + 0.025*"санкц" + 0.022*"рф" + 0.017*"росс" + 0.015*"российск" + 0.014*"сам" + 0.012*"мид" + 0.012*"украин" + 0.011*"представ"
    INFO : topic #12 (0.050): 0.127*"rt" + 0.016*"украин" + 0.014*"счита" + 0.011*"возможн" + 0.011*"цен" + 0.011*"дума" + 0.010*"сша" + 0.010*"люд" + 0.010*"росс" + 0.009*"кин"
    INFO : topic #19 (0.050): 0.040*"rt" + 0.036*"ес" + 0.029*"крым" + 0.028*"политик" + 0.025*"украинск" + 0.024*"чуж" + 0.023*"воен" + 0.022*"днр" + 0.020*"помощ" + 0.020*"фот"
    INFO : topic #0 (0.050): 0.084*"rt" + 0.031*"сир" + 0.020*"сша" + 0.014*"главн" + 0.011*"переговор" + 0.011*"рад" + 0.009*"призва" + 0.009*"минск" + 0.008*"женщин" + 0.008*"игр"
    INFO : topic #7 (0.050): 0.078*"rt" + 0.017*"област" + 0.015*"человек" + 0.012*"март" + 0.012*"пройдет" + 0.011*"фильм" + 0.011*"дня" + 0.011*"люб" + 0.010*"лидер" + 0.010*"отношен"
    INFO : topic diff=0.014896, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #25 = documents up to #2600/9999, outstanding queue size 10
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #5 (0.050): 0.072*"rt" + 0.017*"войн" + 0.015*"украинц" + 0.015*"задержа" + 0.013*"россиян" + 0.012*"ситуац" + 0.011*"смотр" + 0.011*"росс" + 0.010*"предлага" + 0.010*"отправ"
    INFO : topic #19 (0.050): 0.041*"rt" + 0.037*"ес" + 0.031*"крым" + 0.028*"политик" + 0.027*"чуж" + 0.026*"украинск" + 0.024*"воен" + 0.021*"днр" + 0.020*"фот" + 0.019*"украин"
    INFO : topic #17 (0.050): 0.040*"украин" + 0.038*"российск" + 0.033*"мнен" + 0.032*"rt" + 0.028*"выбор" + 0.023*"побед" + 0.021*"арм" + 0.020*"критичн" + 0.018*"газ" + 0.017*"достойн"
    INFO : topic #10 (0.050): 0.084*"rt" + 0.041*"виде" + 0.021*"полиц" + 0.020*"дет" + 0.017*"лучш" + 0.013*"решен" + 0.012*"уб" + 0.011*"метр" + 0.011*"причин" + 0.011*"последн"
    INFO : topic #0 (0.050): 0.085*"rt" + 0.032*"сир" + 0.020*"сша" + 0.014*"главн" + 0.012*"рад" + 0.010*"переговор" + 0.010*"игр" + 0.009*"призва" + 0.009*"увелич" + 0.009*"минск"
    INFO : topic diff=0.012023, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #26 = documents up to #2700/9999, outstanding queue size 6
    INFO : PROGRESS: pass 1, dispatched chunk #27 = documents up to #2800/9999, outstanding queue size 6
    INFO : PROGRESS: pass 1, dispatched chunk #28 = documents up to #2900/9999, outstanding queue size 5
    INFO : PROGRESS: pass 1, dispatched chunk #29 = documents up to #3000/9999, outstanding queue size 6
    INFO : PROGRESS: pass 1, dispatched chunk #30 = documents up to #3100/9999, outstanding queue size 7
    INFO : PROGRESS: pass 1, dispatched chunk #31 = documents up to #3200/9999, outstanding queue size 8
    INFO : PROGRESS: pass 1, dispatched chunk #32 = documents up to #3300/9999, outstanding queue size 9
    INFO : PROGRESS: pass 1, dispatched chunk #33 = documents up to #3400/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #8 (0.050): 0.059*"rt" + 0.024*"рубл" + 0.024*"слов" + 0.018*"сторон" + 0.017*"миноборон" + 0.017*"росс" + 0.016*"получ" + 0.014*"дтп" + 0.014*"украин" + 0.014*"запрет"
    INFO : topic #11 (0.050): 0.132*"rt" + 0.017*"автомобил" + 0.013*"хорош" + 0.012*"поддержива" + 0.011*"депутат" + 0.010*"привет" + 0.010*"числ" + 0.010*"истор" + 0.010*"обам" + 0.009*"совет"
    INFO : topic #9 (0.050): 0.053*"rt" + 0.033*"президент" + 0.019*"рф" + 0.015*"город" + 0.014*"говор" + 0.013*"путин" + 0.012*"росс" + 0.011*"нат" + 0.011*"обсуд" + 0.010*"уф"
    INFO : topic #18 (0.050): 0.074*"rt" + 0.028*"нов" + 0.027*"санкц" + 0.025*"рф" + 0.018*"росс" + 0.016*"сам" + 0.013*"мид" + 0.013*"российск" + 0.011*"украин" + 0.011*"акц"
    INFO : topic #17 (0.050): 0.041*"украин" + 0.039*"российск" + 0.034*"мнен" + 0.032*"rt" + 0.028*"выбор" + 0.025*"побед" + 0.022*"арм" + 0.018*"достойн" + 0.018*"критичн" + 0.018*"донбасс"
    INFO : topic diff=0.012329, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #34 = documents up to #3500/9999, outstanding queue size 10
    INFO : PROGRESS: pass 1, dispatched chunk #35 = documents up to #3600/9999, outstanding queue size 8
    INFO : PROGRESS: pass 1, dispatched chunk #36 = documents up to #3700/9999, outstanding queue size 9
    INFO : PROGRESS: pass 1, dispatched chunk #37 = documents up to #3800/9999, outstanding queue size 10
    INFO : PROGRESS: pass 1, dispatched chunk #38 = documents up to #3900/9999, outstanding queue size 11
    INFO : PROGRESS: pass 1, dispatched chunk #39 = documents up to #4000/9999, outstanding queue size 12
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #4 (0.050): 0.047*"rt" + 0.027*"интересн" + 0.026*"появ" + 0.016*"европ" + 0.015*"закон" + 0.015*"границ" + 0.015*"украин" + 0.013*"очередн" + 0.013*"развит" + 0.012*"трамп"
    INFO : topic #19 (0.050): 0.041*"rt" + 0.035*"крым" + 0.033*"ес" + 0.030*"политик" + 0.028*"чуж" + 0.027*"украинск" + 0.024*"воен" + 0.023*"днр" + 0.020*"украин" + 0.020*"фот"
    INFO : topic #11 (0.050): 0.134*"rt" + 0.016*"автомобил" + 0.013*"депутат" + 0.013*"поддержива" + 0.012*"хорош" + 0.011*"ставропол" + 0.010*"совет" + 0.010*"связ" + 0.010*"привет" + 0.010*"числ"
    INFO : topic #15 (0.050): 0.051*"rt" + 0.030*"quot" + 0.022*"дорог" + 0.021*"сто" + 0.014*"факт" + 0.012*"выход" + 0.012*"узна" + 0.011*"рук" + 0.011*"предлож" + 0.011*"спорт"
    INFO : topic #3 (0.050): 0.059*"rt" + 0.031*"москв" + 0.028*"донецк" + 0.016*"район" + 0.016*"машин" + 0.014*"мэр" + 0.012*"дан" + 0.011*"украин" + 0.011*"прошл" + 0.011*"строительств"
    INFO : topic diff=0.012950, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #40 = documents up to #4100/9999, outstanding queue size 11
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #3 (0.050): 0.057*"rt" + 0.030*"москв" + 0.028*"донецк" + 0.017*"район" + 0.015*"машин" + 0.014*"прошл" + 0.014*"мэр" + 0.011*"дан" + 0.011*"украин" + 0.011*"продаж"
    INFO : topic #19 (0.050): 0.042*"rt" + 0.035*"крым" + 0.032*"ес" + 0.030*"политик" + 0.027*"украинск" + 0.027*"чуж" + 0.024*"днр" + 0.024*"воен" + 0.022*"украин" + 0.021*"помощ"
    INFO : topic #4 (0.050): 0.046*"rt" + 0.028*"появ" + 0.027*"интересн" + 0.017*"европ" + 0.016*"закон" + 0.015*"украин" + 0.015*"границ" + 0.013*"отказа" + 0.012*"очередн" + 0.012*"развит"
    INFO : topic #7 (0.050): 0.080*"rt" + 0.019*"человек" + 0.016*"област" + 0.015*"дня" + 0.013*"пройдет" + 0.013*"новосибирск" + 0.012*"март" + 0.011*"фильм" + 0.011*"лидер" + 0.011*"парт"
    INFO : topic #11 (0.050): 0.136*"rt" + 0.016*"автомобил" + 0.013*"поддержива" + 0.012*"хорош" + 0.011*"депутат" + 0.011*"связ" + 0.011*"привет" + 0.010*"ставропол" + 0.010*"истор" + 0.009*"совет"
    INFO : topic diff=0.012639, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #41 = documents up to #4200/9999, outstanding queue size 6
    INFO : PROGRESS: pass 1, dispatched chunk #42 = documents up to #4300/9999, outstanding queue size 6
    INFO : PROGRESS: pass 1, dispatched chunk #43 = documents up to #4400/9999, outstanding queue size 7
    INFO : PROGRESS: pass 1, dispatched chunk #44 = documents up to #4500/9999, outstanding queue size 8
    INFO : PROGRESS: pass 1, dispatched chunk #45 = documents up to #4600/9999, outstanding queue size 9
    INFO : PROGRESS: pass 1, dispatched chunk #46 = documents up to #4700/9999, outstanding queue size 10
    INFO : PROGRESS: pass 1, dispatched chunk #47 = documents up to #4800/9999, outstanding queue size 11
    INFO : PROGRESS: pass 1, dispatched chunk #48 = documents up to #4900/9999, outstanding queue size 12
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #3 (0.050): 0.057*"rt" + 0.032*"москв" + 0.027*"донецк" + 0.015*"район" + 0.015*"машин" + 0.015*"прошл" + 0.013*"мэр" + 0.011*"продаж" + 0.010*"дан" + 0.010*"украин"
    INFO : topic #14 (0.050): 0.084*"rt" + 0.026*"украин" + 0.026*"петербург" + 0.024*"вер" + 0.017*"новост" + 0.016*"медвед" + 0.015*"пожар" + 0.012*"матч" + 0.011*"восток" + 0.011*"ки"
    INFO : topic #0 (0.050): 0.086*"rt" + 0.034*"сир" + 0.019*"сша" + 0.016*"главн" + 0.014*"рад" + 0.011*"переговор" + 0.010*"призва" + 0.009*"минск" + 0.009*"игр" + 0.009*"увелич"
    INFO : topic #19 (0.050): 0.044*"rt" + 0.038*"крым" + 0.030*"украинск" + 0.030*"ес" + 0.029*"политик" + 0.025*"чуж" + 0.024*"днр" + 0.023*"воен" + 0.021*"сша" + 0.021*"украин"
    INFO : topic #8 (0.050): 0.061*"rt" + 0.027*"рубл" + 0.023*"слов" + 0.018*"росс" + 0.018*"миноборон" + 0.018*"сторон" + 0.016*"получ" + 0.016*"запрет" + 0.013*"дтп" + 0.013*"украин"
    INFO : topic diff=0.012617, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #49 = documents up to #5000/9999, outstanding queue size 9
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #1 (0.050): 0.054*"стран" + 0.043*"rt" + 0.023*"дом" + 0.022*"жител" + 0.020*"пострада" + 0.015*"стал" + 0.013*"днем" + 0.013*"сем" + 0.012*"конц" + 0.011*"массов"
    INFO : topic #6 (0.050): 0.071*"путин" + 0.053*"rt" + 0.040*"росс" + 0.027*"владимир" + 0.022*"мест" + 0.016*"улиц" + 0.014*"назва" + 0.014*"друз" + 0.013*"работ" + 0.012*"александр"
    INFO : topic #18 (0.050): 0.076*"rt" + 0.031*"нов" + 0.028*"санкц" + 0.027*"рф" + 0.018*"сам" + 0.017*"росс" + 0.013*"российск" + 0.013*"мид" + 0.012*"акц" + 0.011*"представ"
    INFO : topic #3 (0.050): 0.059*"rt" + 0.033*"москв" + 0.028*"донецк" + 0.018*"район" + 0.015*"прошл" + 0.015*"машин" + 0.013*"мэр" + 0.012*"продаж" + 0.010*"дан" + 0.010*"украин"
    INFO : topic #12 (0.050): 0.141*"rt" + 0.017*"украин" + 0.015*"люд" + 0.014*"счита" + 0.013*"цен" + 0.012*"возможн" + 0.012*"дума" + 0.011*"сша" + 0.010*"автор" + 0.010*"оруж"
    INFO : topic diff=0.010315, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #50 = documents up to #5100/9999, outstanding queue size 4
    INFO : PROGRESS: pass 1, dispatched chunk #51 = documents up to #5200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 1, dispatched chunk #52 = documents up to #5300/9999, outstanding queue size 3
    INFO : PROGRESS: pass 1, dispatched chunk #53 = documents up to #5400/9999, outstanding queue size 4
    INFO : PROGRESS: pass 1, dispatched chunk #54 = documents up to #5500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 1, dispatched chunk #55 = documents up to #5600/9999, outstanding queue size 6
    INFO : PROGRESS: pass 1, dispatched chunk #56 = documents up to #5700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 1, dispatched chunk #57 = documents up to #5800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 1, dispatched chunk #58 = documents up to #5900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 1, dispatched chunk #59 = documents up to #6000/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #13 (0.050): 0.074*"rt" + 0.030*"готов" + 0.025*"украин" + 0.025*"власт" + 0.022*"перв" + 0.018*"русск" + 0.018*"прав" + 0.015*"сми" + 0.011*"лет" + 0.011*"террорист"
    INFO : topic #4 (0.050): 0.048*"rt" + 0.029*"интересн" + 0.026*"появ" + 0.019*"закон" + 0.017*"европ" + 0.016*"границ" + 0.015*"отказа" + 0.013*"трамп" + 0.013*"украин" + 0.013*"очередн"
    INFO : topic #6 (0.050): 0.074*"путин" + 0.052*"rt" + 0.041*"росс" + 0.027*"владимир" + 0.022*"мест" + 0.016*"улиц" + 0.015*"друз" + 0.015*"назва" + 0.012*"работ" + 0.012*"александр"
    INFO : topic #14 (0.050): 0.087*"rt" + 0.029*"петербург" + 0.027*"украин" + 0.025*"вер" + 0.016*"новост" + 0.015*"медвед" + 0.014*"пожар" + 0.014*"матч" + 0.013*"ки" + 0.013*"восток"
    INFO : topic #5 (0.050): 0.071*"rt" + 0.016*"задержа" + 0.016*"войн" + 0.014*"сотрудник" + 0.013*"смотр" + 0.013*"ситуац" + 0.013*"украинц" + 0.013*"россиян" + 0.013*"отправ" + 0.012*"стат"
    INFO : topic diff=0.012872, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #60 = documents up to #6100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 1, dispatched chunk #61 = documents up to #6200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 1, dispatched chunk #62 = documents up to #6300/9999, outstanding queue size 11
    INFO : PROGRESS: pass 1, dispatched chunk #63 = documents up to #6400/9999, outstanding queue size 12
    INFO : PROGRESS: pass 1, dispatched chunk #64 = documents up to #6500/9999, outstanding queue size 13
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #6 (0.050): 0.081*"путин" + 0.054*"rt" + 0.041*"росс" + 0.026*"владимир" + 0.022*"мест" + 0.015*"улиц" + 0.015*"друз" + 0.014*"жил" + 0.013*"назва" + 0.012*"ход"
    INFO : topic #3 (0.050): 0.057*"rt" + 0.031*"донецк" + 0.030*"москв" + 0.017*"район" + 0.016*"продаж" + 0.015*"машин" + 0.013*"прошл" + 0.013*"мэр" + 0.011*"силовик" + 0.010*"украин"
    INFO : topic #12 (0.050): 0.139*"rt" + 0.016*"украин" + 0.016*"счита" + 0.014*"люд" + 0.013*"дума" + 0.013*"цен" + 0.012*"возможн" + 0.012*"оруж" + 0.012*"автор" + 0.011*"сша"
    INFO : topic #7 (0.050): 0.083*"rt" + 0.025*"человек" + 0.016*"област" + 0.014*"дня" + 0.014*"март" + 0.012*"новосибирск" + 0.012*"суд" + 0.012*"пройдет" + 0.011*"росс" + 0.011*"люб"
    INFO : topic #0 (0.050): 0.083*"rt" + 0.038*"сир" + 0.019*"главн" + 0.016*"сша" + 0.015*"рад" + 0.013*"переговор" + 0.010*"минск" + 0.010*"призва" + 0.009*"игр" + 0.009*"удар"
    INFO : topic diff=0.014680, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #65 = documents up to #6600/9999, outstanding queue size 11
    INFO : PROGRESS: pass 1, dispatched chunk #66 = documents up to #6700/9999, outstanding queue size 10
    INFO : PROGRESS: pass 1, dispatched chunk #67 = documents up to #6800/9999, outstanding queue size 11
    INFO : PROGRESS: pass 1, dispatched chunk #68 = documents up to #6900/9999, outstanding queue size 11
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #5 (0.050): 0.068*"rt" + 0.016*"задержа" + 0.016*"войн" + 0.014*"россиян" + 0.014*"смотр" + 0.014*"сотрудник" + 0.014*"украинц" + 0.013*"предлага" + 0.013*"ситуац" + 0.012*"план"
    INFO : topic #10 (0.050): 0.089*"rt" + 0.041*"виде" + 0.024*"полиц" + 0.023*"дет" + 0.020*"лучш" + 0.014*"последн" + 0.014*"решен" + 0.012*"бизнес" + 0.012*"дела" + 0.012*"уб"
    INFO : topic #9 (0.050): 0.056*"rt" + 0.037*"президент" + 0.023*"рф" + 0.013*"город" + 0.013*"сентябр" + 0.013*"росс" + 0.013*"нат" + 0.013*"путин" + 0.013*"эксперт" + 0.012*"говор"
    INFO : topic #18 (0.050): 0.076*"rt" + 0.038*"нов" + 0.027*"санкц" + 0.025*"рф" + 0.018*"росс" + 0.017*"сам" + 0.012*"мчс" + 0.012*"акц" + 0.012*"российск" + 0.011*"украин"
    INFO : topic #3 (0.050): 0.058*"rt" + 0.032*"донецк" + 0.028*"москв" + 0.016*"район" + 0.014*"продаж" + 0.014*"машин" + 0.013*"мэр" + 0.013*"прошл" + 0.011*"силовик" + 0.010*"украин"
    INFO : topic diff=0.014857, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #69 = documents up to #7000/9999, outstanding queue size 9
    INFO : PROGRESS: pass 1, dispatched chunk #70 = documents up to #7100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 1, dispatched chunk #71 = documents up to #7200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 1, dispatched chunk #72 = documents up to #7300/9999, outstanding queue size 11
    INFO : PROGRESS: pass 1, dispatched chunk #73 = documents up to #7400/9999, outstanding queue size 12
    INFO : PROGRESS: pass 1, dispatched chunk #74 = documents up to #7500/9999, outstanding queue size 13
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #16 (0.050): 0.069*"rt" + 0.034*"нача" + 0.028*"добр" + 0.025*"чита" + 0.025*"пост" + 0.021*"мир" + 0.021*"как" + 0.020*"написа" + 0.018*"потеря" + 0.016*"школ"
    INFO : topic #2 (0.050): 0.057*"rt" + 0.032*"киев" + 0.023*"глав" + 0.017*"жизн" + 0.017*"взрыв" + 0.016*"пыта" + 0.015*"нашл" + 0.013*"погибл" + 0.012*"кита" + 0.011*"страшн"
    INFO : topic #7 (0.050): 0.082*"rt" + 0.024*"человек" + 0.016*"област" + 0.015*"люб" + 0.014*"дня" + 0.013*"март" + 0.012*"фильм" + 0.012*"пройдет" + 0.011*"новосибирск" + 0.011*"суд"
    INFO : topic #6 (0.050): 0.087*"путин" + 0.055*"rt" + 0.043*"росс" + 0.024*"владимир" + 0.021*"мест" + 0.016*"улиц" + 0.015*"назва" + 0.013*"друз" + 0.013*"сборн" + 0.013*"жил"
    INFO : topic #4 (0.050): 0.044*"rt" + 0.031*"интересн" + 0.025*"появ" + 0.021*"закон" + 0.021*"европ" + 0.018*"границ" + 0.016*"украин" + 0.015*"трамп" + 0.014*"отказа" + 0.012*"турц"
    INFO : topic diff=0.011610, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #75 = documents up to #7600/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #1 (0.050): 0.054*"стран" + 0.042*"rt" + 0.023*"пострада" + 0.020*"дом" + 0.018*"жител" + 0.015*"конц" + 0.014*"днем" + 0.014*"сем" + 0.013*"стал" + 0.013*"ан"
    INFO : topic #7 (0.050): 0.081*"rt" + 0.022*"человек" + 0.017*"люб" + 0.015*"област" + 0.015*"дня" + 0.014*"фильм" + 0.013*"пройдет" + 0.012*"суд" + 0.011*"март" + 0.011*"росс"
    INFO : topic #3 (0.050): 0.056*"rt" + 0.035*"донецк" + 0.029*"москв" + 0.018*"район" + 0.014*"прошл" + 0.013*"продаж" + 0.013*"мэр" + 0.012*"машин" + 0.012*"силовик" + 0.010*"дава"
    INFO : topic #9 (0.050): 0.055*"rt" + 0.034*"президент" + 0.023*"рф" + 0.015*"говор" + 0.014*"нат" + 0.014*"город" + 0.013*"росс" + 0.012*"эксперт" + 0.012*"евр" + 0.011*"путин"
    INFO : topic #2 (0.050): 0.055*"rt" + 0.031*"киев" + 0.022*"глав" + 0.018*"жизн" + 0.018*"взрыв" + 0.017*"пыта" + 0.014*"нашл" + 0.014*"погибл" + 0.011*"международн" + 0.011*"кита"
    INFO : topic diff=0.015462, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #76 = documents up to #7700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 1, dispatched chunk #77 = documents up to #7800/9999, outstanding queue size 5
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #5 (0.050): 0.070*"rt" + 0.020*"войн" + 0.018*"задержа" + 0.016*"смотр" + 0.015*"стат" + 0.014*"план" + 0.014*"россиян" + 0.014*"обстрел" + 0.013*"предлага" + 0.013*"сотрудник"
    INFO : topic #8 (0.050): 0.062*"rt" + 0.026*"рубл" + 0.024*"слов" + 0.019*"сторон" + 0.018*"получ" + 0.016*"переп" + 0.015*"росс" + 0.014*"миноборон" + 0.014*"запрет" + 0.013*"новост"
    INFO : topic #15 (0.050): 0.049*"rt" + 0.032*"quot" + 0.026*"дорог" + 0.024*"сто" + 0.015*"памятник" + 0.015*"октябр" + 0.013*"предлож" + 0.013*"выход" + 0.013*"факт" + 0.013*"рук"
    INFO : topic #11 (0.050): 0.137*"rt" + 0.016*"истор" + 0.016*"хорош" + 0.015*"автомобил" + 0.012*"праздник" + 0.012*"поддержива" + 0.012*"связ" + 0.012*"депутат" + 0.011*"обам" + 0.011*"ал"
    INFO : topic #3 (0.050): 0.057*"rt" + 0.035*"донецк" + 0.031*"москв" + 0.023*"район" + 0.013*"мэр" + 0.013*"прошл" + 0.012*"машин" + 0.011*"продаж" + 0.011*"силовик" + 0.010*"дава"
    INFO : topic diff=0.012032, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #78 = documents up to #7900/9999, outstanding queue size 5
    INFO : PROGRESS: pass 1, dispatched chunk #79 = documents up to #8000/9999, outstanding queue size 5
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #8 (0.050): 0.063*"rt" + 0.027*"рубл" + 0.023*"слов" + 0.019*"сторон" + 0.017*"получ" + 0.017*"росс" + 0.015*"переп" + 0.014*"миноборон" + 0.013*"запрет" + 0.012*"след"
    INFO : topic #13 (0.050): 0.074*"rt" + 0.026*"украин" + 0.026*"готов" + 0.022*"перв" + 0.022*"власт" + 0.021*"русск" + 0.018*"террорист" + 0.018*"сми" + 0.014*"прав" + 0.013*"похож"
    INFO : topic #7 (0.050): 0.081*"rt" + 0.020*"человек" + 0.017*"дня" + 0.016*"люб" + 0.016*"пройдет" + 0.015*"област" + 0.014*"фильм" + 0.012*"март" + 0.012*"суд" + 0.011*"отношен"
    INFO : topic #17 (0.050): 0.052*"российск" + 0.050*"украин" + 0.036*"мнен" + 0.034*"rt" + 0.027*"побед" + 0.026*"арм" + 0.026*"выбор" + 0.023*"донбасс" + 0.022*"газ" + 0.019*"критичн"
    INFO : topic #1 (0.050): 0.057*"стран" + 0.045*"rt" + 0.022*"пострада" + 0.018*"дом" + 0.018*"жител" + 0.015*"конц" + 0.014*"днем" + 0.014*"стал" + 0.014*"ожида" + 0.013*"вид"
    INFO : topic diff=0.012132, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #80 = documents up to #8100/9999, outstanding queue size 2
    INFO : PROGRESS: pass 1, dispatched chunk #81 = documents up to #8200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 1, dispatched chunk #82 = documents up to #8300/9999, outstanding queue size 3
    INFO : PROGRESS: pass 1, dispatched chunk #83 = documents up to #8400/9999, outstanding queue size 4
    INFO : PROGRESS: pass 1, dispatched chunk #84 = documents up to #8500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 1, dispatched chunk #85 = documents up to #8600/9999, outstanding queue size 6
    INFO : PROGRESS: pass 1, dispatched chunk #86 = documents up to #8700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 1, dispatched chunk #87 = documents up to #8800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 1, dispatched chunk #88 = documents up to #8900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 1, dispatched chunk #89 = documents up to #9000/9999, outstanding queue size 9
    INFO : PROGRESS: pass 1, dispatched chunk #90 = documents up to #9100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 1, dispatched chunk #91 = documents up to #9200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 1, dispatched chunk #92 = documents up to #9300/9999, outstanding queue size 11
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #17 (0.050): 0.053*"украин" + 0.051*"российск" + 0.036*"мнен" + 0.035*"rt" + 0.028*"побед" + 0.027*"арм" + 0.024*"выбор" + 0.021*"донбасс" + 0.021*"газ" + 0.020*"критичн"
    INFO : topic #8 (0.050): 0.063*"rt" + 0.029*"рубл" + 0.025*"слов" + 0.018*"сторон" + 0.018*"получ" + 0.016*"росс" + 0.014*"переп" + 0.014*"миноборон" + 0.013*"запрет" + 0.012*"дтп"
    INFO : topic #4 (0.050): 0.044*"rt" + 0.034*"интересн" + 0.025*"появ" + 0.021*"европ" + 0.018*"закон" + 0.016*"украин" + 0.016*"трамп" + 0.015*"границ" + 0.012*"отказа" + 0.012*"турц"
    INFO : topic #7 (0.050): 0.080*"rt" + 0.019*"человек" + 0.016*"дня" + 0.016*"люб" + 0.015*"област" + 0.015*"пройдет" + 0.013*"фильм" + 0.013*"суд" + 0.012*"март" + 0.011*"росс"
    INFO : topic #11 (0.050): 0.141*"rt" + 0.017*"автомобил" + 0.016*"хорош" + 0.015*"истор" + 0.013*"депутат" + 0.012*"праздник" + 0.012*"связ" + 0.012*"поддержива" + 0.012*"обам" + 0.011*"час"
    INFO : topic diff=0.012895, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #93 = documents up to #9400/9999, outstanding queue size 11
    INFO : PROGRESS: pass 1, dispatched chunk #94 = documents up to #9500/9999, outstanding queue size 11
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #4 (0.050): 0.043*"rt" + 0.033*"интересн" + 0.025*"появ" + 0.020*"европ" + 0.017*"границ" + 0.017*"закон" + 0.016*"трамп" + 0.016*"украин" + 0.012*"турц" + 0.011*"отказа"
    INFO : topic #11 (0.050): 0.142*"rt" + 0.016*"автомобил" + 0.015*"хорош" + 0.014*"истор" + 0.013*"обам" + 0.013*"депутат" + 0.011*"праздник" + 0.011*"связ" + 0.011*"поддержива" + 0.010*"привет"
    INFO : topic #14 (0.050): 0.080*"rt" + 0.031*"украин" + 0.026*"петербург" + 0.024*"вер" + 0.020*"медвед" + 0.017*"новост" + 0.016*"матч" + 0.015*"пожар" + 0.015*"восток" + 0.013*"ки"
    INFO : topic #5 (0.050): 0.069*"rt" + 0.023*"войн" + 0.021*"задержа" + 0.015*"смотр" + 0.015*"план" + 0.014*"россиян" + 0.014*"предлага" + 0.014*"обстрел" + 0.012*"украинц" + 0.012*"стат"
    INFO : topic #6 (0.050): 0.096*"путин" + 0.054*"rt" + 0.040*"росс" + 0.023*"владимир" + 0.018*"улиц" + 0.017*"сборн" + 0.017*"мест" + 0.014*"ход" + 0.014*"друз" + 0.013*"работ"
    INFO : topic diff=0.013513, rho=0.099020
    INFO : PROGRESS: pass 1, dispatched chunk #95 = documents up to #9600/9999, outstanding queue size 9
    INFO : PROGRESS: pass 1, dispatched chunk #96 = documents up to #9700/9999, outstanding queue size 9
    INFO : PROGRESS: pass 1, dispatched chunk #97 = documents up to #9800/9999, outstanding queue size 10
    INFO : PROGRESS: pass 1, dispatched chunk #98 = documents up to #9900/9999, outstanding queue size 11
    INFO : PROGRESS: pass 1, dispatched chunk #99 = documents up to #9999/9999, outstanding queue size 11
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #11 (0.050): 0.141*"rt" + 0.016*"автомобил" + 0.014*"хорош" + 0.014*"истор" + 0.014*"обам" + 0.013*"депутат" + 0.012*"поддержива" + 0.012*"ставропол" + 0.012*"час" + 0.010*"служб"
    INFO : topic #4 (0.050): 0.043*"rt" + 0.031*"интересн" + 0.023*"появ" + 0.020*"европ" + 0.019*"границ" + 0.016*"трамп" + 0.016*"закон" + 0.015*"украин" + 0.014*"турц" + 0.013*"очередн"
    INFO : topic #15 (0.050): 0.048*"rt" + 0.033*"quot" + 0.029*"дорог" + 0.022*"сто" + 0.014*"октябр" + 0.014*"узна" + 0.013*"выход" + 0.013*"памятник" + 0.012*"факт" + 0.011*"предлож"
    INFO : topic #19 (0.050): 0.049*"rt" + 0.039*"ес" + 0.035*"украинск" + 0.033*"политик" + 0.030*"крым" + 0.029*"чуж" + 0.027*"воен" + 0.025*"сша" + 0.024*"украин" + 0.024*"помощ"
    INFO : topic #9 (0.050): 0.057*"rt" + 0.041*"президент" + 0.024*"рф" + 0.014*"город" + 0.014*"нат" + 0.013*"говор" + 0.011*"эксперт" + 0.011*"приня" + 0.011*"росс" + 0.011*"уф"
    INFO : topic diff=0.014981, rho=0.099020
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.080*"rt" + 0.045*"нов" + 0.028*"санкц" + 0.028*"рф" + 0.022*"росс" + 0.021*"сам" + 0.015*"представ" + 0.015*"мид" + 0.014*"российск" + 0.014*"акц"
    INFO : topic #12 (0.050): 0.146*"rt" + 0.015*"украин" + 0.015*"счита" + 0.015*"сша" + 0.015*"дума" + 0.014*"возможн" + 0.014*"люд" + 0.012*"цен" + 0.012*"росс" + 0.012*"убийств"
    INFO : topic #7 (0.050): 0.083*"rt" + 0.018*"дня" + 0.016*"человек" + 0.016*"пройдет" + 0.015*"област" + 0.015*"люб" + 0.014*"отношен" + 0.014*"март" + 0.014*"фильм" + 0.012*"лидер"
    INFO : topic #1 (0.050): 0.064*"стран" + 0.044*"rt" + 0.025*"дом" + 0.018*"жител" + 0.018*"пострада" + 0.017*"стал" + 0.016*"сем" + 0.015*"конц" + 0.014*"массов" + 0.013*"днем"
    INFO : topic #15 (0.050): 0.048*"rt" + 0.030*"quot" + 0.028*"дорог" + 0.023*"сто" + 0.014*"выход" + 0.013*"факт" + 0.013*"октябр" + 0.013*"памятник" + 0.013*"узна" + 0.012*"заверш"
    INFO : topic diff=0.013079, rho=0.099020
    INFO : -16.045 per-word bound, 67627.0 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : merging changes from 399 documents into a model of 9999 documents
    INFO : topic #13 (0.050): 0.078*"rt" + 0.026*"перв" + 0.025*"готов" + 0.024*"украин" + 0.022*"сми" + 0.019*"русск" + 0.019*"власт" + 0.017*"прав" + 0.014*"похож" + 0.014*"террорист"
    INFO : topic #9 (0.050): 0.059*"rt" + 0.041*"президент" + 0.023*"рф" + 0.017*"город" + 0.014*"говор" + 0.013*"нат" + 0.013*"уф" + 0.012*"немцов" + 0.011*"эксперт" + 0.011*"росс"
    INFO : topic #18 (0.050): 0.079*"rt" + 0.045*"нов" + 0.029*"санкц" + 0.026*"рф" + 0.023*"росс" + 0.021*"сам" + 0.015*"мид" + 0.015*"представ" + 0.014*"российск" + 0.013*"апрел"
    INFO : topic #0 (0.050): 0.086*"rt" + 0.044*"сир" + 0.023*"главн" + 0.021*"рад" + 0.015*"сша" + 0.014*"переговор" + 0.012*"минск" + 0.012*"удар" + 0.010*"игр" + 0.010*"призва"
    INFO : topic #1 (0.050): 0.064*"стран" + 0.045*"rt" + 0.024*"дом" + 0.019*"жител" + 0.019*"пострада" + 0.018*"стал" + 0.017*"массов" + 0.015*"конц" + 0.015*"сем" + 0.014*"днем"
    INFO : topic diff=0.015173, rho=0.099020
    INFO : -15.901 per-word bound, 61173.2 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : PROGRESS: pass 2, dispatched chunk #0 = documents up to #100/9999, outstanding queue size 1
    INFO : PROGRESS: pass 2, dispatched chunk #1 = documents up to #200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 2, dispatched chunk #2 = documents up to #300/9999, outstanding queue size 3
    INFO : PROGRESS: pass 2, dispatched chunk #3 = documents up to #400/9999, outstanding queue size 4
    INFO : PROGRESS: pass 2, dispatched chunk #4 = documents up to #500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 2, dispatched chunk #5 = documents up to #600/9999, outstanding queue size 6
    INFO : PROGRESS: pass 2, dispatched chunk #6 = documents up to #700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 2, dispatched chunk #7 = documents up to #800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 2, dispatched chunk #8 = documents up to #900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 2, dispatched chunk #9 = documents up to #1000/9999, outstanding queue size 10
    INFO : PROGRESS: pass 2, dispatched chunk #10 = documents up to #1100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 2, dispatched chunk #11 = documents up to #1200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 2, dispatched chunk #12 = documents up to #1300/9999, outstanding queue size 11
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #19 (0.050): 0.049*"rt" + 0.040*"ес" + 0.035*"украинск" + 0.030*"политик" + 0.029*"крым" + 0.027*"чуж" + 0.026*"воен" + 0.024*"украин" + 0.024*"сша" + 0.024*"фот"
    INFO : topic #9 (0.050): 0.059*"rt" + 0.041*"президент" + 0.021*"рф" + 0.017*"город" + 0.014*"нат" + 0.013*"говор" + 0.011*"уф" + 0.011*"немцов" + 0.010*"участ" + 0.010*"обсуд"
    INFO : topic #2 (0.050): 0.052*"rt" + 0.030*"киев" + 0.030*"глав" + 0.021*"жизн" + 0.016*"взрыв" + 0.015*"погибл" + 0.015*"пыта" + 0.012*"кита" + 0.012*"южн" + 0.012*"международн"
    INFO : topic #16 (0.050): 0.066*"rt" + 0.031*"нача" + 0.026*"добр" + 0.023*"пост" + 0.023*"чита" + 0.023*"мир" + 0.022*"как" + 0.019*"написа" + 0.016*"школ" + 0.014*"интернет"
    INFO : topic #17 (0.050): 0.054*"украин" + 0.050*"российск" + 0.039*"rt" + 0.036*"мнен" + 0.030*"выбор" + 0.029*"побед" + 0.023*"арм" + 0.021*"донбасс" + 0.019*"учен" + 0.019*"газ"
    INFO : topic diff=0.054126, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #13 = documents up to #1400/9999, outstanding queue size 9
    INFO : PROGRESS: pass 2, dispatched chunk #14 = documents up to #1500/9999, outstanding queue size 7
    INFO : PROGRESS: pass 2, dispatched chunk #15 = documents up to #1600/9999, outstanding queue size 8
    INFO : PROGRESS: pass 2, dispatched chunk #16 = documents up to #1700/9999, outstanding queue size 9
    INFO : PROGRESS: pass 2, dispatched chunk #17 = documents up to #1800/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #5 (0.050): 0.067*"rt" + 0.023*"войн" + 0.019*"задержа" + 0.015*"ситуац" + 0.014*"обстрел" + 0.013*"украинц" + 0.012*"смотр" + 0.012*"предлага" + 0.012*"план" + 0.012*"стат"
    INFO : topic #15 (0.050): 0.046*"rt" + 0.033*"quot" + 0.025*"дорог" + 0.022*"сто" + 0.015*"факт" + 0.012*"выход" + 0.012*"узна" + 0.012*"постро" + 0.011*"предлож" + 0.011*"заверш"
    INFO : topic #16 (0.050): 0.065*"rt" + 0.029*"нача" + 0.025*"чита" + 0.024*"мир" + 0.023*"добр" + 0.022*"пост" + 0.022*"как" + 0.021*"написа" + 0.015*"школ" + 0.015*"уголовн"
    INFO : topic #3 (0.050): 0.055*"rt" + 0.041*"москв" + 0.032*"донецк" + 0.017*"район" + 0.015*"прошл" + 0.011*"центр" + 0.011*"банк" + 0.011*"машин" + 0.010*"силовик" + 0.010*"област"
    INFO : topic #17 (0.050): 0.051*"украин" + 0.046*"российск" + 0.035*"rt" + 0.034*"мнен" + 0.028*"выбор" + 0.027*"побед" + 0.021*"арм" + 0.020*"донбасс" + 0.018*"учен" + 0.017*"достойн"
    INFO : topic diff=0.042899, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #18 = documents up to #1900/9999, outstanding queue size 10
    INFO : PROGRESS: pass 2, dispatched chunk #19 = documents up to #2000/9999, outstanding queue size 9
    INFO : PROGRESS: pass 2, dispatched chunk #20 = documents up to #2100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 2, dispatched chunk #21 = documents up to #2200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 2, dispatched chunk #22 = documents up to #2300/9999, outstanding queue size 11
    INFO : PROGRESS: pass 2, dispatched chunk #23 = documents up to #2400/9999, outstanding queue size 12
    INFO : PROGRESS: pass 2, dispatched chunk #24 = documents up to #2500/9999, outstanding queue size 13
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #1 (0.050): 0.050*"стран" + 0.041*"rt" + 0.022*"дом" + 0.019*"стал" + 0.018*"жител" + 0.018*"пострада" + 0.015*"массов" + 0.015*"сем" + 0.013*"конц" + 0.012*"днем"
    INFO : topic #13 (0.050): 0.068*"rt" + 0.026*"готов" + 0.024*"украин" + 0.023*"перв" + 0.021*"власт" + 0.020*"сми" + 0.017*"прав" + 0.016*"русск" + 0.013*"дел" + 0.012*"террорист"
    INFO : topic #5 (0.050): 0.066*"rt" + 0.023*"войн" + 0.018*"задержа" + 0.014*"ситуац" + 0.014*"украинц" + 0.013*"обстрел" + 0.012*"план" + 0.011*"стат" + 0.011*"смотр" + 0.011*"россиян"
    INFO : topic #19 (0.050): 0.045*"rt" + 0.041*"ес" + 0.036*"украинск" + 0.032*"политик" + 0.030*"крым" + 0.026*"воен" + 0.025*"украин" + 0.025*"чуж" + 0.024*"сша" + 0.023*"фот"
    INFO : topic #16 (0.050): 0.065*"rt" + 0.032*"нача" + 0.023*"написа" + 0.023*"чита" + 0.022*"добр" + 0.021*"мир" + 0.021*"пост" + 0.021*"как" + 0.015*"потеря" + 0.014*"уголовн"
    INFO : topic diff=0.020243, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #25 = documents up to #2600/9999, outstanding queue size 11
    INFO : PROGRESS: pass 2, dispatched chunk #26 = documents up to #2700/9999, outstanding queue size 11
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #5 (0.050): 0.070*"rt" + 0.024*"войн" + 0.018*"задержа" + 0.016*"украинц" + 0.014*"ситуац" + 0.012*"обстрел" + 0.011*"стат" + 0.011*"смотр" + 0.011*"россиян" + 0.011*"план"
    INFO : topic #15 (0.050): 0.048*"rt" + 0.031*"quot" + 0.024*"дорог" + 0.023*"сто" + 0.014*"факт" + 0.012*"предлож" + 0.012*"узна" + 0.011*"рук" + 0.011*"постро" + 0.011*"сезон"
    INFO : topic #4 (0.050): 0.042*"rt" + 0.031*"интересн" + 0.024*"появ" + 0.021*"европ" + 0.017*"границ" + 0.014*"очередн" + 0.014*"трамп" + 0.014*"закон" + 0.014*"турц" + 0.014*"развит"
    INFO : topic #8 (0.050): 0.057*"rt" + 0.032*"рубл" + 0.024*"слов" + 0.021*"сторон" + 0.015*"получ" + 0.015*"росс" + 0.015*"миноборон" + 0.013*"дтп" + 0.013*"запрет" + 0.012*"украин"
    INFO : topic #12 (0.050): 0.137*"rt" + 0.017*"дума" + 0.016*"украин" + 0.015*"возможн" + 0.015*"счита" + 0.014*"люд" + 0.013*"сша" + 0.011*"росс" + 0.010*"кин" + 0.010*"сирийск"
    INFO : topic diff=0.014050, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #27 = documents up to #2800/9999, outstanding queue size 8
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #16 (0.050): 0.063*"rt" + 0.030*"нача" + 0.028*"чита" + 0.026*"мир" + 0.021*"добр" + 0.021*"написа" + 0.020*"пост" + 0.019*"как" + 0.018*"школ" + 0.015*"уголовн"
    INFO : topic #17 (0.050): 0.052*"украин" + 0.048*"российск" + 0.036*"мнен" + 0.035*"rt" + 0.028*"выбор" + 0.027*"побед" + 0.024*"арм" + 0.019*"донбасс" + 0.019*"продолжа" + 0.018*"достойн"
    INFO : topic #19 (0.050): 0.048*"rt" + 0.039*"ес" + 0.036*"украинск" + 0.034*"крым" + 0.032*"политик" + 0.027*"воен" + 0.027*"чуж" + 0.026*"сша" + 0.026*"украин" + 0.024*"фот"
    INFO : topic #6 (0.050): 0.100*"путин" + 0.050*"rt" + 0.039*"росс" + 0.034*"владимир" + 0.019*"улиц" + 0.015*"мест" + 0.014*"сборн" + 0.014*"ход" + 0.014*"друз" + 0.014*"работ"
    INFO : topic #14 (0.050): 0.079*"rt" + 0.030*"петербург" + 0.024*"украин" + 0.022*"вер" + 0.018*"медвед" + 0.014*"новост" + 0.012*"матч" + 0.012*"восток" + 0.011*"дмитр" + 0.011*"пожар"
    INFO : topic diff=0.013596, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #28 = documents up to #2900/9999, outstanding queue size 4
    INFO : PROGRESS: pass 2, dispatched chunk #29 = documents up to #3000/9999, outstanding queue size 2
    INFO : PROGRESS: pass 2, dispatched chunk #30 = documents up to #3100/9999, outstanding queue size 3
    INFO : PROGRESS: pass 2, dispatched chunk #31 = documents up to #3200/9999, outstanding queue size 4
    INFO : PROGRESS: pass 2, dispatched chunk #32 = documents up to #3300/9999, outstanding queue size 5
    INFO : PROGRESS: pass 2, dispatched chunk #33 = documents up to #3400/9999, outstanding queue size 6
    INFO : PROGRESS: pass 2, dispatched chunk #34 = documents up to #3500/9999, outstanding queue size 7
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #15 (0.050): 0.048*"rt" + 0.033*"quot" + 0.024*"сто" + 0.023*"дорог" + 0.014*"факт" + 0.014*"рук" + 0.013*"выход" + 0.013*"предлож" + 0.012*"узна" + 0.011*"солдат"
    INFO : topic #8 (0.050): 0.057*"rt" + 0.030*"рубл" + 0.026*"слов" + 0.022*"сторон" + 0.019*"миноборон" + 0.016*"росс" + 0.016*"получ" + 0.015*"дтп" + 0.014*"запрет" + 0.013*"украин"
    INFO : topic #5 (0.050): 0.070*"rt" + 0.024*"войн" + 0.018*"задержа" + 0.017*"ситуац" + 0.015*"украинц" + 0.014*"обстрел" + 0.013*"отправ" + 0.012*"смотр" + 0.012*"стат" + 0.011*"россиян"
    INFO : topic #16 (0.050): 0.062*"rt" + 0.030*"чита" + 0.030*"нача" + 0.026*"мир" + 0.022*"пост" + 0.020*"написа" + 0.019*"добр" + 0.019*"как" + 0.017*"школ" + 0.015*"уголовн"
    INFO : topic #0 (0.050): 0.084*"rt" + 0.040*"сир" + 0.017*"рад" + 0.015*"главн" + 0.015*"сша" + 0.011*"игр" + 0.011*"удар" + 0.010*"переговор" + 0.010*"минск" + 0.008*"пьян"
    INFO : topic diff=0.014483, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #35 = documents up to #3600/9999, outstanding queue size 7
    INFO : PROGRESS: pass 2, dispatched chunk #36 = documents up to #3700/9999, outstanding queue size 6
    INFO : PROGRESS: pass 2, dispatched chunk #37 = documents up to #3800/9999, outstanding queue size 7
    INFO : PROGRESS: pass 2, dispatched chunk #38 = documents up to #3900/9999, outstanding queue size 8
    INFO : PROGRESS: pass 2, dispatched chunk #39 = documents up to #4000/9999, outstanding queue size 9
    INFO : PROGRESS: pass 2, dispatched chunk #40 = documents up to #4100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 2, dispatched chunk #41 = documents up to #4200/9999, outstanding queue size 11
    INFO : PROGRESS: pass 2, dispatched chunk #42 = documents up to #4300/9999, outstanding queue size 12
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #10 (0.050): 0.085*"rt" + 0.043*"виде" + 0.028*"дет" + 0.022*"полиц" + 0.017*"лучш" + 0.015*"хочет" + 0.014*"бизнес" + 0.013*"начина" + 0.013*"уб" + 0.012*"метр"
    INFO : topic #9 (0.050): 0.054*"rt" + 0.041*"президент" + 0.021*"рф" + 0.016*"город" + 0.014*"росс" + 0.014*"нат" + 0.012*"говор" + 0.012*"обсуд" + 0.011*"участ" + 0.011*"уф"
    INFO : topic #8 (0.050): 0.058*"rt" + 0.033*"рубл" + 0.026*"слов" + 0.023*"сторон" + 0.018*"миноборон" + 0.018*"получ" + 0.016*"росс" + 0.014*"запрет" + 0.014*"дтп" + 0.013*"украин"
    INFO : topic #18 (0.050): 0.076*"rt" + 0.034*"нов" + 0.029*"санкц" + 0.024*"рф" + 0.021*"росс" + 0.020*"сам" + 0.015*"мид" + 0.013*"апрел" + 0.011*"иг" + 0.011*"представ"
    INFO : topic #11 (0.050): 0.132*"rt" + 0.016*"автомобил" + 0.013*"хорош" + 0.012*"ставропол" + 0.012*"поддержива" + 0.012*"депутат" + 0.012*"обам" + 0.012*"числ" + 0.011*"совет" + 0.011*"связ"
    INFO : topic diff=0.013879, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #43 = documents up to #4400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 2, dispatched chunk #44 = documents up to #4500/9999, outstanding queue size 8
    INFO : PROGRESS: pass 2, dispatched chunk #45 = documents up to #4600/9999, outstanding queue size 9
    INFO : PROGRESS: pass 2, dispatched chunk #46 = documents up to #4700/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #8 (0.050): 0.057*"rt" + 0.033*"рубл" + 0.024*"слов" + 0.022*"сторон" + 0.018*"миноборон" + 0.017*"росс" + 0.017*"получ" + 0.016*"запрет" + 0.013*"дтп" + 0.012*"украин"
    INFO : topic #17 (0.050): 0.053*"российск" + 0.052*"украин" + 0.038*"мнен" + 0.037*"rt" + 0.027*"побед" + 0.027*"арм" + 0.026*"выбор" + 0.023*"донбасс" + 0.020*"газ" + 0.020*"достойн"
    INFO : topic #9 (0.050): 0.054*"rt" + 0.045*"президент" + 0.022*"рф" + 0.015*"город" + 0.014*"нат" + 0.014*"росс" + 0.011*"говор" + 0.011*"эксперт" + 0.011*"участ" + 0.011*"сентябр"
    INFO : topic #1 (0.050): 0.057*"стран" + 0.042*"rt" + 0.025*"дом" + 0.017*"стал" + 0.017*"пострада" + 0.016*"сем" + 0.016*"жител" + 0.014*"массов" + 0.014*"днем" + 0.013*"вид"
    INFO : topic #11 (0.050): 0.138*"rt" + 0.017*"автомобил" + 0.013*"истор" + 0.013*"хорош" + 0.012*"обам" + 0.012*"связ" + 0.012*"крушен" + 0.012*"поддержива" + 0.011*"ставропол" + 0.011*"привет"
    INFO : topic diff=0.014379, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #47 = documents up to #4800/9999, outstanding queue size 10
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #0 (0.050): 0.082*"rt" + 0.040*"сир" + 0.017*"главн" + 0.017*"рад" + 0.015*"переговор" + 0.014*"сша" + 0.013*"удар" + 0.011*"игр" + 0.010*"призва" + 0.010*"минск"
    INFO : topic #16 (0.050): 0.062*"rt" + 0.031*"нача" + 0.029*"чита" + 0.027*"мир" + 0.027*"пост" + 0.023*"добр" + 0.018*"написа" + 0.018*"как" + 0.017*"школ" + 0.015*"потеря"
    INFO : topic #3 (0.050): 0.059*"rt" + 0.042*"москв" + 0.031*"донецк" + 0.018*"район" + 0.016*"прошл" + 0.015*"машин" + 0.012*"центр" + 0.012*"продаж" + 0.011*"мэр" + 0.010*"дан"
    INFO : topic #10 (0.050): 0.089*"rt" + 0.041*"виде" + 0.026*"дет" + 0.025*"полиц" + 0.020*"лучш" + 0.016*"хочет" + 0.014*"бизнес" + 0.013*"решен" + 0.013*"уб" + 0.012*"метр"
    INFO : topic #12 (0.050): 0.155*"rt" + 0.021*"люд" + 0.017*"дума" + 0.016*"украин" + 0.015*"возможн" + 0.014*"счита" + 0.013*"сша" + 0.013*"цен" + 0.010*"автор" + 0.010*"кин"
    INFO : topic diff=0.013326, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #48 = documents up to #4900/9999, outstanding queue size 6
    INFO : PROGRESS: pass 2, dispatched chunk #49 = documents up to #5000/9999, outstanding queue size 6
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #9 (0.050): 0.057*"rt" + 0.045*"президент" + 0.024*"рф" + 0.015*"город" + 0.013*"росс" + 0.013*"нат" + 0.012*"говор" + 0.012*"сентябр" + 0.012*"эксперт" + 0.011*"произошл"
    INFO : topic #18 (0.050): 0.076*"rt" + 0.040*"нов" + 0.028*"санкц" + 0.027*"рф" + 0.021*"сам" + 0.019*"росс" + 0.015*"мид" + 0.013*"апрел" + 0.011*"российск" + 0.011*"акц"
    INFO : topic #3 (0.050): 0.059*"rt" + 0.040*"москв" + 0.033*"донецк" + 0.020*"район" + 0.016*"прошл" + 0.014*"машин" + 0.014*"центр" + 0.013*"продаж" + 0.012*"мэр" + 0.011*"силовик"
    INFO : topic #12 (0.050): 0.154*"rt" + 0.022*"люд" + 0.016*"украин" + 0.016*"дума" + 0.015*"возможн" + 0.013*"цен" + 0.012*"сша" + 0.012*"счита" + 0.011*"убийств" + 0.011*"автор"
    INFO : topic #14 (0.050): 0.083*"rt" + 0.035*"петербург" + 0.029*"вер" + 0.024*"украин" + 0.016*"медвед" + 0.016*"новост" + 0.014*"пожар" + 0.012*"ки" + 0.012*"матч" + 0.012*"восток"
    INFO : topic diff=0.013222, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #50 = documents up to #5100/9999, outstanding queue size 3
    INFO : PROGRESS: pass 2, dispatched chunk #51 = documents up to #5200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 2, dispatched chunk #52 = documents up to #5300/9999, outstanding queue size 3
    INFO : PROGRESS: pass 2, dispatched chunk #53 = documents up to #5400/9999, outstanding queue size 4
    INFO : PROGRESS: pass 2, dispatched chunk #54 = documents up to #5500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 2, dispatched chunk #55 = documents up to #5600/9999, outstanding queue size 6
    INFO : PROGRESS: pass 2, dispatched chunk #56 = documents up to #5700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 2, dispatched chunk #57 = documents up to #5800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 2, dispatched chunk #58 = documents up to #5900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 2, dispatched chunk #59 = documents up to #6000/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #14 (0.050): 0.084*"rt" + 0.033*"петербург" + 0.027*"вер" + 0.026*"украин" + 0.016*"медвед" + 0.014*"новост" + 0.014*"пожар" + 0.014*"ки" + 0.014*"матч" + 0.013*"восток"
    INFO : topic #10 (0.050): 0.087*"rt" + 0.042*"виде" + 0.027*"дет" + 0.024*"полиц" + 0.018*"лучш" + 0.016*"последн" + 0.015*"хочет" + 0.014*"бизнес" + 0.014*"уб" + 0.013*"решен"
    INFO : topic #1 (0.050): 0.057*"стран" + 0.044*"rt" + 0.023*"пострада" + 0.022*"жител" + 0.022*"дом" + 0.015*"сем" + 0.015*"стал" + 0.014*"днем" + 0.013*"массов" + 0.013*"конц"
    INFO : topic #19 (0.050): 0.052*"rt" + 0.041*"украинск" + 0.041*"крым" + 0.032*"ес" + 0.031*"украин" + 0.031*"политик" + 0.027*"днр" + 0.027*"воен" + 0.026*"сша" + 0.023*"фот"
    INFO : topic #17 (0.050): 0.057*"украин" + 0.056*"российск" + 0.039*"rt" + 0.033*"мнен" + 0.028*"побед" + 0.026*"выбор" + 0.025*"донбасс" + 0.023*"арм" + 0.021*"газ" + 0.020*"достойн"
    INFO : topic diff=0.014575, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #60 = documents up to #6100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 2, dispatched chunk #61 = documents up to #6200/9999, outstanding queue size 9
    INFO : PROGRESS: pass 2, dispatched chunk #62 = documents up to #6300/9999, outstanding queue size 9
    INFO : PROGRESS: pass 2, dispatched chunk #63 = documents up to #6400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 2, dispatched chunk #64 = documents up to #6500/9999, outstanding queue size 11
    INFO : PROGRESS: pass 2, dispatched chunk #65 = documents up to #6600/9999, outstanding queue size 12
    INFO : PROGRESS: pass 2, dispatched chunk #66 = documents up to #6700/9999, outstanding queue size 13
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #8 (0.050): 0.057*"rt" + 0.031*"рубл" + 0.027*"слов" + 0.019*"сторон" + 0.016*"миноборон" + 0.016*"получ" + 0.015*"росс" + 0.014*"переп" + 0.014*"запрет" + 0.014*"дтп"
    INFO : topic #13 (0.050): 0.078*"rt" + 0.032*"готов" + 0.028*"украин" + 0.026*"перв" + 0.026*"власт" + 0.024*"сми" + 0.021*"русск" + 0.016*"прав" + 0.016*"дел" + 0.013*"репост"
    INFO : topic #10 (0.050): 0.087*"rt" + 0.042*"виде" + 0.028*"дет" + 0.023*"полиц" + 0.021*"лучш" + 0.015*"бизнес" + 0.014*"последн" + 0.014*"решен" + 0.014*"хочет" + 0.014*"уб"
    INFO : topic #12 (0.050): 0.154*"rt" + 0.020*"люд" + 0.017*"дума" + 0.016*"украин" + 0.015*"возможн" + 0.015*"счита" + 0.013*"цен" + 0.012*"сша" + 0.012*"оруж" + 0.011*"автор"
    INFO : topic #7 (0.050): 0.078*"rt" + 0.025*"человек" + 0.018*"област" + 0.015*"дня" + 0.014*"люб" + 0.013*"март" + 0.013*"новосибирск" + 0.012*"фильм" + 0.011*"пройдет" + 0.011*"музык"
    INFO : topic diff=0.012020, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #67 = documents up to #6800/9999, outstanding queue size 10
    INFO : PROGRESS: pass 2, dispatched chunk #68 = documents up to #6900/9999, outstanding queue size 8
    INFO : PROGRESS: pass 2, dispatched chunk #69 = documents up to #7000/9999, outstanding queue size 9
    INFO : PROGRESS: pass 2, dispatched chunk #70 = documents up to #7100/9999, outstanding queue size 10
    INFO : merging changes from 800 documents into a model of 9999 documents
    INFO : topic #3 (0.050): 0.058*"rt" + 0.038*"москв" + 0.037*"донецк" + 0.021*"район" + 0.015*"прошл" + 0.014*"продаж" + 0.014*"машин" + 0.013*"мэр" + 0.013*"центр" + 0.012*"силовик"
    INFO : topic #2 (0.050): 0.056*"rt" + 0.035*"киев" + 0.030*"глав" + 0.019*"жизн" + 0.017*"взрыв" + 0.016*"пыта" + 0.015*"нашл" + 0.015*"кита" + 0.015*"погибл" + 0.013*"результат"
    INFO : topic #16 (0.050): 0.064*"rt" + 0.035*"нача" + 0.029*"чита" + 0.027*"добр" + 0.026*"пост" + 0.025*"мир" + 0.021*"как" + 0.020*"написа" + 0.019*"потеря" + 0.017*"школ"
    INFO : topic #8 (0.050): 0.058*"rt" + 0.030*"рубл" + 0.026*"слов" + 0.019*"сторон" + 0.016*"получ" + 0.015*"росс" + 0.015*"миноборон" + 0.014*"дтп" + 0.014*"переп" + 0.013*"новост"
    INFO : topic #14 (0.050): 0.082*"rt" + 0.031*"петербург" + 0.028*"вер" + 0.026*"украин" + 0.020*"медвед" + 0.014*"новост" + 0.013*"матч" + 0.013*"восток" + 0.013*"пожар" + 0.012*"ки"
    INFO : topic diff=0.011773, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #71 = documents up to #7200/9999, outstanding queue size 6
    INFO : PROGRESS: pass 2, dispatched chunk #72 = documents up to #7300/9999, outstanding queue size 5
    INFO : PROGRESS: pass 2, dispatched chunk #73 = documents up to #7400/9999, outstanding queue size 6
    INFO : PROGRESS: pass 2, dispatched chunk #74 = documents up to #7500/9999, outstanding queue size 7
    INFO : PROGRESS: pass 2, dispatched chunk #75 = documents up to #7600/9999, outstanding queue size 8
    INFO : PROGRESS: pass 2, dispatched chunk #76 = documents up to #7700/9999, outstanding queue size 9
    INFO : PROGRESS: pass 2, dispatched chunk #77 = documents up to #7800/9999, outstanding queue size 10
    INFO : PROGRESS: pass 2, dispatched chunk #78 = documents up to #7900/9999, outstanding queue size 11
    INFO : PROGRESS: pass 2, dispatched chunk #79 = documents up to #8000/9999, outstanding queue size 12
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.076*"rt" + 0.048*"нов" + 0.026*"санкц" + 0.024*"рф" + 0.022*"росс" + 0.021*"сам" + 0.013*"мид" + 0.012*"мчс" + 0.012*"акц" + 0.011*"представ"
    INFO : topic #10 (0.050): 0.089*"rt" + 0.041*"виде" + 0.029*"дет" + 0.026*"лучш" + 0.024*"полиц" + 0.015*"последн" + 0.015*"уб" + 0.014*"бизнес" + 0.013*"решен" + 0.012*"хочет"
    INFO : topic #15 (0.050): 0.049*"rt" + 0.034*"quot" + 0.028*"дорог" + 0.024*"сто" + 0.015*"предлож" + 0.014*"выход" + 0.013*"памятник" + 0.013*"факт" + 0.013*"рук" + 0.012*"действ"
    INFO : topic #6 (0.050): 0.118*"путин" + 0.057*"rt" + 0.036*"росс" + 0.024*"владимир" + 0.020*"мест" + 0.017*"друз" + 0.016*"улиц" + 0.016*"назва" + 0.015*"работ" + 0.014*"сборн"
    INFO : topic #5 (0.050): 0.068*"rt" + 0.023*"войн" + 0.019*"задержа" + 0.016*"план" + 0.015*"стат" + 0.015*"смотр" + 0.015*"обстрел" + 0.014*"украинц" + 0.013*"сотрудник" + 0.013*"ситуац"
    INFO : topic diff=0.012312, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #80 = documents up to #8100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 2, dispatched chunk #81 = documents up to #8200/9999, outstanding queue size 7
    INFO : PROGRESS: pass 2, dispatched chunk #82 = documents up to #8300/9999, outstanding queue size 8
    INFO : PROGRESS: pass 2, dispatched chunk #83 = documents up to #8400/9999, outstanding queue size 9
    INFO : PROGRESS: pass 2, dispatched chunk #84 = documents up to #8500/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #15 (0.050): 0.049*"rt" + 0.033*"quot" + 0.028*"дорог" + 0.023*"сто" + 0.014*"рук" + 0.014*"предлож" + 0.013*"октябр" + 0.013*"памятник" + 0.013*"выход" + 0.012*"действ"
    INFO : topic #3 (0.050): 0.060*"rt" + 0.040*"москв" + 0.036*"донецк" + 0.024*"район" + 0.014*"прошл" + 0.014*"центр" + 0.013*"мэр" + 0.012*"продаж" + 0.012*"машин" + 0.011*"банк"
    INFO : topic #17 (0.050): 0.063*"украин" + 0.061*"российск" + 0.038*"rt" + 0.033*"мнен" + 0.027*"побед" + 0.026*"арм" + 0.026*"выбор" + 0.024*"донбасс" + 0.023*"газ" + 0.020*"новост"
    INFO : topic #13 (0.050): 0.076*"rt" + 0.028*"готов" + 0.027*"украин" + 0.025*"перв" + 0.025*"сми" + 0.023*"власт" + 0.020*"русск" + 0.016*"террорист" + 0.016*"дел" + 0.016*"прав"
    INFO : topic #16 (0.050): 0.065*"rt" + 0.032*"нача" + 0.030*"чита" + 0.028*"пост" + 0.026*"добр" + 0.024*"мир" + 0.023*"как" + 0.017*"написа" + 0.017*"потеря" + 0.017*"школ"
    INFO : topic diff=0.013678, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #85 = documents up to #8600/9999, outstanding queue size 10
    INFO : PROGRESS: pass 2, dispatched chunk #86 = documents up to #8700/9999, outstanding queue size 8
    INFO : PROGRESS: pass 2, dispatched chunk #87 = documents up to #8800/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #16 (0.050): 0.063*"rt" + 0.032*"нача" + 0.028*"чита" + 0.026*"пост" + 0.026*"добр" + 0.024*"мир" + 0.022*"как" + 0.020*"написа" + 0.018*"потеря" + 0.015*"школ"
    INFO : topic #13 (0.050): 0.078*"rt" + 0.027*"готов" + 0.026*"украин" + 0.024*"власт" + 0.024*"сми" + 0.023*"перв" + 0.021*"русск" + 0.017*"террорист" + 0.015*"дел" + 0.014*"прав"
    INFO : topic #19 (0.050): 0.051*"rt" + 0.041*"украинск" + 0.036*"ес" + 0.035*"крым" + 0.032*"воен" + 0.030*"политик" + 0.030*"украин" + 0.025*"чуж" + 0.025*"днр" + 0.025*"сша"
    INFO : topic #18 (0.050): 0.077*"rt" + 0.047*"нов" + 0.031*"санкц" + 0.025*"рф" + 0.022*"росс" + 0.022*"сам" + 0.015*"иг" + 0.014*"мид" + 0.013*"представ" + 0.012*"апрел"
    INFO : topic #3 (0.050): 0.059*"rt" + 0.041*"москв" + 0.034*"донецк" + 0.022*"район" + 0.015*"продаж" + 0.013*"центр" + 0.013*"мэр" + 0.013*"прошл" + 0.013*"машин" + 0.011*"встреч"
    INFO : topic diff=0.015999, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #88 = documents up to #8900/9999, outstanding queue size 9
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #14 (0.050): 0.079*"rt" + 0.030*"петербург" + 0.024*"украин" + 0.024*"вер" + 0.019*"медвед" + 0.016*"пожар" + 0.016*"восток" + 0.016*"матч" + 0.014*"ки" + 0.011*"новост"
    INFO : topic #3 (0.050): 0.059*"rt" + 0.042*"москв" + 0.037*"донецк" + 0.021*"район" + 0.014*"продаж" + 0.014*"центр" + 0.012*"прошл" + 0.012*"известн" + 0.012*"мэр" + 0.011*"машин"
    INFO : topic #9 (0.050): 0.056*"rt" + 0.047*"президент" + 0.025*"рф" + 0.018*"город" + 0.016*"нат" + 0.013*"эксперт" + 0.013*"говор" + 0.012*"росс" + 0.011*"евр" + 0.011*"сентябр"
    INFO : topic #11 (0.050): 0.122*"rt" + 0.017*"обам" + 0.016*"хорош" + 0.016*"истор" + 0.016*"автомобил" + 0.013*"депутат" + 0.012*"связ" + 0.012*"поддержива" + 0.011*"ал" + 0.011*"праздник"
    INFO : topic #10 (0.050): 0.085*"rt" + 0.040*"виде" + 0.031*"дет" + 0.027*"лучш" + 0.023*"полиц" + 0.014*"хочет" + 0.014*"решен" + 0.013*"дела" + 0.013*"последн" + 0.012*"уб"
    INFO : topic diff=0.011727, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #89 = documents up to #9000/9999, outstanding queue size 4
    INFO : PROGRESS: pass 2, dispatched chunk #90 = documents up to #9100/9999, outstanding queue size 2
    INFO : PROGRESS: pass 2, dispatched chunk #91 = documents up to #9200/9999, outstanding queue size 3
    INFO : PROGRESS: pass 2, dispatched chunk #92 = documents up to #9300/9999, outstanding queue size 4
    INFO : PROGRESS: pass 2, dispatched chunk #93 = documents up to #9400/9999, outstanding queue size 5
    INFO : PROGRESS: pass 2, dispatched chunk #94 = documents up to #9500/9999, outstanding queue size 6
    INFO : PROGRESS: pass 2, dispatched chunk #95 = documents up to #9600/9999, outstanding queue size 7
    INFO : PROGRESS: pass 2, dispatched chunk #96 = documents up to #9700/9999, outstanding queue size 8
    INFO : PROGRESS: pass 2, dispatched chunk #97 = documents up to #9800/9999, outstanding queue size 9
    INFO : PROGRESS: pass 2, dispatched chunk #98 = documents up to #9900/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #3 (0.050): 0.059*"rt" + 0.044*"москв" + 0.034*"донецк" + 0.021*"район" + 0.015*"продаж" + 0.014*"центр" + 0.014*"известн" + 0.013*"машин" + 0.013*"прошл" + 0.011*"встреч"
    INFO : topic #1 (0.050): 0.065*"стран" + 0.046*"rt" + 0.023*"дом" + 0.021*"пострада" + 0.021*"жител" + 0.018*"стал" + 0.016*"конц" + 0.014*"днем" + 0.014*"сем" + 0.014*"американск"
    INFO : topic #19 (0.050): 0.054*"rt" + 0.042*"украинск" + 0.039*"ес" + 0.032*"политик" + 0.032*"крым" + 0.030*"воен" + 0.030*"украин" + 0.028*"сша" + 0.027*"чуж" + 0.023*"помощ"
    INFO : topic #15 (0.050): 0.048*"rt" + 0.034*"quot" + 0.029*"дорог" + 0.023*"сто" + 0.014*"узна" + 0.013*"октябр" + 0.013*"рук" + 0.013*"предлож" + 0.012*"заверш" + 0.012*"выход"
    INFO : topic #12 (0.050): 0.165*"rt" + 0.018*"возможн" + 0.017*"люд" + 0.016*"дума" + 0.014*"украин" + 0.014*"сша" + 0.014*"счита" + 0.012*"убийств" + 0.011*"цен" + 0.011*"доллар"
    INFO : topic diff=0.012696, rho=0.098538
    INFO : PROGRESS: pass 2, dispatched chunk #99 = documents up to #9999/9999, outstanding queue size 10
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #19 (0.050): 0.055*"rt" + 0.042*"украинск" + 0.037*"ес" + 0.032*"воен" + 0.030*"политик" + 0.030*"крым" + 0.029*"украин" + 0.026*"сша" + 0.026*"чуж" + 0.024*"помощ"
    INFO : topic #4 (0.050): 0.041*"rt" + 0.034*"интересн" + 0.025*"появ" + 0.022*"европ" + 0.018*"границ" + 0.017*"трамп" + 0.017*"закон" + 0.016*"турц" + 0.014*"очередн" + 0.013*"украин"
    INFO : topic #2 (0.050): 0.054*"rt" + 0.034*"глав" + 0.032*"киев" + 0.021*"жизн" + 0.018*"пыта" + 0.016*"международн" + 0.016*"погибл" + 0.016*"южн" + 0.015*"взрыв" + 0.014*"кита"
    INFO : topic #1 (0.050): 0.066*"стран" + 0.046*"rt" + 0.025*"дом" + 0.020*"жител" + 0.019*"пострада" + 0.017*"стал" + 0.015*"сем" + 0.015*"конц" + 0.014*"массов" + 0.014*"днем"
    INFO : topic #9 (0.050): 0.058*"rt" + 0.048*"президент" + 0.026*"рф" + 0.017*"город" + 0.015*"нат" + 0.013*"уф" + 0.012*"говор" + 0.012*"эксперт" + 0.012*"росс" + 0.011*"приня"
    INFO : topic diff=0.010999, rho=0.098538
    INFO : -15.993 per-word bound, 65240.1 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : merging changes from 399 documents into a model of 9999 documents
    INFO : topic #3 (0.050): 0.062*"rt" + 0.044*"москв" + 0.037*"донецк" + 0.020*"район" + 0.018*"прошл" + 0.014*"центр" + 0.012*"продаж" + 0.012*"машин" + 0.012*"дан" + 0.012*"известн"
    INFO : topic #5 (0.050): 0.064*"rt" + 0.031*"войн" + 0.019*"задержа" + 0.017*"ситуац" + 0.016*"план" + 0.016*"обстрел" + 0.015*"предлага" + 0.013*"смотр" + 0.012*"россиян" + 0.012*"украинц"
    INFO : topic #7 (0.050): 0.076*"rt" + 0.019*"област" + 0.018*"люб" + 0.017*"человек" + 0.016*"дня" + 0.016*"фильм" + 0.014*"пройдет" + 0.014*"отношен" + 0.014*"лидер" + 0.013*"март"
    INFO : topic #15 (0.050): 0.046*"rt" + 0.033*"quot" + 0.029*"дорог" + 0.025*"сто" + 0.016*"факт" + 0.013*"заверш" + 0.013*"выход" + 0.013*"рук" + 0.013*"предлож" + 0.013*"сезон"
    INFO : topic #14 (0.050): 0.080*"rt" + 0.031*"петербург" + 0.025*"вер" + 0.023*"украин" + 0.022*"медвед" + 0.014*"дмитр" + 0.014*"пожар" + 0.013*"восток" + 0.013*"матч" + 0.013*"ки"
    INFO : topic diff=0.013667, rho=0.098538
    INFO : -15.876 per-word bound, 60137.5 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : PROGRESS: pass 3, dispatched chunk #0 = documents up to #100/9999, outstanding queue size 1
    INFO : PROGRESS: pass 3, dispatched chunk #1 = documents up to #200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 3, dispatched chunk #2 = documents up to #300/9999, outstanding queue size 3
    INFO : PROGRESS: pass 3, dispatched chunk #3 = documents up to #400/9999, outstanding queue size 4
    INFO : PROGRESS: pass 3, dispatched chunk #4 = documents up to #500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 3, dispatched chunk #5 = documents up to #600/9999, outstanding queue size 6
    INFO : PROGRESS: pass 3, dispatched chunk #6 = documents up to #700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 3, dispatched chunk #7 = documents up to #800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 3, dispatched chunk #8 = documents up to #900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #9 = documents up to #1000/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #10 = documents up to #1100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #11 = documents up to #1200/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #12 = documents up to #1300/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #13 = documents up to #1400/9999, outstanding queue size 11
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #14 (0.050): 0.079*"rt" + 0.031*"петербург" + 0.024*"вер" + 0.022*"украин" + 0.021*"медвед" + 0.014*"матч" + 0.013*"пожар" + 0.012*"восток" + 0.012*"ки" + 0.012*"дмитр"
    INFO : topic #9 (0.050): 0.059*"rt" + 0.046*"президент" + 0.023*"рф" + 0.017*"город" + 0.014*"нат" + 0.012*"говор" + 0.012*"уф" + 0.011*"обсуд" + 0.011*"росс" + 0.011*"немцов"
    INFO : topic #11 (0.050): 0.116*"rt" + 0.018*"обам" + 0.017*"автомобил" + 0.016*"истор" + 0.015*"хорош" + 0.012*"депутат" + 0.011*"журналист" + 0.011*"водител" + 0.010*"час" + 0.010*"ставропол"
    INFO : topic #0 (0.050): 0.081*"rt" + 0.043*"сир" + 0.020*"рад" + 0.019*"главн" + 0.014*"сша" + 0.014*"переговор" + 0.013*"удар" + 0.011*"минск" + 0.011*"игр" + 0.009*"призва"
    INFO : topic #18 (0.050): 0.075*"rt" + 0.052*"нов" + 0.028*"санкц" + 0.022*"рф" + 0.022*"сам" + 0.022*"росс" + 0.015*"мид" + 0.014*"представ" + 0.013*"иг" + 0.012*"апрел"
    INFO : topic diff=0.049426, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #14 = documents up to #1500/9999, outstanding queue size 11
    INFO : PROGRESS: pass 3, dispatched chunk #15 = documents up to #1600/9999, outstanding queue size 11
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #10 (0.050): 0.081*"rt" + 0.039*"виде" + 0.031*"дет" + 0.020*"лучш" + 0.018*"полиц" + 0.016*"бизнес" + 0.016*"хочет" + 0.016*"метр" + 0.015*"уб" + 0.012*"решен"
    INFO : topic #16 (0.050): 0.061*"rt" + 0.029*"нача" + 0.026*"чита" + 0.026*"мир" + 0.023*"добр" + 0.023*"пост" + 0.021*"написа" + 0.021*"как" + 0.016*"школ" + 0.015*"уголовн"
    INFO : topic #19 (0.050): 0.050*"rt" + 0.042*"украинск" + 0.039*"ес" + 0.031*"политик" + 0.031*"крым" + 0.030*"украин" + 0.028*"воен" + 0.026*"фот" + 0.026*"сша" + 0.025*"чуж"
    INFO : topic #11 (0.050): 0.108*"rt" + 0.017*"обам" + 0.016*"истор" + 0.016*"автомобил" + 0.014*"хорош" + 0.012*"числ" + 0.011*"депутат" + 0.011*"журналист" + 0.011*"крушен" + 0.011*"ставропол"
    INFO : topic #17 (0.050): 0.067*"украин" + 0.058*"российск" + 0.040*"rt" + 0.035*"мнен" + 0.027*"выбор" + 0.026*"побед" + 0.024*"новост" + 0.021*"продолжа" + 0.019*"донбасс" + 0.019*"арм"
    INFO : topic diff=0.041588, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #16 = documents up to #1700/9999, outstanding queue size 8
    INFO : PROGRESS: pass 3, dispatched chunk #17 = documents up to #1800/9999, outstanding queue size 6
    INFO : PROGRESS: pass 3, dispatched chunk #18 = documents up to #1900/9999, outstanding queue size 7
    INFO : PROGRESS: pass 3, dispatched chunk #19 = documents up to #2000/9999, outstanding queue size 8
    INFO : PROGRESS: pass 3, dispatched chunk #20 = documents up to #2100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #21 = documents up to #2200/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #10 (0.050): 0.078*"rt" + 0.040*"виде" + 0.030*"дет" + 0.020*"лучш" + 0.016*"полиц" + 0.016*"уб" + 0.015*"хочет" + 0.015*"бизнес" + 0.014*"метр" + 0.012*"причин"
    INFO : topic #1 (0.050): 0.052*"стран" + 0.043*"rt" + 0.023*"дом" + 0.019*"жител" + 0.018*"пострада" + 0.018*"стал" + 0.015*"массов" + 0.015*"сем" + 0.013*"конц" + 0.012*"днем"
    INFO : topic #17 (0.050): 0.066*"украин" + 0.057*"российск" + 0.037*"rt" + 0.036*"мнен" + 0.027*"выбор" + 0.027*"новост" + 0.024*"побед" + 0.022*"продолжа" + 0.018*"арм" + 0.018*"донбасс"
    INFO : topic #8 (0.050): 0.055*"rt" + 0.031*"рубл" + 0.021*"сторон" + 0.020*"слов" + 0.017*"получ" + 0.016*"миноборон" + 0.015*"росс" + 0.013*"дтп" + 0.012*"запрет" + 0.012*"мужчин"
    INFO : topic #15 (0.050): 0.047*"rt" + 0.032*"quot" + 0.024*"дорог" + 0.022*"сто" + 0.013*"факт" + 0.013*"узна" + 0.013*"постро" + 0.012*"заверш" + 0.011*"предлож" + 0.010*"сезон"
    INFO : topic diff=0.028899, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #22 = documents up to #2300/9999, outstanding queue size 10
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #15 (0.050): 0.047*"rt" + 0.029*"quot" + 0.024*"дорог" + 0.021*"сто" + 0.014*"факт" + 0.014*"предлож" + 0.013*"узна" + 0.012*"рук" + 0.012*"постро" + 0.011*"сезон"
    INFO : topic #19 (0.050): 0.049*"rt" + 0.042*"ес" + 0.041*"украинск" + 0.032*"крым" + 0.032*"политик" + 0.029*"украин" + 0.027*"воен" + 0.027*"сша" + 0.026*"чуж" + 0.025*"фот"
    INFO : topic #8 (0.050): 0.055*"rt" + 0.035*"рубл" + 0.021*"слов" + 0.020*"сторон" + 0.017*"миноборон" + 0.016*"получ" + 0.016*"росс" + 0.014*"дтп" + 0.013*"запрет" + 0.012*"мужчин"
    INFO : topic #13 (0.050): 0.072*"rt" + 0.029*"готов" + 0.024*"перв" + 0.023*"власт" + 0.023*"украин" + 0.023*"сми" + 0.019*"прав" + 0.017*"русск" + 0.017*"дел" + 0.012*"репост"
    INFO : topic #10 (0.050): 0.080*"rt" + 0.042*"виде" + 0.029*"дет" + 0.019*"лучш" + 0.018*"полиц" + 0.016*"уб" + 0.015*"хочет" + 0.015*"метр" + 0.013*"бизнес" + 0.012*"решен"
    INFO : topic diff=0.016672, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #23 = documents up to #2400/9999, outstanding queue size 6
    INFO : PROGRESS: pass 3, dispatched chunk #24 = documents up to #2500/9999, outstanding queue size 6
    INFO : PROGRESS: pass 3, dispatched chunk #25 = documents up to #2600/9999, outstanding queue size 7
    INFO : PROGRESS: pass 3, dispatched chunk #26 = documents up to #2700/9999, outstanding queue size 8
    INFO : PROGRESS: pass 3, dispatched chunk #27 = documents up to #2800/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #28 = documents up to #2900/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #29 = documents up to #3000/9999, outstanding queue size 11
    INFO : PROGRESS: pass 3, dispatched chunk #30 = documents up to #3100/9999, outstanding queue size 12
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #9 (0.050): 0.054*"rt" + 0.042*"президент" + 0.020*"рф" + 0.017*"город" + 0.013*"нат" + 0.013*"обсуд" + 0.012*"росс" + 0.012*"уф" + 0.011*"москв" + 0.011*"говор"
    INFO : topic #12 (0.050): 0.154*"rt" + 0.018*"дума" + 0.018*"люд" + 0.016*"возможн" + 0.016*"счита" + 0.015*"украин" + 0.014*"сша" + 0.011*"цен" + 0.010*"росс" + 0.010*"оруж"
    INFO : topic #10 (0.050): 0.083*"rt" + 0.043*"виде" + 0.027*"дет" + 0.021*"полиц" + 0.018*"лучш" + 0.015*"уб" + 0.015*"хочет" + 0.013*"метр" + 0.013*"решен" + 0.013*"бизнес"
    INFO : topic #17 (0.050): 0.072*"украин" + 0.063*"российск" + 0.042*"rt" + 0.034*"мнен" + 0.031*"новост" + 0.027*"выбор" + 0.023*"побед" + 0.021*"продолжа" + 0.019*"газ" + 0.017*"арм"
    INFO : topic #1 (0.050): 0.050*"стран" + 0.042*"rt" + 0.027*"дом" + 0.020*"жител" + 0.018*"стал" + 0.018*"пострада" + 0.016*"сем" + 0.015*"массов" + 0.014*"днем" + 0.013*"американск"
    INFO : topic diff=0.010635, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #31 = documents up to #3200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #32 = documents up to #3300/9999, outstanding queue size 8
    INFO : PROGRESS: pass 3, dispatched chunk #33 = documents up to #3400/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #34 = documents up to #3500/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #35 = documents up to #3600/9999, outstanding queue size 11
    INFO : PROGRESS: pass 3, dispatched chunk #36 = documents up to #3700/9999, outstanding queue size 12
    INFO : PROGRESS: pass 3, dispatched chunk #37 = documents up to #3800/9999, outstanding queue size 13
    INFO : merging changes from 800 documents into a model of 9999 documents
    INFO : topic #16 (0.050): 0.060*"rt" + 0.031*"нача" + 0.029*"чита" + 0.028*"мир" + 0.022*"пост" + 0.021*"написа" + 0.019*"добр" + 0.017*"школ" + 0.017*"как" + 0.015*"потеря"
    INFO : topic #4 (0.050): 0.041*"rt" + 0.032*"интересн" + 0.026*"появ" + 0.021*"европ" + 0.016*"границ" + 0.016*"турц" + 0.015*"очередн" + 0.015*"развит" + 0.014*"закон" + 0.013*"трамп"
    INFO : topic #6 (0.050): 0.113*"путин" + 0.053*"rt" + 0.032*"владимир" + 0.031*"росс" + 0.019*"улиц" + 0.016*"мест" + 0.015*"работ" + 0.014*"друз" + 0.014*"ход" + 0.014*"назва"
    INFO : topic #11 (0.050): 0.113*"rt" + 0.017*"автомобил" + 0.016*"обам" + 0.015*"хорош" + 0.013*"истор" + 0.013*"ставропол" + 0.012*"совет" + 0.012*"числ" + 0.012*"крушен" + 0.012*"депутат"
    INFO : topic #5 (0.050): 0.068*"rt" + 0.027*"войн" + 0.018*"задержа" + 0.016*"ситуац" + 0.016*"украинц" + 0.014*"обстрел" + 0.013*"отправ" + 0.012*"россиян" + 0.012*"стат" + 0.012*"план"
    INFO : topic diff=0.010905, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #38 = documents up to #3900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #39 = documents up to #4000/9999, outstanding queue size 8
    INFO : PROGRESS: pass 3, dispatched chunk #40 = documents up to #4100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #41 = documents up to #4200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #42 = documents up to #4300/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #14 (0.050): 0.084*"rt" + 0.028*"петербург" + 0.025*"вер" + 0.019*"украин" + 0.017*"медвед" + 0.015*"пожар" + 0.012*"восток" + 0.012*"дмитр" + 0.012*"матч" + 0.011*"ки"
    INFO : topic #4 (0.050): 0.039*"rt" + 0.032*"интересн" + 0.026*"появ" + 0.022*"европ" + 0.017*"турц" + 0.016*"границ" + 0.015*"отказа" + 0.014*"очередн" + 0.013*"развит" + 0.013*"украин"
    INFO : topic #0 (0.050): 0.081*"rt" + 0.043*"сир" + 0.019*"рад" + 0.014*"главн" + 0.014*"сша" + 0.013*"игр" + 0.012*"удар" + 0.011*"переговор" + 0.010*"призва" + 0.010*"пьян"
    INFO : topic #13 (0.050): 0.076*"rt" + 0.030*"готов" + 0.024*"украин" + 0.024*"власт" + 0.023*"сми" + 0.022*"перв" + 0.018*"дел" + 0.017*"прав" + 0.016*"русск" + 0.014*"лет"
    INFO : topic #17 (0.050): 0.071*"украин" + 0.062*"российск" + 0.041*"rt" + 0.035*"мнен" + 0.029*"новост" + 0.027*"выбор" + 0.025*"побед" + 0.021*"донбасс" + 0.021*"арм" + 0.020*"продолжа"
    INFO : topic diff=0.014971, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #43 = documents up to #4400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #44 = documents up to #4500/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #45 = documents up to #4600/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #46 = documents up to #4700/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #8 (0.050): 0.056*"rt" + 0.035*"рубл" + 0.022*"сторон" + 0.022*"слов" + 0.018*"миноборон" + 0.017*"получ" + 0.017*"росс" + 0.016*"запрет" + 0.013*"мужчин" + 0.013*"дтп"
    INFO : topic #12 (0.050): 0.170*"rt" + 0.022*"люд" + 0.018*"дума" + 0.015*"возможн" + 0.015*"украин" + 0.015*"счита" + 0.014*"сша" + 0.013*"цен" + 0.010*"автор" + 0.010*"боевик"
    INFO : topic #17 (0.050): 0.071*"украин" + 0.063*"российск" + 0.042*"rt" + 0.037*"мнен" + 0.031*"новост" + 0.026*"выбор" + 0.025*"побед" + 0.023*"арм" + 0.023*"донбасс" + 0.021*"продолжа"
    INFO : topic #14 (0.050): 0.081*"rt" + 0.032*"петербург" + 0.025*"вер" + 0.018*"украин" + 0.017*"медвед" + 0.015*"пожар" + 0.013*"восток" + 0.012*"ки" + 0.012*"дмитр" + 0.011*"матч"
    INFO : topic #13 (0.050): 0.076*"rt" + 0.033*"готов" + 0.025*"власт" + 0.024*"перв" + 0.024*"сми" + 0.023*"украин" + 0.019*"дел" + 0.018*"прав" + 0.016*"русск" + 0.013*"лет"
    INFO : topic diff=0.013894, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #47 = documents up to #4800/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #48 = documents up to #4900/9999, outstanding queue size 8
    INFO : PROGRESS: pass 3, dispatched chunk #49 = documents up to #5000/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #50 = documents up to #5100/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #11 (0.050): 0.116*"rt" + 0.019*"обам" + 0.017*"истор" + 0.016*"автомобил" + 0.016*"хорош" + 0.014*"связ" + 0.013*"крушен" + 0.013*"поддержива" + 0.012*"депутат" + 0.012*"совет"
    INFO : topic #13 (0.050): 0.077*"rt" + 0.031*"готов" + 0.027*"власт" + 0.025*"перв" + 0.024*"сми" + 0.023*"украин" + 0.019*"дел" + 0.017*"прав" + 0.016*"русск" + 0.012*"лет"
    INFO : topic #8 (0.050): 0.056*"rt" + 0.034*"рубл" + 0.024*"слов" + 0.020*"сторон" + 0.017*"росс" + 0.017*"миноборон" + 0.017*"запрет" + 0.017*"получ" + 0.015*"мужчин" + 0.013*"дтп"
    INFO : topic #9 (0.050): 0.056*"rt" + 0.047*"президент" + 0.024*"рф" + 0.015*"город" + 0.014*"нат" + 0.014*"росс" + 0.012*"эксперт" + 0.011*"обсуд" + 0.011*"произошл" + 0.011*"сентябр"
    INFO : topic #1 (0.050): 0.058*"стран" + 0.043*"rt" + 0.025*"дом" + 0.019*"жител" + 0.018*"пострада" + 0.017*"стал" + 0.017*"американск" + 0.015*"сем" + 0.013*"массов" + 0.013*"днем"
    INFO : topic diff=0.014053, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #51 = documents up to #5200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #52 = documents up to #5300/9999, outstanding queue size 8
    INFO : PROGRESS: pass 3, dispatched chunk #53 = documents up to #5400/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #54 = documents up to #5500/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #3 (0.050): 0.060*"rt" + 0.042*"москв" + 0.031*"донецк" + 0.021*"район" + 0.017*"прошл" + 0.014*"машин" + 0.014*"продаж" + 0.013*"центр" + 0.012*"ран" + 0.012*"мэр"
    INFO : topic #5 (0.050): 0.065*"rt" + 0.024*"войн" + 0.019*"задержа" + 0.015*"план" + 0.015*"ситуац" + 0.014*"украинц" + 0.014*"сотрудник" + 0.013*"стат" + 0.012*"обстрел" + 0.012*"россиян"
    INFO : topic #18 (0.050): 0.074*"rt" + 0.047*"нов" + 0.029*"санкц" + 0.024*"рф" + 0.022*"сам" + 0.020*"росс" + 0.017*"мид" + 0.013*"апрел" + 0.012*"иг" + 0.011*"представ"
    INFO : topic #2 (0.050): 0.051*"rt" + 0.032*"глав" + 0.031*"киев" + 0.022*"жизн" + 0.019*"кита" + 0.017*"погибл" + 0.017*"нашл" + 0.016*"взрыв" + 0.015*"пыта" + 0.014*"международн"
    INFO : topic #9 (0.050): 0.058*"rt" + 0.048*"президент" + 0.024*"рф" + 0.015*"город" + 0.013*"сентябр" + 0.013*"росс" + 0.012*"эксперт" + 0.012*"нат" + 0.011*"произошл" + 0.011*"участ"
    INFO : topic diff=0.010889, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #55 = documents up to #5600/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #56 = documents up to #5700/9999, outstanding queue size 8
    INFO : PROGRESS: pass 3, dispatched chunk #57 = documents up to #5800/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #19 (0.050): 0.057*"rt" + 0.048*"украинск" + 0.042*"крым" + 0.032*"ес" + 0.031*"украин" + 0.030*"воен" + 0.028*"политик" + 0.027*"сша" + 0.026*"фот" + 0.026*"днр"
    INFO : topic #15 (0.050): 0.050*"rt" + 0.033*"quot" + 0.027*"дорог" + 0.020*"сто" + 0.018*"предлож" + 0.015*"рук" + 0.012*"факт" + 0.012*"солдат" + 0.011*"заверш" + 0.011*"действ"
    INFO : topic #16 (0.050): 0.058*"rt" + 0.030*"нача" + 0.029*"чита" + 0.026*"пост" + 0.025*"добр" + 0.024*"мир" + 0.022*"как" + 0.020*"потеря" + 0.019*"написа" + 0.018*"школ"
    INFO : topic #14 (0.050): 0.084*"rt" + 0.037*"петербург" + 0.030*"вер" + 0.019*"украин" + 0.015*"медвед" + 0.015*"пожар" + 0.014*"восток" + 0.014*"матч" + 0.014*"ки" + 0.012*"сообщ"
    INFO : topic #11 (0.050): 0.115*"rt" + 0.021*"истор" + 0.018*"обам" + 0.016*"хорош" + 0.016*"автомобил" + 0.014*"связ" + 0.013*"депутат" + 0.013*"крушен" + 0.013*"числ" + 0.011*"поддержива"
    INFO : topic diff=0.012490, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #58 = documents up to #5900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #59 = documents up to #6000/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #60 = documents up to #6100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #61 = documents up to #6200/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #12 (0.050): 0.169*"rt" + 0.022*"люд" + 0.017*"дума" + 0.015*"украин" + 0.015*"возможн" + 0.014*"счита" + 0.013*"сша" + 0.012*"оруж" + 0.011*"цен" + 0.011*"автор"
    INFO : topic #18 (0.050): 0.074*"rt" + 0.051*"нов" + 0.027*"санкц" + 0.022*"сам" + 0.021*"рф" + 0.021*"росс" + 0.014*"мид" + 0.012*"апрел" + 0.012*"акц" + 0.012*"мчс"
    INFO : topic #7 (0.050): 0.077*"rt" + 0.026*"человек" + 0.020*"област" + 0.016*"люб" + 0.014*"март" + 0.013*"дня" + 0.013*"новосибирск" + 0.013*"суд" + 0.012*"музык" + 0.012*"пройдет"
    INFO : topic #9 (0.050): 0.057*"rt" + 0.048*"президент" + 0.027*"рф" + 0.015*"город" + 0.014*"нат" + 0.014*"эксперт" + 0.013*"росс" + 0.012*"говор" + 0.011*"сентябр" + 0.011*"евр"
    INFO : topic #13 (0.050): 0.079*"rt" + 0.035*"готов" + 0.028*"власт" + 0.027*"украин" + 0.026*"сми" + 0.026*"перв" + 0.021*"русск" + 0.019*"дел" + 0.016*"прав" + 0.012*"скор"
    INFO : topic diff=0.012626, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #62 = documents up to #6300/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #63 = documents up to #6400/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #64 = documents up to #6500/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #65 = documents up to #6600/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #66 = documents up to #6700/9999, outstanding queue size 10
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #19 (0.050): 0.057*"rt" + 0.047*"украинск" + 0.041*"крым" + 0.034*"украин" + 0.033*"воен" + 0.032*"ес" + 0.031*"политик" + 0.025*"днр" + 0.025*"сша" + 0.023*"чуж"
    INFO : topic #12 (0.050): 0.168*"rt" + 0.020*"люд" + 0.016*"дума" + 0.016*"возможн" + 0.016*"украин" + 0.014*"счита" + 0.012*"сша" + 0.012*"оруж" + 0.012*"цен" + 0.011*"убийств"
    INFO : topic #17 (0.050): 0.088*"украин" + 0.063*"российск" + 0.043*"rt" + 0.039*"новост" + 0.030*"мнен" + 0.024*"выбор" + 0.024*"донбасс" + 0.024*"побед" + 0.023*"газ" + 0.021*"арм"
    INFO : topic #4 (0.050): 0.039*"rt" + 0.032*"интересн" + 0.026*"появ" + 0.024*"закон" + 0.023*"европ" + 0.018*"границ" + 0.017*"отказа" + 0.017*"турц" + 0.013*"трамп" + 0.013*"нужн"
    INFO : topic #14 (0.050): 0.083*"rt" + 0.033*"петербург" + 0.030*"вер" + 0.022*"украин" + 0.018*"медвед" + 0.014*"восток" + 0.013*"матч" + 0.013*"ки" + 0.013*"пожар" + 0.011*"польш"
    INFO : topic diff=0.011891, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #67 = documents up to #6800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 3, dispatched chunk #68 = documents up to #6900/9999, outstanding queue size 8
    INFO : PROGRESS: pass 3, dispatched chunk #69 = documents up to #7000/9999, outstanding queue size 7
    INFO : merging changes from 700 documents into a model of 9999 documents
    INFO : topic #7 (0.050): 0.076*"rt" + 0.025*"человек" + 0.020*"област" + 0.018*"люб" + 0.014*"дня" + 0.014*"фильм" + 0.013*"суд" + 0.013*"март" + 0.012*"пройдет" + 0.012*"лидер"
    INFO : topic #2 (0.050): 0.055*"rt" + 0.037*"киев" + 0.032*"глав" + 0.021*"жизн" + 0.018*"взрыв" + 0.016*"пыта" + 0.015*"нашл" + 0.015*"кита" + 0.014*"результат" + 0.014*"погибл"
    INFO : topic #8 (0.050): 0.057*"rt" + 0.032*"рубл" + 0.026*"слов" + 0.019*"сторон" + 0.017*"получ" + 0.015*"миноборон" + 0.015*"переп" + 0.015*"росс" + 0.014*"дтп" + 0.013*"запрет"
    INFO : topic #4 (0.050): 0.039*"rt" + 0.032*"интересн" + 0.026*"появ" + 0.023*"европ" + 0.023*"закон" + 0.017*"границ" + 0.017*"турц" + 0.016*"отказа" + 0.015*"трамп" + 0.012*"очередн"
    INFO : topic #11 (0.050): 0.106*"rt" + 0.020*"истор" + 0.018*"обам" + 0.018*"хорош" + 0.014*"автомобил" + 0.013*"поддержива" + 0.012*"связ" + 0.012*"числ" + 0.012*"крушен" + 0.012*"мост"
    INFO : topic diff=0.011369, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #70 = documents up to #7100/9999, outstanding queue size 4
    INFO : PROGRESS: pass 3, dispatched chunk #71 = documents up to #7200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 3, dispatched chunk #72 = documents up to #7300/9999, outstanding queue size 3
    INFO : PROGRESS: pass 3, dispatched chunk #73 = documents up to #7400/9999, outstanding queue size 4
    INFO : PROGRESS: pass 3, dispatched chunk #74 = documents up to #7500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 3, dispatched chunk #75 = documents up to #7600/9999, outstanding queue size 6
    INFO : PROGRESS: pass 3, dispatched chunk #76 = documents up to #7700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 3, dispatched chunk #77 = documents up to #7800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 3, dispatched chunk #78 = documents up to #7900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #79 = documents up to #8000/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #17 (0.050): 0.087*"украин" + 0.063*"российск" + 0.042*"rt" + 0.041*"новост" + 0.034*"мнен" + 0.023*"выбор" + 0.023*"побед" + 0.023*"донбасс" + 0.022*"газ" + 0.020*"арм"
    INFO : topic #11 (0.050): 0.106*"rt" + 0.020*"истор" + 0.018*"хорош" + 0.017*"обам" + 0.015*"автомобил" + 0.013*"праздник" + 0.013*"связ" + 0.012*"депутат" + 0.012*"числ" + 0.012*"поддержива"
    INFO : topic #6 (0.050): 0.127*"путин" + 0.060*"rt" + 0.032*"росс" + 0.024*"владимир" + 0.020*"мест" + 0.018*"друз" + 0.017*"назва" + 0.017*"улиц" + 0.017*"работ" + 0.015*"сборн"
    INFO : topic #1 (0.050): 0.055*"стран" + 0.043*"rt" + 0.024*"пострада" + 0.024*"американск" + 0.022*"дом" + 0.020*"жител" + 0.016*"стал" + 0.015*"конц" + 0.015*"ожида" + 0.014*"ноч"
    INFO : topic #16 (0.050): 0.064*"rt" + 0.037*"нача" + 0.029*"чита" + 0.025*"добр" + 0.024*"мир" + 0.024*"пост" + 0.023*"как" + 0.020*"потеря" + 0.020*"написа" + 0.016*"школ"
    INFO : topic diff=0.010761, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #80 = documents up to #8100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #81 = documents up to #8200/9999, outstanding queue size 8
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #10 (0.050): 0.088*"rt" + 0.042*"виде" + 0.031*"дет" + 0.030*"лучш" + 0.023*"полиц" + 0.016*"уб" + 0.015*"последн" + 0.014*"бизнес" + 0.014*"хочет" + 0.013*"решен"
    INFO : topic #15 (0.050): 0.050*"rt" + 0.033*"quot" + 0.028*"дорог" + 0.025*"сто" + 0.015*"предлож" + 0.013*"памятник" + 0.013*"действ" + 0.012*"рук" + 0.012*"факт" + 0.012*"выход"
    INFO : topic #6 (0.050): 0.125*"путин" + 0.060*"rt" + 0.033*"росс" + 0.025*"владимир" + 0.019*"мест" + 0.017*"улиц" + 0.017*"друз" + 0.016*"работ" + 0.016*"назва" + 0.015*"сборн"
    INFO : topic #17 (0.050): 0.085*"украин" + 0.065*"российск" + 0.043*"rt" + 0.042*"новост" + 0.033*"мнен" + 0.025*"побед" + 0.024*"выбор" + 0.023*"арм" + 0.023*"донбасс" + 0.022*"газ"
    INFO : topic #8 (0.050): 0.059*"rt" + 0.032*"рубл" + 0.024*"слов" + 0.020*"получ" + 0.018*"сторон" + 0.015*"миноборон" + 0.015*"переп" + 0.015*"росс" + 0.015*"запрет" + 0.013*"мужчин"
    INFO : topic diff=0.010676, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #82 = documents up to #8300/9999, outstanding queue size 8
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #6 (0.050): 0.127*"путин" + 0.059*"rt" + 0.032*"росс" + 0.029*"владимир" + 0.020*"сборн" + 0.019*"улиц" + 0.018*"мест" + 0.017*"друз" + 0.015*"работ" + 0.015*"назва"
    INFO : topic #12 (0.050): 0.174*"rt" + 0.019*"люд" + 0.018*"дума" + 0.017*"возможн" + 0.015*"украин" + 0.014*"сша" + 0.012*"сирийск" + 0.011*"счита" + 0.011*"убийств" + 0.011*"оруж"
    INFO : topic #5 (0.050): 0.064*"rt" + 0.024*"войн" + 0.021*"задержа" + 0.017*"план" + 0.016*"смотр" + 0.015*"обстрел" + 0.015*"предлага" + 0.014*"стат" + 0.013*"россиян" + 0.013*"полицейск"
    INFO : topic #7 (0.050): 0.075*"rt" + 0.021*"человек" + 0.019*"люб" + 0.018*"област" + 0.018*"дня" + 0.016*"пройдет" + 0.013*"суд" + 0.013*"фильм" + 0.012*"отношен" + 0.012*"март"
    INFO : topic #1 (0.050): 0.057*"стран" + 0.046*"rt" + 0.023*"пострада" + 0.021*"дом" + 0.020*"американск" + 0.019*"жител" + 0.018*"стал" + 0.014*"конц" + 0.014*"днем" + 0.014*"ожида"
    INFO : topic diff=0.013232, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #83 = documents up to #8400/9999, outstanding queue size 5
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #0 (0.050): 0.076*"rt" + 0.049*"сир" + 0.022*"рад" + 0.020*"главн" + 0.016*"удар" + 0.015*"переговор" + 0.013*"увелич" + 0.012*"сша" + 0.012*"призва" + 0.011*"игр"
    INFO : topic #6 (0.050): 0.126*"путин" + 0.058*"rt" + 0.032*"росс" + 0.027*"владимир" + 0.020*"улиц" + 0.019*"сборн" + 0.016*"мест" + 0.016*"друз" + 0.014*"работ" + 0.013*"ход"
    INFO : topic #15 (0.050): 0.049*"rt" + 0.034*"quot" + 0.031*"дорог" + 0.023*"сто" + 0.014*"рук" + 0.014*"узна" + 0.014*"октябр" + 0.013*"выход" + 0.013*"предлож" + 0.012*"памятник"
    INFO : topic #9 (0.050): 0.058*"rt" + 0.048*"президент" + 0.027*"рф" + 0.019*"город" + 0.017*"нат" + 0.013*"эксперт" + 0.013*"говор" + 0.012*"росс" + 0.012*"евр" + 0.011*"обсуд"
    INFO : topic #3 (0.050): 0.059*"rt" + 0.045*"москв" + 0.037*"донецк" + 0.025*"район" + 0.017*"центр" + 0.014*"продаж" + 0.013*"известн" + 0.012*"прошл" + 0.012*"мэр" + 0.011*"ран"
    INFO : topic diff=0.010715, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #84 = documents up to #8500/9999, outstanding queue size 2
    INFO : PROGRESS: pass 3, dispatched chunk #85 = documents up to #8600/9999, outstanding queue size 2
    INFO : PROGRESS: pass 3, dispatched chunk #86 = documents up to #8700/9999, outstanding queue size 3
    INFO : PROGRESS: pass 3, dispatched chunk #87 = documents up to #8800/9999, outstanding queue size 4
    INFO : PROGRESS: pass 3, dispatched chunk #88 = documents up to #8900/9999, outstanding queue size 5
    INFO : PROGRESS: pass 3, dispatched chunk #89 = documents up to #9000/9999, outstanding queue size 6
    INFO : PROGRESS: pass 3, dispatched chunk #90 = documents up to #9100/9999, outstanding queue size 7
    INFO : PROGRESS: pass 3, dispatched chunk #91 = documents up to #9200/9999, outstanding queue size 8
    INFO : PROGRESS: pass 3, dispatched chunk #92 = documents up to #9300/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #93 = documents up to #9400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #94 = documents up to #9500/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #95 = documents up to #9600/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #16 (0.050): 0.060*"rt" + 0.033*"нача" + 0.031*"добр" + 0.026*"пост" + 0.025*"как" + 0.024*"чита" + 0.023*"мир" + 0.020*"написа" + 0.017*"потеря" + 0.015*"школ"
    INFO : topic #3 (0.050): 0.058*"rt" + 0.043*"москв" + 0.037*"донецк" + 0.025*"район" + 0.018*"центр" + 0.015*"известн" + 0.014*"продаж" + 0.013*"встреч" + 0.012*"машин" + 0.012*"прошл"
    INFO : topic #1 (0.050): 0.064*"стран" + 0.046*"rt" + 0.024*"дом" + 0.022*"пострада" + 0.022*"американск" + 0.020*"стал" + 0.018*"жител" + 0.017*"конц" + 0.014*"массов" + 0.014*"сем"
    INFO : topic #14 (0.050): 0.079*"rt" + 0.030*"петербург" + 0.026*"вер" + 0.021*"украин" + 0.020*"медвед" + 0.016*"матч" + 0.016*"пожар" + 0.016*"восток" + 0.013*"ки" + 0.010*"луганск"
    INFO : topic #9 (0.050): 0.058*"rt" + 0.050*"президент" + 0.028*"рф" + 0.017*"город" + 0.017*"нат" + 0.013*"говор" + 0.013*"эксперт" + 0.012*"росс" + 0.011*"немцов" + 0.011*"евр"
    INFO : topic diff=0.011192, rho=0.098063
    INFO : PROGRESS: pass 3, dispatched chunk #96 = documents up to #9700/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #97 = documents up to #9800/9999, outstanding queue size 9
    INFO : PROGRESS: pass 3, dispatched chunk #98 = documents up to #9900/9999, outstanding queue size 10
    INFO : PROGRESS: pass 3, dispatched chunk #99 = documents up to #9999/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #5 (0.050): 0.064*"rt" + 0.030*"войн" + 0.020*"задержа" + 0.018*"план" + 0.016*"обстрел" + 0.015*"россиян" + 0.015*"смотр" + 0.014*"предлага" + 0.013*"полицейск" + 0.012*"украинц"
    INFO : topic #18 (0.050): 0.080*"rt" + 0.055*"нов" + 0.029*"санкц" + 0.024*"сам" + 0.022*"рф" + 0.022*"росс" + 0.015*"иг" + 0.014*"апрел" + 0.013*"мид" + 0.013*"представ"
    INFO : topic #2 (0.050): 0.054*"rt" + 0.035*"киев" + 0.031*"глав" + 0.020*"жизн" + 0.018*"пыта" + 0.016*"взрыв" + 0.016*"результат" + 0.016*"международн" + 0.016*"погибл" + 0.015*"южн"
    INFO : topic #4 (0.050): 0.039*"rt" + 0.033*"интересн" + 0.025*"появ" + 0.021*"европ" + 0.020*"границ" + 0.017*"трамп" + 0.016*"закон" + 0.016*"турц" + 0.014*"очередн" + 0.013*"отказа"
    INFO : topic #8 (0.050): 0.061*"rt" + 0.037*"рубл" + 0.027*"слов" + 0.022*"сторон" + 0.022*"получ" + 0.016*"миноборон" + 0.015*"росс" + 0.014*"переп" + 0.013*"министр" + 0.013*"запрет"
    INFO : topic diff=0.013030, rho=0.098063
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #11 (0.050): 0.097*"rt" + 0.020*"хорош" + 0.019*"обам" + 0.018*"автомобил" + 0.016*"истор" + 0.015*"депутат" + 0.013*"ставропол" + 0.012*"час" + 0.012*"поддержива" + 0.011*"серг"
    INFO : topic #5 (0.050): 0.064*"rt" + 0.032*"войн" + 0.019*"задержа" + 0.017*"план" + 0.016*"обстрел" + 0.015*"ситуац" + 0.015*"смотр" + 0.015*"предлага" + 0.014*"полицейск" + 0.014*"россиян"
    INFO : topic #8 (0.050): 0.062*"rt" + 0.037*"рубл" + 0.027*"слов" + 0.024*"получ" + 0.021*"сторон" + 0.015*"миноборон" + 0.015*"росс" + 0.014*"запрет" + 0.013*"переп" + 0.012*"министр"
    INFO : topic #16 (0.050): 0.060*"rt" + 0.033*"нача" + 0.027*"добр" + 0.027*"пост" + 0.024*"как" + 0.024*"чита" + 0.023*"мир" + 0.021*"написа" + 0.018*"школ" + 0.015*"потеря"
    INFO : topic #10 (0.050): 0.085*"rt" + 0.039*"виде" + 0.035*"дет" + 0.026*"лучш" + 0.023*"полиц" + 0.018*"хочет" + 0.016*"московск" + 0.015*"решен" + 0.015*"уб" + 0.014*"метр"
    INFO : topic diff=0.008450, rho=0.098063
    INFO : -15.984 per-word bound, 64805.1 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : merging changes from 299 documents into a model of 9999 documents
    INFO : topic #12 (0.050): 0.184*"rt" + 0.021*"люд" + 0.016*"дума" + 0.016*"возможн" + 0.015*"сша" + 0.015*"счита" + 0.014*"украин" + 0.013*"убийств" + 0.011*"росс" + 0.011*"доллар"
    INFO : topic #17 (0.050): 0.094*"украин" + 0.067*"российск" + 0.048*"rt" + 0.041*"новост" + 0.034*"мнен" + 0.027*"выбор" + 0.025*"побед" + 0.022*"арм" + 0.020*"донбасс" + 0.019*"продолжа"
    INFO : topic #3 (0.050): 0.062*"rt" + 0.044*"москв" + 0.038*"донецк" + 0.021*"район" + 0.019*"прошл" + 0.017*"центр" + 0.013*"известн" + 0.012*"продаж" + 0.012*"дан" + 0.012*"силовик"
    INFO : topic #0 (0.050): 0.081*"rt" + 0.048*"сир" + 0.023*"главн" + 0.023*"рад" + 0.015*"переговор" + 0.014*"удар" + 0.013*"сша" + 0.012*"минск" + 0.012*"игр" + 0.011*"увелич"
    INFO : topic #11 (0.050): 0.097*"rt" + 0.021*"обам" + 0.019*"автомобил" + 0.018*"хорош" + 0.015*"истор" + 0.014*"ставропол" + 0.014*"депутат" + 0.013*"журналист" + 0.012*"крушен" + 0.012*"час"
    INFO : topic diff=0.013887, rho=0.098063
    INFO : -15.839 per-word bound, 58629.5 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : PROGRESS: pass 4, dispatched chunk #0 = documents up to #100/9999, outstanding queue size 1
    INFO : PROGRESS: pass 4, dispatched chunk #1 = documents up to #200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 4, dispatched chunk #2 = documents up to #300/9999, outstanding queue size 3
    INFO : PROGRESS: pass 4, dispatched chunk #3 = documents up to #400/9999, outstanding queue size 4
    INFO : PROGRESS: pass 4, dispatched chunk #4 = documents up to #500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 4, dispatched chunk #5 = documents up to #600/9999, outstanding queue size 6
    INFO : PROGRESS: pass 4, dispatched chunk #6 = documents up to #700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 4, dispatched chunk #7 = documents up to #800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 4, dispatched chunk #8 = documents up to #900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 4, dispatched chunk #9 = documents up to #1000/9999, outstanding queue size 9
    INFO : PROGRESS: pass 4, dispatched chunk #10 = documents up to #1100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 4, dispatched chunk #11 = documents up to #1200/9999, outstanding queue size 11
    INFO : PROGRESS: pass 4, dispatched chunk #12 = documents up to #1300/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #14 (0.050): 0.078*"rt" + 0.031*"петербург" + 0.025*"вер" + 0.021*"медвед" + 0.019*"украин" + 0.014*"матч" + 0.013*"пожар" + 0.013*"дмитр" + 0.013*"восток" + 0.012*"ки"
    INFO : topic #18 (0.050): 0.074*"rt" + 0.060*"нов" + 0.029*"санкц" + 0.024*"сам" + 0.022*"росс" + 0.018*"рф" + 0.015*"мид" + 0.013*"представ" + 0.013*"иг" + 0.013*"апрел"
    INFO : topic #13 (0.050): 0.074*"rt" + 0.029*"сми" + 0.029*"перв" + 0.025*"готов" + 0.022*"украин" + 0.021*"власт" + 0.018*"дел" + 0.017*"русск" + 0.015*"прав" + 0.013*"открыт"
    INFO : topic #0 (0.050): 0.080*"rt" + 0.044*"сир" + 0.021*"рад" + 0.020*"главн" + 0.014*"сша" + 0.013*"удар" + 0.013*"переговор" + 0.011*"минск" + 0.011*"игр" + 0.009*"призва"
    INFO : topic #8 (0.050): 0.060*"rt" + 0.039*"рубл" + 0.024*"сторон" + 0.023*"слов" + 0.020*"получ" + 0.015*"росс" + 0.015*"миноборон" + 0.013*"мужчин" + 0.012*"след" + 0.012*"запрет"
    INFO : topic diff=0.048463, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #13 = documents up to #1400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 4, dispatched chunk #14 = documents up to #1500/9999, outstanding queue size 10
    INFO : PROGRESS: pass 4, dispatched chunk #15 = documents up to #1600/9999, outstanding queue size 11
    INFO : PROGRESS: pass 4, dispatched chunk #16 = documents up to #1700/9999, outstanding queue size 12
    INFO : PROGRESS: pass 4, dispatched chunk #17 = documents up to #1800/9999, outstanding queue size 13
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #1 (0.050): 0.055*"стран" + 0.042*"rt" + 0.022*"дом" + 0.020*"стал" + 0.019*"жител" + 0.019*"американск" + 0.018*"пострада" + 0.016*"массов" + 0.014*"конц" + 0.013*"сем"
    INFO : topic #2 (0.050): 0.048*"rt" + 0.040*"глав" + 0.033*"киев" + 0.022*"жизн" + 0.016*"взрыв" + 0.015*"погибл" + 0.015*"пыта" + 0.014*"кита" + 0.014*"международн" + 0.013*"результат"
    INFO : topic #14 (0.050): 0.077*"rt" + 0.030*"петербург" + 0.023*"вер" + 0.021*"медвед" + 0.018*"украин" + 0.013*"матч" + 0.013*"дмитр" + 0.012*"восток" + 0.012*"сообщ" + 0.011*"пожар"
    INFO : topic #9 (0.050): 0.056*"rt" + 0.048*"президент" + 0.026*"рф" + 0.016*"город" + 0.014*"нат" + 0.012*"говор" + 0.011*"обсуд" + 0.011*"уф" + 0.011*"росс" + 0.010*"участ"
    INFO : topic #18 (0.050): 0.071*"rt" + 0.056*"нов" + 0.028*"санкц" + 0.023*"сам" + 0.021*"росс" + 0.017*"рф" + 0.015*"мид" + 0.012*"иг" + 0.012*"апрел" + 0.012*"сша"
    INFO : topic diff=0.041507, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #18 = documents up to #1900/9999, outstanding queue size 10
    INFO : PROGRESS: pass 4, dispatched chunk #19 = documents up to #2000/9999, outstanding queue size 9
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #5 (0.050): 0.062*"rt" + 0.028*"войн" + 0.019*"задержа" + 0.015*"обстрел" + 0.015*"план" + 0.014*"ситуац" + 0.013*"украинц" + 0.012*"полицейск" + 0.012*"россиян" + 0.011*"стат"
    INFO : topic #9 (0.050): 0.054*"rt" + 0.048*"президент" + 0.025*"рф" + 0.016*"город" + 0.013*"нат" + 0.012*"обсуд" + 0.012*"говор" + 0.011*"москв" + 0.011*"росс" + 0.010*"уф"
    INFO : topic #7 (0.050): 0.068*"rt" + 0.023*"област" + 0.016*"люб" + 0.015*"человек" + 0.014*"лидер" + 0.013*"дня" + 0.013*"отношен" + 0.012*"фильм" + 0.011*"пройдет" + 0.011*"суд"
    INFO : topic #19 (0.050): 0.054*"rt" + 0.045*"украинск" + 0.040*"ес" + 0.032*"воен" + 0.031*"политик" + 0.031*"крым" + 0.030*"украин" + 0.028*"сша" + 0.027*"фот" + 0.024*"чуж"
    INFO : topic #3 (0.050): 0.055*"rt" + 0.043*"москв" + 0.032*"донецк" + 0.022*"район" + 0.015*"центр" + 0.014*"прошл" + 0.014*"встреч" + 0.013*"машин" + 0.012*"банк" + 0.011*"известн"
    INFO : topic diff=0.020931, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #20 = documents up to #2100/9999, outstanding queue size 7
    INFO : PROGRESS: pass 4, dispatched chunk #21 = documents up to #2200/9999, outstanding queue size 7
    INFO : PROGRESS: pass 4, dispatched chunk #22 = documents up to #2300/9999, outstanding queue size 8
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #19 (0.050): 0.055*"rt" + 0.043*"украинск" + 0.042*"ес" + 0.033*"воен" + 0.032*"политик" + 0.032*"крым" + 0.031*"украин" + 0.027*"фот" + 0.027*"сша" + 0.026*"чуж"
    INFO : topic #4 (0.050): 0.037*"rt" + 0.032*"интересн" + 0.024*"появ" + 0.022*"европ" + 0.017*"границ" + 0.017*"турц" + 0.015*"закон" + 0.015*"трамп" + 0.014*"очередн" + 0.014*"развит"
    INFO : topic #12 (0.050): 0.163*"rt" + 0.018*"дума" + 0.018*"люд" + 0.016*"возможн" + 0.014*"украин" + 0.014*"счита" + 0.014*"сша" + 0.010*"кин" + 0.010*"росс" + 0.010*"цен"
    INFO : topic #1 (0.050): 0.054*"стран" + 0.043*"rt" + 0.025*"дом" + 0.021*"жител" + 0.019*"пострада" + 0.018*"стал" + 0.018*"американск" + 0.015*"массов" + 0.015*"днем" + 0.014*"сем"
    INFO : topic #13 (0.050): 0.073*"rt" + 0.029*"готов" + 0.026*"перв" + 0.025*"сми" + 0.024*"власт" + 0.022*"украин" + 0.018*"прав" + 0.018*"дел" + 0.017*"русск" + 0.012*"лет"
    INFO : topic diff=0.014007, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #23 = documents up to #2400/9999, outstanding queue size 5
    INFO : PROGRESS: pass 4, dispatched chunk #24 = documents up to #2500/9999, outstanding queue size 3
    INFO : PROGRESS: pass 4, dispatched chunk #25 = documents up to #2600/9999, outstanding queue size 4
    INFO : PROGRESS: pass 4, dispatched chunk #26 = documents up to #2700/9999, outstanding queue size 5
    INFO : PROGRESS: pass 4, dispatched chunk #27 = documents up to #2800/9999, outstanding queue size 6
    INFO : PROGRESS: pass 4, dispatched chunk #28 = documents up to #2900/9999, outstanding queue size 7
    INFO : PROGRESS: pass 4, dispatched chunk #29 = documents up to #3000/9999, outstanding queue size 8
    INFO : PROGRESS: pass 4, dispatched chunk #30 = documents up to #3100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 4, dispatched chunk #31 = documents up to #3200/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #13 (0.050): 0.073*"rt" + 0.027*"готов" + 0.026*"перв" + 0.025*"сми" + 0.023*"власт" + 0.023*"украин" + 0.020*"дел" + 0.018*"прав" + 0.016*"русск" + 0.013*"похож"
    INFO : topic #5 (0.050): 0.066*"rt" + 0.028*"войн" + 0.018*"задержа" + 0.016*"ситуац" + 0.016*"украинц" + 0.013*"обстрел" + 0.013*"план" + 0.013*"запад" + 0.012*"полицейск" + 0.012*"россиян"
    INFO : topic #11 (0.050): 0.096*"rt" + 0.018*"обам" + 0.018*"автомобил" + 0.016*"хорош" + 0.015*"истор" + 0.012*"крушен" + 0.012*"числ" + 0.012*"ставропол" + 0.011*"совет" + 0.011*"депутат"
    INFO : topic #9 (0.050): 0.055*"rt" + 0.045*"президент" + 0.026*"рф" + 0.017*"город" + 0.014*"нат" + 0.013*"обсуд" + 0.012*"росс" + 0.012*"уф" + 0.012*"говор" + 0.011*"млн"
    INFO : topic #10 (0.050): 0.083*"rt" + 0.043*"виде" + 0.027*"дет" + 0.021*"полиц" + 0.019*"хочет" + 0.018*"лучш" + 0.015*"московск" + 0.015*"уб" + 0.014*"решен" + 0.014*"бизнес"
    INFO : topic diff=0.011053, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #32 = documents up to #3300/9999, outstanding queue size 10
    INFO : PROGRESS: pass 4, dispatched chunk #33 = documents up to #3400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 4, dispatched chunk #34 = documents up to #3500/9999, outstanding queue size 11
    INFO : PROGRESS: pass 4, dispatched chunk #35 = documents up to #3600/9999, outstanding queue size 12
    INFO : PROGRESS: pass 4, dispatched chunk #36 = documents up to #3700/9999, outstanding queue size 11
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #19 (0.050): 0.055*"rt" + 0.044*"украинск" + 0.037*"ес" + 0.036*"крым" + 0.032*"политик" + 0.032*"воен" + 0.032*"украин" + 0.030*"фот" + 0.029*"сша" + 0.026*"чуж"
    INFO : topic #11 (0.050): 0.098*"rt" + 0.018*"обам" + 0.018*"автомобил" + 0.017*"хорош" + 0.014*"ставропол" + 0.014*"истор" + 0.014*"числ" + 0.012*"депутат" + 0.011*"крушен" + 0.011*"поддержива"
    INFO : topic #8 (0.050): 0.056*"rt" + 0.033*"рубл" + 0.025*"слов" + 0.024*"сторон" + 0.019*"миноборон" + 0.018*"получ" + 0.016*"дтп" + 0.015*"росс" + 0.014*"запрет" + 0.012*"переп"
    INFO : topic #5 (0.050): 0.067*"rt" + 0.028*"войн" + 0.018*"задержа" + 0.016*"ситуац" + 0.015*"обстрел" + 0.014*"украинц" + 0.013*"полицейск" + 0.012*"план" + 0.012*"смотр" + 0.012*"россиян"
    INFO : topic #10 (0.050): 0.082*"rt" + 0.045*"виде" + 0.033*"дет" + 0.021*"хочет" + 0.021*"полиц" + 0.018*"лучш" + 0.017*"московск" + 0.014*"бизнес" + 0.014*"уб" + 0.012*"решен"
    INFO : topic diff=0.011369, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #37 = documents up to #3800/9999, outstanding queue size 11
    INFO : PROGRESS: pass 4, dispatched chunk #38 = documents up to #3900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 4, dispatched chunk #39 = documents up to #4000/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #15 (0.050): 0.048*"rt" + 0.033*"quot" + 0.024*"дорог" + 0.023*"сто" + 0.015*"факт" + 0.014*"предлож" + 0.014*"рук" + 0.012*"выход" + 0.012*"солдат" + 0.012*"узна"
    INFO : topic #19 (0.050): 0.056*"rt" + 0.042*"украинск" + 0.035*"ес" + 0.035*"крым" + 0.034*"воен" + 0.032*"украин" + 0.031*"политик" + 0.029*"фот" + 0.028*"сша" + 0.026*"чуж"
    INFO : topic #5 (0.050): 0.069*"rt" + 0.027*"войн" + 0.018*"задержа" + 0.017*"ситуац" + 0.016*"украинц" + 0.016*"полицейск" + 0.016*"обстрел" + 0.015*"отправ" + 0.013*"стат" + 0.012*"план"
    INFO : topic #12 (0.050): 0.172*"rt" + 0.020*"люд" + 0.017*"дума" + 0.017*"возможн" + 0.015*"счита" + 0.014*"сша" + 0.013*"украин" + 0.012*"цен" + 0.011*"убийств" + 0.010*"росс"
    INFO : topic #2 (0.050): 0.050*"rt" + 0.040*"глав" + 0.028*"киев" + 0.020*"пыта" + 0.018*"жизн" + 0.017*"кита" + 0.016*"взрыв" + 0.016*"погибл" + 0.016*"нашл" + 0.015*"южн"
    INFO : topic diff=0.012473, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #40 = documents up to #4100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 4, dispatched chunk #41 = documents up to #4200/9999, outstanding queue size 8
    INFO : PROGRESS: pass 4, dispatched chunk #42 = documents up to #4300/9999, outstanding queue size 9
    INFO : PROGRESS: pass 4, dispatched chunk #43 = documents up to #4400/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.076*"rt" + 0.048*"нов" + 0.028*"санкц" + 0.023*"росс" + 0.022*"сам" + 0.017*"рф" + 0.016*"мид" + 0.015*"апрел" + 0.012*"иг" + 0.011*"представ"
    INFO : topic #0 (0.050): 0.080*"rt" + 0.043*"сир" + 0.019*"рад" + 0.017*"главн" + 0.013*"сша" + 0.013*"игр" + 0.013*"удар" + 0.011*"переговор" + 0.010*"минск" + 0.010*"пьян"
    INFO : topic #8 (0.050): 0.057*"rt" + 0.035*"рубл" + 0.025*"слов" + 0.025*"сторон" + 0.020*"миноборон" + 0.019*"получ" + 0.017*"росс" + 0.016*"запрет" + 0.014*"дтп" + 0.012*"мужчин"
    INFO : topic #15 (0.050): 0.048*"rt" + 0.037*"quot" + 0.023*"сто" + 0.022*"дорог" + 0.016*"предлож" + 0.015*"выход" + 0.014*"факт" + 0.014*"рук" + 0.013*"солдат" + 0.012*"сезон"
    INFO : topic #3 (0.050): 0.060*"rt" + 0.044*"москв" + 0.034*"донецк" + 0.022*"район" + 0.018*"центр" + 0.015*"машин" + 0.014*"мэр" + 0.013*"прошл" + 0.012*"встреч" + 0.012*"известн"
    INFO : topic diff=0.013017, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #44 = documents up to #4500/9999, outstanding queue size 10
    INFO : PROGRESS: pass 4, dispatched chunk #45 = documents up to #4600/9999, outstanding queue size 10
    INFO : PROGRESS: pass 4, dispatched chunk #46 = documents up to #4700/9999, outstanding queue size 11
    INFO : PROGRESS: pass 4, dispatched chunk #47 = documents up to #4800/9999, outstanding queue size 12
    INFO : PROGRESS: pass 4, dispatched chunk #48 = documents up to #4900/9999, outstanding queue size 13
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #10 (0.050): 0.088*"rt" + 0.042*"виде" + 0.028*"дет" + 0.023*"полиц" + 0.020*"лучш" + 0.019*"хочет" + 0.018*"московск" + 0.016*"бизнес" + 0.013*"решен" + 0.012*"метр"
    INFO : topic #4 (0.050): 0.039*"rt" + 0.029*"интересн" + 0.029*"появ" + 0.022*"европ" + 0.020*"турц" + 0.017*"отказа" + 0.016*"закон" + 0.016*"границ" + 0.013*"развит" + 0.012*"очередн"
    INFO : topic #11 (0.050): 0.103*"rt" + 0.019*"обам" + 0.017*"автомобил" + 0.017*"истор" + 0.015*"хорош" + 0.014*"связ" + 0.013*"поддержива" + 0.013*"крушен" + 0.013*"ставропол" + 0.012*"серг"
    INFO : topic #16 (0.050): 0.058*"rt" + 0.033*"нача" + 0.031*"чита" + 0.028*"мир" + 0.026*"пост" + 0.021*"добр" + 0.019*"написа" + 0.017*"как" + 0.017*"школ" + 0.016*"потеря"
    INFO : topic #18 (0.050): 0.077*"rt" + 0.048*"нов" + 0.029*"санкц" + 0.023*"сам" + 0.023*"росс" + 0.018*"рф" + 0.018*"мид" + 0.014*"апрел" + 0.012*"иг" + 0.011*"акц"
    INFO : topic diff=0.011708, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #49 = documents up to #5000/9999, outstanding queue size 11
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #4 (0.050): 0.039*"rt" + 0.030*"появ" + 0.028*"интересн" + 0.020*"турц" + 0.020*"европ" + 0.019*"закон" + 0.018*"отказа" + 0.015*"трамп" + 0.014*"границ" + 0.013*"омск"
    INFO : topic #10 (0.050): 0.088*"rt" + 0.042*"виде" + 0.029*"дет" + 0.025*"полиц" + 0.021*"хочет" + 0.020*"лучш" + 0.018*"московск" + 0.015*"бизнес" + 0.013*"решен" + 0.013*"уб"
    INFO : topic #17 (0.050): 0.103*"украин" + 0.068*"российск" + 0.047*"rt" + 0.042*"новост" + 0.034*"мнен" + 0.024*"побед" + 0.023*"арм" + 0.022*"донбасс" + 0.022*"выбор" + 0.021*"газ"
    INFO : topic #15 (0.050): 0.049*"rt" + 0.035*"quot" + 0.023*"дорог" + 0.020*"сто" + 0.015*"рук" + 0.015*"предлож" + 0.015*"факт" + 0.014*"выход" + 0.013*"солдат" + 0.012*"сезон"
    INFO : topic #6 (0.050): 0.120*"путин" + 0.056*"rt" + 0.030*"росс" + 0.028*"владимир" + 0.022*"мест" + 0.019*"назва" + 0.018*"улиц" + 0.017*"работ" + 0.017*"друз" + 0.012*"жил"
    INFO : topic diff=0.010098, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #50 = documents up to #5100/9999, outstanding queue size 6
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #11 (0.050): 0.101*"rt" + 0.019*"истор" + 0.018*"обам" + 0.016*"хорош" + 0.016*"связ" + 0.016*"автомобил" + 0.014*"депутат" + 0.013*"крушен" + 0.013*"числ" + 0.012*"поддержива"
    INFO : topic #17 (0.050): 0.098*"украин" + 0.070*"российск" + 0.048*"rt" + 0.040*"новост" + 0.032*"мнен" + 0.024*"побед" + 0.024*"донбасс" + 0.023*"выбор" + 0.022*"арм" + 0.022*"газ"
    INFO : topic #13 (0.050): 0.080*"rt" + 0.034*"готов" + 0.029*"власт" + 0.026*"сми" + 0.025*"украин" + 0.025*"перв" + 0.021*"дел" + 0.019*"прав" + 0.018*"русск" + 0.013*"лет"
    INFO : topic #4 (0.050): 0.039*"rt" + 0.032*"интересн" + 0.028*"появ" + 0.021*"европ" + 0.020*"турц" + 0.017*"закон" + 0.017*"отказа" + 0.016*"границ" + 0.013*"трамп" + 0.012*"нужн"
    INFO : topic #3 (0.050): 0.062*"rt" + 0.045*"москв" + 0.033*"донецк" + 0.022*"район" + 0.018*"центр" + 0.016*"прошл" + 0.016*"продаж" + 0.015*"машин" + 0.013*"ран" + 0.012*"мэр"
    INFO : topic diff=0.012585, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #51 = documents up to #5200/9999, outstanding queue size 3
    INFO : PROGRESS: pass 4, dispatched chunk #52 = documents up to #5300/9999, outstanding queue size 2
    INFO : PROGRESS: pass 4, dispatched chunk #53 = documents up to #5400/9999, outstanding queue size 3
    INFO : PROGRESS: pass 4, dispatched chunk #54 = documents up to #5500/9999, outstanding queue size 4
    INFO : PROGRESS: pass 4, dispatched chunk #55 = documents up to #5600/9999, outstanding queue size 5
    INFO : PROGRESS: pass 4, dispatched chunk #56 = documents up to #5700/9999, outstanding queue size 6
    INFO : PROGRESS: pass 4, dispatched chunk #57 = documents up to #5800/9999, outstanding queue size 7
    INFO : PROGRESS: pass 4, dispatched chunk #58 = documents up to #5900/9999, outstanding queue size 8
    INFO : PROGRESS: pass 4, dispatched chunk #59 = documents up to #6000/9999, outstanding queue size 9
    INFO : PROGRESS: pass 4, dispatched chunk #60 = documents up to #6100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 4, dispatched chunk #61 = documents up to #6200/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #17 (0.050): 0.103*"украин" + 0.067*"российск" + 0.047*"rt" + 0.044*"новост" + 0.032*"мнен" + 0.025*"побед" + 0.023*"выбор" + 0.023*"газ" + 0.022*"донбасс" + 0.020*"продолжа"
    INFO : topic #14 (0.050): 0.083*"rt" + 0.034*"петербург" + 0.031*"вер" + 0.020*"украин" + 0.018*"медвед" + 0.014*"ки" + 0.014*"восток" + 0.014*"пожар" + 0.013*"матч" + 0.012*"сообщ"
    INFO : topic #19 (0.050): 0.059*"rt" + 0.046*"украинск" + 0.041*"крым" + 0.036*"украин" + 0.032*"воен" + 0.032*"ес" + 0.029*"политик" + 0.029*"сша" + 0.027*"фот" + 0.025*"днр"
    INFO : topic #6 (0.050): 0.121*"путин" + 0.057*"rt" + 0.028*"росс" + 0.028*"владимир" + 0.022*"мест" + 0.019*"назва" + 0.018*"улиц" + 0.017*"друз" + 0.015*"работ" + 0.015*"жил"
    INFO : topic #18 (0.050): 0.075*"rt" + 0.056*"нов" + 0.027*"санкц" + 0.023*"сам" + 0.021*"росс" + 0.016*"рф" + 0.015*"мид" + 0.013*"апрел" + 0.013*"акц" + 0.011*"мчс"
    INFO : topic diff=0.012055, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #62 = documents up to #6300/9999, outstanding queue size 10
    INFO : PROGRESS: pass 4, dispatched chunk #63 = documents up to #6400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 4, dispatched chunk #64 = documents up to #6500/9999, outstanding queue size 11
    INFO : PROGRESS: pass 4, dispatched chunk #65 = documents up to #6600/9999, outstanding queue size 12
    INFO : PROGRESS: pass 4, dispatched chunk #66 = documents up to #6700/9999, outstanding queue size 13
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #8 (0.050): 0.056*"rt" + 0.033*"рубл" + 0.027*"слов" + 0.021*"сторон" + 0.019*"получ" + 0.018*"миноборон" + 0.015*"переп" + 0.014*"запрет" + 0.014*"дтп" + 0.014*"мужчин"
    INFO : topic #12 (0.050): 0.179*"rt" + 0.021*"люд" + 0.017*"дума" + 0.016*"возможн" + 0.016*"украин" + 0.015*"счита" + 0.013*"сша" + 0.013*"цен" + 0.012*"убийств" + 0.012*"оруж"
    INFO : topic #11 (0.050): 0.097*"rt" + 0.020*"обам" + 0.020*"истор" + 0.017*"хорош" + 0.014*"автомобил" + 0.013*"связ" + 0.013*"поддержива" + 0.013*"депутат" + 0.012*"крушен" + 0.012*"ставропол"
    INFO : topic #0 (0.050): 0.077*"rt" + 0.047*"сир" + 0.021*"рад" + 0.020*"главн" + 0.016*"удар" + 0.015*"переговор" + 0.013*"игр" + 0.012*"сша" + 0.011*"минск" + 0.010*"призва"
    INFO : topic #10 (0.050): 0.087*"rt" + 0.042*"виде" + 0.029*"дет" + 0.023*"полиц" + 0.022*"лучш" + 0.017*"хочет" + 0.015*"бизнес" + 0.014*"московск" + 0.014*"решен" + 0.014*"уб"
    INFO : topic diff=0.010436, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #67 = documents up to #6800/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #6 (0.050): 0.127*"путин" + 0.058*"rt" + 0.030*"росс" + 0.026*"владимир" + 0.020*"мест" + 0.018*"назва" + 0.016*"работ" + 0.016*"улиц" + 0.016*"друз" + 0.014*"сборн"
    INFO : topic #1 (0.050): 0.056*"стран" + 0.044*"rt" + 0.029*"американск" + 0.025*"пострада" + 0.020*"дом" + 0.019*"жител" + 0.015*"днем" + 0.015*"сем" + 0.014*"вид" + 0.014*"конц"
    INFO : topic #10 (0.050): 0.087*"rt" + 0.040*"виде" + 0.029*"дет" + 0.024*"лучш" + 0.023*"полиц" + 0.017*"хочет" + 0.016*"московск" + 0.016*"уб" + 0.014*"бизнес" + 0.013*"последн"
    INFO : topic #16 (0.050): 0.061*"rt" + 0.034*"нача" + 0.029*"чита" + 0.027*"пост" + 0.026*"мир" + 0.024*"добр" + 0.022*"как" + 0.021*"написа" + 0.020*"потеря" + 0.019*"школ"
    INFO : topic #7 (0.050): 0.074*"rt" + 0.026*"человек" + 0.022*"област" + 0.018*"люб" + 0.014*"дня" + 0.013*"суд" + 0.013*"новосибирск" + 0.012*"фильм" + 0.012*"март" + 0.012*"музык"
    INFO : topic diff=0.012545, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #68 = documents up to #6900/9999, outstanding queue size 7
    INFO : PROGRESS: pass 4, dispatched chunk #69 = documents up to #7000/9999, outstanding queue size 5
    INFO : PROGRESS: pass 4, dispatched chunk #70 = documents up to #7100/9999, outstanding queue size 6
    INFO : PROGRESS: pass 4, dispatched chunk #71 = documents up to #7200/9999, outstanding queue size 7
    INFO : PROGRESS: pass 4, dispatched chunk #72 = documents up to #7300/9999, outstanding queue size 8
    INFO : PROGRESS: pass 4, dispatched chunk #73 = documents up to #7400/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #0 (0.050): 0.075*"rt" + 0.051*"сир" + 0.020*"рад" + 0.020*"главн" + 0.018*"переговор" + 0.015*"удар" + 0.013*"игр" + 0.011*"минск" + 0.011*"сша" + 0.010*"пьян"
    INFO : topic #1 (0.050): 0.051*"стран" + 0.044*"rt" + 0.028*"американск" + 0.025*"пострада" + 0.022*"дом" + 0.018*"жител" + 0.015*"днем" + 0.015*"конц" + 0.014*"сем" + 0.013*"ноч"
    INFO : topic #5 (0.050): 0.064*"rt" + 0.023*"войн" + 0.020*"задержа" + 0.018*"обстрел" + 0.016*"полицейск" + 0.016*"план" + 0.016*"сотрудник" + 0.015*"россиян" + 0.014*"смотр" + 0.013*"предлага"
    INFO : topic #16 (0.050): 0.062*"rt" + 0.037*"нача" + 0.030*"чита" + 0.028*"добр" + 0.026*"пост" + 0.025*"мир" + 0.021*"как" + 0.020*"написа" + 0.018*"потеря" + 0.017*"школ"
    INFO : topic #11 (0.050): 0.093*"rt" + 0.021*"истор" + 0.018*"обам" + 0.018*"хорош" + 0.015*"автомобил" + 0.013*"поддержива" + 0.013*"мост" + 0.013*"водител" + 0.013*"связ" + 0.012*"числ"
    INFO : topic diff=0.009354, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #74 = documents up to #7500/9999, outstanding queue size 9
    INFO : PROGRESS: pass 4, dispatched chunk #75 = documents up to #7600/9999, outstanding queue size 8
    INFO : PROGRESS: pass 4, dispatched chunk #76 = documents up to #7700/9999, outstanding queue size 9
    INFO : PROGRESS: pass 4, dispatched chunk #77 = documents up to #7800/9999, outstanding queue size 9
    INFO : PROGRESS: pass 4, dispatched chunk #78 = documents up to #7900/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #7 (0.050): 0.074*"rt" + 0.023*"человек" + 0.022*"люб" + 0.019*"област" + 0.016*"дня" + 0.016*"фильм" + 0.014*"суд" + 0.013*"пройдет" + 0.013*"лидер" + 0.012*"отношен"
    INFO : topic #18 (0.050): 0.076*"rt" + 0.063*"нов" + 0.026*"росс" + 0.025*"санкц" + 0.024*"сам" + 0.014*"рф" + 0.013*"мчс" + 0.012*"западн" + 0.012*"представ" + 0.012*"иг"
    INFO : topic #4 (0.050): 0.036*"rt" + 0.033*"интересн" + 0.025*"европ" + 0.025*"появ" + 0.021*"закон" + 0.018*"турц" + 0.017*"трамп" + 0.016*"границ" + 0.015*"отказа" + 0.014*"очередн"
    INFO : topic #15 (0.050): 0.047*"rt" + 0.037*"quot" + 0.029*"дорог" + 0.022*"сто" + 0.016*"выход" + 0.015*"предлож" + 0.014*"факт" + 0.014*"памятник" + 0.013*"сезон" + 0.013*"рук"
    INFO : topic #12 (0.050): 0.176*"rt" + 0.020*"люд" + 0.018*"возможн" + 0.016*"дума" + 0.015*"украин" + 0.013*"цен" + 0.013*"счита" + 0.012*"сша" + 0.011*"сирийск" + 0.011*"кин"
    INFO : topic diff=0.013424, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #79 = documents up to #8000/9999, outstanding queue size 10
    INFO : PROGRESS: pass 4, dispatched chunk #80 = documents up to #8100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 4, dispatched chunk #81 = documents up to #8200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 4, dispatched chunk #82 = documents up to #8300/9999, outstanding queue size 11
    INFO : merging changes from 800 documents into a model of 9999 documents
    INFO : topic #10 (0.050): 0.090*"rt" + 0.040*"виде" + 0.032*"дет" + 0.030*"лучш" + 0.023*"полиц" + 0.018*"московск" + 0.015*"уб" + 0.015*"последн" + 0.014*"хочет" + 0.013*"бизнес"
    INFO : topic #14 (0.050): 0.080*"rt" + 0.029*"петербург" + 0.029*"вер" + 0.021*"медвед" + 0.020*"украин" + 0.016*"матч" + 0.015*"восток" + 0.015*"пожар" + 0.014*"ки" + 0.012*"дмитр"
    INFO : topic #18 (0.050): 0.078*"rt" + 0.062*"нов" + 0.026*"санкц" + 0.026*"росс" + 0.024*"сам" + 0.016*"рф" + 0.015*"иг" + 0.013*"мид" + 0.012*"мчс" + 0.012*"представ"
    INFO : topic #4 (0.050): 0.037*"rt" + 0.033*"интересн" + 0.027*"появ" + 0.024*"европ" + 0.021*"закон" + 0.018*"турц" + 0.017*"трамп" + 0.016*"границ" + 0.014*"отказа" + 0.013*"очередн"
    INFO : topic #6 (0.050): 0.127*"путин" + 0.058*"rt" + 0.030*"росс" + 0.025*"владимир" + 0.019*"назва" + 0.019*"улиц" + 0.018*"мест" + 0.017*"друз" + 0.017*"сборн" + 0.015*"работ"
    INFO : topic diff=0.007009, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #83 = documents up to #8400/9999, outstanding queue size 6
    INFO : PROGRESS: pass 4, dispatched chunk #84 = documents up to #8500/9999, outstanding queue size 4
    INFO : PROGRESS: pass 4, dispatched chunk #85 = documents up to #8600/9999, outstanding queue size 5
    INFO : PROGRESS: pass 4, dispatched chunk #86 = documents up to #8700/9999, outstanding queue size 6
    INFO : PROGRESS: pass 4, dispatched chunk #87 = documents up to #8800/9999, outstanding queue size 7
    INFO : PROGRESS: pass 4, dispatched chunk #88 = documents up to #8900/9999, outstanding queue size 8
    INFO : PROGRESS: pass 4, dispatched chunk #89 = documents up to #9000/9999, outstanding queue size 9
    INFO : PROGRESS: pass 4, dispatched chunk #90 = documents up to #9100/9999, outstanding queue size 10
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #8 (0.050): 0.060*"rt" + 0.033*"рубл" + 0.026*"слов" + 0.021*"получ" + 0.021*"сторон" + 0.015*"министр" + 0.015*"миноборон" + 0.014*"росс" + 0.014*"переп" + 0.014*"дтп"
    INFO : topic #13 (0.050): 0.078*"rt" + 0.033*"сми" + 0.027*"готов" + 0.026*"перв" + 0.025*"украин" + 0.024*"власт" + 0.019*"русск" + 0.019*"дел" + 0.016*"прав" + 0.015*"террорист"
    INFO : topic #18 (0.050): 0.078*"rt" + 0.061*"нов" + 0.027*"санкц" + 0.025*"росс" + 0.025*"сам" + 0.016*"рф" + 0.015*"иг" + 0.013*"мчс" + 0.013*"апрел" + 0.013*"представ"
    INFO : topic #4 (0.050): 0.036*"rt" + 0.036*"интересн" + 0.025*"появ" + 0.025*"европ" + 0.020*"закон" + 0.017*"турц" + 0.016*"трамп" + 0.015*"границ" + 0.014*"отказа" + 0.013*"очередн"
    INFO : topic #15 (0.050): 0.048*"rt" + 0.035*"quot" + 0.031*"дорог" + 0.023*"сто" + 0.016*"выход" + 0.013*"рук" + 0.013*"предлож" + 0.013*"октябр" + 0.013*"узна" + 0.013*"памятник"
    INFO : topic diff=0.009548, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #91 = documents up to #9200/9999, outstanding queue size 9
    INFO : PROGRESS: pass 4, dispatched chunk #92 = documents up to #9300/9999, outstanding queue size 7
    INFO : PROGRESS: pass 4, dispatched chunk #93 = documents up to #9400/9999, outstanding queue size 8
    INFO : PROGRESS: pass 4, dispatched chunk #94 = documents up to #9500/9999, outstanding queue size 9
    INFO : PROGRESS: pass 4, dispatched chunk #95 = documents up to #9600/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #4 (0.050): 0.036*"rt" + 0.035*"интересн" + 0.025*"появ" + 0.024*"европ" + 0.018*"закон" + 0.017*"границ" + 0.017*"трамп" + 0.016*"турц" + 0.013*"отказа" + 0.013*"очередн"
    INFO : topic #18 (0.050): 0.079*"rt" + 0.064*"нов" + 0.028*"санкц" + 0.026*"росс" + 0.025*"сам" + 0.016*"рф" + 0.015*"представ" + 0.014*"иг" + 0.013*"мчс" + 0.012*"западн"
    INFO : topic #19 (0.050): 0.061*"rt" + 0.043*"украинск" + 0.039*"ес" + 0.036*"воен" + 0.035*"украин" + 0.033*"политик" + 0.032*"крым" + 0.031*"сша" + 0.027*"чуж" + 0.024*"помощ"
    INFO : topic #1 (0.050): 0.062*"стран" + 0.045*"rt" + 0.025*"американск" + 0.024*"дом" + 0.022*"пострада" + 0.019*"стал" + 0.018*"жител" + 0.017*"конц" + 0.015*"сем" + 0.014*"днем"
    INFO : topic #17 (0.050): 0.117*"украин" + 0.064*"российск" + 0.053*"новост" + 0.046*"rt" + 0.035*"мнен" + 0.023*"выбор" + 0.022*"арм" + 0.022*"донбасс" + 0.021*"побед" + 0.020*"газ"
    INFO : topic diff=0.010046, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #96 = documents up to #9700/9999, outstanding queue size 10
    INFO : PROGRESS: pass 4, dispatched chunk #97 = documents up to #9800/9999, outstanding queue size 8
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #6 (0.050): 0.133*"путин" + 0.058*"rt" + 0.030*"росс" + 0.028*"владимир" + 0.019*"улиц" + 0.016*"сборн" + 0.016*"назва" + 0.016*"работ" + 0.016*"мест" + 0.015*"друз"
    INFO : topic #0 (0.050): 0.079*"rt" + 0.048*"сир" + 0.021*"рад" + 0.021*"главн" + 0.016*"переговор" + 0.014*"удар" + 0.013*"игр" + 0.013*"сша" + 0.013*"минск" + 0.011*"призва"
    INFO : topic #3 (0.050): 0.060*"rt" + 0.045*"москв" + 0.036*"донецк" + 0.024*"район" + 0.023*"центр" + 0.015*"прошл" + 0.015*"известн" + 0.014*"продаж" + 0.012*"машин" + 0.012*"встреч"
    INFO : topic #17 (0.050): 0.115*"украин" + 0.067*"российск" + 0.053*"новост" + 0.048*"rt" + 0.036*"мнен" + 0.024*"выбор" + 0.022*"побед" + 0.021*"арм" + 0.020*"донбасс" + 0.019*"газ"
    INFO : topic #10 (0.050): 0.089*"rt" + 0.042*"виде" + 0.033*"дет" + 0.028*"лучш" + 0.023*"полиц" + 0.018*"московск" + 0.018*"хочет" + 0.015*"решен" + 0.014*"уб" + 0.013*"подозрева"
    INFO : topic diff=0.012179, rho=0.097595
    INFO : PROGRESS: pass 4, dispatched chunk #98 = documents up to #9900/9999, outstanding queue size 8
    INFO : PROGRESS: pass 4, dispatched chunk #99 = documents up to #9999/9999, outstanding queue size 8
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #2 (0.050): 0.053*"rt" + 0.039*"глав" + 0.032*"киев" + 0.021*"жизн" + 0.017*"взрыв" + 0.017*"пыта" + 0.017*"южн" + 0.016*"международн" + 0.016*"погибл" + 0.015*"кита"
    INFO : topic #14 (0.050): 0.079*"rt" + 0.031*"петербург" + 0.026*"медвед" + 0.025*"вер" + 0.020*"украин" + 0.015*"дмитр" + 0.014*"восток" + 0.014*"матч" + 0.014*"пожар" + 0.012*"сообщ"
    INFO : topic #6 (0.050): 0.132*"путин" + 0.059*"rt" + 0.031*"росс" + 0.026*"владимир" + 0.021*"назва" + 0.021*"улиц" + 0.019*"работ" + 0.018*"мест" + 0.016*"сборн" + 0.014*"друз"
    INFO : topic #19 (0.050): 0.063*"rt" + 0.043*"украинск" + 0.038*"воен" + 0.037*"ес" + 0.032*"украин" + 0.030*"политик" + 0.029*"крым" + 0.029*"сша" + 0.026*"чуж" + 0.026*"фот"
    INFO : topic #4 (0.050): 0.036*"rt" + 0.033*"интересн" + 0.024*"появ" + 0.024*"европ" + 0.019*"границ" + 0.019*"турц" + 0.018*"трамп" + 0.017*"закон" + 0.015*"очередн" + 0.013*"отказа"
    INFO : topic diff=0.009251, rho=0.097595
    INFO : -15.968 per-word bound, 64107.1 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : merging changes from 399 documents into a model of 9999 documents
    INFO : topic #3 (0.050): 0.062*"rt" + 0.045*"москв" + 0.039*"донецк" + 0.022*"район" + 0.021*"центр" + 0.020*"прошл" + 0.013*"известн" + 0.012*"дан" + 0.012*"продаж" + 0.012*"силовик"
    INFO : topic #2 (0.050): 0.053*"rt" + 0.038*"глав" + 0.032*"киев" + 0.025*"жизн" + 0.017*"взрыв" + 0.016*"пыта" + 0.016*"южн" + 0.016*"погибл" + 0.016*"кита" + 0.015*"международн"
    INFO : topic #4 (0.050): 0.036*"rt" + 0.036*"интересн" + 0.026*"появ" + 0.026*"европ" + 0.020*"границ" + 0.018*"трамп" + 0.016*"турц" + 0.016*"очередн" + 0.016*"закон" + 0.014*"развит"
    INFO : topic #18 (0.050): 0.079*"rt" + 0.064*"нов" + 0.028*"санкц" + 0.026*"росс" + 0.024*"сам" + 0.016*"рф" + 0.015*"мид" + 0.014*"представ" + 0.014*"сша" + 0.014*"апрел"
    INFO : topic #7 (0.050): 0.072*"rt" + 0.021*"област" + 0.020*"люб" + 0.017*"человек" + 0.017*"фильм" + 0.016*"дня" + 0.015*"лидер" + 0.014*"отношен" + 0.014*"пройдет" + 0.013*"март"
    INFO : topic diff=0.011548, rho=0.097595
    INFO : -15.859 per-word bound, 59444.9 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : PROGRESS: pass 5, dispatched chunk #0 = documents up to #100/9999, outstanding queue size 1
    INFO : PROGRESS: pass 5, dispatched chunk #1 = documents up to #200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 5, dispatched chunk #2 = documents up to #300/9999, outstanding queue size 3
    INFO : PROGRESS: pass 5, dispatched chunk #3 = documents up to #400/9999, outstanding queue size 4
    INFO : PROGRESS: pass 5, dispatched chunk #4 = documents up to #500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 5, dispatched chunk #5 = documents up to #600/9999, outstanding queue size 6
    INFO : PROGRESS: pass 5, dispatched chunk #6 = documents up to #700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 5, dispatched chunk #7 = documents up to #800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 5, dispatched chunk #8 = documents up to #900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #9 = documents up to #1000/9999, outstanding queue size 10
    INFO : PROGRESS: pass 5, dispatched chunk #10 = documents up to #1100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #11 = documents up to #1200/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #13 (0.050): 0.074*"rt" + 0.031*"сми" + 0.030*"перв" + 0.025*"готов" + 0.021*"украин" + 0.020*"власт" + 0.020*"дел" + 0.016*"прав" + 0.016*"русск" + 0.013*"открыт"
    INFO : topic #10 (0.050): 0.084*"rt" + 0.039*"виде" + 0.035*"дет" + 0.024*"лучш" + 0.020*"полиц" + 0.019*"хочет" + 0.017*"московск" + 0.017*"бизнес" + 0.015*"уб" + 0.014*"решен"
    INFO : topic #3 (0.050): 0.058*"rt" + 0.048*"москв" + 0.035*"донецк" + 0.022*"центр" + 0.020*"район" + 0.018*"прошл" + 0.013*"встреч" + 0.012*"известн" + 0.012*"дан" + 0.011*"продаж"
    INFO : topic #12 (0.050): 0.175*"rt" + 0.020*"люд" + 0.018*"возможн" + 0.016*"дума" + 0.014*"счита" + 0.014*"сша" + 0.014*"украин" + 0.012*"убийств" + 0.011*"цен" + 0.010*"росс"
    INFO : topic #0 (0.050): 0.079*"rt" + 0.045*"сир" + 0.022*"рад" + 0.020*"главн" + 0.014*"переговор" + 0.014*"удар" + 0.014*"сша" + 0.012*"игр" + 0.012*"минск" + 0.009*"призва"
    INFO : topic diff=0.048103, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #12 = documents up to #1300/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #13 = documents up to #1400/9999, outstanding queue size 8
    INFO : PROGRESS: pass 5, dispatched chunk #14 = documents up to #1500/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #15 = documents up to #1600/9999, outstanding queue size 10
    INFO : PROGRESS: pass 5, dispatched chunk #16 = documents up to #1700/9999, outstanding queue size 11
    INFO : PROGRESS: pass 5, dispatched chunk #17 = documents up to #1800/9999, outstanding queue size 12
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #12 (0.050): 0.168*"rt" + 0.018*"люд" + 0.017*"дума" + 0.017*"возможн" + 0.014*"сша" + 0.014*"украин" + 0.013*"счита" + 0.011*"убийств" + 0.010*"кин" + 0.010*"цен"
    INFO : topic #8 (0.050): 0.057*"rt" + 0.036*"рубл" + 0.024*"слов" + 0.023*"сторон" + 0.020*"получ" + 0.014*"миноборон" + 0.013*"мужчин" + 0.013*"дтп" + 0.013*"министр" + 0.012*"росс"
    INFO : topic #0 (0.050): 0.078*"rt" + 0.044*"сир" + 0.019*"рад" + 0.017*"главн" + 0.014*"удар" + 0.014*"сша" + 0.014*"переговор" + 0.011*"игр" + 0.011*"минск" + 0.009*"призва"
    INFO : topic #13 (0.050): 0.071*"rt" + 0.030*"сми" + 0.028*"перв" + 0.024*"готов" + 0.022*"украин" + 0.021*"власт" + 0.019*"дел" + 0.017*"прав" + 0.015*"русск" + 0.013*"открыт"
    INFO : topic #11 (0.050): 0.083*"rt" + 0.018*"обам" + 0.017*"истор" + 0.016*"автомобил" + 0.016*"хорош" + 0.013*"журналист" + 0.013*"ставропол" + 0.012*"крушен" + 0.011*"числ" + 0.011*"депутат"
    INFO : topic diff=0.040964, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #18 = documents up to #1900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #19 = documents up to #2000/9999, outstanding queue size 7
    INFO : PROGRESS: pass 5, dispatched chunk #20 = documents up to #2100/9999, outstanding queue size 8
    INFO : PROGRESS: pass 5, dispatched chunk #21 = documents up to #2200/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #19 (0.050): 0.056*"rt" + 0.044*"украинск" + 0.040*"ес" + 0.035*"воен" + 0.032*"политик" + 0.032*"украин" + 0.030*"крым" + 0.029*"сша" + 0.028*"фот" + 0.024*"чуж"
    INFO : topic #6 (0.050): 0.116*"путин" + 0.051*"rt" + 0.031*"владимир" + 0.030*"росс" + 0.019*"назва" + 0.017*"улиц" + 0.017*"работ" + 0.017*"сборн" + 0.015*"мест" + 0.014*"друз"
    INFO : topic #15 (0.050): 0.046*"rt" + 0.033*"quot" + 0.025*"дорог" + 0.021*"сто" + 0.015*"факт" + 0.013*"постро" + 0.013*"узна" + 0.011*"заверш" + 0.011*"предлож" + 0.011*"выход"
    INFO : topic #8 (0.050): 0.055*"rt" + 0.034*"рубл" + 0.023*"сторон" + 0.021*"слов" + 0.019*"получ" + 0.017*"миноборон" + 0.014*"дтп" + 0.014*"запрет" + 0.013*"росс" + 0.013*"переп"
    INFO : topic #17 (0.050): 0.111*"украин" + 0.061*"российск" + 0.056*"новост" + 0.045*"rt" + 0.036*"мнен" + 0.024*"выбор" + 0.020*"продолжа" + 0.020*"побед" + 0.017*"газ" + 0.016*"донбасс"
    INFO : topic diff=0.016919, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #22 = documents up to #2300/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #23 = documents up to #2400/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #24 = documents up to #2500/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #25 = documents up to #2600/9999, outstanding queue size 10
    INFO : PROGRESS: pass 5, dispatched chunk #26 = documents up to #2700/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #9 (0.050): 0.052*"rt" + 0.048*"президент" + 0.032*"рф" + 0.015*"город" + 0.014*"нат" + 0.013*"говор" + 0.012*"обсуд" + 0.012*"росс" + 0.011*"млн" + 0.010*"участ"
    INFO : topic #8 (0.050): 0.056*"rt" + 0.038*"рубл" + 0.023*"слов" + 0.021*"сторон" + 0.018*"миноборон" + 0.018*"получ" + 0.015*"дтп" + 0.015*"запрет" + 0.014*"росс" + 0.012*"мужчин"
    INFO : topic #3 (0.050): 0.057*"rt" + 0.043*"москв" + 0.034*"донецк" + 0.023*"район" + 0.018*"центр" + 0.015*"прошл" + 0.013*"встреч" + 0.012*"машин" + 0.012*"банк" + 0.011*"дан"
    INFO : topic #12 (0.050): 0.166*"rt" + 0.019*"дума" + 0.017*"люд" + 0.016*"возможн" + 0.016*"счита" + 0.013*"сша" + 0.013*"украин" + 0.010*"цен" + 0.010*"убийств" + 0.010*"кин"
    INFO : topic #18 (0.050): 0.074*"rt" + 0.061*"нов" + 0.027*"санкц" + 0.023*"росс" + 0.021*"сам" + 0.015*"мид" + 0.014*"рф" + 0.012*"иг" + 0.012*"сильн" + 0.011*"сша"
    INFO : topic diff=0.013565, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #27 = documents up to #2800/9999, outstanding queue size 10
    INFO : PROGRESS: pass 5, dispatched chunk #28 = documents up to #2900/9999, outstanding queue size 8
    INFO : PROGRESS: pass 5, dispatched chunk #29 = documents up to #3000/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #30 = documents up to #3100/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #12 (0.050): 0.170*"rt" + 0.019*"люд" + 0.019*"дума" + 0.017*"возможн" + 0.017*"счита" + 0.014*"сша" + 0.014*"украин" + 0.012*"цен" + 0.010*"убийств" + 0.010*"кин"
    INFO : topic #8 (0.050): 0.056*"rt" + 0.037*"рубл" + 0.024*"слов" + 0.022*"сторон" + 0.018*"миноборон" + 0.018*"получ" + 0.015*"дтп" + 0.014*"запрет" + 0.014*"росс" + 0.012*"млрд"
    INFO : topic #17 (0.050): 0.117*"украин" + 0.066*"российск" + 0.057*"новост" + 0.050*"rt" + 0.034*"мнен" + 0.025*"выбор" + 0.020*"побед" + 0.019*"газ" + 0.018*"продолжа" + 0.015*"арм"
    INFO : topic #9 (0.050): 0.054*"rt" + 0.044*"президент" + 0.032*"рф" + 0.016*"город" + 0.015*"нат" + 0.013*"росс" + 0.013*"обсуд" + 0.012*"уф" + 0.012*"млн" + 0.012*"участ"
    INFO : topic #14 (0.050): 0.081*"rt" + 0.032*"петербург" + 0.024*"вер" + 0.019*"медвед" + 0.018*"украин" + 0.013*"дмитр" + 0.013*"матч" + 0.012*"пожар" + 0.012*"восток" + 0.010*"сообщ"
    INFO : topic diff=0.008900, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #31 = documents up to #3200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 5, dispatched chunk #32 = documents up to #3300/9999, outstanding queue size 8
    INFO : PROGRESS: pass 5, dispatched chunk #33 = documents up to #3400/9999, outstanding queue size 9
    INFO : merging changes from 700 documents into a model of 9999 documents
    INFO : topic #6 (0.050): 0.119*"путин" + 0.055*"rt" + 0.033*"владимир" + 0.028*"росс" + 0.021*"улиц" + 0.017*"назва" + 0.016*"мест" + 0.016*"работ" + 0.014*"друз" + 0.014*"ход"
    INFO : topic #2 (0.050): 0.050*"rt" + 0.038*"глав" + 0.030*"киев" + 0.019*"пыта" + 0.018*"кита" + 0.017*"жизн" + 0.017*"взрыв" + 0.016*"нашл" + 0.016*"погибл" + 0.015*"результат"
    INFO : topic #18 (0.050): 0.076*"rt" + 0.053*"нов" + 0.028*"санкц" + 0.024*"росс" + 0.024*"сам" + 0.015*"мид" + 0.014*"рф" + 0.013*"апрел" + 0.013*"сильн" + 0.012*"иг"
    INFO : topic #10 (0.050): 0.086*"rt" + 0.044*"виде" + 0.030*"дет" + 0.022*"полиц" + 0.021*"хочет" + 0.020*"лучш" + 0.018*"московск" + 0.014*"уб" + 0.013*"бизнес" + 0.012*"решен"
    INFO : topic #5 (0.050): 0.067*"rt" + 0.029*"войн" + 0.018*"полицейск" + 0.018*"задержа" + 0.017*"ситуац" + 0.016*"украинц" + 0.015*"обстрел" + 0.014*"запад" + 0.013*"отправ" + 0.012*"россиян"
    INFO : topic diff=0.010479, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #34 = documents up to #3500/9999, outstanding queue size 6
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #12 (0.050): 0.178*"rt" + 0.021*"люд" + 0.017*"возможн" + 0.017*"дума" + 0.016*"счита" + 0.014*"сша" + 0.013*"украин" + 0.013*"цен" + 0.011*"убийств" + 0.010*"автор"
    INFO : topic #1 (0.050): 0.053*"стран" + 0.044*"rt" + 0.028*"дом" + 0.019*"американск" + 0.018*"жител" + 0.017*"пострада" + 0.017*"стал" + 0.016*"сем" + 0.014*"днем" + 0.013*"росс"
    INFO : topic #0 (0.050): 0.081*"rt" + 0.044*"сир" + 0.022*"рад" + 0.015*"главн" + 0.015*"игр" + 0.013*"сша" + 0.013*"удар" + 0.011*"пьян" + 0.011*"переговор" + 0.010*"минск"
    INFO : topic #2 (0.050): 0.052*"rt" + 0.038*"глав" + 0.029*"киев" + 0.020*"пыта" + 0.019*"жизн" + 0.017*"нашл" + 0.016*"взрыв" + 0.016*"кита" + 0.015*"международн" + 0.015*"погибл"
    INFO : topic #14 (0.050): 0.082*"rt" + 0.030*"петербург" + 0.025*"вер" + 0.019*"медвед" + 0.018*"украин" + 0.015*"пожар" + 0.014*"дмитр" + 0.012*"матч" + 0.012*"восток" + 0.011*"польш"
    INFO : topic diff=0.012046, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #35 = documents up to #3600/9999, outstanding queue size 3
    INFO : PROGRESS: pass 5, dispatched chunk #36 = documents up to #3700/9999, outstanding queue size 2
    INFO : PROGRESS: pass 5, dispatched chunk #37 = documents up to #3800/9999, outstanding queue size 3
    INFO : PROGRESS: pass 5, dispatched chunk #38 = documents up to #3900/9999, outstanding queue size 4
    INFO : PROGRESS: pass 5, dispatched chunk #39 = documents up to #4000/9999, outstanding queue size 5
    INFO : PROGRESS: pass 5, dispatched chunk #40 = documents up to #4100/9999, outstanding queue size 6
    INFO : PROGRESS: pass 5, dispatched chunk #41 = documents up to #4200/9999, outstanding queue size 7
    INFO : PROGRESS: pass 5, dispatched chunk #42 = documents up to #4300/9999, outstanding queue size 8
    INFO : PROGRESS: pass 5, dispatched chunk #43 = documents up to #4400/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #44 = documents up to #4500/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.076*"rt" + 0.053*"нов" + 0.029*"санкц" + 0.026*"росс" + 0.021*"сам" + 0.016*"мид" + 0.015*"рф" + 0.014*"апрел" + 0.012*"иг" + 0.012*"сша"
    INFO : topic #6 (0.050): 0.117*"путин" + 0.054*"rt" + 0.029*"росс" + 0.029*"владимир" + 0.021*"назва" + 0.020*"улиц" + 0.019*"мест" + 0.018*"друз" + 0.017*"работ" + 0.013*"жил"
    INFO : topic #19 (0.050): 0.059*"rt" + 0.041*"украинск" + 0.036*"воен" + 0.035*"крым" + 0.034*"ес" + 0.034*"украин" + 0.031*"политик" + 0.029*"фот" + 0.027*"сша" + 0.024*"чуж"
    INFO : topic #0 (0.050): 0.078*"rt" + 0.043*"сир" + 0.020*"рад" + 0.017*"главн" + 0.014*"игр" + 0.013*"сша" + 0.013*"призва" + 0.012*"удар" + 0.012*"переговор" + 0.010*"пьян"
    INFO : topic #9 (0.050): 0.055*"rt" + 0.050*"президент" + 0.036*"рф" + 0.015*"город" + 0.014*"нат" + 0.014*"росс" + 0.013*"млн" + 0.013*"участ" + 0.012*"обсуд" + 0.011*"говор"
    INFO : topic diff=0.010743, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #45 = documents up to #4600/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #46 = documents up to #4700/9999, outstanding queue size 8
    INFO : PROGRESS: pass 5, dispatched chunk #47 = documents up to #4800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 5, dispatched chunk #48 = documents up to #4900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #49 = documents up to #5000/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.078*"rt" + 0.053*"нов" + 0.030*"санкц" + 0.025*"росс" + 0.023*"сам" + 0.019*"мид" + 0.017*"рф" + 0.013*"апрел" + 0.012*"сша" + 0.011*"акц"
    INFO : topic #8 (0.050): 0.057*"rt" + 0.037*"рубл" + 0.024*"слов" + 0.022*"сторон" + 0.019*"получ" + 0.019*"миноборон" + 0.017*"запрет" + 0.015*"росс" + 0.014*"мужчин" + 0.014*"дтп"
    INFO : topic #1 (0.050): 0.059*"стран" + 0.043*"rt" + 0.024*"дом" + 0.022*"американск" + 0.019*"жител" + 0.017*"пострада" + 0.015*"стал" + 0.015*"сем" + 0.014*"днем" + 0.014*"поздрав"
    INFO : topic #10 (0.050): 0.089*"rt" + 0.040*"виде" + 0.027*"дет" + 0.025*"полиц" + 0.022*"лучш" + 0.021*"хочет" + 0.019*"московск" + 0.014*"бизнес" + 0.014*"решен" + 0.013*"уб"
    INFO : topic #6 (0.050): 0.121*"путин" + 0.056*"rt" + 0.029*"росс" + 0.028*"владимир" + 0.021*"назва" + 0.021*"мест" + 0.019*"улиц" + 0.018*"работ" + 0.018*"друз" + 0.013*"жил"
    INFO : topic diff=0.011150, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #50 = documents up to #5100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 5, dispatched chunk #51 = documents up to #5200/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #52 = documents up to #5300/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #53 = documents up to #5400/9999, outstanding queue size 10
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #9 (0.050): 0.057*"rt" + 0.049*"президент" + 0.037*"рф" + 0.015*"город" + 0.014*"нат" + 0.013*"росс" + 0.013*"участ" + 0.013*"сентябр" + 0.012*"млн" + 0.012*"эксперт"
    INFO : topic #14 (0.050): 0.082*"rt" + 0.034*"петербург" + 0.031*"вер" + 0.018*"медвед" + 0.017*"украин" + 0.015*"пожар" + 0.013*"восток" + 0.012*"матч" + 0.012*"ки" + 0.012*"дмитр"
    INFO : topic #6 (0.050): 0.123*"путин" + 0.057*"rt" + 0.028*"росс" + 0.028*"владимир" + 0.023*"мест" + 0.021*"назва" + 0.019*"улиц" + 0.019*"работ" + 0.016*"друз" + 0.013*"жил"
    INFO : topic #8 (0.050): 0.057*"rt" + 0.037*"рубл" + 0.024*"слов" + 0.021*"сторон" + 0.019*"получ" + 0.018*"миноборон" + 0.016*"запрет" + 0.016*"росс" + 0.015*"дтп" + 0.014*"мужчин"
    INFO : topic #18 (0.050): 0.076*"rt" + 0.055*"нов" + 0.030*"санкц" + 0.023*"росс" + 0.023*"сам" + 0.017*"мид" + 0.016*"рф" + 0.013*"апрел" + 0.012*"иг" + 0.012*"сша"
    INFO : topic diff=0.007576, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #54 = documents up to #5500/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #3 (0.050): 0.062*"rt" + 0.047*"москв" + 0.033*"донецк" + 0.025*"район" + 0.020*"центр" + 0.016*"прошл" + 0.015*"продаж" + 0.014*"ран" + 0.013*"машин" + 0.012*"силовик"
    INFO : topic #10 (0.050): 0.089*"rt" + 0.041*"виде" + 0.028*"дет" + 0.024*"полиц" + 0.020*"хочет" + 0.019*"лучш" + 0.018*"московск" + 0.015*"уб" + 0.014*"последн" + 0.013*"бизнес"
    INFO : topic #5 (0.050): 0.067*"rt" + 0.025*"войн" + 0.019*"задержа" + 0.019*"полицейск" + 0.015*"план" + 0.014*"ситуац" + 0.014*"стат" + 0.014*"сотрудник" + 0.013*"украинц" + 0.013*"отправ"
    INFO : topic #8 (0.050): 0.057*"rt" + 0.037*"рубл" + 0.024*"слов" + 0.021*"сторон" + 0.019*"получ" + 0.017*"миноборон" + 0.016*"запрет" + 0.016*"мужчин" + 0.015*"дтп" + 0.015*"росс"
    INFO : topic #13 (0.050): 0.080*"rt" + 0.034*"готов" + 0.029*"власт" + 0.027*"украин" + 0.026*"сми" + 0.026*"перв" + 0.023*"дел" + 0.019*"прав" + 0.018*"русск" + 0.012*"скор"
    INFO : topic diff=0.011586, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #55 = documents up to #5600/9999, outstanding queue size 6
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #15 (0.050): 0.047*"rt" + 0.034*"quot" + 0.031*"дорог" + 0.023*"сто" + 0.018*"предлож" + 0.013*"рук" + 0.013*"выход" + 0.012*"крупн" + 0.012*"солдат" + 0.012*"узна"
    INFO : topic #10 (0.050): 0.087*"rt" + 0.040*"виде" + 0.027*"дет" + 0.022*"полиц" + 0.020*"лучш" + 0.019*"хочет" + 0.017*"московск" + 0.015*"уб" + 0.015*"решен" + 0.015*"последн"
    INFO : topic #12 (0.050): 0.182*"rt" + 0.022*"люд" + 0.017*"дума" + 0.015*"возможн" + 0.014*"счита" + 0.013*"украин" + 0.013*"сша" + 0.012*"цен" + 0.012*"убийств" + 0.012*"оруж"
    INFO : topic #2 (0.050): 0.055*"rt" + 0.038*"глав" + 0.035*"киев" + 0.019*"жизн" + 0.018*"кита" + 0.017*"взрыв" + 0.016*"результат" + 0.016*"нашл" + 0.016*"международн" + 0.016*"погибл"
    INFO : topic #19 (0.050): 0.062*"rt" + 0.046*"украинск" + 0.040*"крым" + 0.038*"воен" + 0.037*"украин" + 0.032*"ес" + 0.030*"политик" + 0.027*"сша" + 0.027*"фот" + 0.026*"днр"
    INFO : topic diff=0.011346, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #56 = documents up to #5700/9999, outstanding queue size 3
    INFO : PROGRESS: pass 5, dispatched chunk #57 = documents up to #5800/9999, outstanding queue size 2
    INFO : PROGRESS: pass 5, dispatched chunk #58 = documents up to #5900/9999, outstanding queue size 3
    INFO : PROGRESS: pass 5, dispatched chunk #59 = documents up to #6000/9999, outstanding queue size 4
    INFO : PROGRESS: pass 5, dispatched chunk #60 = documents up to #6100/9999, outstanding queue size 5
    INFO : PROGRESS: pass 5, dispatched chunk #61 = documents up to #6200/9999, outstanding queue size 6
    INFO : PROGRESS: pass 5, dispatched chunk #62 = documents up to #6300/9999, outstanding queue size 7
    INFO : PROGRESS: pass 5, dispatched chunk #63 = documents up to #6400/9999, outstanding queue size 8
    INFO : PROGRESS: pass 5, dispatched chunk #64 = documents up to #6500/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #65 = documents up to #6600/9999, outstanding queue size 10
    INFO : PROGRESS: pass 5, dispatched chunk #66 = documents up to #6700/9999, outstanding queue size 10
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #0 (0.050): 0.077*"rt" + 0.049*"сир" + 0.022*"рад" + 0.019*"главн" + 0.016*"переговор" + 0.016*"удар" + 0.013*"игр" + 0.012*"сша" + 0.011*"минск" + 0.010*"призва"
    INFO : topic #17 (0.050): 0.125*"украин" + 0.065*"российск" + 0.062*"новост" + 0.050*"rt" + 0.031*"мнен" + 0.023*"газ" + 0.022*"выбор" + 0.022*"побед" + 0.021*"донбасс" + 0.018*"арм"
    INFO : topic #6 (0.050): 0.127*"путин" + 0.058*"rt" + 0.029*"росс" + 0.027*"владимир" + 0.021*"мест" + 0.019*"назва" + 0.018*"улиц" + 0.016*"работ" + 0.016*"друз" + 0.013*"жил"
    INFO : topic #1 (0.050): 0.055*"стран" + 0.044*"rt" + 0.032*"американск" + 0.024*"пострада" + 0.021*"дом" + 0.021*"жител" + 0.015*"конц" + 0.015*"днем" + 0.013*"вид" + 0.013*"росс"
    INFO : topic #7 (0.050): 0.074*"rt" + 0.027*"человек" + 0.022*"област" + 0.017*"люб" + 0.015*"дня" + 0.015*"суд" + 0.014*"март" + 0.013*"новосибирск" + 0.012*"фильм" + 0.012*"лидер"
    INFO : topic diff=0.011277, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #67 = documents up to #6800/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #68 = documents up to #6900/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #7 (0.050): 0.074*"rt" + 0.025*"человек" + 0.022*"област" + 0.019*"люб" + 0.015*"дня" + 0.014*"суд" + 0.013*"фильм" + 0.013*"музык" + 0.012*"март" + 0.012*"новосибирск"
    INFO : topic #15 (0.050): 0.048*"rt" + 0.035*"quot" + 0.031*"дорог" + 0.023*"сто" + 0.016*"предлож" + 0.014*"рук" + 0.013*"сезон" + 0.013*"выход" + 0.013*"памятник" + 0.012*"действ"
    INFO : topic #18 (0.050): 0.077*"rt" + 0.068*"нов" + 0.028*"санкц" + 0.026*"росс" + 0.022*"сам" + 0.014*"рф" + 0.012*"иг" + 0.012*"акц" + 0.012*"мчс" + 0.012*"мид"
    INFO : topic #13 (0.050): 0.079*"rt" + 0.031*"сми" + 0.031*"готов" + 0.029*"перв" + 0.028*"украин" + 0.026*"власт" + 0.023*"дел" + 0.020*"русск" + 0.017*"прав" + 0.012*"скор"
    INFO : topic #19 (0.050): 0.062*"rt" + 0.044*"украинск" + 0.039*"воен" + 0.038*"крым" + 0.037*"украин" + 0.030*"ес" + 0.029*"политик" + 0.027*"сша" + 0.025*"фот" + 0.024*"днр"
    INFO : topic diff=0.010970, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #69 = documents up to #7000/9999, outstanding queue size 7
    INFO : PROGRESS: pass 5, dispatched chunk #70 = documents up to #7100/9999, outstanding queue size 5
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #15 (0.050): 0.047*"rt" + 0.036*"quot" + 0.029*"дорог" + 0.024*"сто" + 0.016*"предлож" + 0.015*"выход" + 0.013*"памятник" + 0.013*"рук" + 0.013*"сезон" + 0.013*"факт"
    INFO : topic #3 (0.050): 0.060*"rt" + 0.042*"москв" + 0.039*"донецк" + 0.025*"район" + 0.022*"центр" + 0.015*"прошл" + 0.013*"продаж" + 0.013*"мэр" + 0.012*"силовик" + 0.012*"машин"
    INFO : topic #4 (0.050): 0.034*"rt" + 0.033*"интересн" + 0.026*"появ" + 0.026*"европ" + 0.024*"закон" + 0.018*"турц" + 0.017*"границ" + 0.016*"трамп" + 0.015*"отказа" + 0.013*"очередн"
    INFO : topic #5 (0.050): 0.063*"rt" + 0.024*"войн" + 0.020*"задержа" + 0.018*"полицейск" + 0.017*"обстрел" + 0.016*"план" + 0.015*"сотрудник" + 0.015*"россиян" + 0.013*"ситуац" + 0.013*"смотр"
    INFO : topic #16 (0.050): 0.065*"rt" + 0.039*"нача" + 0.029*"чита" + 0.028*"добр" + 0.026*"мир" + 0.025*"пост" + 0.021*"как" + 0.020*"написа" + 0.020*"рассказа" + 0.018*"потеря"
    INFO : topic diff=0.009909, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #71 = documents up to #7200/9999, outstanding queue size 5
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #12 (0.050): 0.179*"rt" + 0.020*"люд" + 0.018*"возможн" + 0.017*"дума" + 0.014*"счита" + 0.013*"цен" + 0.012*"сша" + 0.012*"украин" + 0.012*"убийств" + 0.012*"кин"
    INFO : topic #11 (0.050): 0.088*"rt" + 0.021*"истор" + 0.020*"хорош" + 0.018*"обам" + 0.016*"автомобил" + 0.014*"праздник" + 0.014*"связ" + 0.012*"поддержива" + 0.012*"числ" + 0.012*"депутат"
    INFO : topic #8 (0.050): 0.058*"rt" + 0.033*"рубл" + 0.026*"слов" + 0.022*"получ" + 0.019*"сторон" + 0.017*"переп" + 0.016*"миноборон" + 0.015*"запрет" + 0.015*"дтп" + 0.014*"мужчин"
    INFO : topic #2 (0.050): 0.056*"rt" + 0.038*"киев" + 0.037*"глав" + 0.019*"взрыв" + 0.018*"жизн" + 0.018*"пыта" + 0.015*"нашл" + 0.014*"погибл" + 0.014*"международн" + 0.014*"южн"
    INFO : topic #19 (0.050): 0.062*"rt" + 0.044*"украинск" + 0.038*"воен" + 0.036*"крым" + 0.035*"ес" + 0.034*"украин" + 0.031*"политик" + 0.027*"фот" + 0.026*"сша" + 0.024*"помощ"
    INFO : topic diff=0.009231, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #72 = documents up to #7300/9999, outstanding queue size 2
    INFO : PROGRESS: pass 5, dispatched chunk #73 = documents up to #7400/9999, outstanding queue size 2
    INFO : PROGRESS: pass 5, dispatched chunk #74 = documents up to #7500/9999, outstanding queue size 3
    INFO : PROGRESS: pass 5, dispatched chunk #75 = documents up to #7600/9999, outstanding queue size 4
    INFO : PROGRESS: pass 5, dispatched chunk #76 = documents up to #7700/9999, outstanding queue size 5
    INFO : PROGRESS: pass 5, dispatched chunk #77 = documents up to #7800/9999, outstanding queue size 6
    INFO : PROGRESS: pass 5, dispatched chunk #78 = documents up to #7900/9999, outstanding queue size 7
    INFO : PROGRESS: pass 5, dispatched chunk #79 = documents up to #8000/9999, outstanding queue size 8
    INFO : PROGRESS: pass 5, dispatched chunk #80 = documents up to #8100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #81 = documents up to #8200/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #82 = documents up to #8300/9999, outstanding queue size 10
    INFO : PROGRESS: pass 5, dispatched chunk #83 = documents up to #8400/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #13 (0.050): 0.077*"rt" + 0.032*"сми" + 0.028*"готов" + 0.028*"перв" + 0.027*"украин" + 0.022*"власт" + 0.021*"дел" + 0.019*"русск" + 0.017*"террорист" + 0.016*"прав"
    INFO : topic #19 (0.050): 0.063*"rt" + 0.044*"украинск" + 0.040*"воен" + 0.034*"украин" + 0.033*"крым" + 0.033*"ес" + 0.029*"политик" + 0.028*"фот" + 0.026*"сша" + 0.023*"помощ"
    INFO : topic #15 (0.050): 0.048*"rt" + 0.034*"quot" + 0.027*"дорог" + 0.026*"сто" + 0.015*"предлож" + 0.014*"памятник" + 0.013*"факт" + 0.013*"октябр" + 0.013*"выход" + 0.013*"действ"
    INFO : topic #4 (0.050): 0.035*"rt" + 0.033*"интересн" + 0.025*"появ" + 0.025*"европ" + 0.021*"закон" + 0.020*"турц" + 0.019*"трамп" + 0.015*"отказа" + 0.015*"границ" + 0.013*"очередн"
    INFO : topic #2 (0.050): 0.055*"rt" + 0.037*"киев" + 0.034*"глав" + 0.019*"жизн" + 0.018*"пыта" + 0.017*"взрыв" + 0.015*"международн" + 0.014*"южн" + 0.014*"результат" + 0.014*"нашл"
    INFO : topic diff=0.009384, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #84 = documents up to #8500/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #11 (0.050): 0.085*"rt" + 0.020*"обам" + 0.020*"истор" + 0.019*"хорош" + 0.017*"автомобил" + 0.014*"поддержива" + 0.013*"праздник" + 0.013*"депутат" + 0.013*"связ" + 0.012*"числ"
    INFO : topic #14 (0.050): 0.079*"rt" + 0.030*"петербург" + 0.027*"вер" + 0.021*"медвед" + 0.018*"украин" + 0.017*"матч" + 0.016*"восток" + 0.016*"пожар" + 0.015*"ки" + 0.013*"дмитр"
    INFO : topic #3 (0.050): 0.061*"rt" + 0.045*"москв" + 0.037*"донецк" + 0.028*"район" + 0.022*"центр" + 0.014*"встреч" + 0.013*"прошл" + 0.013*"мэр" + 0.013*"известн" + 0.012*"машин"
    INFO : topic #18 (0.050): 0.080*"rt" + 0.065*"нов" + 0.028*"санкц" + 0.028*"росс" + 0.024*"сам" + 0.016*"рф" + 0.016*"иг" + 0.014*"мид" + 0.013*"представ" + 0.012*"западн"
    INFO : topic #17 (0.050): 0.118*"украин" + 0.072*"российск" + 0.059*"новост" + 0.049*"rt" + 0.034*"мнен" + 0.022*"побед" + 0.022*"выбор" + 0.020*"газ" + 0.020*"донбасс" + 0.020*"арм"
    INFO : topic diff=0.014248, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #85 = documents up to #8600/9999, outstanding queue size 7
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #11 (0.050): 0.083*"rt" + 0.020*"обам" + 0.019*"хорош" + 0.019*"истор" + 0.019*"автомобил" + 0.014*"депутат" + 0.014*"связ" + 0.013*"праздник" + 0.013*"поддержива" + 0.013*"журналист"
    INFO : topic #2 (0.050): 0.054*"rt" + 0.036*"киев" + 0.034*"глав" + 0.022*"пыта" + 0.018*"жизн" + 0.017*"погибл" + 0.017*"взрыв" + 0.016*"результат" + 0.015*"южн" + 0.015*"международн"
    INFO : topic #1 (0.050): 0.060*"стран" + 0.046*"rt" + 0.025*"американск" + 0.022*"дом" + 0.021*"пострада" + 0.020*"стал" + 0.017*"конц" + 0.017*"жител" + 0.014*"массов" + 0.014*"ожида"
    INFO : topic #15 (0.050): 0.048*"rt" + 0.034*"quot" + 0.031*"дорог" + 0.024*"сто" + 0.015*"октябр" + 0.014*"выход" + 0.014*"рук" + 0.014*"узна" + 0.013*"предлож" + 0.013*"памятник"
    INFO : topic #16 (0.050): 0.062*"rt" + 0.034*"нача" + 0.027*"добр" + 0.026*"чита" + 0.025*"пост" + 0.024*"как" + 0.024*"мир" + 0.019*"написа" + 0.018*"рассказа" + 0.017*"школ"
    INFO : topic diff=0.011478, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #86 = documents up to #8700/9999, outstanding queue size 4
    INFO : PROGRESS: pass 5, dispatched chunk #87 = documents up to #8800/9999, outstanding queue size 2
    INFO : PROGRESS: pass 5, dispatched chunk #88 = documents up to #8900/9999, outstanding queue size 3
    INFO : PROGRESS: pass 5, dispatched chunk #89 = documents up to #9000/9999, outstanding queue size 4
    INFO : PROGRESS: pass 5, dispatched chunk #90 = documents up to #9100/9999, outstanding queue size 5
    INFO : PROGRESS: pass 5, dispatched chunk #91 = documents up to #9200/9999, outstanding queue size 6
    INFO : PROGRESS: pass 5, dispatched chunk #92 = documents up to #9300/9999, outstanding queue size 7
    INFO : PROGRESS: pass 5, dispatched chunk #93 = documents up to #9400/9999, outstanding queue size 8
    INFO : PROGRESS: pass 5, dispatched chunk #94 = documents up to #9500/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #3 (0.050): 0.060*"rt" + 0.046*"москв" + 0.038*"донецк" + 0.026*"район" + 0.023*"центр" + 0.015*"известн" + 0.014*"встреч" + 0.014*"продаж" + 0.012*"прошл" + 0.012*"машин"
    INFO : topic #5 (0.050): 0.065*"rt" + 0.030*"войн" + 0.022*"задержа" + 0.017*"план" + 0.017*"обстрел" + 0.016*"полицейск" + 0.016*"смотр" + 0.015*"россиян" + 0.015*"предлага" + 0.014*"запад"
    INFO : topic #17 (0.050): 0.130*"украин" + 0.068*"российск" + 0.057*"новост" + 0.050*"rt" + 0.034*"мнен" + 0.023*"выбор" + 0.022*"побед" + 0.021*"арм" + 0.020*"донбасс" + 0.019*"газ"
    INFO : topic #14 (0.050): 0.077*"rt" + 0.030*"петербург" + 0.027*"вер" + 0.022*"медвед" + 0.018*"украин" + 0.017*"матч" + 0.016*"пожар" + 0.016*"восток" + 0.014*"ки" + 0.012*"дмитр"
    INFO : topic #19 (0.050): 0.062*"rt" + 0.043*"украинск" + 0.041*"воен" + 0.040*"ес" + 0.035*"украин" + 0.032*"политик" + 0.031*"сша" + 0.030*"крым" + 0.027*"чуж" + 0.025*"фот"
    INFO : topic diff=0.009777, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #95 = documents up to #9600/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #96 = documents up to #9700/9999, outstanding queue size 9
    INFO : PROGRESS: pass 5, dispatched chunk #97 = documents up to #9800/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #9 (0.050): 0.057*"rt" + 0.048*"президент" + 0.044*"рф" + 0.017*"нат" + 0.015*"город" + 0.013*"говор" + 0.013*"уф" + 0.012*"росс" + 0.012*"приня" + 0.011*"эксперт"
    INFO : topic #8 (0.050): 0.062*"rt" + 0.038*"рубл" + 0.028*"слов" + 0.023*"получ" + 0.023*"сторон" + 0.016*"миноборон" + 0.015*"переп" + 0.014*"министр" + 0.014*"росс" + 0.013*"запрет"
    INFO : topic #7 (0.050): 0.075*"rt" + 0.021*"област" + 0.019*"люб" + 0.018*"человек" + 0.018*"пройдет" + 0.018*"дня" + 0.016*"март" + 0.015*"суд" + 0.015*"отношен" + 0.013*"лидер"
    INFO : topic #5 (0.050): 0.064*"rt" + 0.031*"войн" + 0.021*"задержа" + 0.018*"план" + 0.017*"полицейск" + 0.016*"обстрел" + 0.015*"запад" + 0.015*"россиян" + 0.015*"смотр" + 0.014*"предлага"
    INFO : topic #13 (0.050): 0.083*"rt" + 0.031*"перв" + 0.031*"сми" + 0.026*"готов" + 0.025*"украин" + 0.024*"дел" + 0.023*"власт" + 0.020*"русск" + 0.015*"прав" + 0.015*"заяв"
    INFO : topic diff=0.012212, rho=0.097133
    INFO : PROGRESS: pass 5, dispatched chunk #98 = documents up to #9900/9999, outstanding queue size 8
    INFO : PROGRESS: pass 5, dispatched chunk #99 = documents up to #9999/9999, outstanding queue size 6
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #4 (0.050): 0.035*"интересн" + 0.035*"rt" + 0.027*"появ" + 0.025*"европ" + 0.020*"границ" + 0.018*"трамп" + 0.018*"турц" + 0.016*"закон" + 0.015*"очередн" + 0.013*"отказа"
    INFO : topic #3 (0.050): 0.062*"rt" + 0.046*"москв" + 0.036*"донецк" + 0.024*"центр" + 0.023*"район" + 0.017*"прошл" + 0.015*"известн" + 0.013*"встреч" + 0.012*"продаж" + 0.012*"банк"
    INFO : topic #14 (0.050): 0.077*"rt" + 0.033*"петербург" + 0.025*"медвед" + 0.025*"вер" + 0.018*"украин" + 0.016*"матч" + 0.015*"дмитр" + 0.015*"пожар" + 0.015*"восток" + 0.013*"ки"
    INFO : topic #19 (0.050): 0.065*"rt" + 0.043*"украинск" + 0.042*"воен" + 0.040*"ес" + 0.033*"украин" + 0.031*"политик" + 0.030*"сша" + 0.027*"крым" + 0.027*"чуж" + 0.026*"фот"
    INFO : topic #12 (0.050): 0.189*"rt" + 0.018*"люд" + 0.017*"дума" + 0.017*"возможн" + 0.016*"сша" + 0.015*"счита" + 0.014*"убийств" + 0.012*"цен" + 0.011*"доллар" + 0.011*"кин"
    INFO : topic diff=0.007950, rho=0.097133
    INFO : merging changes from 399 documents into a model of 9999 documents
    INFO : topic #7 (0.050): 0.073*"rt" + 0.020*"област" + 0.019*"люб" + 0.017*"дня" + 0.017*"фильм" + 0.017*"человек" + 0.015*"пройдет" + 0.015*"лидер" + 0.015*"отношен" + 0.014*"суд"
    INFO : topic #1 (0.050): 0.065*"стран" + 0.046*"rt" + 0.025*"дом" + 0.020*"американск" + 0.020*"жител" + 0.020*"пострада" + 0.019*"стал" + 0.016*"массов" + 0.015*"сем" + 0.015*"росс"
    INFO : topic #16 (0.050): 0.061*"rt" + 0.034*"нача" + 0.028*"добр" + 0.027*"пост" + 0.025*"мир" + 0.024*"как" + 0.023*"чита" + 0.019*"рассказа" + 0.019*"написа" + 0.018*"школ"
    INFO : topic #3 (0.050): 0.064*"rt" + 0.048*"москв" + 0.037*"донецк" + 0.023*"центр" + 0.023*"район" + 0.019*"прошл" + 0.013*"известн" + 0.013*"встреч" + 0.012*"продаж" + 0.012*"дан"
    INFO : topic #15 (0.050): 0.045*"rt" + 0.033*"quot" + 0.029*"дорог" + 0.025*"сто" + 0.017*"факт" + 0.014*"выход" + 0.013*"заверш" + 0.013*"сезон" + 0.013*"рук" + 0.012*"предлож"
    INFO : topic diff=0.013172, rho=0.097133
    INFO : -15.862 per-word bound, 59575.5 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : PROGRESS: pass 6, dispatched chunk #0 = documents up to #100/9999, outstanding queue size 1
    INFO : PROGRESS: pass 6, dispatched chunk #1 = documents up to #200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 6, dispatched chunk #2 = documents up to #300/9999, outstanding queue size 3
    INFO : PROGRESS: pass 6, dispatched chunk #3 = documents up to #400/9999, outstanding queue size 4
    INFO : PROGRESS: pass 6, dispatched chunk #4 = documents up to #500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 6, dispatched chunk #5 = documents up to #600/9999, outstanding queue size 6
    INFO : PROGRESS: pass 6, dispatched chunk #6 = documents up to #700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 6, dispatched chunk #7 = documents up to #800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 6, dispatched chunk #8 = documents up to #900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 6, dispatched chunk #9 = documents up to #1000/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #10 = documents up to #1100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #11 = documents up to #1200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #12 = documents up to #1300/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #13 = documents up to #1400/9999, outstanding queue size 11
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #11 (0.050): 0.083*"rt" + 0.020*"обам" + 0.018*"автомобил" + 0.017*"хорош" + 0.017*"истор" + 0.015*"журналист" + 0.013*"депутат" + 0.012*"ставропол" + 0.012*"крушен" + 0.011*"серг"
    INFO : topic #1 (0.050): 0.060*"стран" + 0.044*"rt" + 0.024*"дом" + 0.020*"американск" + 0.019*"пострада" + 0.018*"жител" + 0.018*"стал" + 0.016*"массов" + 0.014*"конц" + 0.014*"сем"
    INFO : topic #14 (0.050): 0.077*"rt" + 0.030*"петербург" + 0.025*"вер" + 0.024*"медвед" + 0.016*"дмитр" + 0.016*"украин" + 0.014*"матч" + 0.013*"восток" + 0.012*"пожар" + 0.012*"сообщ"
    INFO : topic #4 (0.050): 0.035*"интересн" + 0.034*"rt" + 0.024*"европ" + 0.024*"появ" + 0.019*"трамп" + 0.018*"границ" + 0.017*"турц" + 0.015*"закон" + 0.014*"очередн" + 0.012*"отказа"
    INFO : topic #3 (0.050): 0.060*"rt" + 0.049*"москв" + 0.034*"донецк" + 0.024*"центр" + 0.021*"район" + 0.017*"прошл" + 0.014*"встреч" + 0.012*"известн" + 0.011*"силовик" + 0.011*"банк"
    INFO : topic diff=0.049875, rho=0.096678
    INFO : PROGRESS: pass 6, dispatched chunk #14 = documents up to #1500/9999, outstanding queue size 9
    INFO : PROGRESS: pass 6, dispatched chunk #15 = documents up to #1600/9999, outstanding queue size 7
    INFO : PROGRESS: pass 6, dispatched chunk #16 = documents up to #1700/9999, outstanding queue size 8
    INFO : PROGRESS: pass 6, dispatched chunk #17 = documents up to #1800/9999, outstanding queue size 9
    INFO : PROGRESS: pass 6, dispatched chunk #18 = documents up to #1900/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #19 = documents up to #2000/9999, outstanding queue size 11
    INFO : PROGRESS: pass 6, dispatched chunk #20 = documents up to #2100/9999, outstanding queue size 12
    INFO : PROGRESS: pass 6, dispatched chunk #21 = documents up to #2200/9999, outstanding queue size 13
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #3 (0.050): 0.058*"rt" + 0.048*"москв" + 0.034*"донецк" + 0.025*"район" + 0.022*"центр" + 0.015*"прошл" + 0.014*"встреч" + 0.012*"банк" + 0.012*"известн" + 0.011*"машин"
    INFO : topic #1 (0.050): 0.055*"стран" + 0.042*"rt" + 0.023*"дом" + 0.021*"американск" + 0.019*"жител" + 0.019*"стал" + 0.017*"пострада" + 0.016*"массов" + 0.015*"сем" + 0.014*"росс"
    INFO : topic #5 (0.050): 0.062*"rt" + 0.028*"войн" + 0.020*"задержа" + 0.019*"полицейск" + 0.016*"обстрел" + 0.016*"ситуац" + 0.015*"план" + 0.014*"запад" + 0.013*"украинц" + 0.012*"смотр"
    INFO : topic #17 (0.050): 0.119*"украин" + 0.068*"российск" + 0.057*"новост" + 0.048*"rt" + 0.033*"мнен" + 0.023*"выбор" + 0.021*"побед" + 0.018*"продолжа" + 0.016*"арм" + 0.016*"донбасс"
    INFO : topic #16 (0.050): 0.059*"rt" + 0.034*"нача" + 0.025*"чита" + 0.024*"мир" + 0.023*"добр" + 0.022*"пост" + 0.022*"как" + 0.021*"написа" + 0.019*"рассказа" + 0.016*"школ"
    INFO : topic diff=0.040719, rho=0.096678
    INFO : PROGRESS: pass 6, dispatched chunk #22 = documents up to #2300/9999, outstanding queue size 12
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.075*"rt" + 0.060*"нов" + 0.028*"санкц" + 0.024*"росс" + 0.022*"сам" + 0.016*"мид" + 0.015*"рф" + 0.013*"иг" + 0.012*"сша" + 0.012*"апрел"
    INFO : topic #6 (0.050): 0.118*"путин" + 0.052*"rt" + 0.032*"владимир" + 0.028*"росс" + 0.020*"назва" + 0.020*"улиц" + 0.020*"работ" + 0.017*"сборн" + 0.015*"мест" + 0.014*"друз"
    INFO : topic #4 (0.050): 0.033*"rt" + 0.032*"интересн" + 0.026*"появ" + 0.022*"европ" + 0.019*"границ" + 0.018*"турц" + 0.017*"трамп" + 0.015*"развит" + 0.015*"очередн" + 0.014*"закон"
    INFO : topic #16 (0.050): 0.059*"rt" + 0.034*"нача" + 0.025*"чита" + 0.024*"мир" + 0.023*"написа" + 0.022*"добр" + 0.021*"пост" + 0.020*"как" + 0.019*"рассказа" + 0.016*"школ"
    INFO : topic #14 (0.050): 0.077*"rt" + 0.031*"петербург" + 0.023*"вер" + 0.021*"медвед" + 0.014*"украин" + 0.014*"матч" + 0.013*"дмитр" + 0.013*"пожар" + 0.012*"сообщ" + 0.012*"восток"
    INFO : topic diff=0.013244, rho=0.096678
    INFO : PROGRESS: pass 6, dispatched chunk #23 = documents up to #2400/9999, outstanding queue size 7
    INFO : PROGRESS: pass 6, dispatched chunk #24 = documents up to #2500/9999, outstanding queue size 7
    INFO : PROGRESS: pass 6, dispatched chunk #25 = documents up to #2600/9999, outstanding queue size 8
    INFO : PROGRESS: pass 6, dispatched chunk #26 = documents up to #2700/9999, outstanding queue size 9
    INFO : PROGRESS: pass 6, dispatched chunk #27 = documents up to #2800/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #28 = documents up to #2900/9999, outstanding queue size 11
    INFO : merging changes from 700 documents into a model of 9999 documents
    INFO : topic #17 (0.050): 0.124*"украин" + 0.068*"российск" + 0.058*"новост" + 0.051*"rt" + 0.033*"мнен" + 0.024*"выбор" + 0.021*"побед" + 0.018*"продолжа" + 0.017*"газ" + 0.016*"арм"
    INFO : topic #9 (0.050): 0.053*"rt" + 0.048*"президент" + 0.040*"рф" + 0.017*"город" + 0.014*"нат" + 0.013*"говор" + 0.012*"обсуд" + 0.012*"млн" + 0.012*"росс" + 0.011*"уф"
    INFO : topic #7 (0.050): 0.069*"rt" + 0.021*"област" + 0.017*"человек" + 0.016*"люб" + 0.014*"суд" + 0.014*"дня" + 0.013*"фильм" + 0.013*"лидер" + 0.012*"новосибирск" + 0.012*"март"
    INFO : topic #12 (0.050): 0.174*"rt" + 0.019*"люд" + 0.018*"дума" + 0.017*"возможн" + 0.016*"счита" + 0.015*"сша" + 0.011*"цен" + 0.011*"убийств" + 0.010*"боевик" + 0.010*"кин"
    INFO : topic #14 (0.050): 0.079*"rt" + 0.031*"петербург" + 0.024*"вер" + 0.020*"медвед" + 0.015*"дмитр" + 0.013*"украин" + 0.013*"матч" + 0.013*"пожар" + 0.012*"восток" + 0.011*"польш"
    INFO : topic diff=0.006415, rho=0.096678
    INFO : PROGRESS: pass 6, dispatched chunk #29 = documents up to #3000/9999, outstanding queue size 6
    INFO : PROGRESS: pass 6, dispatched chunk #30 = documents up to #3100/9999, outstanding queue size 4
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #1 (0.050): 0.056*"стран" + 0.043*"rt" + 0.026*"дом" + 0.020*"жител" + 0.020*"американск" + 0.018*"стал" + 0.017*"пострада" + 0.016*"сем" + 0.014*"днем" + 0.014*"росс"
    INFO : topic #7 (0.050): 0.069*"rt" + 0.021*"област" + 0.019*"человек" + 0.015*"люб" + 0.014*"дня" + 0.014*"суд" + 0.013*"лидер" + 0.013*"пройдет" + 0.013*"фильм" + 0.012*"новосибирск"
    INFO : topic #3 (0.050): 0.061*"rt" + 0.046*"москв" + 0.035*"донецк" + 0.024*"район" + 0.020*"центр" + 0.015*"машин" + 0.014*"встреч" + 0.014*"прошл" + 0.013*"мэр" + 0.013*"дан"
    INFO : topic #14 (0.050): 0.079*"rt" + 0.030*"петербург" + 0.025*"вер" + 0.019*"медвед" + 0.015*"дмитр" + 0.014*"украин" + 0.014*"матч" + 0.013*"пожар" + 0.012*"восток" + 0.011*"польш"
    INFO : topic #6 (0.050): 0.120*"путин" + 0.054*"rt" + 0.032*"владимир" + 0.026*"росс" + 0.020*"улиц" + 0.020*"назва" + 0.018*"работ" + 0.016*"мест" + 0.014*"сборн" + 0.014*"друз"
    INFO : topic diff=0.010717, rho=0.096678
    INFO : PROGRESS: pass 6, dispatched chunk #31 = documents up to #3200/9999, outstanding queue size 3
    INFO : PROGRESS: pass 6, dispatched chunk #32 = documents up to #3300/9999, outstanding queue size 2
    INFO : PROGRESS: pass 6, dispatched chunk #33 = documents up to #3400/9999, outstanding queue size 3
    INFO : PROGRESS: pass 6, dispatched chunk #34 = documents up to #3500/9999, outstanding queue size 4
    INFO : PROGRESS: pass 6, dispatched chunk #35 = documents up to #3600/9999, outstanding queue size 5
    INFO : PROGRESS: pass 6, dispatched chunk #36 = documents up to #3700/9999, outstanding queue size 6
    INFO : PROGRESS: pass 6, dispatched chunk #37 = documents up to #3800/9999, outstanding queue size 7
    INFO : PROGRESS: pass 6, dispatched chunk #38 = documents up to #3900/9999, outstanding queue size 8
    INFO : PROGRESS: pass 6, dispatched chunk #39 = documents up to #4000/9999, outstanding queue size 9
    INFO : PROGRESS: pass 6, dispatched chunk #40 = documents up to #4100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #41 = documents up to #4200/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #9 (0.050): 0.054*"rt" + 0.047*"президент" + 0.042*"рф" + 0.016*"город" + 0.014*"нат" + 0.013*"участ" + 0.013*"росс" + 0.012*"млн" + 0.012*"обсуд" + 0.012*"говор"
    INFO : topic #15 (0.050): 0.047*"rt" + 0.035*"quot" + 0.023*"дорог" + 0.023*"сто" + 0.016*"факт" + 0.015*"выход" + 0.014*"рук" + 0.014*"предлож" + 0.012*"сезон" + 0.012*"узна"
    INFO : topic #6 (0.050): 0.119*"путин" + 0.055*"rt" + 0.030*"владимир" + 0.026*"росс" + 0.022*"назва" + 0.019*"улиц" + 0.018*"работ" + 0.018*"мест" + 0.017*"друз" + 0.014*"сборн"
    INFO : topic #2 (0.050): 0.051*"rt" + 0.041*"глав" + 0.029*"киев" + 0.020*"пыта" + 0.020*"жизн" + 0.016*"кита" + 0.016*"нашл" + 0.016*"международн" + 0.016*"взрыв" + 0.016*"погибл"
    INFO : topic #19 (0.050): 0.059*"rt" + 0.041*"украинск" + 0.040*"воен" + 0.034*"ес" + 0.034*"украин" + 0.032*"крым" + 0.031*"фот" + 0.031*"политик" + 0.028*"сша" + 0.026*"чуж"
    INFO : topic diff=0.011337, rho=0.096678
    INFO : PROGRESS: pass 6, dispatched chunk #42 = documents up to #4300/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #43 = documents up to #4400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #44 = documents up to #4500/9999, outstanding queue size 11
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.078*"rt" + 0.054*"нов" + 0.029*"росс" + 0.028*"санкц" + 0.021*"сам" + 0.017*"мид" + 0.015*"рф" + 0.014*"апрел" + 0.012*"сша" + 0.012*"иг"
    INFO : topic #14 (0.050): 0.080*"rt" + 0.032*"петербург" + 0.025*"вер" + 0.019*"медвед" + 0.015*"пожар" + 0.014*"дмитр" + 0.014*"восток" + 0.013*"ки" + 0.013*"украин" + 0.012*"матч"
    INFO : topic #0 (0.050): 0.078*"rt" + 0.044*"сир" + 0.020*"рад" + 0.017*"главн" + 0.013*"призва" + 0.013*"игр" + 0.013*"сша" + 0.013*"удар" + 0.012*"переговор" + 0.010*"пьян"
    INFO : topic #1 (0.050): 0.057*"стран" + 0.044*"rt" + 0.025*"дом" + 0.020*"американск" + 0.017*"жител" + 0.017*"пострада" + 0.016*"сем" + 0.016*"стал" + 0.015*"росс" + 0.014*"массов"
    INFO : topic #7 (0.050): 0.069*"rt" + 0.020*"област" + 0.020*"человек" + 0.017*"дня" + 0.016*"люб" + 0.014*"лидер" + 0.014*"суд" + 0.013*"пройдет" + 0.013*"фильм" + 0.013*"отношен"
    INFO : topic diff=0.010201, rho=0.096678
    INFO : PROGRESS: pass 6, dispatched chunk #45 = documents up to #4600/9999, outstanding queue size 9
    INFO : PROGRESS: pass 6, dispatched chunk #46 = documents up to #4700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 6, dispatched chunk #47 = documents up to #4800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 6, dispatched chunk #48 = documents up to #4900/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #15 (0.050): 0.045*"rt" + 0.035*"quot" + 0.022*"дорог" + 0.021*"сто" + 0.015*"предлож" + 0.014*"факт" + 0.014*"рук" + 0.014*"выход" + 0.012*"сезон" + 0.012*"солдат"
    INFO : topic #18 (0.050): 0.079*"rt" + 0.054*"нов" + 0.029*"санкц" + 0.028*"росс" + 0.022*"сам" + 0.020*"мид" + 0.016*"рф" + 0.013*"апрел" + 0.012*"сша" + 0.012*"иг"
    INFO : topic #19 (0.050): 0.061*"rt" + 0.043*"украинск" + 0.042*"воен" + 0.035*"крым" + 0.035*"украин" + 0.031*"сша" + 0.031*"ес" + 0.030*"политик" + 0.029*"фот" + 0.024*"чуж"
    INFO : topic #5 (0.050): 0.064*"rt" + 0.025*"войн" + 0.022*"полицейск" + 0.020*"задержа" + 0.017*"ситуац" + 0.015*"украинц" + 0.015*"обстрел" + 0.015*"стат" + 0.015*"сотрудник" + 0.014*"запад"
    INFO : topic #8 (0.050): 0.059*"rt" + 0.037*"рубл" + 0.025*"слов" + 0.023*"сторон" + 0.020*"получ" + 0.018*"миноборон" + 0.017*"запрет" + 0.016*"мужчин" + 0.014*"росс" + 0.014*"дтп"
    INFO : topic diff=0.010847, rho=0.096678
    INFO : PROGRESS: pass 6, dispatched chunk #49 = documents up to #5000/9999, outstanding queue size 9
    INFO : PROGRESS: pass 6, dispatched chunk #50 = documents up to #5100/9999, outstanding queue size 8
    INFO : PROGRESS: pass 6, dispatched chunk #51 = documents up to #5200/9999, outstanding queue size 9
    INFO : PROGRESS: pass 6, dispatched chunk #52 = documents up to #5300/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #53 = documents up to #5400/9999, outstanding queue size 11
    INFO : PROGRESS: pass 6, dispatched chunk #54 = documents up to #5500/9999, outstanding queue size 12
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #13 (0.050): 0.082*"rt" + 0.032*"готов" + 0.028*"власт" + 0.027*"украин" + 0.026*"сми" + 0.026*"перв" + 0.025*"дел" + 0.018*"русск" + 0.017*"прав" + 0.013*"росс"
    INFO : topic #14 (0.050): 0.080*"rt" + 0.035*"петербург" + 0.031*"вер" + 0.018*"медвед" + 0.016*"пожар" + 0.013*"восток" + 0.013*"матч" + 0.013*"ки" + 0.012*"дмитр" + 0.012*"украин"
    INFO : topic #9 (0.050): 0.056*"rt" + 0.051*"президент" + 0.047*"рф" + 0.014*"город" + 0.014*"нат" + 0.012*"росс" + 0.012*"участ" + 0.012*"сентябр" + 0.012*"эксперт" + 0.012*"млн"
    INFO : topic #18 (0.050): 0.079*"rt" + 0.056*"нов" + 0.029*"санкц" + 0.026*"росс" + 0.023*"сам" + 0.018*"мид" + 0.016*"рф" + 0.013*"апрел" + 0.012*"иг" + 0.012*"сша"
    INFO : topic #4 (0.050): 0.035*"rt" + 0.030*"появ" + 0.028*"интересн" + 0.021*"турц" + 0.021*"закон" + 0.020*"европ" + 0.018*"отказа" + 0.016*"трамп" + 0.014*"границ" + 0.013*"очередн"
    INFO : topic diff=0.007320, rho=0.096678
    INFO : PROGRESS: pass 6, dispatched chunk #55 = documents up to #5600/9999, outstanding queue size 9
    INFO : PROGRESS: pass 6, dispatched chunk #56 = documents up to #5700/9999, outstanding queue size 8
    INFO : PROGRESS: pass 6, dispatched chunk #57 = documents up to #5800/9999, outstanding queue size 9
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #4 (0.050): 0.034*"rt" + 0.030*"интересн" + 0.029*"появ" + 0.023*"европ" + 0.020*"закон" + 0.020*"турц" + 0.017*"границ" + 0.017*"отказа" + 0.015*"трамп" + 0.014*"омск"
    INFO : topic #14 (0.050): 0.080*"rt" + 0.035*"петербург" + 0.031*"вер" + 0.018*"медвед" + 0.015*"пожар" + 0.014*"восток" + 0.014*"матч" + 0.014*"ки" + 0.012*"сообщ" + 0.012*"дмитр"
    INFO : topic #19 (0.050): 0.063*"rt" + 0.044*"украинск" + 0.040*"воен" + 0.038*"крым" + 0.037*"украин" + 0.032*"ес" + 0.030*"сша" + 0.030*"фот" + 0.029*"политик" + 0.025*"днр"
    INFO : topic #8 (0.050): 0.058*"rt" + 0.037*"рубл" + 0.024*"слов" + 0.021*"сторон" + 0.020*"получ" + 0.016*"миноборон" + 0.016*"мужчин" + 0.016*"запрет" + 0.015*"дтп" + 0.014*"росс"
    INFO : topic #18 (0.050): 0.078*"rt" + 0.060*"нов" + 0.028*"санкц" + 0.026*"росс" + 0.022*"сам" + 0.016*"мид" + 0.015*"рф" + 0.013*"апрел" + 0.012*"акц" + 0.012*"представ"
    INFO : topic diff=0.009268, rho=0.096678
    INFO : PROGRESS: pass 6, dispatched chunk #58 = documents up to #5900/9999, outstanding queue size 6
    INFO : PROGRESS: pass 6, dispatched chunk #59 = documents up to #6000/9999, outstanding queue size 4
    INFO : PROGRESS: pass 6, dispatched chunk #60 = documents up to #6100/9999, outstanding queue size 5
    INFO : PROGRESS: pass 6, dispatched chunk #61 = documents up to #6200/9999, outstanding queue size 6
    INFO : PROGRESS: pass 6, dispatched chunk #62 = documents up to #6300/9999, outstanding queue size 7
    INFO : PROGRESS: pass 6, dispatched chunk #63 = documents up to #6400/9999, outstanding queue size 8
    INFO : PROGRESS: pass 6, dispatched chunk #64 = documents up to #6500/9999, outstanding queue size 9
    INFO : PROGRESS: pass 6, dispatched chunk #65 = documents up to #6600/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #1 (0.050): 0.057*"стран" + 0.044*"rt" + 0.031*"американск" + 0.023*"пострада" + 0.021*"дом" + 0.021*"жител" + 0.015*"днем" + 0.014*"конц" + 0.014*"росс" + 0.013*"стал"
    INFO : topic #10 (0.050): 0.087*"rt" + 0.040*"виде" + 0.030*"дет" + 0.024*"полиц" + 0.024*"лучш" + 0.019*"хочет" + 0.017*"московск" + 0.014*"уб" + 0.014*"бизнес" + 0.014*"решен"
    INFO : topic #12 (0.050): 0.185*"rt" + 0.022*"люд" + 0.017*"дума" + 0.016*"возможн" + 0.015*"счита" + 0.013*"сша" + 0.013*"цен" + 0.012*"убийств" + 0.011*"оруж" + 0.011*"автор"
    INFO : topic #8 (0.050): 0.058*"rt" + 0.034*"рубл" + 0.028*"слов" + 0.022*"сторон" + 0.020*"получ" + 0.019*"миноборон" + 0.017*"переп" + 0.015*"мужчин" + 0.015*"запрет" + 0.014*"дтп"
    INFO : topic #16 (0.050): 0.060*"rt" + 0.034*"нача" + 0.027*"чита" + 0.026*"добр" + 0.026*"пост" + 0.026*"мир" + 0.023*"рассказа" + 0.022*"как" + 0.022*"написа" + 0.019*"потеря"
    INFO : topic diff=0.012813, rho=0.096678
    INFO : PROGRESS: pass 6, dispatched chunk #66 = documents up to #6700/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #67 = documents up to #6800/9999, outstanding queue size 9
    INFO : PROGRESS: pass 6, dispatched chunk #68 = documents up to #6900/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #69 = documents up to #7000/9999, outstanding queue size 11
    INFO : PROGRESS: pass 6, dispatched chunk #70 = documents up to #7100/9999, outstanding queue size 12
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #11 (0.050): 0.084*"rt" + 0.021*"обам" + 0.020*"истор" + 0.018*"хорош" + 0.015*"автомобил" + 0.013*"крушен" + 0.013*"депутат" + 0.013*"связ" + 0.013*"поддержива" + 0.013*"ставропол"
    INFO : topic #19 (0.050): 0.063*"rt" + 0.045*"украинск" + 0.044*"воен" + 0.037*"украин" + 0.037*"крым" + 0.030*"ес" + 0.030*"политик" + 0.028*"сша" + 0.026*"фот" + 0.024*"чуж"
    INFO : topic #17 (0.050): 0.136*"украин" + 0.067*"российск" + 0.062*"новост" + 0.052*"rt" + 0.031*"мнен" + 0.022*"выбор" + 0.020*"донбасс" + 0.020*"газ" + 0.020*"арм" + 0.020*"побед"
    INFO : topic #7 (0.050): 0.073*"rt" + 0.026*"человек" + 0.021*"област" + 0.018*"люб" + 0.016*"дня" + 0.014*"суд" + 0.013*"фильм" + 0.012*"март" + 0.012*"музык" + 0.012*"лидер"
    INFO : topic #4 (0.050): 0.033*"интересн" + 0.032*"rt" + 0.027*"появ" + 0.024*"европ" + 0.024*"закон" + 0.019*"турц" + 0.018*"границ" + 0.016*"отказа" + 0.015*"трамп" + 0.012*"нужн"
    INFO : topic diff=0.010329, rho=0.096678
    INFO : PROGRESS: pass 6, dispatched chunk #71 = documents up to #7200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #72 = documents up to #7300/9999, outstanding queue size 8
    INFO : PROGRESS: pass 6, dispatched chunk #73 = documents up to #7400/9999, outstanding queue size 9
    INFO : PROGRESS: pass 6, dispatched chunk #74 = documents up to #7500/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #17 (0.050): 0.134*"украин" + 0.065*"российск" + 0.064*"новост" + 0.052*"rt" + 0.032*"мнен" + 0.022*"выбор" + 0.020*"донбасс" + 0.020*"газ" + 0.018*"арм" + 0.018*"побед"
    INFO : topic #3 (0.050): 0.062*"rt" + 0.046*"москв" + 0.039*"донецк" + 0.026*"район" + 0.024*"центр" + 0.016*"прошл" + 0.013*"продаж" + 0.013*"мэр" + 0.012*"машин" + 0.012*"встреч"
    INFO : topic #15 (0.050): 0.048*"rt" + 0.036*"quot" + 0.027*"дорог" + 0.023*"сто" + 0.016*"предлож" + 0.014*"выход" + 0.014*"памятник" + 0.013*"рук" + 0.013*"сезон" + 0.013*"действ"
    INFO : topic #10 (0.050): 0.087*"rt" + 0.038*"виде" + 0.030*"дет" + 0.028*"лучш" + 0.025*"полиц" + 0.018*"московск" + 0.017*"хочет" + 0.015*"уб" + 0.014*"последн" + 0.013*"решен"
    INFO : topic #7 (0.050): 0.073*"rt" + 0.025*"человек" + 0.019*"люб" + 0.019*"област" + 0.016*"дня" + 0.014*"фильм" + 0.014*"суд" + 0.013*"март" + 0.012*"пройдет" + 0.012*"лидер"
    INFO : topic diff=0.009030, rho=0.096678
    INFO : PROGRESS: pass 6, dispatched chunk #75 = documents up to #7600/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #76 = documents up to #7700/9999, outstanding queue size 10
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #11 (0.050): 0.083*"rt" + 0.019*"истор" + 0.019*"хорош" + 0.018*"обам" + 0.015*"автомобил" + 0.014*"праздник" + 0.013*"связ" + 0.013*"поддержива" + 0.013*"депутат" + 0.012*"серг"
    INFO : topic #14 (0.050): 0.077*"rt" + 0.031*"петербург" + 0.030*"вер" + 0.022*"медвед" + 0.016*"матч" + 0.014*"дмитр" + 0.014*"пожар" + 0.014*"восток" + 0.013*"ки" + 0.011*"украин"
    INFO : topic #2 (0.050): 0.055*"rt" + 0.039*"глав" + 0.037*"киев" + 0.019*"жизн" + 0.018*"взрыв" + 0.017*"пыта" + 0.015*"нашл" + 0.015*"международн" + 0.014*"погибл" + 0.014*"южн"
    INFO : topic #8 (0.050): 0.058*"rt" + 0.032*"рубл" + 0.027*"слов" + 0.022*"получ" + 0.021*"сторон" + 0.018*"переп" + 0.016*"миноборон" + 0.015*"министр" + 0.014*"запрет" + 0.014*"дтп"
    INFO : topic #7 (0.050): 0.073*"rt" + 0.023*"человек" + 0.020*"люб" + 0.018*"дня" + 0.018*"област" + 0.016*"фильм" + 0.014*"суд" + 0.014*"пройдет" + 0.012*"лидер" + 0.011*"март"
    INFO : topic diff=0.009053, rho=0.096678
    INFO : PROGRESS: pass 6, dispatched chunk #77 = documents up to #7800/9999, outstanding queue size 6
    INFO : PROGRESS: pass 6, dispatched chunk #78 = documents up to #7900/9999, outstanding queue size 4
    INFO : PROGRESS: pass 6, dispatched chunk #79 = documents up to #8000/9999, outstanding queue size 5
    INFO : PROGRESS: pass 6, dispatched chunk #80 = documents up to #8100/9999, outstanding queue size 6
    INFO : PROGRESS: pass 6, dispatched chunk #81 = documents up to #8200/9999, outstanding queue size 7
    INFO : PROGRESS: pass 6, dispatched chunk #82 = documents up to #8300/9999, outstanding queue size 8
    INFO : PROGRESS: pass 6, dispatched chunk #83 = documents up to #8400/9999, outstanding queue size 9
    INFO : PROGRESS: pass 6, dispatched chunk #84 = documents up to #8500/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #85 = documents up to #8600/9999, outstanding queue size 11
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.082*"rt" + 0.061*"нов" + 0.031*"росс" + 0.028*"санкц" + 0.024*"сам" + 0.017*"иг" + 0.016*"рф" + 0.014*"мид" + 0.013*"представ" + 0.012*"сильн"
    INFO : topic #19 (0.050): 0.063*"rt" + 0.045*"воен" + 0.044*"украинск" + 0.033*"украин" + 0.033*"крым" + 0.032*"ес" + 0.030*"фот" + 0.028*"сша" + 0.028*"политик" + 0.023*"чуж"
    INFO : topic #7 (0.050): 0.073*"rt" + 0.023*"человек" + 0.019*"люб" + 0.018*"дня" + 0.018*"област" + 0.016*"пройдет" + 0.015*"фильм" + 0.013*"суд" + 0.012*"март" + 0.012*"лидер"
    INFO : topic #15 (0.050): 0.048*"rt" + 0.034*"quot" + 0.027*"дорог" + 0.024*"сто" + 0.015*"октябр" + 0.014*"рук" + 0.014*"предлож" + 0.014*"памятник" + 0.013*"выход" + 0.013*"факт"
    INFO : topic #5 (0.050): 0.063*"rt" + 0.027*"войн" + 0.022*"задержа" + 0.018*"полицейск" + 0.017*"обстрел" + 0.016*"предлага" + 0.016*"план" + 0.016*"смотр" + 0.014*"сотрудник" + 0.013*"россиян"
    INFO : topic diff=0.011453, rho=0.096678
    INFO : PROGRESS: pass 6, dispatched chunk #86 = documents up to #8700/9999, outstanding queue size 11
    INFO : PROGRESS: pass 6, dispatched chunk #87 = documents up to #8800/9999, outstanding queue size 11
    INFO : PROGRESS: pass 6, dispatched chunk #88 = documents up to #8900/9999, outstanding queue size 12
    INFO : PROGRESS: pass 6, dispatched chunk #89 = documents up to #9000/9999, outstanding queue size 13
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #9 (0.050): 0.055*"rt" + 0.047*"президент" + 0.045*"рф" + 0.018*"нат" + 0.018*"город" + 0.014*"эксперт" + 0.013*"говор" + 0.012*"евр" + 0.012*"обсуд" + 0.011*"росс"
    INFO : topic #13 (0.050): 0.082*"rt" + 0.032*"сми" + 0.027*"украин" + 0.027*"готов" + 0.025*"перв" + 0.024*"дел" + 0.023*"власт" + 0.019*"русск" + 0.017*"террорист" + 0.016*"заяв"
    INFO : topic #14 (0.050): 0.077*"rt" + 0.030*"петербург" + 0.026*"вер" + 0.021*"медвед" + 0.016*"матч" + 0.016*"пожар" + 0.016*"восток" + 0.014*"дмитр" + 0.014*"ки" + 0.011*"неизвестн"
    INFO : topic #5 (0.050): 0.063*"rt" + 0.026*"войн" + 0.022*"задержа" + 0.019*"полицейск" + 0.017*"смотр" + 0.017*"обстрел" + 0.016*"план" + 0.015*"предлага" + 0.013*"сотрудник" + 0.013*"запад"
    INFO : topic #8 (0.050): 0.062*"rt" + 0.035*"рубл" + 0.025*"слов" + 0.022*"получ" + 0.022*"сторон" + 0.016*"переп" + 0.015*"миноборон" + 0.015*"министр" + 0.014*"росс" + 0.014*"дтп"
    INFO : topic diff=0.011675, rho=0.096678
    INFO : PROGRESS: pass 6, dispatched chunk #90 = documents up to #9100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #91 = documents up to #9200/9999, outstanding queue size 8
    INFO : PROGRESS: pass 6, dispatched chunk #92 = documents up to #9300/9999, outstanding queue size 9
    INFO : PROGRESS: pass 6, dispatched chunk #93 = documents up to #9400/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #16 (0.050): 0.061*"rt" + 0.034*"нача" + 0.030*"добр" + 0.026*"пост" + 0.026*"чита" + 0.024*"как" + 0.024*"мир" + 0.019*"рассказа" + 0.018*"написа" + 0.016*"школ"
    INFO : topic #11 (0.050): 0.079*"rt" + 0.019*"истор" + 0.019*"обам" + 0.018*"хорош" + 0.017*"автомобил" + 0.014*"депутат" + 0.014*"журналист" + 0.013*"поддержива" + 0.013*"связ" + 0.013*"праздник"
    INFO : topic #0 (0.050): 0.077*"rt" + 0.050*"сир" + 0.025*"рад" + 0.019*"главн" + 0.016*"удар" + 0.015*"переговор" + 0.012*"увелич" + 0.011*"минск" + 0.011*"игр" + 0.011*"сша"
    INFO : topic #14 (0.050): 0.076*"rt" + 0.031*"петербург" + 0.027*"вер" + 0.022*"медвед" + 0.017*"матч" + 0.015*"пожар" + 0.015*"восток" + 0.014*"ки" + 0.013*"дмитр" + 0.011*"сообщ"
    INFO : topic #15 (0.050): 0.048*"rt" + 0.034*"quot" + 0.029*"дорог" + 0.022*"сто" + 0.015*"октябр" + 0.014*"рук" + 0.014*"памятник" + 0.013*"выход" + 0.013*"предлож" + 0.012*"узна"
    INFO : topic diff=0.011029, rho=0.096678
    INFO : PROGRESS: pass 6, dispatched chunk #94 = documents up to #9500/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #95 = documents up to #9600/9999, outstanding queue size 10
    INFO : PROGRESS: pass 6, dispatched chunk #96 = documents up to #9700/9999, outstanding queue size 11
    INFO : PROGRESS: pass 6, dispatched chunk #97 = documents up to #9800/9999, outstanding queue size 12
    INFO : PROGRESS: pass 6, dispatched chunk #98 = documents up to #9900/9999, outstanding queue size 13
    INFO : PROGRESS: pass 6, dispatched chunk #99 = documents up to #9999/9999, outstanding queue size 12
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #4 (0.050): 0.034*"интересн" + 0.033*"rt" + 0.026*"появ" + 0.022*"европ" + 0.019*"границ" + 0.018*"трамп" + 0.018*"закон" + 0.018*"турц" + 0.013*"отказа" + 0.013*"очередн"
    INFO : topic #0 (0.050): 0.078*"rt" + 0.049*"сир" + 0.024*"рад" + 0.019*"главн" + 0.015*"переговор" + 0.015*"удар" + 0.012*"увелич" + 0.012*"сша" + 0.012*"минск" + 0.012*"игр"
    INFO : topic #8 (0.050): 0.062*"rt" + 0.037*"рубл" + 0.029*"слов" + 0.023*"сторон" + 0.023*"получ" + 0.016*"миноборон" + 0.016*"переп" + 0.015*"министр" + 0.013*"росс" + 0.013*"запрет"
    INFO : topic #6 (0.050): 0.130*"путин" + 0.056*"rt" + 0.027*"владимир" + 0.026*"росс" + 0.020*"улиц" + 0.020*"работ" + 0.019*"назва" + 0.017*"сборн" + 0.016*"мест" + 0.015*"друз"
    INFO : topic #19 (0.050): 0.065*"rt" + 0.044*"воен" + 0.043*"украинск" + 0.037*"ес" + 0.034*"украин" + 0.031*"политик" + 0.031*"сша" + 0.029*"крым" + 0.028*"фот" + 0.027*"чуж"
    INFO : topic diff=0.009614, rho=0.096678
    INFO : -15.966 per-word bound, 64014.0 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #17 (0.050): 0.132*"украин" + 0.070*"российск" + 0.057*"новост" + 0.055*"rt" + 0.032*"мнен" + 0.025*"выбор" + 0.022*"побед" + 0.020*"арм" + 0.018*"донбасс" + 0.017*"учен"
    INFO : topic #7 (0.050): 0.072*"rt" + 0.019*"люб" + 0.019*"област" + 0.018*"человек" + 0.018*"дня" + 0.016*"пройдет" + 0.014*"март" + 0.014*"суд" + 0.014*"отношен" + 0.014*"фильм"
    INFO : topic #4 (0.050): 0.036*"интересн" + 0.033*"rt" + 0.029*"появ" + 0.024*"европ" + 0.019*"границ" + 0.019*"турц" + 0.018*"трамп" + 0.017*"закон" + 0.015*"очередн" + 0.014*"отказа"
    INFO : topic #11 (0.050): 0.078*"rt" + 0.020*"хорош" + 0.019*"обам" + 0.017*"автомобил" + 0.017*"истор" + 0.015*"депутат" + 0.014*"журналист" + 0.012*"поддержива" + 0.012*"ставропол" + 0.012*"час"
    INFO : topic #2 (0.050): 0.052*"rt" + 0.042*"глав" + 0.033*"киев" + 0.022*"жизн" + 0.019*"пыта" + 0.016*"южн" + 0.016*"взрыв" + 0.016*"результат" + 0.015*"погибл" + 0.015*"кита"
    INFO : topic diff=0.009141, rho=0.096678
    INFO : -15.967 per-word bound, 64048.5 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : merging changes from 499 documents into a model of 9999 documents
    INFO : topic #15 (0.050): 0.046*"rt" + 0.033*"quot" + 0.028*"дорог" + 0.024*"сто" + 0.016*"факт" + 0.014*"выход" + 0.014*"рук" + 0.013*"заверш" + 0.012*"предлож" + 0.012*"действ"
    INFO : topic #10 (0.050): 0.085*"rt" + 0.037*"виде" + 0.036*"дет" + 0.027*"лучш" + 0.023*"полиц" + 0.020*"хочет" + 0.019*"московск" + 0.016*"бизнес" + 0.015*"решен" + 0.014*"метр"
    INFO : topic #2 (0.050): 0.053*"rt" + 0.043*"глав" + 0.032*"киев" + 0.024*"жизн" + 0.017*"пыта" + 0.017*"южн" + 0.017*"погибл" + 0.017*"взрыв" + 0.015*"кита" + 0.015*"международн"
    INFO : topic #6 (0.050): 0.126*"путин" + 0.056*"rt" + 0.028*"владимир" + 0.027*"росс" + 0.026*"назва" + 0.022*"работ" + 0.021*"улиц" + 0.018*"сборн" + 0.017*"мест" + 0.014*"друз"
    INFO : topic #0 (0.050): 0.081*"rt" + 0.051*"сир" + 0.026*"рад" + 0.022*"главн" + 0.015*"переговор" + 0.015*"удар" + 0.013*"минск" + 0.012*"игр" + 0.011*"сша" + 0.011*"увелич"
    INFO : topic diff=0.011818, rho=0.096678
    INFO : -15.879 per-word bound, 60260.1 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : PROGRESS: pass 7, dispatched chunk #0 = documents up to #100/9999, outstanding queue size 1
    INFO : PROGRESS: pass 7, dispatched chunk #1 = documents up to #200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 7, dispatched chunk #2 = documents up to #300/9999, outstanding queue size 3
    INFO : PROGRESS: pass 7, dispatched chunk #3 = documents up to #400/9999, outstanding queue size 4
    INFO : PROGRESS: pass 7, dispatched chunk #4 = documents up to #500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 7, dispatched chunk #5 = documents up to #600/9999, outstanding queue size 5
    INFO : PROGRESS: pass 7, dispatched chunk #6 = documents up to #700/9999, outstanding queue size 6
    INFO : PROGRESS: pass 7, dispatched chunk #7 = documents up to #800/9999, outstanding queue size 7
    INFO : PROGRESS: pass 7, dispatched chunk #8 = documents up to #900/9999, outstanding queue size 8
    INFO : PROGRESS: pass 7, dispatched chunk #9 = documents up to #1000/9999, outstanding queue size 9
    INFO : PROGRESS: pass 7, dispatched chunk #10 = documents up to #1100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 7, dispatched chunk #11 = documents up to #1200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 7, dispatched chunk #12 = documents up to #1300/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.078*"rt" + 0.067*"нов" + 0.029*"санкц" + 0.028*"росс" + 0.024*"сам" + 0.016*"мид" + 0.014*"рф" + 0.014*"представ" + 0.014*"сша" + 0.014*"иг"
    INFO : topic #12 (0.050): 0.180*"rt" + 0.020*"люд" + 0.018*"возможн" + 0.016*"дума" + 0.015*"сша" + 0.013*"счита" + 0.013*"убийств" + 0.011*"цен" + 0.011*"сирийск" + 0.010*"кин"
    INFO : topic #2 (0.050): 0.050*"rt" + 0.045*"глав" + 0.032*"киев" + 0.021*"жизн" + 0.016*"взрыв" + 0.016*"пыта" + 0.015*"погибл" + 0.015*"кита" + 0.015*"международн" + 0.015*"южн"
    INFO : topic #6 (0.050): 0.118*"путин" + 0.054*"rt" + 0.029*"росс" + 0.029*"владимир" + 0.026*"назва" + 0.021*"работ" + 0.020*"улиц" + 0.017*"сборн" + 0.016*"мест" + 0.015*"друз"
    INFO : topic #9 (0.050): 0.058*"rt" + 0.051*"президент" + 0.045*"рф" + 0.017*"город" + 0.016*"нат" + 0.012*"говор" + 0.011*"сил" + 0.011*"обсуд" + 0.011*"участ" + 0.011*"млн"
    INFO : topic diff=0.048018, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #13 = documents up to #1400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 7, dispatched chunk #14 = documents up to #1500/9999, outstanding queue size 10
    INFO : PROGRESS: pass 7, dispatched chunk #15 = documents up to #1600/9999, outstanding queue size 11
    INFO : PROGRESS: pass 7, dispatched chunk #16 = documents up to #1700/9999, outstanding queue size 12
    INFO : PROGRESS: pass 7, dispatched chunk #17 = documents up to #1800/9999, outstanding queue size 13
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #7 (0.050): 0.068*"rt" + 0.021*"област" + 0.018*"люб" + 0.016*"человек" + 0.015*"дня" + 0.014*"фильм" + 0.013*"суд" + 0.013*"лидер" + 0.012*"летн" + 0.012*"отношен"
    INFO : topic #19 (0.050): 0.058*"rt" + 0.043*"украинск" + 0.041*"воен" + 0.037*"ес" + 0.034*"украин" + 0.032*"фот" + 0.029*"политик" + 0.029*"сша" + 0.028*"крым" + 0.024*"чуж"
    INFO : topic #1 (0.050): 0.054*"стран" + 0.042*"rt" + 0.023*"дом" + 0.021*"американск" + 0.018*"стал" + 0.018*"пострада" + 0.017*"жител" + 0.015*"массов" + 0.014*"конц" + 0.014*"сем"
    INFO : topic #4 (0.050): 0.035*"интересн" + 0.033*"rt" + 0.025*"появ" + 0.022*"европ" + 0.018*"трамп" + 0.018*"турц" + 0.017*"границ" + 0.015*"очередн" + 0.014*"закон" + 0.013*"отказа"
    INFO : topic #16 (0.050): 0.058*"rt" + 0.030*"нача" + 0.027*"чита" + 0.025*"мир" + 0.024*"добр" + 0.023*"пост" + 0.021*"как" + 0.021*"написа" + 0.020*"рассказа" + 0.016*"школ"
    INFO : topic diff=0.040118, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #18 = documents up to #1900/9999, outstanding queue size 9
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #9 (0.050): 0.053*"rt" + 0.050*"президент" + 0.042*"рф" + 0.016*"город" + 0.014*"нат" + 0.012*"говор" + 0.012*"обсуд" + 0.011*"млн" + 0.010*"росс" + 0.010*"участ"
    INFO : topic #11 (0.050): 0.078*"rt" + 0.018*"обам" + 0.018*"автомобил" + 0.016*"хорош" + 0.015*"истор" + 0.014*"журналист" + 0.013*"крушен" + 0.011*"серг" + 0.011*"депутат" + 0.011*"числ"
    INFO : topic #19 (0.050): 0.057*"rt" + 0.043*"украинск" + 0.041*"воен" + 0.040*"ес" + 0.034*"украин" + 0.031*"политик" + 0.030*"фот" + 0.028*"сша" + 0.028*"крым" + 0.023*"чуж"
    INFO : topic #1 (0.050): 0.050*"стран" + 0.042*"rt" + 0.023*"дом" + 0.020*"американск" + 0.019*"жител" + 0.018*"пострада" + 0.018*"стал" + 0.015*"массов" + 0.015*"сем" + 0.014*"конц"
    INFO : topic #16 (0.050): 0.059*"rt" + 0.034*"нача" + 0.025*"чита" + 0.024*"написа" + 0.023*"мир" + 0.022*"добр" + 0.021*"пост" + 0.021*"рассказа" + 0.019*"как" + 0.016*"школ"
    INFO : topic diff=0.013813, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #19 = documents up to #2000/9999, outstanding queue size 5
    INFO : PROGRESS: pass 7, dispatched chunk #20 = documents up to #2100/9999, outstanding queue size 4
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #7 (0.050): 0.068*"rt" + 0.020*"област" + 0.017*"люб" + 0.016*"человек" + 0.015*"суд" + 0.014*"дня" + 0.013*"лидер" + 0.013*"март" + 0.012*"фильм" + 0.012*"летн"
    INFO : topic #18 (0.050): 0.076*"rt" + 0.060*"нов" + 0.030*"санкц" + 0.027*"росс" + 0.022*"сам" + 0.016*"мид" + 0.014*"рф" + 0.013*"иг" + 0.012*"сильн" + 0.012*"сша"
    INFO : topic #15 (0.050): 0.047*"rt" + 0.032*"quot" + 0.023*"дорог" + 0.023*"сто" + 0.014*"факт" + 0.014*"предлож" + 0.012*"рук" + 0.012*"узна" + 0.012*"постро" + 0.011*"сезон"
    INFO : topic #10 (0.050): 0.081*"rt" + 0.041*"виде" + 0.030*"дет" + 0.020*"лучш" + 0.019*"полиц" + 0.019*"хочет" + 0.016*"московск" + 0.016*"уб" + 0.013*"бизнес" + 0.013*"метр"
    INFO : topic #3 (0.050): 0.060*"rt" + 0.051*"москв" + 0.032*"донецк" + 0.024*"район" + 0.020*"центр" + 0.015*"встреч" + 0.014*"прошл" + 0.012*"банк" + 0.012*"машин" + 0.011*"мэр"
    INFO : topic diff=0.010880, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #21 = documents up to #2200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 7, dispatched chunk #22 = documents up to #2300/9999, outstanding queue size 2
    INFO : PROGRESS: pass 7, dispatched chunk #23 = documents up to #2400/9999, outstanding queue size 3
    INFO : PROGRESS: pass 7, dispatched chunk #24 = documents up to #2500/9999, outstanding queue size 4
    INFO : PROGRESS: pass 7, dispatched chunk #25 = documents up to #2600/9999, outstanding queue size 5
    INFO : PROGRESS: pass 7, dispatched chunk #26 = documents up to #2700/9999, outstanding queue size 6
    INFO : PROGRESS: pass 7, dispatched chunk #27 = documents up to #2800/9999, outstanding queue size 7
    INFO : PROGRESS: pass 7, dispatched chunk #28 = documents up to #2900/9999, outstanding queue size 8
    INFO : PROGRESS: pass 7, dispatched chunk #29 = documents up to #3000/9999, outstanding queue size 9
    INFO : PROGRESS: pass 7, dispatched chunk #30 = documents up to #3100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 7, dispatched chunk #31 = documents up to #3200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 7, dispatched chunk #32 = documents up to #3300/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #0 (0.050): 0.081*"rt" + 0.045*"сир" + 0.022*"рад" + 0.016*"главн" + 0.014*"удар" + 0.013*"сша" + 0.013*"игр" + 0.011*"переговор" + 0.010*"минск" + 0.009*"пьян"
    INFO : topic #12 (0.050): 0.175*"rt" + 0.020*"люд" + 0.019*"дума" + 0.017*"возможн" + 0.016*"счита" + 0.015*"сша" + 0.012*"цен" + 0.011*"убийств" + 0.010*"боевик" + 0.010*"кин"
    INFO : topic #6 (0.050): 0.119*"путин" + 0.053*"rt" + 0.031*"владимир" + 0.027*"росс" + 0.021*"назва" + 0.021*"улиц" + 0.020*"работ" + 0.016*"мест" + 0.015*"друз" + 0.014*"сборн"
    INFO : topic #4 (0.050): 0.033*"интересн" + 0.032*"rt" + 0.025*"появ" + 0.021*"европ" + 0.018*"турц" + 0.017*"границ" + 0.016*"очередн" + 0.016*"закон" + 0.015*"развит" + 0.014*"отказа"
    INFO : topic #17 (0.050): 0.136*"украин" + 0.067*"российск" + 0.059*"новост" + 0.053*"rt" + 0.033*"мнен" + 0.023*"выбор" + 0.021*"побед" + 0.018*"продолжа" + 0.016*"газ" + 0.015*"арм"
    INFO : topic diff=0.009060, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #33 = documents up to #3400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 7, dispatched chunk #34 = documents up to #3500/9999, outstanding queue size 10
    INFO : merging changes from 800 documents into a model of 9999 documents
    INFO : topic #8 (0.050): 0.058*"rt" + 0.036*"рубл" + 0.026*"слов" + 0.024*"сторон" + 0.021*"получ" + 0.019*"миноборон" + 0.016*"дтп" + 0.015*"запрет" + 0.014*"мужчин" + 0.013*"министр"
    INFO : topic #17 (0.050): 0.133*"украин" + 0.068*"российск" + 0.058*"новост" + 0.053*"rt" + 0.034*"мнен" + 0.022*"выбор" + 0.022*"побед" + 0.018*"арм" + 0.017*"продолжа" + 0.016*"газ"
    INFO : topic #0 (0.050): 0.081*"rt" + 0.045*"сир" + 0.022*"рад" + 0.016*"главн" + 0.013*"удар" + 0.013*"игр" + 0.013*"сша" + 0.011*"переговор" + 0.010*"минск" + 0.010*"пьян"
    INFO : topic #16 (0.050): 0.058*"rt" + 0.033*"нача" + 0.031*"чита" + 0.027*"мир" + 0.024*"пост" + 0.021*"добр" + 0.021*"рассказа" + 0.021*"написа" + 0.019*"школ" + 0.017*"как"
    INFO : topic #12 (0.050): 0.178*"rt" + 0.020*"люд" + 0.018*"дума" + 0.017*"возможн" + 0.016*"счита" + 0.015*"сша" + 0.012*"цен" + 0.012*"убийств" + 0.010*"кин" + 0.010*"боевик"
    INFO : topic diff=0.009295, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #35 = documents up to #3600/9999, outstanding queue size 4
    INFO : PROGRESS: pass 7, dispatched chunk #36 = documents up to #3700/9999, outstanding queue size 2
    INFO : PROGRESS: pass 7, dispatched chunk #37 = documents up to #3800/9999, outstanding queue size 3
    INFO : PROGRESS: pass 7, dispatched chunk #38 = documents up to #3900/9999, outstanding queue size 4
    INFO : PROGRESS: pass 7, dispatched chunk #39 = documents up to #4000/9999, outstanding queue size 5
    INFO : PROGRESS: pass 7, dispatched chunk #40 = documents up to #4100/9999, outstanding queue size 6
    INFO : PROGRESS: pass 7, dispatched chunk #41 = documents up to #4200/9999, outstanding queue size 7
    INFO : PROGRESS: pass 7, dispatched chunk #42 = documents up to #4300/9999, outstanding queue size 8
    INFO : PROGRESS: pass 7, dispatched chunk #43 = documents up to #4400/9999, outstanding queue size 9
    INFO : PROGRESS: pass 7, dispatched chunk #44 = documents up to #4500/9999, outstanding queue size 10
    INFO : PROGRESS: pass 7, dispatched chunk #45 = documents up to #4600/9999, outstanding queue size 11
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #7 (0.050): 0.071*"rt" + 0.021*"человек" + 0.020*"област" + 0.016*"люб" + 0.015*"дня" + 0.015*"суд" + 0.013*"новосибирск" + 0.013*"лидер" + 0.013*"пройдет" + 0.013*"фильм"
    INFO : topic #2 (0.050): 0.052*"rt" + 0.041*"глав" + 0.028*"киев" + 0.019*"жизн" + 0.019*"пыта" + 0.018*"погибл" + 0.017*"кита" + 0.016*"нашл" + 0.016*"результат" + 0.015*"взрыв"
    INFO : topic #8 (0.050): 0.057*"rt" + 0.036*"рубл" + 0.025*"слов" + 0.023*"сторон" + 0.021*"получ" + 0.019*"миноборон" + 0.016*"запрет" + 0.015*"мужчин" + 0.015*"дтп" + 0.014*"росс"
    INFO : topic #0 (0.050): 0.080*"rt" + 0.044*"сир" + 0.021*"рад" + 0.016*"главн" + 0.016*"удар" + 0.013*"игр" + 0.013*"сша" + 0.011*"переговор" + 0.011*"призва" + 0.010*"минск"
    INFO : topic #10 (0.050): 0.085*"rt" + 0.042*"виде" + 0.030*"дет" + 0.022*"полиц" + 0.021*"лучш" + 0.020*"хочет" + 0.019*"московск" + 0.015*"бизнес" + 0.013*"уб" + 0.013*"решен"
    INFO : topic diff=0.010537, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #46 = documents up to #4700/9999, outstanding queue size 9
    INFO : PROGRESS: pass 7, dispatched chunk #47 = documents up to #4800/9999, outstanding queue size 9
    INFO : PROGRESS: pass 7, dispatched chunk #48 = documents up to #4900/9999, outstanding queue size 10
    INFO : PROGRESS: pass 7, dispatched chunk #49 = documents up to #5000/9999, outstanding queue size 11
    INFO : PROGRESS: pass 7, dispatched chunk #50 = documents up to #5100/9999, outstanding queue size 12
    INFO : PROGRESS: pass 7, dispatched chunk #51 = documents up to #5200/9999, outstanding queue size 13
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.079*"rt" + 0.055*"нов" + 0.030*"санкц" + 0.027*"росс" + 0.024*"сам" + 0.017*"мид" + 0.015*"рф" + 0.014*"апрел" + 0.012*"сша" + 0.012*"иг"
    INFO : topic #12 (0.050): 0.185*"rt" + 0.024*"люд" + 0.018*"дума" + 0.016*"возможн" + 0.014*"сша" + 0.014*"счита" + 0.013*"цен" + 0.012*"убийств" + 0.010*"оруж" + 0.010*"кин"
    INFO : topic #11 (0.050): 0.087*"rt" + 0.020*"обам" + 0.017*"автомобил" + 0.017*"хорош" + 0.017*"истор" + 0.015*"депутат" + 0.013*"связ" + 0.013*"крушен" + 0.012*"ставропол" + 0.012*"поддержива"
    INFO : topic #3 (0.050): 0.062*"rt" + 0.053*"москв" + 0.032*"донецк" + 0.022*"район" + 0.021*"центр" + 0.015*"прошл" + 0.014*"встреч" + 0.014*"машин" + 0.012*"мэр" + 0.012*"продаж"
    INFO : topic #15 (0.050): 0.048*"rt" + 0.033*"quot" + 0.023*"дорог" + 0.022*"сто" + 0.015*"рук" + 0.014*"предлож" + 0.013*"факт" + 0.012*"выход" + 0.012*"солдат" + 0.011*"сезон"
    INFO : topic diff=0.010136, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #52 = documents up to #5300/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #15 (0.050): 0.047*"rt" + 0.034*"quot" + 0.026*"дорог" + 0.020*"сто" + 0.016*"предлож" + 0.015*"рук" + 0.013*"факт" + 0.013*"солдат" + 0.012*"выход" + 0.012*"заверш"
    INFO : topic #1 (0.050): 0.053*"стран" + 0.043*"rt" + 0.024*"дом" + 0.024*"жител" + 0.022*"американск" + 0.021*"пострада" + 0.016*"стал" + 0.015*"ожида" + 0.015*"сем" + 0.013*"конц"
    INFO : topic #17 (0.050): 0.134*"украин" + 0.068*"российск" + 0.056*"новост" + 0.055*"rt" + 0.030*"мнен" + 0.022*"побед" + 0.022*"выбор" + 0.019*"донбасс" + 0.018*"газ" + 0.018*"продолжа"
    INFO : topic #18 (0.050): 0.079*"rt" + 0.057*"нов" + 0.028*"санкц" + 0.025*"сам" + 0.025*"росс" + 0.016*"мид" + 0.016*"рф" + 0.013*"апрел" + 0.012*"иг" + 0.011*"акц"
    INFO : topic #14 (0.050): 0.080*"rt" + 0.038*"петербург" + 0.031*"вер" + 0.017*"медвед" + 0.014*"дмитр" + 0.014*"пожар" + 0.013*"матч" + 0.013*"ки" + 0.013*"смерт" + 0.012*"восток"
    INFO : topic diff=0.010461, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #53 = documents up to #5400/9999, outstanding queue size 6
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #17 (0.050): 0.136*"украин" + 0.068*"российск" + 0.055*"rt" + 0.054*"новост" + 0.030*"мнен" + 0.022*"побед" + 0.021*"выбор" + 0.020*"газ" + 0.019*"донбасс" + 0.018*"продолжа"
    INFO : topic #0 (0.050): 0.079*"rt" + 0.047*"сир" + 0.021*"рад" + 0.021*"главн" + 0.015*"переговор" + 0.015*"удар" + 0.013*"игр" + 0.012*"минск" + 0.011*"пьян" + 0.011*"сша"
    INFO : topic #12 (0.050): 0.181*"rt" + 0.022*"люд" + 0.018*"дума" + 0.015*"возможн" + 0.014*"счита" + 0.013*"сша" + 0.013*"убийств" + 0.012*"цен" + 0.012*"оруж" + 0.011*"доллар"
    INFO : topic #13 (0.050): 0.082*"rt" + 0.031*"готов" + 0.029*"украин" + 0.027*"сми" + 0.026*"дел" + 0.026*"власт" + 0.026*"перв" + 0.020*"русск" + 0.017*"прав" + 0.016*"заяв"
    INFO : topic #18 (0.050): 0.078*"rt" + 0.060*"нов" + 0.027*"санкц" + 0.025*"росс" + 0.024*"сам" + 0.015*"мид" + 0.014*"рф" + 0.013*"апрел" + 0.013*"акц" + 0.012*"иг"
    INFO : topic diff=0.010744, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #54 = documents up to #5500/9999, outstanding queue size 2
    INFO : PROGRESS: pass 7, dispatched chunk #55 = documents up to #5600/9999, outstanding queue size 2
    INFO : PROGRESS: pass 7, dispatched chunk #56 = documents up to #5700/9999, outstanding queue size 3
    INFO : PROGRESS: pass 7, dispatched chunk #57 = documents up to #5800/9999, outstanding queue size 4
    INFO : PROGRESS: pass 7, dispatched chunk #58 = documents up to #5900/9999, outstanding queue size 5
    INFO : PROGRESS: pass 7, dispatched chunk #59 = documents up to #6000/9999, outstanding queue size 6
    INFO : PROGRESS: pass 7, dispatched chunk #60 = documents up to #6100/9999, outstanding queue size 7
    INFO : PROGRESS: pass 7, dispatched chunk #61 = documents up to #6200/9999, outstanding queue size 8
    INFO : PROGRESS: pass 7, dispatched chunk #62 = documents up to #6300/9999, outstanding queue size 9
    INFO : PROGRESS: pass 7, dispatched chunk #63 = documents up to #6400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 7, dispatched chunk #64 = documents up to #6500/9999, outstanding queue size 9
    INFO : PROGRESS: pass 7, dispatched chunk #65 = documents up to #6600/9999, outstanding queue size 10
    INFO : PROGRESS: pass 7, dispatched chunk #66 = documents up to #6700/9999, outstanding queue size 11
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #15 (0.050): 0.049*"rt" + 0.036*"quot" + 0.030*"дорог" + 0.020*"сто" + 0.017*"предлож" + 0.015*"рук" + 0.012*"памятник" + 0.012*"постро" + 0.012*"узна" + 0.011*"выход"
    INFO : topic #18 (0.050): 0.079*"rt" + 0.063*"нов" + 0.028*"санкц" + 0.026*"росс" + 0.024*"сам" + 0.014*"рф" + 0.013*"мид" + 0.012*"апрел" + 0.012*"акц" + 0.011*"иг"
    INFO : topic #9 (0.050): 0.055*"rt" + 0.051*"президент" + 0.049*"рф" + 0.016*"нат" + 0.016*"город" + 0.013*"росс" + 0.012*"эксперт" + 0.012*"говор" + 0.012*"сентябр" + 0.011*"млн"
    INFO : topic #6 (0.050): 0.124*"путин" + 0.055*"rt" + 0.027*"владимир" + 0.026*"росс" + 0.022*"назва" + 0.020*"мест" + 0.018*"работ" + 0.018*"улиц" + 0.015*"друз" + 0.014*"крут"
    INFO : topic #0 (0.050): 0.079*"rt" + 0.048*"сир" + 0.023*"рад" + 0.020*"главн" + 0.016*"удар" + 0.015*"переговор" + 0.012*"игр" + 0.011*"сша" + 0.011*"минск" + 0.010*"пьян"
    INFO : topic diff=0.009841, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #67 = documents up to #6800/9999, outstanding queue size 10
    INFO : PROGRESS: pass 7, dispatched chunk #68 = documents up to #6900/9999, outstanding queue size 10
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.080*"rt" + 0.067*"нов" + 0.028*"росс" + 0.027*"санкц" + 0.023*"сам" + 0.013*"рф" + 0.013*"иг" + 0.012*"акц" + 0.012*"мид" + 0.012*"мчс"
    INFO : topic #19 (0.050): 0.063*"rt" + 0.045*"украинск" + 0.042*"воен" + 0.037*"крым" + 0.037*"украин" + 0.031*"ес" + 0.028*"фот" + 0.028*"политик" + 0.026*"сша" + 0.024*"днр"
    INFO : topic #13 (0.050): 0.080*"rt" + 0.031*"сми" + 0.031*"украин" + 0.029*"готов" + 0.028*"перв" + 0.025*"дел" + 0.025*"власт" + 0.020*"русск" + 0.017*"прав" + 0.016*"заяв"
    INFO : topic #5 (0.050): 0.061*"rt" + 0.023*"войн" + 0.019*"задержа" + 0.018*"полицейск" + 0.017*"план" + 0.015*"обстрел" + 0.015*"ситуац" + 0.014*"смотр" + 0.014*"сотрудник" + 0.014*"россиян"
    INFO : topic #8 (0.050): 0.056*"rt" + 0.033*"рубл" + 0.027*"слов" + 0.020*"получ" + 0.020*"сторон" + 0.017*"миноборон" + 0.017*"дтп" + 0.016*"переп" + 0.014*"министр" + 0.014*"запрет"
    INFO : topic diff=0.010361, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #69 = documents up to #7000/9999, outstanding queue size 7
    INFO : PROGRESS: pass 7, dispatched chunk #70 = documents up to #7100/9999, outstanding queue size 5
    INFO : PROGRESS: pass 7, dispatched chunk #71 = documents up to #7200/9999, outstanding queue size 6
    INFO : PROGRESS: pass 7, dispatched chunk #72 = documents up to #7300/9999, outstanding queue size 7
    INFO : PROGRESS: pass 7, dispatched chunk #73 = documents up to #7400/9999, outstanding queue size 8
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #12 (0.050): 0.177*"rt" + 0.019*"люд" + 0.018*"возможн" + 0.017*"дума" + 0.015*"цен" + 0.013*"счита" + 0.013*"сша" + 0.012*"убийств" + 0.011*"оруж" + 0.011*"доллар"
    INFO : topic #11 (0.050): 0.081*"rt" + 0.021*"истор" + 0.019*"хорош" + 0.018*"обам" + 0.014*"автомобил" + 0.014*"водител" + 0.013*"поддержива" + 0.013*"числ" + 0.012*"журналист" + 0.012*"мост"
    INFO : topic #18 (0.050): 0.081*"rt" + 0.066*"нов" + 0.029*"росс" + 0.026*"санкц" + 0.023*"сам" + 0.013*"рф" + 0.013*"представ" + 0.013*"сильн" + 0.012*"иг" + 0.012*"мчс"
    INFO : topic #16 (0.050): 0.063*"rt" + 0.037*"нача" + 0.030*"чита" + 0.027*"добр" + 0.026*"мир" + 0.024*"пост" + 0.022*"рассказа" + 0.021*"как" + 0.020*"написа" + 0.017*"школ"
    INFO : topic #6 (0.050): 0.128*"путин" + 0.055*"rt" + 0.027*"росс" + 0.024*"владимир" + 0.023*"назва" + 0.019*"мест" + 0.019*"работ" + 0.018*"улиц" + 0.015*"сборн" + 0.013*"друз"
    INFO : topic diff=0.010026, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #74 = documents up to #7500/9999, outstanding queue size 8
    INFO : PROGRESS: pass 7, dispatched chunk #75 = documents up to #7600/9999, outstanding queue size 7
    INFO : PROGRESS: pass 7, dispatched chunk #76 = documents up to #7700/9999, outstanding queue size 8
    INFO : PROGRESS: pass 7, dispatched chunk #77 = documents up to #7800/9999, outstanding queue size 9
    INFO : PROGRESS: pass 7, dispatched chunk #78 = documents up to #7900/9999, outstanding queue size 10
    INFO : PROGRESS: pass 7, dispatched chunk #79 = documents up to #8000/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #17 (0.050): 0.138*"украин" + 0.065*"российск" + 0.063*"новост" + 0.053*"rt" + 0.033*"мнен" + 0.021*"выбор" + 0.019*"побед" + 0.019*"газ" + 0.019*"донбасс" + 0.017*"арм"
    INFO : topic #8 (0.050): 0.058*"rt" + 0.032*"рубл" + 0.026*"слов" + 0.023*"получ" + 0.020*"сторон" + 0.017*"переп" + 0.017*"миноборон" + 0.015*"дтп" + 0.015*"запрет" + 0.014*"мужчин"
    INFO : topic #5 (0.050): 0.064*"rt" + 0.024*"войн" + 0.021*"задержа" + 0.017*"план" + 0.016*"полицейск" + 0.016*"смотр" + 0.016*"обстрел" + 0.014*"россиян" + 0.014*"запад" + 0.014*"предлага"
    INFO : topic #9 (0.050): 0.054*"rt" + 0.048*"рф" + 0.045*"президент" + 0.018*"нат" + 0.015*"город" + 0.014*"говор" + 0.013*"росс" + 0.013*"млн" + 0.012*"евр" + 0.012*"обсуд"
    INFO : topic #2 (0.050): 0.055*"rt" + 0.042*"глав" + 0.037*"киев" + 0.019*"взрыв" + 0.019*"жизн" + 0.018*"пыта" + 0.015*"погибл" + 0.014*"южн" + 0.014*"нашл" + 0.013*"международн"
    INFO : topic diff=0.009340, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #80 = documents up to #8100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 7, dispatched chunk #81 = documents up to #8200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 7, dispatched chunk #82 = documents up to #8300/9999, outstanding queue size 11
    INFO : PROGRESS: pass 7, dispatched chunk #83 = documents up to #8400/9999, outstanding queue size 12
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #5 (0.050): 0.064*"rt" + 0.027*"войн" + 0.022*"задержа" + 0.017*"полицейск" + 0.017*"план" + 0.016*"предлага" + 0.015*"обстрел" + 0.015*"смотр" + 0.014*"стат" + 0.014*"запад"
    INFO : topic #4 (0.050): 0.034*"интересн" + 0.032*"rt" + 0.025*"появ" + 0.023*"европ" + 0.020*"турц" + 0.020*"закон" + 0.019*"трамп" + 0.015*"границ" + 0.015*"отказа" + 0.014*"поддержк"
    INFO : topic #15 (0.050): 0.048*"rt" + 0.034*"quot" + 0.027*"дорог" + 0.025*"сто" + 0.014*"предлож" + 0.014*"памятник" + 0.014*"октябр" + 0.013*"факт" + 0.013*"выход" + 0.013*"действ"
    INFO : topic #7 (0.050): 0.071*"rt" + 0.022*"человек" + 0.020*"люб" + 0.019*"дня" + 0.016*"област" + 0.016*"пройдет" + 0.015*"фильм" + 0.014*"суд" + 0.012*"март" + 0.012*"лидер"
    INFO : topic #18 (0.050): 0.082*"rt" + 0.061*"нов" + 0.031*"росс" + 0.026*"санкц" + 0.024*"сам" + 0.017*"иг" + 0.016*"рф" + 0.014*"мид" + 0.013*"сильн" + 0.012*"западн"
    INFO : topic diff=0.009023, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #84 = documents up to #8500/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #5 (0.050): 0.063*"rt" + 0.027*"войн" + 0.021*"задержа" + 0.018*"полицейск" + 0.018*"смотр" + 0.017*"обстрел" + 0.016*"план" + 0.015*"предлага" + 0.014*"запад" + 0.013*"ситуац"
    INFO : topic #17 (0.050): 0.135*"украин" + 0.072*"российск" + 0.059*"новост" + 0.053*"rt" + 0.033*"мнен" + 0.022*"побед" + 0.022*"выбор" + 0.019*"арм" + 0.019*"газ" + 0.019*"донбасс"
    INFO : topic #4 (0.050): 0.035*"интересн" + 0.032*"rt" + 0.028*"появ" + 0.022*"европ" + 0.020*"закон" + 0.018*"турц" + 0.017*"трамп" + 0.016*"границ" + 0.015*"отказа" + 0.013*"очередн"
    INFO : topic #3 (0.050): 0.063*"rt" + 0.053*"москв" + 0.036*"донецк" + 0.028*"район" + 0.022*"центр" + 0.015*"встреч" + 0.013*"мэр" + 0.013*"прошл" + 0.013*"известн" + 0.011*"машин"
    INFO : topic #6 (0.050): 0.125*"путин" + 0.056*"rt" + 0.027*"владимир" + 0.025*"росс" + 0.022*"назва" + 0.020*"улиц" + 0.018*"работ" + 0.018*"сборн" + 0.016*"мест" + 0.016*"друз"
    INFO : topic diff=0.013927, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #85 = documents up to #8600/9999, outstanding queue size 7
    INFO : PROGRESS: pass 7, dispatched chunk #86 = documents up to #8700/9999, outstanding queue size 6
    INFO : PROGRESS: pass 7, dispatched chunk #87 = documents up to #8800/9999, outstanding queue size 7
    INFO : PROGRESS: pass 7, dispatched chunk #88 = documents up to #8900/9999, outstanding queue size 8
    INFO : PROGRESS: pass 7, dispatched chunk #89 = documents up to #9000/9999, outstanding queue size 8
    INFO : PROGRESS: pass 7, dispatched chunk #90 = documents up to #9100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 7, dispatched chunk #91 = documents up to #9200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 7, dispatched chunk #92 = documents up to #9300/9999, outstanding queue size 11
    INFO : PROGRESS: pass 7, dispatched chunk #93 = documents up to #9400/9999, outstanding queue size 12
    INFO : PROGRESS: pass 7, dispatched chunk #94 = documents up to #9500/9999, outstanding queue size 13
    INFO : merging changes from 700 documents into a model of 9999 documents
    INFO : topic #11 (0.050): 0.078*"rt" + 0.019*"обам" + 0.019*"хорош" + 0.019*"истор" + 0.017*"автомобил" + 0.014*"депутат" + 0.013*"журналист" + 0.013*"поддержива" + 0.013*"праздник" + 0.012*"связ"
    INFO : topic #13 (0.050): 0.081*"rt" + 0.031*"сми" + 0.027*"украин" + 0.026*"перв" + 0.026*"готов" + 0.025*"дел" + 0.023*"власт" + 0.019*"русск" + 0.017*"заяв" + 0.016*"террорист"
    INFO : topic #19 (0.050): 0.063*"rt" + 0.044*"воен" + 0.043*"украинск" + 0.038*"ес" + 0.035*"украин" + 0.032*"фот" + 0.031*"крым" + 0.030*"политик" + 0.029*"сша" + 0.026*"чуж"
    INFO : topic #3 (0.050): 0.063*"rt" + 0.053*"москв" + 0.037*"донецк" + 0.027*"район" + 0.022*"центр" + 0.014*"встреч" + 0.013*"известн" + 0.012*"прошл" + 0.012*"продаж" + 0.012*"мэр"
    INFO : topic #8 (0.050): 0.062*"rt" + 0.035*"рубл" + 0.026*"слов" + 0.024*"получ" + 0.023*"сторон" + 0.015*"переп" + 0.015*"министр" + 0.015*"миноборон" + 0.014*"дтп" + 0.013*"росс"
    INFO : topic diff=0.009153, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #95 = documents up to #9600/9999, outstanding queue size 10
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #12 (0.050): 0.188*"rt" + 0.019*"возможн" + 0.018*"дума" + 0.017*"люд" + 0.016*"сша" + 0.014*"счита" + 0.013*"убийств" + 0.012*"цен" + 0.011*"сирийск" + 0.011*"боевик"
    INFO : topic #11 (0.050): 0.079*"rt" + 0.019*"обам" + 0.019*"хорош" + 0.018*"автомобил" + 0.018*"истор" + 0.015*"депутат" + 0.014*"ставропол" + 0.014*"поддержива" + 0.013*"журналист" + 0.012*"водител"
    INFO : topic #7 (0.050): 0.074*"rt" + 0.020*"област" + 0.019*"люб" + 0.019*"человек" + 0.018*"пройдет" + 0.018*"дня" + 0.015*"суд" + 0.014*"март" + 0.014*"отношен" + 0.013*"фильм"
    INFO : topic #6 (0.050): 0.130*"путин" + 0.057*"rt" + 0.029*"владимир" + 0.025*"росс" + 0.021*"работ" + 0.020*"улиц" + 0.019*"назва" + 0.018*"сборн" + 0.015*"мест" + 0.015*"друз"
    INFO : topic #17 (0.050): 0.139*"украин" + 0.071*"российск" + 0.057*"новост" + 0.055*"rt" + 0.033*"мнен" + 0.022*"выбор" + 0.021*"побед" + 0.019*"арм" + 0.017*"донбасс" + 0.017*"газ"
    INFO : topic diff=0.010405, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #96 = documents up to #9700/9999, outstanding queue size 6
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #3 (0.050): 0.065*"rt" + 0.056*"москв" + 0.033*"донецк" + 0.026*"район" + 0.024*"центр" + 0.016*"прошл" + 0.015*"встреч" + 0.014*"известн" + 0.012*"продаж" + 0.012*"мэр"
    INFO : topic #4 (0.050): 0.035*"интересн" + 0.032*"rt" + 0.026*"появ" + 0.023*"европ" + 0.019*"турц" + 0.019*"трамп" + 0.018*"границ" + 0.017*"закон" + 0.015*"очередн" + 0.014*"отказа"
    INFO : topic #15 (0.050): 0.046*"rt" + 0.030*"quot" + 0.027*"дорог" + 0.024*"сто" + 0.014*"факт" + 0.014*"выход" + 0.014*"октябр" + 0.013*"заверш" + 0.013*"рук" + 0.013*"узна"
    INFO : topic #13 (0.050): 0.083*"rt" + 0.031*"сми" + 0.028*"перв" + 0.027*"украин" + 0.027*"дел" + 0.026*"готов" + 0.020*"власт" + 0.019*"русск" + 0.018*"заяв" + 0.015*"террорист"
    INFO : topic #17 (0.050): 0.135*"украин" + 0.072*"российск" + 0.056*"rt" + 0.056*"новост" + 0.031*"мнен" + 0.024*"выбор" + 0.022*"побед" + 0.019*"арм" + 0.017*"донбасс" + 0.016*"газ"
    INFO : topic diff=0.008542, rho=0.096230
    INFO : PROGRESS: pass 7, dispatched chunk #97 = documents up to #9800/9999, outstanding queue size 2
    INFO : PROGRESS: pass 7, dispatched chunk #98 = documents up to #9900/9999, outstanding queue size 2
    INFO : PROGRESS: pass 7, dispatched chunk #99 = documents up to #9999/9999, outstanding queue size 3
    INFO : merging changes from 399 documents into a model of 9999 documents
    INFO : topic #4 (0.050): 0.037*"интересн" + 0.032*"rt" + 0.028*"появ" + 0.025*"европ" + 0.020*"границ" + 0.018*"трамп" + 0.017*"турц" + 0.016*"закон" + 0.016*"очередн" + 0.014*"развит"
    INFO : topic #10 (0.050): 0.086*"rt" + 0.039*"виде" + 0.036*"дет" + 0.026*"лучш" + 0.022*"полиц" + 0.020*"московск" + 0.019*"хочет" + 0.016*"бизнес" + 0.015*"решен" + 0.015*"метр"
    INFO : topic #5 (0.050): 0.062*"rt" + 0.033*"войн" + 0.020*"задержа" + 0.019*"полицейск" + 0.018*"ситуац" + 0.016*"обстрел" + 0.016*"план" + 0.016*"запад" + 0.015*"предлага" + 0.014*"смотр"
    INFO : topic #18 (0.050): 0.082*"rt" + 0.066*"нов" + 0.030*"росс" + 0.029*"санкц" + 0.024*"сам" + 0.016*"мид" + 0.015*"рф" + 0.014*"апрел" + 0.014*"иг" + 0.014*"представ"
    INFO : topic #12 (0.050): 0.189*"rt" + 0.020*"люд" + 0.017*"возможн" + 0.017*"дума" + 0.015*"сша" + 0.015*"счита" + 0.014*"убийств" + 0.011*"кин" + 0.011*"сирийск" + 0.011*"цен"
    INFO : topic diff=0.010898, rho=0.096230
    INFO : -15.864 per-word bound, 59645.1 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : PROGRESS: pass 8, dispatched chunk #0 = documents up to #100/9999, outstanding queue size 1
    INFO : PROGRESS: pass 8, dispatched chunk #1 = documents up to #200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 8, dispatched chunk #2 = documents up to #300/9999, outstanding queue size 3
    INFO : PROGRESS: pass 8, dispatched chunk #3 = documents up to #400/9999, outstanding queue size 4
    INFO : PROGRESS: pass 8, dispatched chunk #4 = documents up to #500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 8, dispatched chunk #5 = documents up to #600/9999, outstanding queue size 6
    INFO : PROGRESS: pass 8, dispatched chunk #6 = documents up to #700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 8, dispatched chunk #7 = documents up to #800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 8, dispatched chunk #8 = documents up to #900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 8, dispatched chunk #9 = documents up to #1000/9999, outstanding queue size 10
    INFO : PROGRESS: pass 8, dispatched chunk #10 = documents up to #1100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 8, dispatched chunk #11 = documents up to #1200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 8, dispatched chunk #12 = documents up to #1300/9999, outstanding queue size 11
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #7 (0.050): 0.070*"rt" + 0.021*"област" + 0.021*"люб" + 0.017*"дня" + 0.016*"человек" + 0.016*"суд" + 0.015*"фильм" + 0.014*"лидер" + 0.013*"пройдет" + 0.013*"отношен"
    INFO : topic #19 (0.050): 0.063*"rt" + 0.045*"украинск" + 0.043*"воен" + 0.039*"ес" + 0.035*"фот" + 0.033*"украин" + 0.029*"политик" + 0.029*"крым" + 0.025*"сша" + 0.024*"чуж"
    INFO : topic #3 (0.050): 0.061*"rt" + 0.057*"москв" + 0.034*"донецк" + 0.023*"центр" + 0.021*"район" + 0.017*"прошл" + 0.015*"встреч" + 0.012*"банк" + 0.012*"известн" + 0.011*"дан"
    INFO : topic #18 (0.050): 0.078*"rt" + 0.068*"нов" + 0.028*"росс" + 0.028*"санкц" + 0.024*"сам" + 0.016*"мид" + 0.014*"рф" + 0.014*"сша" + 0.014*"иг" + 0.014*"представ"
    INFO : topic #2 (0.050): 0.051*"rt" + 0.046*"глав" + 0.033*"киев" + 0.021*"жизн" + 0.017*"взрыв" + 0.016*"пыта" + 0.015*"международн" + 0.015*"погибл" + 0.015*"кита" + 0.014*"южн"
    INFO : topic diff=0.046131, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #13 = documents up to #1400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 8, dispatched chunk #14 = documents up to #1500/9999, outstanding queue size 10
    INFO : PROGRESS: pass 8, dispatched chunk #15 = documents up to #1600/9999, outstanding queue size 11
    INFO : PROGRESS: pass 8, dispatched chunk #16 = documents up to #1700/9999, outstanding queue size 12
    INFO : PROGRESS: pass 8, dispatched chunk #17 = documents up to #1800/9999, outstanding queue size 13
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #3 (0.050): 0.059*"rt" + 0.058*"москв" + 0.032*"донецк" + 0.023*"центр" + 0.022*"район" + 0.015*"прошл" + 0.015*"встреч" + 0.012*"банк" + 0.011*"известн" + 0.010*"машин"
    INFO : topic #4 (0.050): 0.035*"интересн" + 0.032*"rt" + 0.026*"появ" + 0.023*"европ" + 0.018*"трамп" + 0.017*"границ" + 0.017*"турц" + 0.016*"очередн" + 0.014*"закон" + 0.014*"сша"
    INFO : topic #16 (0.050): 0.059*"rt" + 0.031*"нача" + 0.027*"чита" + 0.025*"мир" + 0.023*"добр" + 0.022*"пост" + 0.021*"как" + 0.021*"написа" + 0.020*"рассказа" + 0.017*"школ"
    INFO : topic #18 (0.050): 0.074*"rt" + 0.063*"нов" + 0.028*"санкц" + 0.027*"росс" + 0.022*"сам" + 0.017*"мид" + 0.014*"рф" + 0.013*"иг" + 0.013*"сша" + 0.012*"представ"
    INFO : topic #17 (0.050): 0.132*"украин" + 0.069*"российск" + 0.057*"новост" + 0.051*"rt" + 0.032*"мнен" + 0.023*"выбор" + 0.021*"побед" + 0.017*"продолжа" + 0.016*"росс" + 0.016*"донбасс"
    INFO : topic diff=0.039576, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #18 = documents up to #1900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 8, dispatched chunk #19 = documents up to #2000/9999, outstanding queue size 8
    INFO : PROGRESS: pass 8, dispatched chunk #20 = documents up to #2100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 8, dispatched chunk #21 = documents up to #2200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 8, dispatched chunk #22 = documents up to #2300/9999, outstanding queue size 11
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #15 (0.050): 0.046*"rt" + 0.032*"quot" + 0.024*"дорог" + 0.022*"сто" + 0.015*"факт" + 0.013*"постро" + 0.013*"узна" + 0.012*"заверш" + 0.011*"предлож" + 0.011*"сезон"
    INFO : topic #7 (0.050): 0.067*"rt" + 0.021*"област" + 0.017*"люб" + 0.015*"человек" + 0.014*"лидер" + 0.013*"дня" + 0.013*"летн" + 0.013*"суд" + 0.012*"отношен" + 0.012*"фильм"
    INFO : topic #9 (0.050): 0.053*"rt" + 0.048*"президент" + 0.043*"рф" + 0.016*"город" + 0.014*"нат" + 0.012*"обсуд" + 0.012*"говор" + 0.011*"млн" + 0.011*"участ" + 0.010*"росс"
    INFO : topic #3 (0.050): 0.058*"rt" + 0.055*"москв" + 0.032*"донецк" + 0.023*"район" + 0.020*"центр" + 0.016*"встреч" + 0.014*"прошл" + 0.013*"банк" + 0.012*"машин" + 0.011*"официальн"
    INFO : topic #16 (0.050): 0.060*"rt" + 0.034*"нача" + 0.025*"чита" + 0.023*"написа" + 0.023*"мир" + 0.021*"рассказа" + 0.021*"добр" + 0.020*"пост" + 0.019*"как" + 0.016*"школ"
    INFO : topic diff=0.015402, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #23 = documents up to #2400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 8, dispatched chunk #24 = documents up to #2500/9999, outstanding queue size 10
    INFO : PROGRESS: pass 8, dispatched chunk #25 = documents up to #2600/9999, outstanding queue size 11
    INFO : PROGRESS: pass 8, dispatched chunk #26 = documents up to #2700/9999, outstanding queue size 12
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #10 (0.050): 0.080*"rt" + 0.041*"виде" + 0.030*"дет" + 0.020*"лучш" + 0.019*"полиц" + 0.018*"хочет" + 0.017*"московск" + 0.016*"уб" + 0.014*"метр" + 0.013*"правительств"
    INFO : topic #15 (0.050): 0.047*"rt" + 0.029*"quot" + 0.023*"дорог" + 0.022*"сто" + 0.015*"факт" + 0.015*"предлож" + 0.013*"рук" + 0.013*"постро" + 0.012*"узна" + 0.011*"сезон"
    INFO : topic #4 (0.050): 0.031*"rt" + 0.030*"интересн" + 0.026*"появ" + 0.024*"европ" + 0.018*"турц" + 0.018*"границ" + 0.017*"закон" + 0.015*"трамп" + 0.015*"развит" + 0.014*"очередн"
    INFO : topic #9 (0.050): 0.052*"rt" + 0.047*"президент" + 0.044*"рф" + 0.017*"город" + 0.014*"нат" + 0.013*"обсуд" + 0.013*"говор" + 0.012*"млн" + 0.011*"росс" + 0.010*"участ"
    INFO : topic #8 (0.050): 0.058*"rt" + 0.038*"рубл" + 0.022*"слов" + 0.021*"сторон" + 0.019*"получ" + 0.019*"миноборон" + 0.015*"дтп" + 0.015*"мужчин" + 0.015*"запрет" + 0.014*"министр"
    INFO : topic diff=0.012925, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #27 = documents up to #2800/9999, outstanding queue size 10
    INFO : PROGRESS: pass 8, dispatched chunk #28 = documents up to #2900/9999, outstanding queue size 10
    INFO : PROGRESS: pass 8, dispatched chunk #29 = documents up to #3000/9999, outstanding queue size 11
    INFO : PROGRESS: pass 8, dispatched chunk #30 = documents up to #3100/9999, outstanding queue size 12
    INFO : PROGRESS: pass 8, dispatched chunk #31 = documents up to #3200/9999, outstanding queue size 13
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #16 (0.050): 0.059*"rt" + 0.033*"нача" + 0.028*"чита" + 0.026*"мир" + 0.022*"написа" + 0.021*"пост" + 0.019*"добр" + 0.019*"рассказа" + 0.018*"школ" + 0.016*"как"
    INFO : topic #5 (0.050): 0.067*"rt" + 0.030*"войн" + 0.019*"задержа" + 0.017*"полицейск" + 0.016*"ситуац" + 0.016*"украинц" + 0.015*"запад" + 0.013*"обстрел" + 0.013*"план" + 0.012*"россиян"
    INFO : topic #4 (0.050): 0.034*"интересн" + 0.031*"rt" + 0.024*"появ" + 0.023*"европ" + 0.018*"турц" + 0.017*"границ" + 0.016*"очередн" + 0.016*"закон" + 0.016*"развит" + 0.014*"трамп"
    INFO : topic #8 (0.050): 0.058*"rt" + 0.037*"рубл" + 0.024*"слов" + 0.023*"сторон" + 0.019*"получ" + 0.019*"миноборон" + 0.015*"дтп" + 0.014*"запрет" + 0.013*"мужчин" + 0.013*"росс"
    INFO : topic #7 (0.050): 0.069*"rt" + 0.021*"област" + 0.017*"человек" + 0.017*"люб" + 0.016*"суд" + 0.014*"новосибирск" + 0.014*"дня" + 0.013*"март" + 0.013*"летн" + 0.013*"фильм"
    INFO : topic diff=0.008868, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #32 = documents up to #3300/9999, outstanding queue size 11
    INFO : PROGRESS: pass 8, dispatched chunk #33 = documents up to #3400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 8, dispatched chunk #34 = documents up to #3500/9999, outstanding queue size 11
    INFO : PROGRESS: pass 8, dispatched chunk #35 = documents up to #3600/9999, outstanding queue size 12
    INFO : PROGRESS: pass 8, dispatched chunk #36 = documents up to #3700/9999, outstanding queue size 12
    INFO : PROGRESS: pass 8, dispatched chunk #37 = documents up to #3800/9999, outstanding queue size 13
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #0 (0.050): 0.081*"rt" + 0.043*"сир" + 0.021*"рад" + 0.016*"главн" + 0.014*"игр" + 0.014*"удар" + 0.012*"сша" + 0.011*"переговор" + 0.011*"пьян" + 0.011*"минск"
    INFO : topic #7 (0.050): 0.070*"rt" + 0.021*"област" + 0.018*"человек" + 0.016*"суд" + 0.015*"люб" + 0.014*"новосибирск" + 0.014*"дня" + 0.013*"пройдет" + 0.013*"лидер" + 0.012*"фильм"
    INFO : topic #15 (0.050): 0.047*"rt" + 0.032*"quot" + 0.025*"сто" + 0.023*"дорог" + 0.014*"факт" + 0.014*"рук" + 0.014*"предлож" + 0.013*"узна" + 0.012*"сезон" + 0.012*"выход"
    INFO : topic #16 (0.050): 0.059*"rt" + 0.032*"нача" + 0.031*"чита" + 0.027*"мир" + 0.021*"пост" + 0.021*"написа" + 0.020*"рассказа" + 0.019*"добр" + 0.019*"школ" + 0.017*"как"
    INFO : topic #1 (0.050): 0.054*"стран" + 0.044*"rt" + 0.025*"дом" + 0.021*"американск" + 0.020*"жител" + 0.018*"стал" + 0.017*"пострада" + 0.015*"сем" + 0.015*"днем" + 0.014*"массов"
    INFO : topic diff=0.010538, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #38 = documents up to #3900/9999, outstanding queue size 11
    INFO : PROGRESS: pass 8, dispatched chunk #39 = documents up to #4000/9999, outstanding queue size 11
    INFO : PROGRESS: pass 8, dispatched chunk #40 = documents up to #4100/9999, outstanding queue size 12
    INFO : PROGRESS: pass 8, dispatched chunk #41 = documents up to #4200/9999, outstanding queue size 13
    INFO : PROGRESS: pass 8, dispatched chunk #42 = documents up to #4300/9999, outstanding queue size 14
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #1 (0.050): 0.053*"стран" + 0.044*"rt" + 0.027*"дом" + 0.019*"жител" + 0.019*"американск" + 0.018*"пострада" + 0.017*"стал" + 0.016*"сем" + 0.014*"днем" + 0.014*"массов"
    INFO : topic #5 (0.050): 0.068*"rt" + 0.028*"войн" + 0.020*"полицейск" + 0.018*"задержа" + 0.017*"ситуац" + 0.016*"украинц" + 0.015*"обстрел" + 0.014*"отправ" + 0.014*"запад" + 0.013*"план"
    INFO : topic #10 (0.050): 0.083*"rt" + 0.045*"виде" + 0.031*"дет" + 0.022*"полиц" + 0.022*"хочет" + 0.020*"лучш" + 0.018*"московск" + 0.014*"бизнес" + 0.014*"уб" + 0.013*"правительств"
    INFO : topic #18 (0.050): 0.080*"rt" + 0.054*"нов" + 0.030*"росс" + 0.028*"санкц" + 0.022*"сам" + 0.016*"мид" + 0.014*"апрел" + 0.014*"рф" + 0.013*"иг" + 0.012*"сильн"
    INFO : topic #14 (0.050): 0.080*"rt" + 0.032*"петербург" + 0.025*"вер" + 0.017*"медвед" + 0.015*"дмитр" + 0.013*"пожар" + 0.013*"восток" + 0.013*"матч" + 0.011*"ки" + 0.010*"польш"
    INFO : topic diff=0.011336, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #43 = documents up to #4400/9999, outstanding queue size 11
    INFO : PROGRESS: pass 8, dispatched chunk #44 = documents up to #4500/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #9 (0.050): 0.055*"rt" + 0.050*"президент" + 0.049*"рф" + 0.016*"город" + 0.015*"нат" + 0.014*"млн" + 0.013*"росс" + 0.013*"участ" + 0.012*"обсуд" + 0.011*"говор"
    INFO : topic #12 (0.050): 0.180*"rt" + 0.022*"люд" + 0.018*"дума" + 0.018*"возможн" + 0.015*"сша" + 0.015*"счита" + 0.014*"цен" + 0.011*"убийств" + 0.011*"автор" + 0.010*"оруж"
    INFO : topic #4 (0.050): 0.032*"rt" + 0.031*"интересн" + 0.030*"появ" + 0.024*"европ" + 0.020*"турц" + 0.018*"закон" + 0.016*"границ" + 0.015*"отказа" + 0.014*"очередн" + 0.014*"развит"
    INFO : topic #16 (0.050): 0.058*"rt" + 0.034*"нача" + 0.034*"чита" + 0.027*"мир" + 0.027*"пост" + 0.021*"рассказа" + 0.020*"добр" + 0.020*"написа" + 0.019*"школ" + 0.016*"интернет"
    INFO : topic #18 (0.050): 0.079*"rt" + 0.054*"нов" + 0.030*"росс" + 0.029*"санкц" + 0.022*"сам" + 0.017*"мид" + 0.014*"рф" + 0.014*"апрел" + 0.012*"иг" + 0.012*"сша"
    INFO : topic diff=0.012763, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #45 = documents up to #4600/9999, outstanding queue size 9
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #3 (0.050): 0.063*"rt" + 0.058*"москв" + 0.031*"донецк" + 0.022*"район" + 0.020*"центр" + 0.016*"прошл" + 0.015*"машин" + 0.014*"встреч" + 0.012*"мэр" + 0.011*"официальн"
    INFO : topic #17 (0.050): 0.139*"украин" + 0.066*"российск" + 0.056*"rt" + 0.056*"новост" + 0.032*"мнен" + 0.023*"побед" + 0.021*"выбор" + 0.020*"арм" + 0.019*"газ" + 0.019*"донбасс"
    INFO : topic #6 (0.050): 0.120*"путин" + 0.055*"rt" + 0.029*"владимир" + 0.025*"росс" + 0.023*"назва" + 0.022*"работ" + 0.020*"мест" + 0.019*"улиц" + 0.017*"друз" + 0.013*"жил"
    INFO : topic #11 (0.050): 0.088*"rt" + 0.020*"обам" + 0.017*"автомобил" + 0.017*"хорош" + 0.017*"истор" + 0.014*"крушен" + 0.013*"связ" + 0.013*"поддержива" + 0.013*"депутат" + 0.013*"ставропол"
    INFO : topic #4 (0.050): 0.032*"rt" + 0.030*"интересн" + 0.030*"появ" + 0.022*"европ" + 0.021*"турц" + 0.021*"закон" + 0.017*"отказа" + 0.015*"границ" + 0.014*"очередн" + 0.014*"трамп"
    INFO : topic diff=0.009721, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #46 = documents up to #4700/9999, outstanding queue size 5
    INFO : PROGRESS: pass 8, dispatched chunk #47 = documents up to #4800/9999, outstanding queue size 4
    INFO : PROGRESS: pass 8, dispatched chunk #48 = documents up to #4900/9999, outstanding queue size 4
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #14 (0.050): 0.082*"rt" + 0.034*"петербург" + 0.031*"вер" + 0.018*"медвед" + 0.016*"пожар" + 0.014*"дмитр" + 0.014*"восток" + 0.013*"матч" + 0.013*"ки" + 0.012*"смерт"
    INFO : topic #12 (0.050): 0.186*"rt" + 0.024*"люд" + 0.017*"дума" + 0.016*"возможн" + 0.015*"сша" + 0.014*"счита" + 0.013*"цен" + 0.012*"убийств" + 0.011*"оруж" + 0.010*"кин"
    INFO : topic #11 (0.050): 0.088*"rt" + 0.019*"обам" + 0.018*"хорош" + 0.017*"истор" + 0.016*"автомобил" + 0.015*"депутат" + 0.015*"связ" + 0.013*"крушен" + 0.013*"числ" + 0.013*"поддержива"
    INFO : topic #13 (0.050): 0.083*"rt" + 0.033*"готов" + 0.027*"власт" + 0.027*"дел" + 0.026*"перв" + 0.025*"сми" + 0.025*"украин" + 0.018*"русск" + 0.017*"прав" + 0.015*"заяв"
    INFO : topic #6 (0.050): 0.123*"путин" + 0.055*"rt" + 0.029*"владимир" + 0.024*"росс" + 0.024*"назва" + 0.023*"мест" + 0.021*"работ" + 0.020*"улиц" + 0.016*"друз" + 0.012*"жил"
    INFO : topic diff=0.008694, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #49 = documents up to #5000/9999, outstanding queue size 4
    INFO : PROGRESS: pass 8, dispatched chunk #50 = documents up to #5100/9999, outstanding queue size 2
    INFO : PROGRESS: pass 8, dispatched chunk #51 = documents up to #5200/9999, outstanding queue size 3
    INFO : PROGRESS: pass 8, dispatched chunk #52 = documents up to #5300/9999, outstanding queue size 4
    INFO : PROGRESS: pass 8, dispatched chunk #53 = documents up to #5400/9999, outstanding queue size 5
    INFO : PROGRESS: pass 8, dispatched chunk #54 = documents up to #5500/9999, outstanding queue size 6
    INFO : PROGRESS: pass 8, dispatched chunk #55 = documents up to #5600/9999, outstanding queue size 7
    INFO : PROGRESS: pass 8, dispatched chunk #56 = documents up to #5700/9999, outstanding queue size 8
    INFO : PROGRESS: pass 8, dispatched chunk #57 = documents up to #5800/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #17 (0.050): 0.139*"украин" + 0.072*"российск" + 0.057*"rt" + 0.053*"новост" + 0.028*"мнен" + 0.022*"побед" + 0.020*"выбор" + 0.019*"донбасс" + 0.019*"газ" + 0.018*"арм"
    INFO : topic #5 (0.050): 0.066*"rt" + 0.025*"войн" + 0.019*"задержа" + 0.019*"полицейск" + 0.014*"план" + 0.014*"стат" + 0.014*"ситуац" + 0.014*"сотрудник" + 0.013*"украинц" + 0.013*"смотр"
    INFO : topic #15 (0.050): 0.048*"rt" + 0.033*"quot" + 0.027*"дорог" + 0.021*"сто" + 0.018*"предлож" + 0.015*"рук" + 0.013*"факт" + 0.012*"солдат" + 0.012*"выход" + 0.011*"заверш"
    INFO : topic #14 (0.050): 0.082*"rt" + 0.037*"петербург" + 0.032*"вер" + 0.016*"медвед" + 0.015*"матч" + 0.015*"пожар" + 0.015*"восток" + 0.015*"ки" + 0.012*"дмитр" + 0.012*"сообщ"
    INFO : topic #11 (0.050): 0.088*"rt" + 0.021*"истор" + 0.019*"обам" + 0.018*"хорош" + 0.016*"автомобил" + 0.015*"связ" + 0.015*"депутат" + 0.014*"числ" + 0.013*"крушен" + 0.012*"поддержива"
    INFO : topic diff=0.011035, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #58 = documents up to #5900/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #10 (0.050): 0.085*"rt" + 0.040*"виде" + 0.029*"дет" + 0.022*"полиц" + 0.020*"лучш" + 0.019*"хочет" + 0.017*"московск" + 0.016*"уб" + 0.015*"последн" + 0.015*"решен"
    INFO : topic #6 (0.050): 0.127*"путин" + 0.055*"rt" + 0.028*"владимир" + 0.023*"росс" + 0.023*"назва" + 0.022*"мест" + 0.019*"улиц" + 0.018*"работ" + 0.017*"друз" + 0.014*"жил"
    INFO : topic #19 (0.050): 0.063*"rt" + 0.046*"украинск" + 0.043*"воен" + 0.039*"крым" + 0.038*"украин" + 0.032*"ес" + 0.031*"фот" + 0.029*"политик" + 0.026*"сша" + 0.026*"днр"
    INFO : topic #9 (0.050): 0.055*"rt" + 0.051*"рф" + 0.049*"президент" + 0.016*"город" + 0.015*"нат" + 0.014*"росс" + 0.013*"эксперт" + 0.012*"войск" + 0.012*"участ" + 0.012*"говор"
    INFO : topic #5 (0.050): 0.064*"rt" + 0.025*"войн" + 0.019*"задержа" + 0.017*"полицейск" + 0.017*"план" + 0.015*"обстрел" + 0.015*"украинц" + 0.014*"сотрудник" + 0.014*"стат" + 0.013*"смотр"
    INFO : topic diff=0.010953, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #59 = documents up to #6000/9999, outstanding queue size 6
    INFO : PROGRESS: pass 8, dispatched chunk #60 = documents up to #6100/9999, outstanding queue size 4
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #4 (0.050): 0.033*"интересн" + 0.031*"rt" + 0.026*"появ" + 0.023*"закон" + 0.023*"европ" + 0.020*"границ" + 0.018*"турц" + 0.018*"отказа" + 0.014*"нужн" + 0.013*"трамп"
    INFO : topic #16 (0.050): 0.061*"rt" + 0.035*"нача" + 0.029*"чита" + 0.027*"добр" + 0.025*"пост" + 0.025*"мир" + 0.023*"написа" + 0.022*"рассказа" + 0.022*"как" + 0.021*"потеря"
    INFO : topic #9 (0.050): 0.056*"rt" + 0.051*"рф" + 0.050*"президент" + 0.016*"нат" + 0.016*"город" + 0.013*"росс" + 0.013*"сентябр" + 0.012*"эксперт" + 0.012*"войск" + 0.012*"евр"
    INFO : topic #12 (0.050): 0.182*"rt" + 0.021*"люд" + 0.018*"дума" + 0.017*"возможн" + 0.016*"счита" + 0.014*"сша" + 0.013*"оруж" + 0.013*"цен" + 0.013*"убийств" + 0.011*"автор"
    INFO : topic #3 (0.050): 0.062*"rt" + 0.051*"москв" + 0.036*"донецк" + 0.024*"район" + 0.020*"центр" + 0.015*"прошл" + 0.015*"продаж" + 0.013*"машин" + 0.012*"ран" + 0.012*"банк"
    INFO : topic diff=0.010950, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #61 = documents up to #6200/9999, outstanding queue size 4
    INFO : PROGRESS: pass 8, dispatched chunk #62 = documents up to #6300/9999, outstanding queue size 2
    INFO : PROGRESS: pass 8, dispatched chunk #63 = documents up to #6400/9999, outstanding queue size 3
    INFO : PROGRESS: pass 8, dispatched chunk #64 = documents up to #6500/9999, outstanding queue size 4
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #9 (0.050): 0.055*"rt" + 0.052*"рф" + 0.049*"президент" + 0.016*"нат" + 0.014*"город" + 0.014*"войск" + 0.014*"росс" + 0.013*"эксперт" + 0.013*"сентябр" + 0.013*"млн"
    INFO : topic #7 (0.050): 0.073*"rt" + 0.026*"человек" + 0.021*"област" + 0.019*"люб" + 0.015*"суд" + 0.014*"дня" + 0.013*"летн" + 0.012*"новосибирск" + 0.012*"фильм" + 0.012*"март"
    INFO : topic #17 (0.050): 0.150*"украин" + 0.068*"российск" + 0.062*"новост" + 0.056*"rt" + 0.030*"мнен" + 0.022*"выбор" + 0.020*"газ" + 0.019*"побед" + 0.019*"донбасс" + 0.019*"арм"
    INFO : topic #2 (0.050): 0.055*"rt" + 0.042*"глав" + 0.039*"киев" + 0.021*"жизн" + 0.020*"взрыв" + 0.017*"пыта" + 0.016*"нашл" + 0.015*"кита" + 0.015*"результат" + 0.014*"страшн"
    INFO : topic #10 (0.050): 0.086*"rt" + 0.039*"виде" + 0.030*"дет" + 0.025*"лучш" + 0.023*"полиц" + 0.017*"хочет" + 0.017*"московск" + 0.017*"уб" + 0.014*"последн" + 0.013*"решен"
    INFO : topic diff=0.011521, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #65 = documents up to #6600/9999, outstanding queue size 4
    INFO : PROGRESS: pass 8, dispatched chunk #66 = documents up to #6700/9999, outstanding queue size 2
    INFO : PROGRESS: pass 8, dispatched chunk #67 = documents up to #6800/9999, outstanding queue size 3
    INFO : PROGRESS: pass 8, dispatched chunk #68 = documents up to #6900/9999, outstanding queue size 4
    INFO : PROGRESS: pass 8, dispatched chunk #69 = documents up to #7000/9999, outstanding queue size 5
    INFO : PROGRESS: pass 8, dispatched chunk #70 = documents up to #7100/9999, outstanding queue size 6
    INFO : PROGRESS: pass 8, dispatched chunk #71 = documents up to #7200/9999, outstanding queue size 7
    INFO : PROGRESS: pass 8, dispatched chunk #72 = documents up to #7300/9999, outstanding queue size 8
    INFO : PROGRESS: pass 8, dispatched chunk #73 = documents up to #7400/9999, outstanding queue size 9
    INFO : PROGRESS: pass 8, dispatched chunk #74 = documents up to #7500/9999, outstanding queue size 10
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #1 (0.050): 0.051*"стран" + 0.044*"rt" + 0.029*"американск" + 0.025*"пострада" + 0.021*"дом" + 0.019*"жител" + 0.015*"сем" + 0.015*"конц" + 0.015*"днем" + 0.013*"ноч"
    INFO : topic #10 (0.050): 0.087*"rt" + 0.039*"виде" + 0.030*"дет" + 0.028*"лучш" + 0.024*"полиц" + 0.018*"московск" + 0.016*"хочет" + 0.016*"уб" + 0.015*"последн" + 0.013*"решен"
    INFO : topic #0 (0.050): 0.076*"rt" + 0.052*"сир" + 0.023*"рад" + 0.019*"главн" + 0.018*"переговор" + 0.015*"удар" + 0.013*"игр" + 0.012*"минск" + 0.010*"сша" + 0.010*"пьян"
    INFO : topic #3 (0.050): 0.063*"rt" + 0.051*"москв" + 0.038*"донецк" + 0.025*"район" + 0.023*"центр" + 0.014*"прошл" + 0.013*"мэр" + 0.013*"продаж" + 0.012*"машин" + 0.012*"силовик"
    INFO : topic #17 (0.050): 0.149*"украин" + 0.066*"российск" + 0.064*"новост" + 0.055*"rt" + 0.031*"мнен" + 0.022*"выбор" + 0.021*"газ" + 0.018*"побед" + 0.018*"донбасс" + 0.017*"арм"
    INFO : topic diff=0.009027, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #75 = documents up to #7600/9999, outstanding queue size 9
    INFO : PROGRESS: pass 8, dispatched chunk #76 = documents up to #7700/9999, outstanding queue size 9
    INFO : PROGRESS: pass 8, dispatched chunk #77 = documents up to #7800/9999, outstanding queue size 10
    INFO : PROGRESS: pass 8, dispatched chunk #78 = documents up to #7900/9999, outstanding queue size 11
    INFO : PROGRESS: pass 8, dispatched chunk #79 = documents up to #8000/9999, outstanding queue size 12
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #17 (0.050): 0.143*"украин" + 0.065*"российск" + 0.063*"новост" + 0.055*"rt" + 0.033*"мнен" + 0.020*"выбор" + 0.020*"газ" + 0.019*"побед" + 0.018*"донбасс" + 0.018*"арм"
    INFO : topic #4 (0.050): 0.034*"интересн" + 0.031*"rt" + 0.026*"европ" + 0.025*"появ" + 0.022*"закон" + 0.019*"турц" + 0.018*"трамп" + 0.016*"границ" + 0.015*"отказа" + 0.014*"очередн"
    INFO : topic #8 (0.050): 0.058*"rt" + 0.033*"рубл" + 0.026*"слов" + 0.022*"получ" + 0.020*"сторон" + 0.017*"переп" + 0.017*"миноборон" + 0.015*"запрет" + 0.015*"дтп" + 0.015*"министр"
    INFO : topic #15 (0.050): 0.047*"rt" + 0.035*"quot" + 0.028*"дорог" + 0.024*"сто" + 0.015*"памятник" + 0.015*"предлож" + 0.014*"выход" + 0.013*"факт" + 0.013*"действ" + 0.013*"сезон"
    INFO : topic #11 (0.050): 0.081*"rt" + 0.021*"истор" + 0.020*"хорош" + 0.018*"обам" + 0.016*"автомобил" + 0.015*"праздник" + 0.014*"связ" + 0.012*"числ" + 0.012*"депутат" + 0.012*"ставропол"
    INFO : topic diff=0.009233, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #80 = documents up to #8100/9999, outstanding queue size 10
    INFO : PROGRESS: pass 8, dispatched chunk #81 = documents up to #8200/9999, outstanding queue size 9
    INFO : PROGRESS: pass 8, dispatched chunk #82 = documents up to #8300/9999, outstanding queue size 10
    INFO : PROGRESS: pass 8, dispatched chunk #83 = documents up to #8400/9999, outstanding queue size 11
    INFO : PROGRESS: pass 8, dispatched chunk #84 = documents up to #8500/9999, outstanding queue size 12
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.082*"rt" + 0.064*"нов" + 0.032*"росс" + 0.027*"санкц" + 0.024*"сам" + 0.017*"иг" + 0.016*"рф" + 0.014*"мид" + 0.013*"сильн" + 0.012*"западн"
    INFO : topic #17 (0.050): 0.139*"украин" + 0.068*"российск" + 0.062*"новост" + 0.055*"rt" + 0.032*"мнен" + 0.021*"побед" + 0.021*"выбор" + 0.020*"арм" + 0.020*"газ" + 0.018*"донбасс"
    INFO : topic #1 (0.050): 0.052*"стран" + 0.044*"rt" + 0.027*"американск" + 0.022*"пострада" + 0.020*"дом" + 0.019*"жител" + 0.015*"конц" + 0.014*"ожида" + 0.014*"днем" + 0.014*"вид"
    INFO : topic #0 (0.050): 0.074*"rt" + 0.052*"сир" + 0.024*"рад" + 0.020*"главн" + 0.017*"переговор" + 0.017*"удар" + 0.013*"игр" + 0.012*"минск" + 0.012*"призва" + 0.011*"пьян"
    INFO : topic #8 (0.050): 0.059*"rt" + 0.034*"рубл" + 0.026*"слов" + 0.022*"получ" + 0.020*"сторон" + 0.017*"переп" + 0.016*"миноборон" + 0.015*"запрет" + 0.015*"министр" + 0.014*"мужчин"
    INFO : topic diff=0.009391, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #85 = documents up to #8600/9999, outstanding queue size 11
    INFO : PROGRESS: pass 8, dispatched chunk #86 = documents up to #8700/9999, outstanding queue size 11
    INFO : PROGRESS: pass 8, dispatched chunk #87 = documents up to #8800/9999, outstanding queue size 12
    INFO : PROGRESS: pass 8, dispatched chunk #88 = documents up to #8900/9999, outstanding queue size 13
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #0 (0.050): 0.075*"rt" + 0.051*"сир" + 0.023*"рад" + 0.020*"главн" + 0.016*"переговор" + 0.016*"удар" + 0.013*"увелич" + 0.012*"игр" + 0.012*"призва" + 0.011*"минск"
    INFO : topic #11 (0.050): 0.078*"rt" + 0.020*"хорош" + 0.019*"истор" + 0.019*"обам" + 0.018*"автомобил" + 0.014*"праздник" + 0.013*"поддержива" + 0.013*"депутат" + 0.012*"связ" + 0.012*"журналист"
    INFO : topic #19 (0.050): 0.062*"rt" + 0.046*"воен" + 0.044*"украинск" + 0.036*"украин" + 0.035*"ес" + 0.032*"крым" + 0.031*"фот" + 0.029*"политик" + 0.025*"сша" + 0.023*"чуж"
    INFO : topic #16 (0.050): 0.062*"rt" + 0.034*"нача" + 0.030*"чита" + 0.027*"пост" + 0.026*"добр" + 0.023*"мир" + 0.023*"как" + 0.019*"рассказа" + 0.019*"написа" + 0.017*"школ"
    INFO : topic #13 (0.050): 0.080*"rt" + 0.031*"сми" + 0.028*"украин" + 0.027*"готов" + 0.026*"перв" + 0.025*"дел" + 0.022*"власт" + 0.021*"русск" + 0.018*"заяв" + 0.017*"террорист"
    INFO : topic diff=0.009964, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #89 = documents up to #9000/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #9 (0.050): 0.057*"rt" + 0.049*"президент" + 0.048*"рф" + 0.018*"нат" + 0.017*"город" + 0.013*"говор" + 0.012*"эксперт" + 0.012*"росс" + 0.012*"обсуд" + 0.011*"евр"
    INFO : topic #4 (0.050): 0.035*"интересн" + 0.031*"rt" + 0.026*"появ" + 0.023*"европ" + 0.020*"закон" + 0.018*"границ" + 0.018*"турц" + 0.018*"трамп" + 0.014*"отказа" + 0.013*"поддержк"
    INFO : topic #6 (0.050): 0.127*"путин" + 0.055*"rt" + 0.025*"росс" + 0.024*"владимир" + 0.021*"назва" + 0.020*"работ" + 0.020*"улиц" + 0.018*"мест" + 0.017*"сборн" + 0.016*"друз"
    INFO : topic #7 (0.050): 0.072*"rt" + 0.020*"люб" + 0.020*"человек" + 0.018*"дня" + 0.018*"област" + 0.016*"пройдет" + 0.014*"суд" + 0.013*"фильм" + 0.013*"март" + 0.013*"лидер"
    INFO : topic #16 (0.050): 0.062*"rt" + 0.034*"нача" + 0.028*"добр" + 0.027*"чита" + 0.025*"пост" + 0.025*"как" + 0.023*"мир" + 0.018*"рассказа" + 0.017*"написа" + 0.017*"школ"
    INFO : topic diff=0.010858, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #90 = documents up to #9100/9999, outstanding queue size 6
    INFO : PROGRESS: pass 8, dispatched chunk #91 = documents up to #9200/9999, outstanding queue size 4
    INFO : PROGRESS: pass 8, dispatched chunk #92 = documents up to #9300/9999, outstanding queue size 5
    INFO : PROGRESS: pass 8, dispatched chunk #93 = documents up to #9400/9999, outstanding queue size 6
    INFO : PROGRESS: pass 8, dispatched chunk #94 = documents up to #9500/9999, outstanding queue size 7
    INFO : PROGRESS: pass 8, dispatched chunk #95 = documents up to #9600/9999, outstanding queue size 8
    INFO : PROGRESS: pass 8, dispatched chunk #96 = documents up to #9700/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.084*"rt" + 0.065*"нов" + 0.030*"росс" + 0.029*"санкц" + 0.025*"сам" + 0.016*"иг" + 0.015*"рф" + 0.014*"представ" + 0.013*"сша" + 0.013*"сильн"
    INFO : topic #8 (0.050): 0.060*"rt" + 0.036*"рубл" + 0.029*"слов" + 0.024*"получ" + 0.023*"сторон" + 0.017*"миноборон" + 0.016*"переп" + 0.016*"министр" + 0.013*"мужчин" + 0.013*"оборон"
    INFO : topic #9 (0.050): 0.058*"rt" + 0.053*"рф" + 0.049*"президент" + 0.017*"нат" + 0.016*"город" + 0.013*"говор" + 0.012*"млн" + 0.012*"росс" + 0.011*"эксперт" + 0.011*"обсуд"
    INFO : topic #6 (0.050): 0.127*"путин" + 0.056*"rt" + 0.026*"росс" + 0.025*"владимир" + 0.022*"работ" + 0.020*"назва" + 0.019*"улиц" + 0.016*"сборн" + 0.016*"мест" + 0.016*"друз"
    INFO : topic #2 (0.050): 0.054*"rt" + 0.039*"глав" + 0.037*"киев" + 0.019*"жизн" + 0.019*"пыта" + 0.018*"погибл" + 0.018*"взрыв" + 0.017*"результат" + 0.016*"южн" + 0.015*"сет"
    INFO : topic diff=0.009537, rho=0.095787
    INFO : PROGRESS: pass 8, dispatched chunk #97 = documents up to #9800/9999, outstanding queue size 9
    INFO : PROGRESS: pass 8, dispatched chunk #98 = documents up to #9900/9999, outstanding queue size 8
    INFO : PROGRESS: pass 8, dispatched chunk #99 = documents up to #9999/9999, outstanding queue size 8
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #12 (0.050): 0.188*"rt" + 0.018*"дума" + 0.018*"возможн" + 0.018*"сша" + 0.017*"люд" + 0.015*"счита" + 0.013*"убийств" + 0.013*"цен" + 0.012*"доллар" + 0.011*"кин"
    INFO : topic #19 (0.050): 0.065*"rt" + 0.045*"воен" + 0.044*"украинск" + 0.036*"ес" + 0.036*"украин" + 0.030*"политик" + 0.029*"фот" + 0.028*"крым" + 0.028*"сша" + 0.026*"чуж"
    INFO : topic #18 (0.050): 0.083*"rt" + 0.066*"нов" + 0.031*"росс" + 0.028*"санкц" + 0.024*"сам" + 0.016*"иг" + 0.015*"рф" + 0.015*"представ" + 0.015*"мид" + 0.014*"сша"
    INFO : topic #10 (0.050): 0.085*"rt" + 0.040*"виде" + 0.034*"дет" + 0.027*"лучш" + 0.023*"полиц" + 0.019*"московск" + 0.019*"хочет" + 0.015*"уб" + 0.015*"решен" + 0.014*"бизнес"
    INFO : topic #3 (0.050): 0.065*"rt" + 0.058*"москв" + 0.034*"донецк" + 0.025*"центр" + 0.024*"район" + 0.015*"прошл" + 0.015*"известн" + 0.013*"встреч" + 0.013*"банк" + 0.012*"продаж"
    INFO : topic diff=0.011258, rho=0.095787
    INFO : -15.967 per-word bound, 64055.5 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #1 (0.050): 0.063*"стран" + 0.045*"rt" + 0.024*"дом" + 0.022*"американск" + 0.020*"пострада" + 0.019*"жител" + 0.018*"стал" + 0.016*"конц" + 0.016*"массов" + 0.015*"сем"
    INFO : topic #9 (0.050): 0.058*"rt" + 0.053*"рф" + 0.050*"президент" + 0.017*"город" + 0.016*"нат" + 0.013*"говор" + 0.013*"уф" + 0.012*"немцов" + 0.012*"росс" + 0.012*"млн"
    INFO : topic #19 (0.050): 0.066*"rt" + 0.044*"украинск" + 0.043*"воен" + 0.037*"ес" + 0.034*"украин" + 0.033*"фот" + 0.030*"политик" + 0.027*"крым" + 0.027*"сша" + 0.026*"чуж"
    INFO : topic #8 (0.050): 0.063*"rt" + 0.040*"рубл" + 0.029*"слов" + 0.025*"получ" + 0.023*"сторон" + 0.015*"миноборон" + 0.015*"запрет" + 0.014*"министр" + 0.014*"переп" + 0.013*"мужчин"
    INFO : topic #12 (0.050): 0.189*"rt" + 0.020*"люд" + 0.017*"возможн" + 0.017*"сша" + 0.017*"дума" + 0.016*"счита" + 0.014*"убийств" + 0.012*"доллар" + 0.011*"цен" + 0.011*"кин"
    INFO : topic diff=0.010321, rho=0.095787
    INFO : -15.977 per-word bound, 64477.4 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : merging changes from 99 documents into a model of 9999 documents
    INFO : topic #12 (0.050): 0.190*"rt" + 0.018*"люд" + 0.018*"дума" + 0.018*"сша" + 0.016*"возможн" + 0.015*"счита" + 0.014*"сирийск" + 0.013*"убийств" + 0.012*"кин" + 0.011*"населен"
    INFO : topic #11 (0.050): 0.078*"rt" + 0.024*"обам" + 0.018*"хорош" + 0.017*"автомобил" + 0.016*"журналист" + 0.015*"мост" + 0.015*"ставропол" + 0.015*"истор" + 0.014*"числ" + 0.013*"депутат"
    INFO : topic #15 (0.050): 0.046*"rt" + 0.031*"quot" + 0.030*"дорог" + 0.027*"сто" + 0.016*"выход" + 0.016*"рук" + 0.015*"сезон" + 0.015*"факт" + 0.013*"постро" + 0.012*"предлож"
    INFO : topic #3 (0.050): 0.071*"rt" + 0.057*"москв" + 0.044*"донецк" + 0.023*"центр" + 0.021*"район" + 0.019*"прошл" + 0.014*"банк" + 0.013*"дан" + 0.012*"мирн" + 0.012*"област"
    INFO : topic #4 (0.050): 0.044*"интересн" + 0.030*"rt" + 0.024*"появ" + 0.023*"европ" + 0.022*"границ" + 0.017*"трамп" + 0.016*"турц" + 0.015*"закон" + 0.014*"очередн" + 0.014*"омск"
    INFO : topic diff=0.022102, rho=0.095787
    INFO : -15.678 per-word bound, 52425.0 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : PROGRESS: pass 9, dispatched chunk #0 = documents up to #100/9999, outstanding queue size 1
    INFO : PROGRESS: pass 9, dispatched chunk #1 = documents up to #200/9999, outstanding queue size 2
    INFO : PROGRESS: pass 9, dispatched chunk #2 = documents up to #300/9999, outstanding queue size 3
    INFO : PROGRESS: pass 9, dispatched chunk #3 = documents up to #400/9999, outstanding queue size 4
    INFO : PROGRESS: pass 9, dispatched chunk #4 = documents up to #500/9999, outstanding queue size 5
    INFO : PROGRESS: pass 9, dispatched chunk #5 = documents up to #600/9999, outstanding queue size 6
    INFO : PROGRESS: pass 9, dispatched chunk #6 = documents up to #700/9999, outstanding queue size 7
    INFO : PROGRESS: pass 9, dispatched chunk #7 = documents up to #800/9999, outstanding queue size 8
    INFO : PROGRESS: pass 9, dispatched chunk #8 = documents up to #900/9999, outstanding queue size 9
    INFO : PROGRESS: pass 9, dispatched chunk #9 = documents up to #1000/9999, outstanding queue size 10
    INFO : PROGRESS: pass 9, dispatched chunk #10 = documents up to #1100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 9, dispatched chunk #11 = documents up to #1200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 9, dispatched chunk #12 = documents up to #1300/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #18 (0.050): 0.077*"rt" + 0.069*"нов" + 0.030*"росс" + 0.027*"санкц" + 0.023*"сам" + 0.015*"сша" + 0.014*"мид" + 0.013*"апрел" + 0.013*"иг" + 0.013*"представ"
    INFO : topic #4 (0.050): 0.042*"интересн" + 0.031*"rt" + 0.023*"европ" + 0.022*"появ" + 0.019*"границ" + 0.016*"трамп" + 0.016*"турц" + 0.014*"закон" + 0.014*"событ" + 0.013*"очередн"
    INFO : topic #19 (0.050): 0.062*"rt" + 0.044*"ес" + 0.044*"воен" + 0.043*"украинск" + 0.038*"украин" + 0.033*"крым" + 0.032*"политик" + 0.031*"фот" + 0.027*"сша" + 0.024*"чуж"
    INFO : topic #2 (0.050): 0.052*"rt" + 0.045*"глав" + 0.033*"киев" + 0.026*"жизн" + 0.018*"взрыв" + 0.016*"кита" + 0.015*"погибл" + 0.015*"страшн" + 0.014*"пыта" + 0.014*"международн"
    INFO : topic #16 (0.050): 0.062*"rt" + 0.030*"нача" + 0.029*"добр" + 0.026*"мир" + 0.024*"чита" + 0.024*"пост" + 0.020*"как" + 0.019*"рассказа" + 0.019*"школ" + 0.018*"написа"
    INFO : topic diff=0.047088, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #13 = documents up to #1400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 9, dispatched chunk #14 = documents up to #1500/9999, outstanding queue size 8
    INFO : PROGRESS: pass 9, dispatched chunk #15 = documents up to #1600/9999, outstanding queue size 9
    INFO : PROGRESS: pass 9, dispatched chunk #16 = documents up to #1700/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #1 (0.050): 0.053*"стран" + 0.043*"rt" + 0.022*"дом" + 0.021*"американск" + 0.021*"жител" + 0.018*"пострада" + 0.018*"стал" + 0.016*"днем" + 0.016*"массов" + 0.015*"росс"
    INFO : topic #17 (0.050): 0.147*"украин" + 0.063*"новост" + 0.063*"российск" + 0.054*"rt" + 0.033*"мнен" + 0.021*"побед" + 0.021*"продолжа" + 0.020*"выбор" + 0.018*"росс" + 0.018*"газ"
    INFO : topic #5 (0.050): 0.062*"rt" + 0.028*"войн" + 0.022*"задержа" + 0.016*"план" + 0.016*"обстрел" + 0.016*"полицейск" + 0.015*"ситуац" + 0.014*"запад" + 0.013*"предлага" + 0.013*"стат"
    INFO : topic #19 (0.050): 0.058*"rt" + 0.043*"ес" + 0.042*"воен" + 0.040*"украинск" + 0.040*"украин" + 0.033*"политик" + 0.033*"крым" + 0.031*"фот" + 0.030*"сша" + 0.024*"чуж"
    INFO : topic #12 (0.050): 0.172*"rt" + 0.019*"дума" + 0.017*"сша" + 0.017*"люд" + 0.015*"возможн" + 0.013*"кин" + 0.012*"сирийск" + 0.012*"счита" + 0.011*"убийств" + 0.010*"боевик"
    INFO : topic diff=0.040351, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #17 = documents up to #1800/9999, outstanding queue size 10
    INFO : merging changes from 700 documents into a model of 9999 documents
    INFO : topic #15 (0.050): 0.047*"rt" + 0.031*"quot" + 0.025*"дорог" + 0.022*"сто" + 0.014*"постро" + 0.014*"факт" + 0.013*"сезон" + 0.012*"выход" + 0.012*"рук" + 0.012*"узна"
    INFO : topic #5 (0.050): 0.061*"rt" + 0.027*"войн" + 0.022*"задержа" + 0.016*"полицейск" + 0.016*"план" + 0.015*"обстрел" + 0.014*"ситуац" + 0.013*"запад" + 0.012*"украинц" + 0.012*"предлага"
    INFO : topic #0 (0.050): 0.079*"rt" + 0.044*"сир" + 0.020*"рад" + 0.016*"главн" + 0.016*"удар" + 0.012*"игр" + 0.012*"сша" + 0.012*"переговор" + 0.009*"минск" + 0.009*"пьян"
    INFO : topic #17 (0.050): 0.146*"украин" + 0.063*"российск" + 0.062*"новост" + 0.052*"rt" + 0.032*"мнен" + 0.021*"выбор" + 0.020*"продолжа" + 0.020*"побед" + 0.018*"газ" + 0.018*"росс"
    INFO : topic #11 (0.050): 0.076*"rt" + 0.021*"обам" + 0.016*"автомобил" + 0.015*"истор" + 0.015*"хорош" + 0.015*"журналист" + 0.013*"числ" + 0.013*"ставропол" + 0.012*"крушен" + 0.011*"мост"
    INFO : topic diff=0.019957, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #18 = documents up to #1900/9999, outstanding queue size 4
    INFO : PROGRESS: pass 9, dispatched chunk #19 = documents up to #2000/9999, outstanding queue size 2
    INFO : PROGRESS: pass 9, dispatched chunk #20 = documents up to #2100/9999, outstanding queue size 3
    INFO : PROGRESS: pass 9, dispatched chunk #21 = documents up to #2200/9999, outstanding queue size 4
    INFO : PROGRESS: pass 9, dispatched chunk #22 = documents up to #2300/9999, outstanding queue size 5
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #17 (0.050): 0.147*"украин" + 0.065*"российск" + 0.062*"новост" + 0.054*"rt" + 0.034*"мнен" + 0.021*"выбор" + 0.019*"продолжа" + 0.019*"побед" + 0.018*"газ" + 0.018*"росс"
    INFO : topic #9 (0.050): 0.055*"rt" + 0.045*"рф" + 0.045*"президент" + 0.016*"город" + 0.013*"нат" + 0.013*"обсуд" + 0.013*"млн" + 0.011*"говор" + 0.011*"войск" + 0.011*"росс"
    INFO : topic #12 (0.050): 0.169*"rt" + 0.020*"дума" + 0.017*"люд" + 0.016*"сша" + 0.016*"возможн" + 0.013*"счита" + 0.012*"кин" + 0.011*"сирийск" + 0.010*"убийств" + 0.010*"боевик"
    INFO : topic #14 (0.050): 0.075*"rt" + 0.031*"петербург" + 0.026*"вер" + 0.019*"медвед" + 0.015*"дмитр" + 0.014*"матч" + 0.012*"восток" + 0.011*"луганск" + 0.011*"пожар" + 0.011*"смерт"
    INFO : topic #16 (0.050): 0.059*"rt" + 0.029*"нача" + 0.026*"мир" + 0.025*"чита" + 0.023*"добр" + 0.022*"пост" + 0.020*"написа" + 0.019*"школ" + 0.018*"как" + 0.018*"рассказа"
    INFO : topic diff=0.012552, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #23 = documents up to #2400/9999, outstanding queue size 5
    INFO : PROGRESS: pass 9, dispatched chunk #24 = documents up to #2500/9999, outstanding queue size 4
    INFO : PROGRESS: pass 9, dispatched chunk #25 = documents up to #2600/9999, outstanding queue size 5
    INFO : PROGRESS: pass 9, dispatched chunk #26 = documents up to #2700/9999, outstanding queue size 6
    INFO : PROGRESS: pass 9, dispatched chunk #27 = documents up to #2800/9999, outstanding queue size 7
    INFO : PROGRESS: pass 9, dispatched chunk #28 = documents up to #2900/9999, outstanding queue size 8
    INFO : PROGRESS: pass 9, dispatched chunk #29 = documents up to #3000/9999, outstanding queue size 9
    INFO : PROGRESS: pass 9, dispatched chunk #30 = documents up to #3100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 9, dispatched chunk #31 = documents up to #3200/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #2 (0.050): 0.052*"rt" + 0.040*"глав" + 0.033*"киев" + 0.022*"жизн" + 0.018*"взрыв" + 0.017*"погибл" + 0.017*"пыта" + 0.016*"кита" + 0.015*"результат" + 0.014*"страшн"
    INFO : topic #8 (0.050): 0.056*"rt" + 0.037*"рубл" + 0.026*"слов" + 0.026*"сторон" + 0.020*"получ" + 0.017*"миноборон" + 0.015*"министр" + 0.014*"дтп" + 0.013*"запрет" + 0.013*"мужчин"
    INFO : topic #3 (0.050): 0.068*"rt" + 0.057*"москв" + 0.034*"донецк" + 0.022*"район" + 0.019*"центр" + 0.014*"встреч" + 0.013*"банк" + 0.012*"прошл" + 0.012*"дан" + 0.011*"известн"
    INFO : topic #19 (0.050): 0.058*"rt" + 0.043*"украинск" + 0.042*"воен" + 0.041*"ес" + 0.037*"крым" + 0.036*"украин" + 0.031*"фот" + 0.031*"политик" + 0.028*"сша" + 0.026*"чуж"
    INFO : topic #17 (0.050): 0.152*"украин" + 0.066*"российск" + 0.063*"новост" + 0.056*"rt" + 0.033*"мнен" + 0.020*"выбор" + 0.019*"побед" + 0.018*"продолжа" + 0.018*"росс" + 0.017*"газ"
    INFO : topic diff=0.009855, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #32 = documents up to #3300/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #9 (0.050): 0.055*"rt" + 0.048*"президент" + 0.047*"рф" + 0.015*"город" + 0.014*"нат" + 0.014*"росс" + 0.013*"млн" + 0.013*"обсуд" + 0.013*"говор" + 0.012*"участ"
    INFO : topic #6 (0.050): 0.118*"путин" + 0.055*"rt" + 0.037*"владимир" + 0.025*"росс" + 0.021*"назва" + 0.020*"улиц" + 0.019*"работ" + 0.015*"друз" + 0.015*"ход" + 0.014*"мест"
    INFO : topic #11 (0.050): 0.082*"rt" + 0.021*"обам" + 0.017*"хорош" + 0.017*"автомобил" + 0.016*"ставропол" + 0.016*"числ" + 0.015*"истор" + 0.013*"журналист" + 0.013*"депутат" + 0.012*"мост"
    INFO : topic #10 (0.050): 0.083*"rt" + 0.044*"виде" + 0.033*"дет" + 0.022*"полиц" + 0.022*"хочет" + 0.019*"лучш" + 0.019*"московск" + 0.015*"уб" + 0.014*"бизнес" + 0.013*"метр"
    INFO : topic #2 (0.050): 0.052*"rt" + 0.043*"глав" + 0.032*"киев" + 0.021*"жизн" + 0.018*"пыта" + 0.018*"погибл" + 0.017*"взрыв" + 0.017*"кита" + 0.015*"результат" + 0.014*"нашл"
    INFO : topic diff=0.010371, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #33 = documents up to #3400/9999, outstanding queue size 7
    INFO : PROGRESS: pass 9, dispatched chunk #34 = documents up to #3500/9999, outstanding queue size 7
    INFO : PROGRESS: pass 9, dispatched chunk #35 = documents up to #3600/9999, outstanding queue size 8
    INFO : PROGRESS: pass 9, dispatched chunk #36 = documents up to #3700/9999, outstanding queue size 9
    INFO : PROGRESS: pass 9, dispatched chunk #37 = documents up to #3800/9999, outstanding queue size 10
    INFO : PROGRESS: pass 9, dispatched chunk #38 = documents up to #3900/9999, outstanding queue size 11
    INFO : PROGRESS: pass 9, dispatched chunk #39 = documents up to #4000/9999, outstanding queue size 12
    INFO : PROGRESS: pass 9, dispatched chunk #40 = documents up to #4100/9999, outstanding queue size 13
    INFO : PROGRESS: pass 9, dispatched chunk #41 = documents up to #4200/9999, outstanding queue size 12
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #15 (0.050): 0.047*"rt" + 0.034*"quot" + 0.024*"сто" + 0.024*"дорог" + 0.016*"выход" + 0.016*"рук" + 0.014*"предлож" + 0.014*"факт" + 0.013*"сезон" + 0.012*"узна"
    INFO : topic #9 (0.050): 0.056*"rt" + 0.048*"рф" + 0.046*"президент" + 0.016*"город" + 0.014*"нат" + 0.013*"росс" + 0.013*"млн" + 0.013*"обсуд" + 0.013*"участ" + 0.012*"уф"
    INFO : topic #18 (0.050): 0.078*"rt" + 0.052*"нов" + 0.030*"росс" + 0.029*"санкц" + 0.022*"сам" + 0.016*"мид" + 0.015*"апрел" + 0.013*"рф" + 0.012*"иг" + 0.012*"сильн"
    INFO : topic #6 (0.050): 0.116*"путин" + 0.055*"rt" + 0.035*"владимир" + 0.024*"росс" + 0.022*"назва" + 0.019*"улиц" + 0.018*"работ" + 0.017*"друз" + 0.016*"мест" + 0.014*"ход"
    INFO : topic #4 (0.050): 0.035*"интересн" + 0.031*"rt" + 0.027*"появ" + 0.021*"европ" + 0.019*"турц" + 0.018*"границ" + 0.015*"отказа" + 0.014*"закон" + 0.014*"очередн" + 0.013*"трамп"
    INFO : topic diff=0.010359, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #42 = documents up to #4300/9999, outstanding queue size 10
    INFO : PROGRESS: pass 9, dispatched chunk #43 = documents up to #4400/9999, outstanding queue size 10
    INFO : PROGRESS: pass 9, dispatched chunk #44 = documents up to #4500/9999, outstanding queue size 11
    INFO : PROGRESS: pass 9, dispatched chunk #45 = documents up to #4600/9999, outstanding queue size 12
    INFO : PROGRESS: pass 9, dispatched chunk #46 = documents up to #4700/9999, outstanding queue size 13
    INFO : merging changes from 600 documents into a model of 9999 documents
    INFO : topic #4 (0.050): 0.034*"интересн" + 0.031*"rt" + 0.027*"появ" + 0.021*"европ" + 0.020*"турц" + 0.017*"границ" + 0.016*"отказа" + 0.016*"закон" + 0.013*"очередн" + 0.013*"развит"
    INFO : topic #0 (0.050): 0.082*"rt" + 0.044*"сир" + 0.021*"рад" + 0.017*"главн" + 0.016*"удар" + 0.015*"игр" + 0.013*"сша" + 0.011*"переговор" + 0.010*"минск" + 0.010*"увелич"
    INFO : topic #13 (0.050): 0.082*"rt" + 0.028*"готов" + 0.027*"сми" + 0.025*"власт" + 0.024*"дел" + 0.024*"перв" + 0.018*"прав" + 0.018*"украин" + 0.018*"русск" + 0.017*"заяв"
    INFO : topic #7 (0.050): 0.070*"rt" + 0.023*"человек" + 0.020*"област" + 0.015*"новосибирск" + 0.015*"дня" + 0.015*"лидер" + 0.015*"суд" + 0.015*"люб" + 0.014*"фильм" + 0.013*"пройдет"
    INFO : topic #5 (0.050): 0.066*"rt" + 0.026*"войн" + 0.020*"полицейск" + 0.019*"задержа" + 0.016*"ситуац" + 0.016*"украинц" + 0.015*"стат" + 0.015*"запад" + 0.014*"обстрел" + 0.014*"отправ"
    INFO : topic diff=0.009116, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #47 = documents up to #4800/9999, outstanding queue size 9
    INFO : PROGRESS: pass 9, dispatched chunk #48 = documents up to #4900/9999, outstanding queue size 7
    INFO : PROGRESS: pass 9, dispatched chunk #49 = documents up to #5000/9999, outstanding queue size 8
    INFO : PROGRESS: pass 9, dispatched chunk #50 = documents up to #5100/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #16 (0.050): 0.059*"rt" + 0.031*"чита" + 0.030*"нача" + 0.028*"мир" + 0.025*"пост" + 0.023*"рассказа" + 0.022*"добр" + 0.019*"школ" + 0.018*"написа" + 0.017*"как"
    INFO : topic #11 (0.050): 0.086*"rt" + 0.022*"обам" + 0.017*"хорош" + 0.016*"истор" + 0.016*"автомобил" + 0.015*"ставропол" + 0.014*"депутат" + 0.014*"числ" + 0.014*"связ" + 0.013*"крушен"
    INFO : topic #4 (0.050): 0.033*"интересн" + 0.031*"rt" + 0.029*"появ" + 0.021*"закон" + 0.020*"турц" + 0.019*"европ" + 0.017*"отказа" + 0.015*"границ" + 0.015*"трамп" + 0.014*"очередн"
    INFO : topic #0 (0.050): 0.080*"rt" + 0.045*"сир" + 0.023*"рад" + 0.017*"главн" + 0.017*"удар" + 0.014*"игр" + 0.013*"переговор" + 0.012*"сша" + 0.011*"увелич" + 0.009*"минск"
    INFO : topic #17 (0.050): 0.150*"украин" + 0.065*"российск" + 0.058*"новост" + 0.058*"rt" + 0.032*"мнен" + 0.022*"побед" + 0.019*"арм" + 0.019*"газ" + 0.019*"донбасс" + 0.019*"выбор"
    INFO : topic diff=0.013173, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #51 = documents up to #5200/9999, outstanding queue size 9
    INFO : PROGRESS: pass 9, dispatched chunk #52 = documents up to #5300/9999, outstanding queue size 8
    INFO : PROGRESS: pass 9, dispatched chunk #53 = documents up to #5400/9999, outstanding queue size 8
    INFO : PROGRESS: pass 9, dispatched chunk #54 = documents up to #5500/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #15 (0.050): 0.047*"rt" + 0.033*"quot" + 0.027*"дорог" + 0.021*"сто" + 0.016*"рук" + 0.014*"предлож" + 0.014*"выход" + 0.013*"факт" + 0.013*"солдат" + 0.012*"сезон"
    INFO : topic #0 (0.050): 0.081*"rt" + 0.047*"сир" + 0.023*"рад" + 0.018*"главн" + 0.017*"переговор" + 0.015*"удар" + 0.013*"игр" + 0.012*"минск" + 0.011*"сша" + 0.011*"увелич"
    INFO : topic #2 (0.050): 0.053*"rt" + 0.042*"глав" + 0.033*"киев" + 0.022*"жизн" + 0.020*"кита" + 0.018*"погибл" + 0.017*"взрыв" + 0.016*"нашл" + 0.015*"пыта" + 0.014*"результат"
    INFO : topic #11 (0.050): 0.086*"rt" + 0.022*"обам" + 0.019*"истор" + 0.018*"хорош" + 0.015*"депутат" + 0.015*"числ" + 0.015*"автомобил" + 0.013*"ставропол" + 0.013*"связ" + 0.012*"крушен"
    INFO : topic #19 (0.050): 0.063*"rt" + 0.043*"воен" + 0.042*"украинск" + 0.041*"крым" + 0.038*"украин" + 0.033*"фот" + 0.033*"ес" + 0.030*"сша" + 0.029*"политик" + 0.023*"чуж"
    INFO : topic diff=0.010707, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #55 = documents up to #5600/9999, outstanding queue size 9
    INFO : PROGRESS: pass 9, dispatched chunk #56 = documents up to #5700/9999, outstanding queue size 8
    INFO : PROGRESS: pass 9, dispatched chunk #57 = documents up to #5800/9999, outstanding queue size 9
    INFO : PROGRESS: pass 9, dispatched chunk #58 = documents up to #5900/9999, outstanding queue size 10
    INFO : PROGRESS: pass 9, dispatched chunk #59 = documents up to #6000/9999, outstanding queue size 11
    INFO : PROGRESS: pass 9, dispatched chunk #60 = documents up to #6100/9999, outstanding queue size 11
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #14 (0.050): 0.081*"rt" + 0.036*"петербург" + 0.033*"вер" + 0.017*"медвед" + 0.014*"матч" + 0.014*"восток" + 0.014*"ки" + 0.014*"дмитр" + 0.014*"пожар" + 0.012*"смерт"
    INFO : topic #3 (0.050): 0.069*"rt" + 0.059*"москв" + 0.036*"донецк" + 0.024*"район" + 0.021*"центр" + 0.015*"прошл" + 0.013*"машин" + 0.013*"продаж" + 0.013*"ран" + 0.012*"мэр"
    INFO : topic #7 (0.050): 0.074*"rt" + 0.025*"человек" + 0.020*"област" + 0.015*"дня" + 0.015*"новосибирск" + 0.014*"люб" + 0.014*"фильм" + 0.014*"лидер" + 0.014*"март" + 0.014*"суд"
    INFO : topic #15 (0.050): 0.048*"rt" + 0.031*"quot" + 0.028*"дорог" + 0.023*"сто" + 0.016*"предлож" + 0.016*"рук" + 0.014*"выход" + 0.013*"действ" + 0.013*"солдат" + 0.012*"факт"
    INFO : topic #9 (0.050): 0.057*"rt" + 0.051*"рф" + 0.049*"президент" + 0.015*"город" + 0.014*"росс" + 0.014*"нат" + 0.013*"млн" + 0.013*"эксперт" + 0.012*"участ" + 0.012*"сентябр"
    INFO : topic diff=0.010338, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #61 = documents up to #6200/9999, outstanding queue size 10
    INFO : PROGRESS: pass 9, dispatched chunk #62 = documents up to #6300/9999, outstanding queue size 8
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #17 (0.050): 0.159*"украин" + 0.065*"российск" + 0.059*"новост" + 0.057*"rt" + 0.029*"мнен" + 0.021*"побед" + 0.020*"выбор" + 0.019*"газ" + 0.019*"донбасс" + 0.018*"росс"
    INFO : topic #14 (0.050): 0.080*"rt" + 0.036*"петербург" + 0.032*"вер" + 0.018*"медвед" + 0.015*"дмитр" + 0.015*"восток" + 0.014*"матч" + 0.013*"ки" + 0.012*"пожар" + 0.011*"смерт"
    INFO : topic #5 (0.050): 0.063*"rt" + 0.023*"войн" + 0.020*"задержа" + 0.017*"полицейск" + 0.016*"план" + 0.016*"обстрел" + 0.015*"сотрудник" + 0.014*"россиян" + 0.014*"смотр" + 0.014*"предлага"
    INFO : topic #4 (0.050): 0.035*"интересн" + 0.030*"rt" + 0.026*"появ" + 0.023*"закон" + 0.020*"европ" + 0.019*"границ" + 0.019*"турц" + 0.017*"отказа" + 0.013*"трамп" + 0.013*"омск"
    INFO : topic #11 (0.050): 0.083*"rt" + 0.022*"обам" + 0.018*"истор" + 0.018*"хорош" + 0.015*"депутат" + 0.014*"ставропол" + 0.014*"автомобил" + 0.013*"числ" + 0.013*"мост" + 0.013*"поддержива"
    INFO : topic diff=0.011246, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #63 = documents up to #6400/9999, outstanding queue size 7
    INFO : PROGRESS: pass 9, dispatched chunk #64 = documents up to #6500/9999, outstanding queue size 7
    INFO : PROGRESS: pass 9, dispatched chunk #65 = documents up to #6600/9999, outstanding queue size 8
    INFO : PROGRESS: pass 9, dispatched chunk #66 = documents up to #6700/9999, outstanding queue size 9
    INFO : PROGRESS: pass 9, dispatched chunk #67 = documents up to #6800/9999, outstanding queue size 10
    INFO : merging changes from 800 documents into a model of 9999 documents
    INFO : topic #7 (0.050): 0.073*"rt" + 0.026*"человек" + 0.020*"област" + 0.017*"люб" + 0.015*"дня" + 0.015*"фильм" + 0.014*"суд" + 0.013*"новосибирск" + 0.013*"март" + 0.013*"лидер"
    INFO : topic #19 (0.050): 0.061*"rt" + 0.044*"украинск" + 0.044*"воен" + 0.039*"украин" + 0.039*"крым" + 0.032*"ес" + 0.030*"фот" + 0.029*"политик" + 0.028*"сша" + 0.024*"днр"
    INFO : topic #6 (0.050): 0.127*"путин" + 0.056*"rt" + 0.029*"владимир" + 0.024*"росс" + 0.022*"назва" + 0.021*"мест" + 0.019*"работ" + 0.018*"улиц" + 0.015*"друз" + 0.013*"сборн"
    INFO : topic #15 (0.050): 0.049*"rt" + 0.034*"quot" + 0.030*"дорог" + 0.023*"сто" + 0.016*"предлож" + 0.015*"рук" + 0.013*"выход" + 0.013*"памятник" + 0.013*"действ" + 0.012*"сезон"
    INFO : topic #9 (0.050): 0.056*"rt" + 0.051*"рф" + 0.049*"президент" + 0.016*"нат" + 0.014*"город" + 0.014*"росс" + 0.013*"эксперт" + 0.013*"войск" + 0.013*"сентябр" + 0.013*"млн"
    INFO : topic diff=0.009281, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #68 = documents up to #6900/9999, outstanding queue size 4
    INFO : PROGRESS: pass 9, dispatched chunk #69 = documents up to #7000/9999, outstanding queue size 3
    INFO : PROGRESS: pass 9, dispatched chunk #70 = documents up to #7100/9999, outstanding queue size 4
    INFO : PROGRESS: pass 9, dispatched chunk #71 = documents up to #7200/9999, outstanding queue size 5
    INFO : PROGRESS: pass 9, dispatched chunk #72 = documents up to #7300/9999, outstanding queue size 6
    INFO : PROGRESS: pass 9, dispatched chunk #73 = documents up to #7400/9999, outstanding queue size 7
    INFO : PROGRESS: pass 9, dispatched chunk #74 = documents up to #7500/9999, outstanding queue size 8
    INFO : PROGRESS: pass 9, dispatched chunk #75 = documents up to #7600/9999, outstanding queue size 9
    INFO : PROGRESS: pass 9, dispatched chunk #76 = documents up to #7700/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #14 (0.050): 0.078*"rt" + 0.034*"петербург" + 0.033*"вер" + 0.020*"медвед" + 0.015*"матч" + 0.015*"дмитр" + 0.014*"восток" + 0.014*"ки" + 0.013*"пожар" + 0.011*"губернатор"
    INFO : topic #11 (0.050): 0.081*"rt" + 0.020*"хорош" + 0.019*"обам" + 0.019*"истор" + 0.015*"автомобил" + 0.014*"депутат" + 0.014*"связ" + 0.013*"числ" + 0.013*"мост" + 0.013*"ставропол"
    INFO : topic #5 (0.050): 0.062*"rt" + 0.024*"войн" + 0.022*"задержа" + 0.017*"полицейск" + 0.017*"план" + 0.015*"обстрел" + 0.015*"смотр" + 0.014*"предлага" + 0.014*"сотрудник" + 0.014*"россиян"
    INFO : topic #1 (0.050): 0.055*"стран" + 0.043*"rt" + 0.028*"американск" + 0.023*"пострада" + 0.022*"дом" + 0.021*"жител" + 0.015*"днем" + 0.014*"ожида" + 0.014*"росс" + 0.013*"стал"
    INFO : topic #13 (0.050): 0.081*"rt" + 0.031*"сми" + 0.029*"перв" + 0.029*"готов" + 0.025*"дел" + 0.024*"власт" + 0.019*"русск" + 0.017*"прав" + 0.017*"украин" + 0.017*"заяв"
    INFO : topic diff=0.013527, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #77 = documents up to #7800/9999, outstanding queue size 9
    INFO : PROGRESS: pass 9, dispatched chunk #78 = documents up to #7900/9999, outstanding queue size 7
    INFO : PROGRESS: pass 9, dispatched chunk #79 = documents up to #8000/9999, outstanding queue size 8
    INFO : PROGRESS: pass 9, dispatched chunk #80 = documents up to #8100/9999, outstanding queue size 9
    INFO : PROGRESS: pass 9, dispatched chunk #81 = documents up to #8200/9999, outstanding queue size 10
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #7 (0.050): 0.072*"rt" + 0.024*"человек" + 0.019*"люб" + 0.019*"дня" + 0.017*"област" + 0.016*"пройдет" + 0.016*"фильм" + 0.015*"суд" + 0.012*"март" + 0.012*"лидер"
    INFO : topic #19 (0.050): 0.064*"rt" + 0.045*"воен" + 0.044*"украинск" + 0.037*"украин" + 0.036*"крым" + 0.035*"ес" + 0.032*"политик" + 0.031*"фот" + 0.028*"сша" + 0.024*"чуж"
    INFO : topic #2 (0.050): 0.056*"rt" + 0.041*"глав" + 0.039*"киев" + 0.021*"жизн" + 0.018*"взрыв" + 0.017*"пыта" + 0.015*"международн" + 0.015*"погибл" + 0.015*"кита" + 0.014*"южн"
    INFO : topic #16 (0.050): 0.064*"rt" + 0.035*"нача" + 0.030*"чита" + 0.026*"мир" + 0.025*"добр" + 0.024*"как" + 0.023*"пост" + 0.019*"рассказа" + 0.018*"школ" + 0.018*"написа"
    INFO : topic #10 (0.050): 0.089*"rt" + 0.040*"виде" + 0.032*"дет" + 0.028*"лучш" + 0.024*"полиц" + 0.019*"московск" + 0.016*"уб" + 0.016*"хочет" + 0.015*"бизнес" + 0.014*"последн"
    INFO : topic diff=0.008534, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #82 = documents up to #8300/9999, outstanding queue size 10
    INFO : PROGRESS: pass 9, dispatched chunk #83 = documents up to #8400/9999, outstanding queue size 9
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #17 (0.050): 0.153*"украин" + 0.068*"российск" + 0.060*"новост" + 0.056*"rt" + 0.031*"мнен" + 0.022*"побед" + 0.020*"арм" + 0.019*"газ" + 0.019*"выбор" + 0.018*"росс"
    INFO : topic #0 (0.050): 0.078*"rt" + 0.050*"сир" + 0.024*"рад" + 0.019*"главн" + 0.017*"удар" + 0.015*"переговор" + 0.013*"игр" + 0.012*"увелич" + 0.011*"минск" + 0.011*"призва"
    INFO : topic #13 (0.050): 0.081*"rt" + 0.030*"сми" + 0.027*"готов" + 0.027*"перв" + 0.027*"дел" + 0.023*"власт" + 0.021*"русск" + 0.018*"заяв" + 0.017*"террорист" + 0.016*"украин"
    INFO : topic #19 (0.050): 0.064*"rt" + 0.046*"воен" + 0.045*"украинск" + 0.038*"украин" + 0.036*"ес" + 0.035*"крым" + 0.033*"фот" + 0.030*"политик" + 0.027*"сша" + 0.024*"чуж"
    INFO : topic #2 (0.050): 0.055*"rt" + 0.039*"глав" + 0.037*"киев" + 0.020*"жизн" + 0.020*"пыта" + 0.017*"взрыв" + 0.016*"результат" + 0.015*"погибл" + 0.015*"южн" + 0.015*"международн"
    INFO : topic diff=0.010035, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #84 = documents up to #8500/9999, outstanding queue size 8
    INFO : PROGRESS: pass 9, dispatched chunk #85 = documents up to #8600/9999, outstanding queue size 8
    INFO : PROGRESS: pass 9, dispatched chunk #86 = documents up to #8700/9999, outstanding queue size 9
    INFO : PROGRESS: pass 9, dispatched chunk #87 = documents up to #8800/9999, outstanding queue size 10
    INFO : PROGRESS: pass 9, dispatched chunk #88 = documents up to #8900/9999, outstanding queue size 11
    INFO : PROGRESS: pass 9, dispatched chunk #89 = documents up to #9000/9999, outstanding queue size 12
    INFO : PROGRESS: pass 9, dispatched chunk #90 = documents up to #9100/9999, outstanding queue size 13
    INFO : merging changes from 400 documents into a model of 9999 documents
    INFO : topic #15 (0.050): 0.046*"rt" + 0.035*"quot" + 0.030*"дорог" + 0.024*"сто" + 0.014*"октябр" + 0.014*"выход" + 0.013*"рук" + 0.013*"предлож" + 0.013*"узна" + 0.012*"действ"
    INFO : topic #13 (0.050): 0.083*"rt" + 0.030*"сми" + 0.028*"готов" + 0.025*"перв" + 0.025*"дел" + 0.023*"власт" + 0.022*"русск" + 0.018*"заяв" + 0.017*"террорист" + 0.015*"украин"
    INFO : topic #4 (0.050): 0.038*"интересн" + 0.031*"rt" + 0.027*"появ" + 0.022*"европ" + 0.020*"закон" + 0.017*"турц" + 0.016*"границ" + 0.016*"трамп" + 0.015*"отказа" + 0.013*"омск"
    INFO : topic #18 (0.050): 0.080*"rt" + 0.063*"нов" + 0.031*"росс" + 0.030*"санкц" + 0.025*"сам" + 0.015*"иг" + 0.015*"рф" + 0.014*"апрел" + 0.014*"мид" + 0.013*"мчс"
    INFO : topic #9 (0.050): 0.058*"rt" + 0.050*"рф" + 0.047*"президент" + 0.019*"город" + 0.017*"нат" + 0.013*"эксперт" + 0.013*"говор" + 0.012*"евр" + 0.012*"обсуд" + 0.011*"росс"
    INFO : topic diff=0.011085, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #91 = documents up to #9200/9999, outstanding queue size 11
    INFO : PROGRESS: pass 9, dispatched chunk #92 = documents up to #9300/9999, outstanding queue size 10
    INFO : PROGRESS: pass 9, dispatched chunk #93 = documents up to #9400/9999, outstanding queue size 11
    INFO : PROGRESS: pass 9, dispatched chunk #94 = documents up to #9500/9999, outstanding queue size 12
    INFO : PROGRESS: pass 9, dispatched chunk #95 = documents up to #9600/9999, outstanding queue size 13
    INFO : merging changes from 700 documents into a model of 9999 documents
    INFO : topic #3 (0.050): 0.067*"rt" + 0.059*"москв" + 0.037*"донецк" + 0.026*"район" + 0.023*"центр" + 0.014*"встреч" + 0.014*"продаж" + 0.014*"известн" + 0.013*"банк" + 0.012*"мирн"
    INFO : topic #7 (0.050): 0.072*"rt" + 0.020*"человек" + 0.018*"люб" + 0.018*"дня" + 0.017*"пройдет" + 0.017*"област" + 0.015*"фильм" + 0.014*"суд" + 0.014*"лидер" + 0.013*"март"
    INFO : topic #6 (0.050): 0.126*"путин" + 0.057*"rt" + 0.028*"владимир" + 0.023*"росс" + 0.020*"работ" + 0.019*"назва" + 0.019*"улиц" + 0.018*"сборн" + 0.016*"мест" + 0.016*"друз"
    INFO : topic #8 (0.050): 0.061*"rt" + 0.036*"рубл" + 0.028*"слов" + 0.024*"сторон" + 0.022*"получ" + 0.017*"миноборон" + 0.015*"министр" + 0.015*"переп" + 0.014*"росс" + 0.013*"дтп"
    INFO : topic #4 (0.050): 0.037*"интересн" + 0.030*"rt" + 0.027*"появ" + 0.021*"европ" + 0.019*"закон" + 0.017*"границ" + 0.017*"турц" + 0.016*"трамп" + 0.014*"отказа" + 0.013*"омск"
    INFO : topic diff=0.007288, rho=0.095351
    INFO : PROGRESS: pass 9, dispatched chunk #96 = documents up to #9700/9999, outstanding queue size 9
    INFO : PROGRESS: pass 9, dispatched chunk #97 = documents up to #9800/9999, outstanding queue size 9
    INFO : PROGRESS: pass 9, dispatched chunk #98 = documents up to #9900/9999, outstanding queue size 10
    INFO : PROGRESS: pass 9, dispatched chunk #99 = documents up to #9999/9999, outstanding queue size 11
    INFO : merging changes from 500 documents into a model of 9999 documents
    INFO : topic #14 (0.050): 0.076*"rt" + 0.033*"петербург" + 0.026*"вер" + 0.022*"медвед" + 0.016*"матч" + 0.016*"восток" + 0.016*"дмитр" + 0.015*"пожар" + 0.014*"ки" + 0.012*"сообщ"
    INFO : topic #19 (0.050): 0.063*"rt" + 0.046*"воен" + 0.043*"украинск" + 0.038*"ес" + 0.037*"украин" + 0.031*"фот" + 0.030*"политик" + 0.030*"крым" + 0.030*"сша" + 0.026*"чуж"
    INFO : topic #8 (0.050): 0.061*"rt" + 0.038*"рубл" + 0.027*"слов" + 0.024*"получ" + 0.024*"сторон" + 0.015*"миноборон" + 0.015*"переп" + 0.015*"запрет" + 0.014*"министр" + 0.013*"росс"
    INFO : topic #6 (0.050): 0.129*"путин" + 0.058*"rt" + 0.030*"владимир" + 0.024*"росс" + 0.023*"назва" + 0.020*"работ" + 0.020*"улиц" + 0.018*"сборн" + 0.016*"мест" + 0.015*"друз"
    INFO : topic #3 (0.050): 0.068*"rt" + 0.060*"москв" + 0.036*"донецк" + 0.025*"центр" + 0.023*"район" + 0.014*"встреч" + 0.014*"известн" + 0.013*"прошл" + 0.013*"банк" + 0.013*"продаж"
    INFO : topic diff=0.012323, rho=0.095351
    INFO : -15.903 per-word bound, 61266.6 perplexity estimate based on a held-out corpus of 99 documents with 661 words
    INFO : merging changes from 699 documents into a model of 9999 documents
    INFO : topic #9 (0.050): 0.059*"rt" + 0.052*"рф" + 0.049*"президент" + 0.018*"город" + 0.016*"нат" + 0.013*"уф" + 0.013*"говор" + 0.011*"эксперт" + 0.011*"участ" + 0.011*"росс"
    INFO : topic #11 (0.050): 0.077*"rt" + 0.021*"обам" + 0.020*"хорош" + 0.018*"автомобил" + 0.016*"истор" + 0.015*"депутат" + 0.015*"ставропол" + 0.014*"журналист" + 0.012*"числ" + 0.012*"час"
    INFO : topic #18 (0.050): 0.080*"rt" + 0.064*"нов" + 0.030*"росс" + 0.030*"санкц" + 0.024*"сам" + 0.016*"мид" + 0.015*"рф" + 0.015*"апрел" + 0.014*"иг" + 0.014*"представ"
    INFO : topic #19 (0.050): 0.065*"rt" + 0.045*"воен" + 0.043*"украинск" + 0.039*"ес" + 0.036*"украин" + 0.033*"фот" + 0.030*"политик" + 0.030*"крым" + 0.029*"сша" + 0.026*"чуж"
    INFO : topic #15 (0.050): 0.045*"rt" + 0.033*"quot" + 0.030*"дорог" + 0.025*"сто" + 0.015*"факт" + 0.014*"выход" + 0.013*"рук" + 0.013*"заверш" + 0.013*"действ" + 0.012*"октябр"
    INFO : topic diff=0.009347, rho=0.095351
    INFO : -15.858 per-word bound, 59382.6 perplexity estimate based on a held-out corpus of 99 documents with 661 words


    Current Time = 10:54:54



```python
# Print the Keyword in the 20 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
```

    INFO : topic #0 (0.050): 0.081*"rt" + 0.049*"сир" + 0.025*"рад" + 0.022*"главн" + 0.015*"удар" + 0.014*"переговор" + 0.013*"игр" + 0.012*"минск" + 0.012*"сша" + 0.011*"увелич"
    INFO : topic #1 (0.050): 0.064*"стран" + 0.047*"rt" + 0.024*"дом" + 0.022*"американск" + 0.021*"жител" + 0.019*"пострада" + 0.018*"стал" + 0.016*"массов" + 0.015*"росс" + 0.014*"конц"
    INFO : topic #2 (0.050): 0.054*"rt" + 0.044*"глав" + 0.034*"киев" + 0.023*"жизн" + 0.018*"пыта" + 0.017*"погибл" + 0.017*"взрыв" + 0.016*"южн" + 0.016*"международн" + 0.015*"кита"
    INFO : topic #3 (0.050): 0.070*"rt" + 0.060*"москв" + 0.037*"донецк" + 0.023*"центр" + 0.023*"район" + 0.016*"прошл" + 0.013*"встреч" + 0.013*"банк" + 0.012*"известн" + 0.012*"мирн"
    INFO : topic #4 (0.050): 0.037*"интересн" + 0.030*"rt" + 0.027*"появ" + 0.023*"европ" + 0.019*"границ" + 0.018*"трамп" + 0.017*"турц" + 0.017*"закон" + 0.015*"очередн" + 0.013*"отказа"
    INFO : topic #5 (0.050): 0.062*"rt" + 0.031*"войн" + 0.021*"задержа" + 0.019*"полицейск" + 0.017*"план" + 0.017*"ситуац" + 0.016*"обстрел" + 0.016*"запад" + 0.015*"предлага" + 0.015*"смотр"
    INFO : topic #6 (0.050): 0.127*"путин" + 0.058*"rt" + 0.032*"владимир" + 0.024*"росс" + 0.024*"назва" + 0.022*"работ" + 0.021*"улиц" + 0.018*"сборн" + 0.016*"мест" + 0.014*"друз"
    INFO : topic #7 (0.050): 0.071*"rt" + 0.019*"област" + 0.019*"человек" + 0.018*"люб" + 0.018*"дня" + 0.016*"фильм" + 0.016*"пройдет" + 0.015*"суд" + 0.014*"лидер" + 0.013*"отношен"
    INFO : topic #8 (0.050): 0.062*"rt" + 0.039*"рубл" + 0.027*"слов" + 0.025*"сторон" + 0.025*"получ" + 0.015*"запрет" + 0.015*"миноборон" + 0.014*"министр" + 0.014*"переп" + 0.014*"росс"
    INFO : topic #9 (0.050): 0.059*"rt" + 0.052*"рф" + 0.049*"президент" + 0.018*"город" + 0.016*"нат" + 0.013*"уф" + 0.013*"говор" + 0.011*"эксперт" + 0.011*"участ" + 0.011*"росс"
    INFO : topic #10 (0.050): 0.086*"rt" + 0.040*"виде" + 0.036*"дет" + 0.026*"лучш" + 0.022*"полиц" + 0.020*"московск" + 0.019*"хочет" + 0.015*"бизнес" + 0.015*"решен" + 0.014*"уб"
    INFO : topic #11 (0.050): 0.077*"rt" + 0.021*"обам" + 0.020*"хорош" + 0.018*"автомобил" + 0.016*"истор" + 0.015*"депутат" + 0.015*"ставропол" + 0.014*"журналист" + 0.012*"числ" + 0.012*"час"
    INFO : topic #12 (0.050): 0.189*"rt" + 0.019*"люд" + 0.018*"дума" + 0.017*"возможн" + 0.016*"сша" + 0.014*"счита" + 0.013*"убийств" + 0.012*"сирийск" + 0.011*"кин" + 0.011*"цен"
    INFO : topic #13 (0.050): 0.085*"rt" + 0.031*"сми" + 0.028*"перв" + 0.027*"готов" + 0.025*"дел" + 0.021*"власт" + 0.019*"русск" + 0.019*"заяв" + 0.016*"прав" + 0.015*"террорист"
    INFO : topic #14 (0.050): 0.076*"rt" + 0.033*"петербург" + 0.027*"вер" + 0.022*"медвед" + 0.017*"дмитр" + 0.015*"матч" + 0.015*"восток" + 0.014*"пожар" + 0.014*"ки" + 0.012*"сообщ"
    INFO : topic #15 (0.050): 0.045*"rt" + 0.033*"quot" + 0.030*"дорог" + 0.025*"сто" + 0.015*"факт" + 0.014*"выход" + 0.013*"рук" + 0.013*"заверш" + 0.013*"действ" + 0.012*"октябр"
    INFO : topic #16 (0.050): 0.060*"rt" + 0.032*"нача" + 0.027*"пост" + 0.027*"добр" + 0.025*"мир" + 0.025*"чита" + 0.024*"как" + 0.020*"рассказа" + 0.019*"написа" + 0.018*"школ"
    INFO : topic #17 (0.050): 0.154*"украин" + 0.069*"российск" + 0.059*"rt" + 0.056*"новост" + 0.031*"мнен" + 0.022*"побед" + 0.022*"выбор" + 0.019*"арм" + 0.019*"росс" + 0.017*"продолжа"
    INFO : topic #18 (0.050): 0.080*"rt" + 0.064*"нов" + 0.030*"росс" + 0.030*"санкц" + 0.024*"сам" + 0.016*"мид" + 0.015*"рф" + 0.015*"апрел" + 0.014*"иг" + 0.014*"представ"
    INFO : topic #19 (0.050): 0.065*"rt" + 0.045*"воен" + 0.043*"украинск" + 0.039*"ес" + 0.036*"украин" + 0.033*"фот" + 0.030*"политик" + 0.030*"крым" + 0.029*"сша" + 0.026*"чуж"


    [(0,
      '0.081*"rt" + 0.049*"сир" + 0.025*"рад" + 0.022*"главн" + 0.015*"удар" + '
      '0.014*"переговор" + 0.013*"игр" + 0.012*"минск" + 0.012*"сша" + '
      '0.011*"увелич"'),
     (1,
      '0.064*"стран" + 0.047*"rt" + 0.024*"дом" + 0.022*"американск" + '
      '0.021*"жител" + 0.019*"пострада" + 0.018*"стал" + 0.016*"массов" + '
      '0.015*"росс" + 0.014*"конц"'),
     (2,
      '0.054*"rt" + 0.044*"глав" + 0.034*"киев" + 0.023*"жизн" + 0.018*"пыта" + '
      '0.017*"погибл" + 0.017*"взрыв" + 0.016*"южн" + 0.016*"международн" + '
      '0.015*"кита"'),
     (3,
      '0.070*"rt" + 0.060*"москв" + 0.037*"донецк" + 0.023*"центр" + 0.023*"район" '
      '+ 0.016*"прошл" + 0.013*"встреч" + 0.013*"банк" + 0.012*"известн" + '
      '0.012*"мирн"'),
     (4,
      '0.037*"интересн" + 0.030*"rt" + 0.027*"появ" + 0.023*"европ" + '
      '0.019*"границ" + 0.018*"трамп" + 0.017*"турц" + 0.017*"закон" + '
      '0.015*"очередн" + 0.013*"отказа"'),
     (5,
      '0.062*"rt" + 0.031*"войн" + 0.021*"задержа" + 0.019*"полицейск" + '
      '0.017*"план" + 0.017*"ситуац" + 0.016*"обстрел" + 0.016*"запад" + '
      '0.015*"предлага" + 0.015*"смотр"'),
     (6,
      '0.127*"путин" + 0.058*"rt" + 0.032*"владимир" + 0.024*"росс" + '
      '0.024*"назва" + 0.022*"работ" + 0.021*"улиц" + 0.018*"сборн" + 0.016*"мест" '
      '+ 0.014*"друз"'),
     (7,
      '0.071*"rt" + 0.019*"област" + 0.019*"человек" + 0.018*"люб" + 0.018*"дня" + '
      '0.016*"фильм" + 0.016*"пройдет" + 0.015*"суд" + 0.014*"лидер" + '
      '0.013*"отношен"'),
     (8,
      '0.062*"rt" + 0.039*"рубл" + 0.027*"слов" + 0.025*"сторон" + 0.025*"получ" + '
      '0.015*"запрет" + 0.015*"миноборон" + 0.014*"министр" + 0.014*"переп" + '
      '0.014*"росс"'),
     (9,
      '0.059*"rt" + 0.052*"рф" + 0.049*"президент" + 0.018*"город" + 0.016*"нат" + '
      '0.013*"уф" + 0.013*"говор" + 0.011*"эксперт" + 0.011*"участ" + '
      '0.011*"росс"'),
     (10,
      '0.086*"rt" + 0.040*"виде" + 0.036*"дет" + 0.026*"лучш" + 0.022*"полиц" + '
      '0.020*"московск" + 0.019*"хочет" + 0.015*"бизнес" + 0.015*"решен" + '
      '0.014*"уб"'),
     (11,
      '0.077*"rt" + 0.021*"обам" + 0.020*"хорош" + 0.018*"автомобил" + '
      '0.016*"истор" + 0.015*"депутат" + 0.015*"ставропол" + 0.014*"журналист" + '
      '0.012*"числ" + 0.012*"час"'),
     (12,
      '0.189*"rt" + 0.019*"люд" + 0.018*"дума" + 0.017*"возможн" + 0.016*"сша" + '
      '0.014*"счита" + 0.013*"убийств" + 0.012*"сирийск" + 0.011*"кин" + '
      '0.011*"цен"'),
     (13,
      '0.085*"rt" + 0.031*"сми" + 0.028*"перв" + 0.027*"готов" + 0.025*"дел" + '
      '0.021*"власт" + 0.019*"русск" + 0.019*"заяв" + 0.016*"прав" + '
      '0.015*"террорист"'),
     (14,
      '0.076*"rt" + 0.033*"петербург" + 0.027*"вер" + 0.022*"медвед" + '
      '0.017*"дмитр" + 0.015*"матч" + 0.015*"восток" + 0.014*"пожар" + 0.014*"ки" '
      '+ 0.012*"сообщ"'),
     (15,
      '0.045*"rt" + 0.033*"quot" + 0.030*"дорог" + 0.025*"сто" + 0.015*"факт" + '
      '0.014*"выход" + 0.013*"рук" + 0.013*"заверш" + 0.013*"действ" + '
      '0.012*"октябр"'),
     (16,
      '0.060*"rt" + 0.032*"нача" + 0.027*"пост" + 0.027*"добр" + 0.025*"мир" + '
      '0.025*"чита" + 0.024*"как" + 0.020*"рассказа" + 0.019*"написа" + '
      '0.018*"школ"'),
     (17,
      '0.154*"украин" + 0.069*"российск" + 0.059*"rt" + 0.056*"новост" + '
      '0.031*"мнен" + 0.022*"побед" + 0.022*"выбор" + 0.019*"арм" + 0.019*"росс" + '
      '0.017*"продолжа"'),
     (18,
      '0.080*"rt" + 0.064*"нов" + 0.030*"росс" + 0.030*"санкц" + 0.024*"сам" + '
      '0.016*"мид" + 0.015*"рф" + 0.015*"апрел" + 0.014*"иг" + 0.014*"представ"'),
     (19,
      '0.065*"rt" + 0.045*"воен" + 0.043*"украинск" + 0.039*"ес" + 0.036*"украин" '
      '+ 0.033*"фот" + 0.030*"политик" + 0.030*"крым" + 0.029*"сша" + 0.026*"чуж"')]


