+++
title = "Manufacturing Consent on Sanctions"
subtitle = "Or How I Learned to Stop Worrying and Love Word Embeddings"

# Add a summary to display on homepage (optional).
summary = ""

date = 2020-09-21T10:11:22-04:00
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
  caption = "Photo by Morning Brew on Unsplash"

  # Focal point (optional)
  # Options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
  focal_point = ""
+++

In my [previous post](https://www.seantnorton.net/post/polmeth-2020/) on the Russian disinformation campaign, I used a topic model to 
examine what the Russian Internet Research Agency was posting about and how that varied over time, focusing only on the Russian language tweets.
Topic models are a nice way to begin exploration of a corpus, and particularly a large corpus,
because they reveal interesting and often substantively-meaningful patterns in the data 
with little researcher specification needed.

However, they have some weaknesses. One is their inability to zero in on a specific 
word/phrase/anything smaller than a topic (which contains many words, and can be quite broad).
Another big weakness is the overly simplistic way topics models consider context.When you run a topic model you reduce each document to a bag of words in which the order of the words, grammar, etc. don't matter at all, then infer the topics based on words that tend to co-occur in those "bags"
across documents. 

This is where word-embeddings come in handy as an alternative unsupervised approach to extracting meaning 
from large corpora. Word embeddings handle context in a much more sensible, though still uncomplicated, way - they are based on the simple 
linguistic idea that words that occur near each other frequently are likely to have similar meanings. 

Word embeddings accomplish this through feeding words to a neural network, which then *embeds* that word 
in a lower-dimensional space by representing it as a vector. Vectors that are closer to 
each other in that space are more similar, meaning the words have similar meanings.

The general process is quite simple:

1. Convert every word in the corpus to a one-hot encoding. This creates 
an extremely high dimensional vector representation for each word, with the number of dimensions equal to the number of 
unique words in the corpus. A "1" in any position in the vector corresponds to an individual word.
2. Decide on some context window - i.e. how many words surrounding the target word (in both directions) you'll 
input into the model. This window will "slide" over the corpus as we fit the model.
3. Feed the input to an *embedding layer*, which projects that high-dimensional vector on to a lower dimensional 
vector. (You choose the number of dimensions.)
4. Using those embeddings, predict either the words in the window surrounding the target word ("skip-gram") or use 
the words in the window to predict the target word ("continuous bag-of-words").
5. Back-propagate the errors until some convergence criteria is reached.
6. Output the vectors learned in the embedding layer.

Pretty neat! You take something that at face level doesn't seem like a predictive question (what words are similar?),
turn it into a predictive question, then leverage the power of neural networks to generate predictions for high-dimensional 
data. In more traditional statistical terms, you're learning the position of every word in a latent space by assuming that 
the context of words arises from that latent position.

Circling back to my dataset of Russian-language tweets associated with the Russian disinformation campaign, this allows me 
to specifically dig into the context of specific words and how it changes over time. While a topic model is good for describing the data 
as a whole, word embeddings allow me to zero in on topics in a much more specific sense. 

An unsurprising result from my topic model was the prominence of tweets discussing sanctions in the America 
and European Union topics it returned. Sanctions, and Russian counter-sanctions, were harmful to the Russian 
economy. In general, the regime's approach to manipulating the narrative around sanctions was three-pronged:
minimize the impact, claim sanctions harmed the US/EU more than Russia, and argue that sanctions were an example 
of a hostile West afraid of Russia's return to great power status. 

So what do we see in the Twitter dataset, particularly as the sanctions themselves changed over time? To find out, 
I used word2vec, a common and fast implementation of word embeddings. First, I selected only tweets posted between 
March 1st 2014, shortly after the start of the Crimean crisis, January 31st 2015, when the frequency of Russian-language 
tweets begins to taper off. I then split each month into its own dataset, and fit a word2vec model using a skip-gram with 
a window of 5. I then queried each month for the 25 most similar words to sanctions, using the cosine distance as the metric 
of similarity. 

Before I get into the results, it would be helpful to walk through the timeline of sanctions and
the Ukrainian crisis over this timespan. This [timeline](https://www.rferl.org/a/russia-sanctions-timeline/29477179.html) 
is excellent if you want a detailed overview of sanctions, but doesn't mention the war in detail, so I'll hit the highlights here:

* *March 2014*: Russia begins occupation of Crimea in late February, annexes Crimea on March 18th. Throughout the latter half 
of March, the US/EU target visa restrictions and asset freezes at specific Russian/Crimean officials. The US also sanctions 
Bank Rossiya and forbids exporting defense products to Russia. Russia responds with targeted visa bans against EU/US politicians 
and officials, including John McCain.
* *April 2014*: Situation in the Donbass rapidly escalates - heavily-armed insurgents and protestors seize government buildings, 
police stations, etc. throughout the region. Ukrainian government launches its first "anti-terrorist operation" to retake 
Sloviansk, leading to first combat fatalities of the war. US and EU sanction additional individuals and companies, and US restricts exports 
of certain dual-use goods. 
* *May 2014*: Intense fighting in Sloviansk, fighting/rioting in Donetsk and Mariupol. Donetsk and Luhansk declare independence, 
form unrecognized state of Novorossiya. EU targets sanctions at more individuals/companies
* *June 2014*: Fighting continues throughout the Southeast and escalates, Ukraine and US allege that Russia has sent tanks to separatists. 
Ceasefire, frequently broken, between June 20th-30th. US targets sanctions at separatist commanders/political leaders.
* *July 2014*: Intense fighting around Donetsk, with a government counter-offensive pushing towards the city of Donetsk itself; separatists suffer heavy losses. Separatists accidentally shoot down MH17 over Donetsk on July 17th. Prior to MH17, the EU adds more individuals to 
its sanction list, and the US escalates further by sanctioning two important banks, major energy companies, and the Russian defense industry. 
After the shootdown of MH17, the US again escalates with broad, sectoral sanctions on the Russian energy, finance, and defense industries. 
The EU partially follows suit, restricting access to European capital markets and placing an embargo on arms and dual-use technology in both 
the defense and oil sectors.
* *August 2014*: Government counter-offensive pushes into the cities of Luhansk and Donetsk, with the separatists close to collapse by the 
end of the month. Significant numbers of Russian troops begin to join the conflict, and Ukrainian government advance is halted and pushed back 
as the month ends. US significantly restricts export of energy sector technologies. For the first time, Russia answers in a significant way: 
a near-complete embargo on the import of agricultural products from the US, the EU, and all other countries that had imposed sanctions against Russia.
* *September 2014*: First Minsk Protocol ceasefire negotiated; fighting lessens substantially with scattered but significant violations of ceasefire. Towards the end of the month, fighting heats up around Donetsk International Airport.
* *October 2014*: Increasingly intense fighting, despite ceasefire technically remaining in force. No new sanctions.
* *November 2014*: Minsk ceasefire falls apart. Separatists hold elections in controlled territories, in violation of Minsk Protocols. Heavy fighting resumes in the Donbass, NATO and Ukraine observe large quantities of military equipment crossing the border. No new sanctions.
* *December 2014*: Fighting lessens, after a temporary ceasefire to acknowledge 1,000 deaths due to the war. No new sanctions.
* *January 2015*: Separatists in Donetsk launch offensive against Donetsk International Airport, resulting in intense fighting. New round of Minsk talks fails to start when separatist leaders refuse to attend. Separatists push attack on Ukrainian line of control after taking airport. US and EU ban import/export of goods/services from Crimea. 

With that context in mind, let's get into the results of the word2vec model. I present an abbreviated version here, using only some of the 
words from the 25 closest words to sanctions, to make the results more interpretable and spare me from having to translate 250 words. I pulled many tweets with these keywords to help reach these conclusions, but to keep this relatively short, I'll stick to summary.

* *March 2014*: 
    - Keywords: miscalculate, criteria, isolation, anti-Russian, "real person" (as in a person in the legal sense)
    - Interpretation: Repeating government line that this is an attack on Russia meant to isolate it, the criteria of who was put on the sanction list were arbitrary, and the idea that this was a strategic error on the part of the EU and US
    
* *April 2014*: 
     - Keywords: boomerang, skeptical, agricultural embargo, counter-sanctions, "special measures"
     - Interpretation: Discussion of Russian government counter-sanctions against US and EU officials, very early discussion of a potential embargo on agricultural imports ("special measures" will end up being part of the title of the embargo decree), and repeating the Russian government's warning that sanctions will have a "boomerang effect"
     
* *May 2014*: 
    - Keywords: presidential, intrigues, bluff, EU, symmetrical
    - Interpretation: Discussion of presidential election in Ukraine alongside discussion of new EU sanctions and potential Russian response

* *June 2014*: 
    - Keywords: negotiations, dialogue, boomerang, Jackson-Vanik, Russophobic, senseless
    - Interpretation: Discussion of negotiations over ceasefire in Ukraine and potential impact of that on US/EU sanctions. Accusations that the sanctions are due to Russophobia, and again pushing the line that sanctions are poorly thought out and will boomerang. Interestingly, the Jackson-Vanik Amendment (which limited trade relations with Soviet-bloc countries), comes up. In 2012, its application to Russia was ended and it was replaced by the Magnitsky Act. 
    
* *July 2014*: 
    - Keywords: convey, undermine, countermeasures, isolation
    - Interpretation: No clear interpretation here. Russian government clearly on the PR defensive after MH17 shoot-down, and most sanctions in response to MH17 come at the very end of the month.
    
* *August 2014*: 
    - Keywords: weapons, materials, Americans, Europeans, detrimental, countermeasures, counter-productive, reciprocity
    - Interpretation: Decrying the marked escalation in EU and US sanctions, which heavily target the energy and defense sectors, which the Russians argue hurt the US and EU due to future effects on hydrocarbon supply. Amplifying the justification for the agricultural embargo, which is the first proportional Russian response to sanctions.
    
* *September 2014*: 
    - Keywords: West, fellow citizens, conspiracy, undermine, crumple, energy independence, weaken, freeze, force hand, illogical
    - Interpretation: Talking up the effect of Russian counter-sanctions on Europe, and also reminding Russians the the EU is dependent on Russian energy. Implies that counter-sanctions will eventually cause the EU to break. Again, parroting rhetoric on sanctions as an anti-Russian conspiracy at their core.
    
* *October 2014*: 
    - Keywords: war, West, laughable, plot, sadomasochism, purposeless, dead-end, anti-sanction, tolerate/be patient
    - Interpretation: As conflict heats back up, emphasis on the idea that sanctions are counter-productive and harmful to the US and EU (sadomasochistic), and that all the Russians have to do is wait for the anti-sanction movement in the EU to gain steam
    
* *November 2014*:
    - Keywords: EU, destabilization, asymmetrical, incomprehensible, agricultural embargo
    - Interpretation: Emphasis again on irrationality/counter-productiveness of sanctions and effectiveness of Russian counter-sanctions.
    
* *December 2014*: 
   - Keywords: effect, hostile, incomprehensible, rapprochement, undermine, weaken, EU
   - Interpretation: Sanctions and counter-sanctions drag on; talk of reconsideration of sanctions as conflict in Ukraine seems to cool after collapse of Minsk ceasefire in November
   
* *January 2014*:
   - EU, Rosoboronexport (Russian defense industry exporter), cheese maker, reciprocal, reasoning, absurd
   - Interpretation: Attack against the reasoning behind new sanctions, discussion of countermeasures. The "cheese maker" similarity is due to Russian state emphasizing response of Russian agricultural industry to sanctions and their success in duplicating banned European cheeses (some better than others...)
   
   
That was a deep dive that leads to some interesting conclusions:

1. The IRA was reacting quickly to events on the ground and shifting the narrative surrounding sanctions to match. In particular, the response to sanctions on the energy sector were really evident (see August 2014, September 2014)
2. There was a lot of discipline in pushing the official government narrative on sanctions - that they would "boomerang" back on the US and the EU and harm them as substantially or more than Russia. Going back to a tweet I pulled for my topic models, the narrative could be nicely summarized as: "Russia hasn’t even felt a thing yet, but the EU itself can’t withstand the effect of its own sanctions!"
3. Once the Russian agricultural embargo was in place, the campaign really played up the impact on the EU and the EU anti-sanctions movement. This was relatively grounded in reality - the EU losses in the agricultural sector were significant. 

Again, this is interesting in contrast to the Chinese case, where social media propaganda was largely intended to distract. On sanctions, distraction seems like a potentially good strategy, particularly as 2014 drags on. Oil prices collapsed throughout 2014, and sanctions hit the Russian oil and financial industries even harder as a result. The agricultural embargo hurt the EU, but it also had a noticeable impact on Russian consumers - something like 50% of Russia's meat, fish, dairy, vegetables, and fruits came from the EU, and that vanished overnight. This caused an immediate and sharp increase in the price of food, in the midst of a recession and already high inflation of the Russian ruble. While Russia could easily blame the West for the financial and hydrocarbon sectoral sanctions, the agricultural embargo was the regime's own doing. Instead of distracting, the IRA disinfo campaign went on the attack - playing up the significant economic fallout in the EU, speculating that the EU would have to cave any day now due to discontent over sanctions, and towards the end of the campaign, even emphasizing the opportunities it would create for Russian agriculture (which they were correct about). In effect, the campaign fairly neatly integrated sanctions into the dominant national narrative on Crimea, arguing they were a reaction to Russia's triumphant return to global power status, and Russia would not only endure but was capable of taking the fight to the West. It seems authoritarian regimes are capable of taking more traditional propaganda campaigns online, though it remains an open question whether they work.

Especially interesting is the clear choice to use messaging targeted at regime supporters (or at least those who agree with Russian foreign policy). This dovetails neatly with the literature on polarization and social media, which intuitively finds that people are more likely to repost information that is congruent with their ideological position. This is particularly true on Twitter, where Barbera, Jost et al. (2015) find that 75% of retweets on political topics (in the US) involve users of similar ideological background. In some sense, this is a low-cost strategy that seems more likely, at face level, to succeed. With citizens already primed to view these issues in a partisan light by state media/official statements, simply inserting/amplifying similar content could lead to quick and sustained spread among co-partisans. It's also consistent with journalistic accounts of the Internet Research Agency's operation, in which former employees recount that the emphasis was on quantity of posts over quality. Targeting the opposition would require signficant thought into clever ways to get them to engage with disinformation and move them in whatever direction the regime preferred. Simply engaging with the dominant media narrative generates content much more quickly and easily.

Another interesting piece of the puzzle is the relationship between regime support and social media platform choice in Russia. Reuter and Szyakoni (2011) find that Twitter and Facebook are the preffered social media platforms of the Russian *opposition*. There's 3 potential explanations for so many resources being invested in creating Twitter content:

1. The partisan composition of Russian Twitter users changed between 2011-2014. This isn't implausible, as this was a period of rapid worldwide growth in Twitter usage.
2. This was an attempt to drive a wedge in the Russian opposition over the Ukraine crisis. Many critics of the Putin regime nevertheless support the annexation of Crimea (as do most Russians). Taking advantage of the sudden salience of foreign policy over domestic issues seems like a good oppurtunity for the Russian regime to cultivate support or fragment the opposition further.
3. It was a misuse of resources. Authoritarian regimes frequently make poor decisions. 

As always, get in touch if you have any questions/suggestions/corrections!
