+++
title = "Rally Around the Tweet"
subtitle = "The Internal Russian Disinformation Campaign"
date = 2019-07-05
draft = false

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ['Disinfo']

# Project summary to display on homepage.
summary = ""

# Slides (optional).
#   Associate this page with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references 
#   `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides = ""

# Optional external URL for project (replaces project detail page).
external_link = ""

# Links (optional).
url_pdf = ""
url_code = ""
url_dataset = ""
url_slides = ""
url_video = ""
url_poster = ""

# Custom links (optional).
#   Uncomment line below to enable. For multiple links, use the form `[{...}, {...}, {...}]`.
# links = [{icon_pack = "fab", icon="twitter", name="Follow", url = "https://twitter.com"}]

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
[image]
  # Caption (optional)
  caption = "Photo by Prateek Katyal on Unsplash"

  # Focal point (optional)
  # Options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
  focal_point = ""
+++

The rise of internet communication technologies (ICTs), and in particular social media, has allowed people to communicate and coordinate in a decentralized manner that is difficult for the state to mediate and understand. In particular, ICTs have proven to be a powerful tool for anti-authoritarian actors. Increasingly, authoritarians have developed sophisticated methods to control online conversation, from covertly throttling connections to certain sites, burying unwanted conversations in irrelevant posts, or using their own supporters and bots to counter-protest online. While it has been established that authoritarians have a broad menu of tactics available for contesting online space, little thought has been given to how and why different regimes deploy different tactics. 

I argue that the presence of routinized competition, namely regular semi-competitive elections, fundamentally structures the way in which authoritarian regimes engage with the public. While all authoritarian regimes seek to cultivate support, electoral authoritarians must regularly turn out supporters to win, ideally with the least fraud and coercion necessary. These authoritarians also face persistent and institutionalized opposition that contests the streets as well as the ballot boxes, incentivizing electoral authoritarian regimes to build state-mobilized movements to demonstrate the regime's strength while reducing the need for risky and expensive displays of overt coercive force. The threats from elections, protest, and social media to these authoritarian regimes are inextricably linked; social media is not inherently dangerous to authoritarians except in its ability to spread information and coordinate action. It would come as no surprise if states took advantage of ICTs in the same way as their opponents: as a mobilizational tool. 

The data for this project come from the Twitter Election Integrity Dataset, which includes nearly 10 million tweets from accounts run by Russian state-affiliated actors. Of these tweets, nearly 4.8 million are in Russian and were sent between January 2015 and January 2016, the most intense period of the international crisis triggered by Russia's annexation of Crimea. Using both topic models and word embeddings, I extracted both the dominant themes of the tweets as well as the specific semantic context surrounding discussion of sanctions against Russian over time. 

Using natural language processing techniques and nearly 5 million tweets accounts created by Russian state-affiliated actors, I demonstrate the Russian Twitter disinformation campaign was a unique prong of a multi-media mobilizational campaign intended to harness a dramatic rise in both Russian nationalism and the popularity of Vladimir Putin. While traditional state-affiliated media pushed state-friendly narratives, the Twitter operation involved actors expending great effort to appear as ordinary, Russian-speaking Twitter users. This "sockpuppeting" lent credibility to their efforts to boost content from state-affiliated media, amplify real state-friendly users, and redirect the conversation on sanctions in line with state-sponsored narratives. Much as with state-mobilized movements on the streets, the intent was to mobilize supporters, demonstrate the regime's strength, and marginalize opposition narratives.

Using a dataset of nearly 5 million tweets from Russian state-affiliated actors during the height of the Ukraine crisis, I utilize natural language processing techniques to reverse engineer the goals of this authoritarian disinformation campaign. Both topic models and word embeddings show that the campaign was highly coordinated and reactive to events on the ground, pushed dominant narratives from Russian state-owned traditional media, and attempted to spin and control potentially damaging discourses on sanctions. While the Twitter disinformation campaign was highly coordinated with the regime's traditional media offensive, the Twitter operation involved actors expending great effort to appear as ordinary, Russian-speaking Twitter users. This "sockpuppeting" lent credibility to their efforts to boost content from state-affiliated media, amplify real state-friendly users, and redirect the conversation on sanctions in line with state-sponsored narratives. Much as with state-mobilized movements on the streets, the intent was to mobilize supporters, demonstrate the regime's strength, and marginalize opposition narratives.

This project, currently under review, addresses two major gaps in the literature on disinformation by bringing the state back in and presenting a case study of a disinformation campaign in an non-democratic, non-Western country. Future work in this agenda will also utilize the Twitter election dataset by taking advantage of the thousands of images attached to state-affiliated accounts tweets. Modern social media is fundamentally visual, yet analysis of social media content often ignores this as a matter of convenience. Using recent advances in the analysis of images and video, I plan to dissect the visual language of the Russian disinformation campaign much as I have already dissected the text.