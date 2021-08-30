+++
title = "Decentralizing States, Decentralizing Protest"
date = "2019-05-07"
draft = false

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["Protest"]

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
  caption = "Photo by Damien Checoury on Unsplash"

  # Focal point (optional)
  # Options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
  focal_point = "Smart"
+++

*(with Kaitlin Alper)*

Contentious politics is intricately tied to the creation of the modern nation-state, with changes to the state driving new forms and patterns of contention and contention forcing state adaptations. Chief among these reciprocal changes was the centralization of the state; driven initially by the need to more efficiently extract resources, centralization crosscut highly local identities and repertoires of protest, leading contentious actors to centralize in response. The centralization of contention allowed citizens to extract key concessions from early modern states, paving the way for both democracy and modern social movements. However, states did not stop developing after centralization; in recent decades, many of the world's developed democracies have chosen to devolve power to lower levels of government, often seeking more efficient and responsive government. Existing theories would predict that contentious politics should also respond to decentralization, and yet little research has attempted to determine whether this is true.

Our initial contribution to this research program is a co-authored article, in which we develop a theoretical framework that explains why we expect contentious politics to decentralize with the state and subject it to an initial quantitative test. Theoretically, we synthesize the literatures on multilevel governance, contentious politics, and state-making to argue that protest should be responsive to state-level decentralization through two mechanisms: the existence and formation of local communities and the increase in political access points. We also develop a measure of the centralization of protest: synchronicity. It follows from the centralization of protest that protest should largely be centrally organized, whether by institutionalized social movements or through the use of the "connective action" of the internet. This centralization leads protest events to occur at the same time in a relatively small number of places. In contrast, decentralized protest, being driven by factors that are not generally replicated throughout the country, should not synchronize with protests in other localities except by chance or during national protest waves.

We then test for a relationship between decentralization and synchronicity using a spatio-temporal network of protest events in developed democracies from 2001-2020, the Regional Authority Index (RAI), and a battery of socioeconomic controls. The spatio-temporal network is constructed by creating an edge between any given protest and all other protests in the same country that occur either in a window two weeks before or after the given protest. These edges are then weighted using the time between protests in days and the inverse of the spatial distance between the protests. This captures both the temporal and spatial clustering of protest. As regional authority increases, we expect edge weights to increase, reflecting the tendency of protest to occur within a single region and on a timeline that is not synchronous with other protests across the country. A hierarchical modification of the additive and mixed effects network model will be used to analyze the data, allowing for intuitive interpretation of results.

This project is currently in the modeling stage, with data collection and cleaning already done. The scale of the data requires me to code a custom version of the additive and mixed effects model capable of relatively fast sampling on a large network and appropriately modeling within-country dependencies. Once the model is complete, we expect to submit the article for peer review. Future work will include case studies of the relationship between decentralization and contention as well as analysis of how the particular modalities of decentralization, such as the creation of regional representative institutions, influence this process.


