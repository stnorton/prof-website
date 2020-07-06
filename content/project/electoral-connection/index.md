+++
title = "The Electoral Consequences of Political Protest"
date = 2019-07-05
draft = false

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["Protest", "Urban Politics"]

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
  caption = "Photo by Kirill Zharkoi on Unsplash"

  # Focal point (optional)
  # Options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
  focal_point = ""
+++

*(with Graeme Robertson)*

In recent years, protest in both democracies and authoritarian regimes has taken a central place in political science as urban protests have proliferated across the world. Much of this recent literature is focused upon the effects of protest and in particular the effects of protests on political attitudes, whether of protesters (Converse and Philips 1991, Pop-Eleches et al. 2018) or bystanders (Banaszak and Ondercin 2016, Tertychnaya and Lankina 2018, Wouters 2018). In this paper, we follow in this tradition, but take the analysis beyond changes in public opinion to look at whether and how protest might lead to increased voting for opposition forces in a paradigmatic electoral authoritarian regime, Russia. Using a dataset of protest events and electoral returns in Russia's cities with > 1 million people, we examine the effect waves of protest have on subsequent opposition vote totals.

Given that latent discontent with authotarian rule drives both opposition voting and protest, we adopt a latent variable approach that aims to identify whether protest has an effect on opposition voting *beyond* general discontent. We accomplish this using a state-space model, where protest and opposition voting are treated as noisy signals of latent discontent, with protest also used as an explanatory variable in the emission distribution, which measures opposition voting. This approach provides a measure of latent discontent and controls for it in a principled manner, taking into account measurment uncertainty, while also eliminating the autocorrelation inherent to data on protest and elections.
