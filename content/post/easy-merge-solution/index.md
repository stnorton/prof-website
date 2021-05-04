+++
title = "A Quick and Easy Solution to Problems with dplyr Left Join"
subtitle = ""

# Add a summary to display on homepage (optional).
summary = ""

date = 2021-05-04T10:46:39-04:00
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

A frequent problem with any join, but in particular dplyr `left_join`, is key columns on the "left" dataset 
mapping to multiple rows in the "right" dataset, creating new rows in the left dataset. When there's only 
a few unique combinations of the key values, this is fairly easy to resolve by hand - identify the misspelled or 
otherwise wrong key on either side and fix it. When you have a lot of unique key combinations, this is a pain. This post 
will show you my quick and easy way to identify the problematic keys so they can be corrected.

(I debated about whether to write this because it's so simple, but most Stack Overflow answers I found on the subject 
were ways to work around the problem, such as dropping any duplicate matches, and not fix it.)

It only takes 4 steps:

1. Create a unique ID column in the left dataset (i.e. the one you're merging data into): `data$unique_id <- 1:nrow(data)`
2. Run `left_join`
3. Identify observations that generated multiple matches: `dups <- duplicated(data$unique_id)`
4. Get the key(s) that created the problem: `unique(data[dups, c(key_1, key_2)])`

And there you go - you've now found the key values or combinations of key values that aren't generating unique matches and can 
easily correct them (hopefully).
