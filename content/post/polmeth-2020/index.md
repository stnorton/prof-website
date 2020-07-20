+++
title = "Who Do You Think You're Fooling"
subtitle = "The Internal Russian Disinformation Campaign"

# Add a summary to display on homepage (optional).
summary = ""

date = 2020-07-20T09:58:02-04:00
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

I recently presented my work on the Russian Twitter disinformation campaign at PolMeth 2020. 
Kudos to the organizers of the event - they put together an excellent online conference in very 
little time. I encourage you to check out some of the other posters, which are available on the
[the conference website](https://polmeth.theopenscholar.com/polmeth2020/graduate-students-posters).

My poster is below, along with some more detailed commentary. You can pop the poster out 
in a separate tab [here](/img/poster.jpg).

![Poster](/img/poster.jpg)

This project started thanks to a suggestion from Deen Freelon at UNC (via my advisor) 
to look at the other half of the Twitter Election Integrity Dataset, which is primarily
in Russian. These tweets were part of a disinformation campaign conducted by the state-affiliated 
Internet Research Agency (IRA) in Russia. While the disinformation campaign targeted at the 2016 US 
presidential election is fairly well-known, the Russian language tweets are comparatively understudied.

Early on in this project, a puzzle emerged. The Russian-language campaign, consisting of a little 
shy of 4.9 million tweets and well over half the IRA users, peaked immediately after the 
Russian annexation of Crimea and the US/EU imposition of sanctions on Russia. Excellent work by 
Jennifer Pan, Margaret Roberts, Gary King, and others has established that the Chinese state uses 
social media as a tool for censorship via distraction. Simplifying quite a bit, the argument goes that direct censorship 
is costly and difficult, but by flooding social media with irrelevant information the state makes it harder 
for citizens to see undesirable information. Unless you're an activist, posts about protests, corruption, etc. 
might not come across your feed very often, and if your feed is filled with distracting content you're considerably 
less likely to see those posts.

Keen Russia observers will already see the puzzle - the Russian disinformation campaign peaked during 
what's known as the "Russian Spring", a dramatic revival of Russian nationalism and almost overnight 
meteoric rises in approval ratings for Putin, United Russia, and the Russian government as a whole. 
While dissenters certainly exist, the Russian Spring is remarkable as a moment of collective 
effervescence - citizens' opinions on Ukraine, the Russian state, the EU, and the US rapidly 
converged. United Russia and Putin, somewhat unexpectedly, found themselves in the strongest 
domestic position they'd ever been in. Given a political context in which Russians largely supported the 
official narrative on events and overwhelmingly approved of the government's actions, why 
would they possibly want to distract their citizenry? In other words, if everyone likes what 
you're doing, it doesn't make sense to try to move attention to irrelevant topics.

Authoritarian governments certainly can do, and often do, the wrong thing - and that's what I initially 
expected to find here. Instead, I found active participation in the discourse surrounding Ukraine, 
the EU, the US and sanctions. 

After a long and computationally expensive tuning process (4.9 million documents, even tweets, is 
fairly big data) I settled on a 15 topic model. That in itself is somewhat remarkable - that 4.9
million documents could be explained reasonably well by 15 topics suggests a high level of message coordination. 
Five particularly substantively interesting topics emerged:

1. *Pushing the news*: news stories the IRA was attempting to promote, largely through quote tweets and 
replies. e.g. “Thoughtfully written. The Russian people are sincerely worried about the situation
in Ukraine.” ~ 12% of total tweets. 
2. *Putin and Patriotism*: tweets hyping up Vladimir Putin, using national symbols important to 
the regime, and general nationalist dreck. e.g. "Don’t get it confused: the government is the government, but the state is each
of us!"; ~ 8% of total tweets. If you check the graph in the middle panel of the poster for this topic, you'll 
notice a large peak in prevelance around the 70th anniversary of the Soviet victory in WW2, an important event for the regime.
3. *"Personal" Opinions*: (Emphasis on the air-qoutes around personal!) General personal commentary on current events. 
While the Pushing the News topic is commentary on current events as well, this tended to not be comments on articles
or particularly retweet heavy. I think of it as the IRA users playing the role of a politically-engaged citizen. 
~ 6% of total tweets. e.g. "What do you think would happen? Ukraine would just ride off into the sunset?"
4. *America*: Predictably, largely discussion of American santions. Also, plenty of negative news stories on things happening in the
US (we've been generating plenty of those). Surprisingly, not a lot about the American election - 
probably because the Russian disinfo campaign had tapered off in favor of the campaign targeted at the US election. 
~ 9% of total tweets. e.g. "Why is Russia feeding the US?" (referring to Russian grain exports to the US)
5. *Russia vs. the EU*: Unsurprisingly, largely discussion of EU sanctions. Heavily pushing the narrative that EU sanctions 
are worse for the EU than Russia. ~ 7% of total tweets. e.g.  "Russia hasn’t even felt a thing yet, but the EU itself can’t withstand the effect of
its own sanctions!"

These topics track pretty well with events on the ground, which I take as an indicator 
of good model fit. There's a couple interesting trends:

* The Pushing the News topic, the most prevalent topic in the data, 
drops off in the period immediately following the annexation of Crimea in favor of 
the Putin and Patriotism topic. Favorable news was clearly less of an emphasis for the 
IRA for February-March 2014 - instead they attempted to activate/amplify feelings of patriotism.
* I see something similar with "Personal" Opinions following the imposition of sanctions, this time 
with the topic dropping off in favor of the News topic, the America topic, and the EU topic. 
For this period, it was less important for the IRA to offer "personal" commentary on events than it was 
to promote the idea that not only would Russia withstand sanctions, they would backfire on the US 
and the EU.

My initial conclusion from this analysis is that the Russian government wasn't using the disinfo campaign to 
distract, or at least not in the way the Chinese state distracts. Rather, they were reacting 
to events on the ground and actively engaging with the Twitter discussion surrounding them. It's important 
to note that these discussions weren't particularly manufactured or controlled: there really was strong 
Russian support for the regime and its actions. In other words, the goal appears to have been to *amplify* 
the surge of support for the regime, rather than create it or control it (both of which are nearly 
impossible tasks in the decentralized world of social media).

The results raise an interesting substantive question that I've been grappling with for some time: 
do distraction or engagement have differential effects? Even if (like me) you spend far too much time 
on Twitter, you still have limited time. If the IRA engaging in certain topics boosted their 
visibility on Twitter, users were inevitably distracted from other content. 

Future work on this project aims to untangle exactly that question by looking at the *effect* 
of these tweets, particularly on emotional engagement. A separate project will also look at the 
use of images in the dataset - something I think it is very important to explore. Working with 
images is much more difficult than working with text, but modern social media is a very 
visual phenomenon. If we keep ignoring image data, we're discarding tons of potentially 
useful information.

If you have any questions or feedback, feel free to contact me! I am always happy to 
talk about this subject.




