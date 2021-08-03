---
layout: page
title: About
tagline: Why and what is that ?
permalink: /about.html
ref: about
---

I wrote this blog with the idea of sharing my knowledge and my thoughts on various topics.
It is mostly focused on data science and geeky stuff.
Most of the images used in the posts comes from the royalty free image bank [pexels](https://www.pexels.com/).

This a a long-term project I have been working for months (years?) but it is hard to conciliate free-time and working on it!
If you want to support my work, please consider making a donation :)

[![Donate with Bitcoin](https://en.cryptobadges.io/badge/small/12eAEKU4rgvhLCxvdkxKJYocJdEFRyNrta)](https://en.cryptobadges.io/donate/12eAEKU4rgvhLCxvdkxKJYocJdEFRyNrta)

# How this website was made ?

This blog uses [Jekyll](https://jekyllrb.com/) and is hosted on [Github pages](https://pages.github.com/). 
It is mostly based from the modified `Cayman Blog` theme by `lorepirri`. As the author describes it:

>Cayman Blog is a Jekyll theme for GitHub Pages. It is based on the nice [Cayman theme](https://pages-themes.github.io/cayman/), with blogging features added. You can [preview the theme to see what it looks like](http://lorepirri.github.io/cayman-blog), or even [use it today](https://github.com/pages-themes/cayman).

It is really nice because it allows anyone to build for free his own website in a few minutes! *Of course, if you are like me and want to optimize and modify the theme, it will mostly take you weeks...*

For the jupyter notebooks, the templates were heavily inspired from the wonderfull <a href="http://peterroelants.github.io/">peterroelants</a> blog (you should definitivelly check it!).
I am using [jupytext](https://github.com/mwouts/jupytext) to get the notebooks out of python files, and [nbconvert](https://github.com/jupyter/nbconvert) to build the html. References, post metadata and collapse buttons are injected using python via [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/).

You may see sometimes a badge like this one ![](https://mybinder.org/badge_logo.svg), this means that the post is interactive and compatible with [Binder](https://mybinder.readthedocs.io/en/latest/introduction.html).
Binder spawn for you a standalone web-based jupyter server to let you play with the underlying notebook and data, so you can modify the code as you want without any consequence!

Most of the icons are svg embedded in html from [iconmonstr](https://iconmonstr.com/).

The tags and posts pages were inspired from [codinfox](https://codinfox.github.io/dev/2015/03/06/use-tags-and-categories-in-your-jekyll-based-github-pages/).

Check the source code for the website at [{{ site.github.repository_name }}]({{ site.github.repository_url }}).

[Go to the Home Page]({{ '/' | absolute_url }})