import sys
import os
import argparse
import re
import codecs
from datetime import date

from bs4 import BeautifulSoup
from citeproc import Citation
from citeproc import CitationItem
from citeproc import CitationStylesBibliography
from citeproc import CitationStylesStyle
from citeproc import formatter
from citeproc.source.bibtex import BibTeX

def parse_input():
    """Parse the input from the command line."""
    parser = argparse.ArgumentParser(description='Convert ipynb html files so that the input field are collapsable.')
    parser.add_argument('html_file', type=argparse.FileType('r'), help='ipynb as html file to process.')
    args = parser.parse_args()
    return args.html_file

def _cite_warn(citation_item):
    print("WARNING: Reference with key '{}' not found in the bibliography."
          .format(citation_item.key))

def write_bib_to_html(cite_entries):
    # html references section to add at the end of the notebook
    html = '''<div class="cell border-box-sizing text_cell rendered">
    <div class="prompt input_prompt">
    </div>
    <div class="inner_cell">
    <div class="text_cell_render border-box-sizing rendered_html">
    <h2 id="References">
        References
        <a class="anchor-link" href="#References">
        &#182;
        </a>
    </h2>
    </div>
    </div>
    </div>
    <div class="cell border-box-sizing text_cell rendered">
    <div class="prompt input_prompt">
    </div>
    <div class="inner_cell">
    <div class="text_cell_render border-box-sizing rendered_html">
    '''
    for key in cite_entries:
        html += '<div id=\"{}\">\n'.format(key)
        html += '<p>' + cite_entries[key][1] + '</p>\n'
        html += '</div>\n'
    html += '''</div>
    </div>
    </div>
    '''
    return html

def get_bib():
    bib_source = BibTeX('notebooks/bibliography.bib', encoding='utf8')
    bib_style = CitationStylesStyle('notebooks/springer.csl', validate=False)

    bibliography = CitationStylesBibliography(bib_style, bib_source, formatter.html)
    bib_cites = [Citation([CitationItem(item)]) for item in bib_source]

    for item in bib_cites:
        bibliography.register(item)
    for item in bib_cites:
        bibliography.cite(item, _cite_warn)
    
    num = len(bibliography.keys)
    bib_entries = dict()
    for i in range(num):
        bib = ''.join(bibliography.bibliography()[i])
        # remove beginning digits and \. from bib entries
        bib = '{}.&emsp;' + re.sub("^\d+\.", "", bib)
        bib_entries[bibliography.keys[i]] = bib

    return bib_entries

def get_title_and_remove(soup):
    """Insert the collapse buttons on the code input field of the nb."""
    input_area = soup.find_all(["h1"])[0]
    title = input_area.text.strip()[:-1].strip()
    input_area = soup.select('div.cell')[0]
    input_area.decompose()
    
    return title

def get_tags_category_and_remove(soup):
    """Insert the collapse buttons on the code input field of the nb."""
    input_areas = soup.select('div.cell')
    tags = []
    category = []
    tag_found = False
    for idx, input_area in enumerate(input_areas):
        if input_area.text.strip()[:-1].strip() == "Tags":
            tag_found = True
            break
    if tag_found:
        tags = "".join(input_areas[idx + 1].text.strip().split(";")[1:])
        category = input_areas[idx + 1].text.strip().split(";")[0]
        input_areas[idx + 1].decompose()
        input_areas[idx].decompose()

    return tags, category

def write_refs(soup, bib_entries):
    """Insert the collapse buttons on the code input field of the nb."""
    cite_entries = dict()
    id_ref = 1
    input_areas = soup.select('cite')
    for input_area in input_areas:
        if input_area.string is not None:
            ref = input_area.string.strip()
            # does the provided ref exists in the bib?
            if ref in bib_entries.keys():
                # was it already created or not?
                if not ref in cite_entries.keys():
                    cite_entries[ref] = [id_ref, bib_entries[ref].format(id_ref)]
                    id_ref = id_ref + 1
                # replace provided <cite> tag with updated <cite>
                cite_tag = soup.new_tag('cite')
                ref_tag = soup.new_tag('a', href="#{}".format(ref))
                ref_tag.string = "[{}]".format(cite_entries[ref][0])
                cite_tag.append(ref_tag)
                input_area.replace_with(cite_tag)

    return cite_entries

def insert_collapse_buttons(soup):
    """Insert the collapse buttons on the code input field of the nb."""
    input_areas = soup.select('div.inner_cell > div.input_area')
    for idx, input_area in enumerate(input_areas):
        # Add the collapse/expand button
        collapse_expand_button_tag = soup.new_tag('div')
        collapse_expand_button_tag['class'] = 'collapse_expand_button far fa-1x fa-minus-square'
        input_area.insert(0, collapse_expand_button_tag)
        # If the cell starts with ##, by default it will be collapsed
        if re.search(".*\n##", input_area.text):
            input_area['class'].append('collapsed')

def write_front_matter(soup, html_file, title, tags, category):
    """Add the front matter at the beginning of the html."""
    front_matter = '''---
layout: post
title: {}
date: {}
category: {}
tags:  {}
img: /notebooks/imgs/{}
file: /notebooks/{}
excerpt_separator: <h2
---
    '''
    # post date
    curr_date = date.today().strftime('%Y-%m-%d')
    # python file
    input_file = os.path.basename(html_file.name).split('.')[:-1]
    input_file = ".".join(input_file)
    python_file = input_file + '.py'
    # post picture, jpg or png
    post_img = os.path.join(input_file, 'post_img.jpg')
    if not os.path.exists('notebooks/imgs/{}'.format(post_img)):
        post_img = os.path.join(input_file, 'post_img.png')

    return front_matter.format(title, curr_date, category, tags, post_img, python_file)

def add_binder(soup, html_file):
    """Add the binder badge"""
    input_file = os.path.basename(html_file.name).split('.')[:-1][0]
#     html = '''<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
# </div><div class="inner_cell">
# <div class="text_cell_render border-box-sizing rendered_html">
# <p> <a href="https://mybinder.org/v2/gh/ltetrel/ltetrel.github.io/master?filepath=notebooks%2Fipynb%2F{}.ipynb"><img src="https://mybinder.org/badge_logo.svg"></a> (interactive version) </p>
# </div>
# </div>
# </div>
#     '''.format(input_file)
    
    input_areas = soup.select('binder')
    for input_area in input_areas:
        # replace <binder> tag with binder badge
        tag = soup.new_tag('p')
        ref_tag = soup.new_tag('a', href="https://mybinder.org/v2/gh/ltetrel/ltetrel.github.io/master?filepath=notebooks%2Fipynb%2F{}.ipynb".format(input_file))
        img_tag = soup.new_tag('img', src="https://mybinder.org/badge_logo.svg")
        ref_tag.append(img_tag)
        tag.append(ref_tag)
        question_tag = soup.new_tag('a', href="/about.html")
        sup_tag = soup.new_tag('sup')
        sup_tag.append(" (?)")
        question_tag.append(sup_tag)
        tag.append(question_tag)
        input_area.replace_with(tag)

def main():
    html_file = parse_input()
    soup = BeautifulSoup(html_file, 'html.parser', exclude_encodings="ascii")
    # cells wil be collapsed if they begins with "##""
    insert_collapse_buttons(soup)
    # if <binder> tag is deteccted, it is replace with the binder link
    add_binder(soup, html_file)
    # bibliography
    bib_entries = get_bib()
    cite_entries = write_refs(soup, bib_entries)
    # title
    title = get_title_and_remove(soup)
    # category and tags
    tags, category = get_tags_category_and_remove(soup)
    # front matter pre-pending
    front_matter = write_front_matter(soup, html_file, title, tags, category)
    with open(html_file.name, "w") as file:
        file.write(front_matter)
    file.close()
    # overwrite original html
    modifiedHtml = str(soup)
    with open(html_file.name, "a") as file:
        file.write(modifiedHtml)
    file.close()
    if cite_entries != {}:
        with open(html_file.name, "a") as file:
            file.write(write_bib_to_html(cite_entries))
        file.close()
    html_file.close()

if __name__ == "__main__":
    main()