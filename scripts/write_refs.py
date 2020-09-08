import sys
import argparse
import re
import codecs

from bs4 import BeautifulSoup
from citeproc import Citation
from citeproc import CitationItem
from citeproc import CitationStylesBibliography
from citeproc import CitationStylesStyle
from citeproc import formatter
from citeproc.source.bibtex import BibTeX

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
        bib = '{}.&emsp;' + bib[2:]
        bib_entries[bibliography.keys[i]] = bib

    return bib_entries

def parse_input():
    """Parse the input from the command line."""
    parser = argparse.ArgumentParser(description='Convert ipynb html files so that the input field are collapsable.')
    parser.add_argument('html_file', type=argparse.FileType('r'), help='ipynb as html file to process.')
    args = parser.parse_args()
    return args.html_file

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
                if not ref in cite_entries.keys() :
                    cite_entries[ref] = [id_ref, bib_entries[ref].format(id_ref)]
                    id_ref = id_ref + 1
                # replace provided <cite> tag with updated <cite>
                cite_tag = soup.new_tag('cite')
                ref_tag = soup.new_tag('a', href="#{}".format(ref))
                ref_tag.string = "[{}]".format(cite_entries[ref][0])
                cite_tag.append(ref_tag)
                input_area.replace_with(cite_tag)

    return cite_entries

def main():
    html_file = parse_input()
    soup = BeautifulSoup(html_file, 'html.parser', exclude_encodings="ascii")
    bib_entries = get_bib()
    cite_entries = write_refs(soup, bib_entries)

    if cite_entries != {}:
        modifiedHtml = str(soup)
        # overwrite original file
        with open(html_file.name, "w") as file:
            file.write(modifiedHtml)
        file.close()
        with open(html_file.name, "a") as file:
            file.write(write_bib_to_html(cite_entries))
        file.close()
    html_file.close()

if __name__ == "__main__":
    main()
