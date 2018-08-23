import sys
import argparse
from bs4 import BeautifulSoup


def parse_input():
    """Parse the input from the command line."""
    parser = argparse.ArgumentParser(description='Convert ipynb html files so that the input field are collapsable.')
    parser.add_argument('html_file', type=argparse.FileType('r'), help='ipynb as html file to process.')
    args = parser.parse_args()
    return args.html_file

def insert_collapse_buttons(soup):
    """Insert the collapse buttons on the code input field of the nb."""
    input_areas = soup.select('div.inner_cell > div.input_area')
    for idx, input_area in enumerate(input_areas):
        # Add the collapse/expand button
        collapse_expand_button_tag = soup.new_tag('div')
        collapse_expand_button_tag['class'] = 'collapse_expand_button far fa-1x fa-minus-square'
        input_area.insert(0, collapse_expand_button_tag)
        input_area['class'].append('collapsed')

def main():
    html_file = parse_input()
    soup = BeautifulSoup(html_file, 'html.parser', exclude_encodings="ascii")
    insert_collapse_buttons(soup)
    html_file.close()
    modifiedHtml = str(soup.prettify('ascii'), 'ascii')
##     Overwrite original file
    with open(html_file.name, "w") as file:
        file.write(modifiedHtml)
    file.close()

if __name__ == "__main__":
    main()