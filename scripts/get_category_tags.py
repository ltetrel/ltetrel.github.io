import sys
import argparse
import re
from bs4 import BeautifulSoup


def parse_input():
    """Parse the input from the command line."""
    parser = argparse.ArgumentParser(description='Convert ipynb html files so that the input field are collapsable.')
    parser.add_argument('html_file', type=argparse.FileType('r'), help='ipynb as html file to process.')
    parser.add_argument('--print_category', action='store_true', help='print the category.')
    parser.add_argument('--remove_cell', action='store_true', help='remove the tags cell from the html.')
    args = parser.parse_args()
    return args.html_file, args.print_category, args.remove_cell

def get_tags_category_and_remove(soup, remove_cell):
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
        if remove_cell:
            input_areas[idx + 1].decompose()
            input_areas[idx].decompose()

    return tags, category

def main():
    html_file, print_category, remove_cell = parse_input()
    soup = BeautifulSoup(html_file, 'html.parser', exclude_encodings="ascii")
    tags, category = get_tags_category_and_remove(soup, remove_cell)
    # get category and tags
    if print_category:
        print(category)
    else:
        print(tags)

    html_file.close()
    modifiedHtml = str(soup)
    # overwrite original file
    with open(html_file.name, "w") as file:
        file.write(modifiedHtml)
    file.close()

    html_file.close()

if __name__ == "__main__":
    main()