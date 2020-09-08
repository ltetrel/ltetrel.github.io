import sys
import argparse
import re
from bs4 import BeautifulSoup


def parse_input():
    """Parse the input from the command line."""
    parser = argparse.ArgumentParser(description='Convert ipynb html files so that the input field are collapsable.')
    parser.add_argument('html_file', type=argparse.FileType('r'), help='ipynb as html file to process.')
    args = parser.parse_args()
    return args.html_file

def get_title_and_remove(soup):
    """Insert the collapse buttons on the code input field of the nb."""
    input_area = soup.find_all(["h1"])[0]
    title = input_area.text.strip()[:-1].strip()
    input_area = soup.select('div.cell')[0]
    input_area.decompose()
    
    return title

def main():
    html_file = parse_input()
    soup = BeautifulSoup(html_file, 'html.parser', exclude_encodings="ascii")
    print(get_title_and_remove(soup))

    html_file.close()
    modifiedHtml = str(soup)
    # overwrite original file
    with open(html_file.name, "w") as file:
        file.write(modifiedHtml)
    file.close()

    html_file.close()

if __name__ == "__main__":
    main()