#!/bin/bash

if [ "$1" != "" ]; then
    py_files=$1
else
    py_files=*
fi

nb_list=$(ls notebooks/$py_files.py)

echo $nb_list

for nb in ${nb_list[*]}; do
	jupytext --to notebook --execute $nb
	jupyter nbconvert --to html --template basic ${nb%%.*}.ipynb
	rm ${nb%%.*}.ipynb
	nb_html=${nb%%.*}.html
	if [ -f $nb_html ]; then
		python3 scripts/insertCollapseTags.py $nb_html
		echo "---
$(cat $nb_html)" > $nb_html
		echo "layout: post
$(cat $nb_html)" > $nb_html
		echo "---
$(cat $nb_html)" > $nb_html
		mv $nb_html ./posts/
	fi
done

cp -r notebooks/imgs posts/
rm -r posts/iframe_figures
mv notebooks/iframe_figures posts/
