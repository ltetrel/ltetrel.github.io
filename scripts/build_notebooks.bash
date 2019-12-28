#!/bin/bash

nb_list=$(ls notebooks/*.ipynb)

for nb in ${nb_list[*]}; do
	#jupytext --to notebook --execute notebooks/BayesModel.py
	jupyter nbconvert --to html --template basic $nb
	nb_html=${nb%%.*}.html
	#nb_html_copy=${nb%%.*}_copy.html
	if [ -f $nb_html ]; then
		python3 scripts/insertCollapseTags.py $nb_html
		echo "---
$(cat $nb_html)" > $nb_html
		echo "layout: post
$(cat $nb_html)" > $nb_html
		echo "---
$(cat $nb_html)" > $nb_html
		#cat $nb_html_copy >> $nb_html
		#rm $nb_html_copy
		mv $nb_html ./posts/
	fi
done

cp -r notebooks/imgs posts/
