#!/bin/bash

if [ "$1" != "" ]; then
    nb_list=$(basename -a $1)
else
    nb_list=$(basename -a $(ls notebooks/*.py))
fi

for nb in ${nb_list[*]}; do
	nb_filepath="notebooks/$nb"
	jupytext --to notebook --execute $nb_filepath
	jupyter nbconvert --to html --template basic ${nb_filepath%%.*}.ipynb
	mv ${nb_filepath%%.*}.ipynb notebooks/ipynb/
	html=${nb%%.*}.html
	html_filepath="notebooks/$html"
	if [ -f $html_filepath ]; then
		# copying iframe folder and deletion of previous iframes (if exists)
		if [ -d "notebooks/iframe_figures" ]; then
			dir_iframe="assets/iframes/${nb%%.*}"
			mkdir -p $dir_iframe
			mv notebooks/iframe_figures/* $dir_iframe
			rm -r notebooks/iframe_figures
			# replacing iframes paths
			sed -i "s|src=\"iframe_figures|src=\"/assets/iframes/${nb%%.*}|" $html_filepath
		fi
		# replacing image path in html
		sed -i "s|src=\"imgs|src=\"/notebooks/imgs/|" $html_filepath
		# replacing data paths in ipynb files for binder
		sed -i "s|data/|/notebooks/data/|" notebooks/ipynb/${nb%%.*}.ipynb
		# if [ ! -d "assets/imgs/${nb%%.*}" ]; then
		# 	cd assets/imgs/
		# 	ln -s ../../_notebooks/imgs/${nb%%.*} ${nb%%.*}
		# 	cd ../../
			# cp -r _notebooks/imgs/${nb%%.*} assets/imgs/
		# fi
		# linking python file to asset dir (to enable download)
		# if [ ! -f "assets/notebooks/${nb%%.*}.py" ]; then
		# 	cd assets/notebooks/
		# 	ln -s ../../_notebooks/${nb%%.*}.py ${nb%%.*}.py
		# 	cd ../../
		# 	# cp $nb_filepath assets/notebooks/
		# fi
		# inserting collapse icons, managing references, title, tags in front matter
		python3 scripts/html_parser.py $html_filepath
		# moving the updated html
		mv $html_filepath _posts/2000-01-01-$html
        echo "------"
	fi
done
