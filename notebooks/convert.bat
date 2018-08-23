@echo off

for %%s in (likelihood.ipynb) do (
	jupyter nbconvert --to html --template basic %%s 
	if exist %%~ns.html (
		ren %%~ns.html %%~ns_copy.html
		python insertCollapseTags.py %%~ns_copy.html
		echo --- >> %%~ns.html
		echo layout: post >> %%~ns.html
		echo --- >> %%~ns.html
		type %%~ns_copy.html >> %%~ns.html
		del %%~ns_copy.html
		robocopy %cd% ..\posts %%~ns.html /mov /nfl /ndl /njh /njs /nc /ns /np
		)
)

pause