@echo off

for %%s in (Test3Dline.ipynb) do (
	jupyter nbconvert --to html --template basic %%s 
	if exist %%~ns.html (
		ren %%~ns.html %%~ns_copy.html
		REM python insertCollapseTags.py %%~ns_copy.html 4 > %%~ns_copy.html
		echo --- >> %%~ns.html
		echo layout: default >> %%~ns.html
		echo --- >> %%~ns.html
		type %%~ns_copy.html >> %%~ns.html
		del %%~ns_copy.html
		robocopy %cd% ..\posts %%~ns.html /mov /nfl /ndl /njh /njs /nc /ns /np
		)
)

pause