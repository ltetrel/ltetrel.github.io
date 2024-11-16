Install ruby deps

```
sudo apt update
sudo apt install ruby-bundle
```

Remove all previous `Gemfile.lock` files.

Run
```
bundle install
```

Now install jupyter deps:
```
pip install -r build-requirements.txt
python -m ipykernel install --user --name=ltetrel.io
```