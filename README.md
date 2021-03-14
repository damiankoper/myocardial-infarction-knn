# Recognising myocardial infarction using kNN classifier

## Setup
This project uses `pipenv` for python virtual environment. Tu install it run:
```sh 
pip install --user pipenv
```
or if using Debian like OS:
```sh 
sudo apt install pipenv
```
Then to install all required dependencies run:
```
pipenv install
```

## How to run
Enter env shell and run `src/main.py`:
```
pipenv shell
python src/main.py
```

## Docs
Python script generates data that is saved in `docs/src/{img,data}` directory and
automatically included in the Latex main file `docs/src/main.tex`. 
PDF is rebuild on every data change.

If you work with VsCode install
[Latex Workshop extension](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) 
and make sure to have Latex disto (e.g. texlive) and `latexmk` available from your PATH.