.PHONY: rufflint ruffformat runisort codeimprove

rufflint:
	ruff check mlalgo --fix

ruffformat:
	ruff format mlalgo

runisort:
	isort mlalgo

codeimprove: ruffformat rufflint runisort


