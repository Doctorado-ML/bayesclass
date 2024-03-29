SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: coverage deps help lint push test doc build

coverage:  ## Run tests with coverage
	pytest --cov

deps:  ## Install dependencies
	pip install -r requirements.txt

devdeps:  ## Install development dependencies
	pip install black pip-audit flake8 mypy coverage

lint:  ## Lint and static-check
	black bayesclass
	flake8 bayesclass
	mypy bayesclass

push:  ## Push code with tags
	git push && git push --tags

test:  ## Run tests
	python -m doctest bayesclass.clfs.py
	pytest

doc:  ## Update documentation
	make -C doc --makefile=Makefile html

build:  ## Build package
	rm -fr dist/*
	rm -fr build/*
	python setup.py sdist bdist_wheel

doc-clean:  ## Update documentation
	make -C docs --makefile=Makefile clean

audit: ## Audit pip
	pip-audit

version:
	@echo "Current Python version .....: $(shell python --version)"
	@echo "Current Bayesclass version .: $(shell python -c "from bayesclass import _version; print(_version.__version__)")"
	@echo "Installed Bayesclass version: $(shell pip show bayesclass | grep Version | cut -d' ' -f2)"
	@echo "Installed pgmpy version ....: $(shell pip show pgmpy | grep Version | cut -d' ' -f2)"

help: ## Show help message
	@IFS=$$'\n' ; \
	help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%s\n\n" "Usage: make [task]"; \
	printf "%-20s %s\n" "task" "help" ; \
	printf "%-20s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
		IFS=$$':' ; \
		help_split=($$help_line) ; \
		help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		printf '\033[36m'; \
		printf "%-20s %s" $$help_command ; \
		printf '\033[0m'; \
		printf "%s\n" $$help_info; \
	done
