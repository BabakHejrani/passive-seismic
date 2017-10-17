.PHONY: help clean clean-pyc clean-build list test test-all coverage docs release sdist

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"

clean: clean-build clean-pyc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

docs:
	rm -f docs/seismic.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ seismic
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	open docs/_build/html/index.html

lint:
	pytest --junit-xml=test_output/flake8/results.xml \
	    --flake8 -p no:regtest	--cache-clear seismic

test:
	pytest --junit-xml=test_output/pytest/results.xml --cache-clear

coverage:
	mpirun --allow-run-as-root -n 2 pytest tests/test_pyasdf.py
	pytest --junit-xml=test_output/pytest/results.xml --cov \
	    --cov-report=html:test_output/coverage --cov-fail-under=50 \
	    --cache-clear ./tests
