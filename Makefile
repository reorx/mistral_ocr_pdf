.PHONY: build publish clean

build:
	python -m build

publish: build
	python -m twine upload dist/*

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +