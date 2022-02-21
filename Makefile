PYTHON=python
python_version_full := $(wordlist 2,4,$(subst ., ,$(shell ${PYTHON} --version 2>&1)))
python_version_major := $(word 1,${python_version_full})
PIP=pip
vv=$(shell $(PYTHON) setup.py get_version )

all: build

build :
	$(PYTHON) setup.py build

install:
	$(PIP) install .

install-user:
	$(PIP) install . --user

clean:
	$(PYTHON) setup.py clean -a

sdist:
	$(PYTHON) setup.py sdist

check: build
	pip install --no-build-isolation --no-deps --no-cache --upgrade --target tests/pysndfile_inst_dir .
	touch tests/pysndfile_inst_dir/__init__.py
	cd tests; $(PYTHON) pysndfile_test.py
