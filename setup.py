import pathlib
from setuptools import find_packages, setup

def parse_requirements(filename):
    with pathlib.Path(filename).open() as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

install_requires = parse_requirements("requirements.txt")

long_description = pathlib.Path("README.md").read_text(encoding="utf8")

setup(
    name="pysortgui",
    version="0.4.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'pysortgui=pysortgui.main:launch_app',
            'pysortgui-cli=pysortgui.plugins.cli:launch_cli'
        ],
    },
    install_requires=install_requires,
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        'pysortgui': ['External/bins/*',
                      'UI/style.qss',
                      #   'plugins/*.*',
                      #   'datafun/*.*',
                      #   'icons/*.*'
                      ],
    },

    # metadata for upload to PyPI
    author="Harry Wang",
    author_email="harry.wang0210@gmail.com",
    description="This package provides a spike sorter for neural data",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/HarryWang0210/PySortGUI",
    license="MIT",
    keywords="ephys neurons spike sorting",
    python_requires='>=3.6',
    # url="http://example.com/HelloWorld/",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)
