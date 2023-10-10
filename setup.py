"""EdgeColoring setup file."""

from setuptools import setup, find_packages

setup(
    name="edge-coloring",
    version="0.0.1",
    description="Edge-coloring bipartite graphs",
    author="Alexander Ivrii",
    author_email="alexi@il.ibm.com",
    packages=find_packages(),
    url="https://github.com/alexanderivrii/edge_coloring",
    keywords="Graphs, Coloring, Bipartite",
    install_requires=["rustworkx>=0.13.0"],
    python_requires=">=3.7",
)
