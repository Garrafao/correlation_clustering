# Correlation Clustering

A python implementation of Correlation Clustering [(Bansal et al., 2004)](https://link.springer.com/article/10.1023/B:MACH.0000033116.57574.95). Correlation Clustering is a weighted graph clustering technique minimizing the sum of cluster disagreements, i.e., the sum of negative edge weights within clusters plus the sum of positive edge weights across clusters. It has some nice properties, e.g.:

- finds number of clusters by itself
- robust to errors by minimizing a global loss
- optimizes an intuitive quality criterion
- our implementation is fast by using multiprocessing

If you use this software for academic research, please [cite](#bibtex) these papers:

- Dominik Schlechtweg, Nina Tahmasebi, Simon Hengchen, Haim Dubossarsky, Barbara McGillivray. 2021. [DWUG: A large Resource of Diachronic Word Usage Graphs in Four Languages](https://aclanthology.org/2021.emnlp-main.567/). In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing.
- Dominik Schlechtweg. 2023. [Human and Computational Measurement of Lexical Semantic Change](http://dx.doi.org/10.18419/opus-12833). PhD thesis. University of Stuttgart.

Find further extensive experiments testing and optimizing this implementation in:

- Benjamin Tunc. [Optimierung von Clustering von Wortverwendungsgraphen](https://elib.uni-stuttgart.de/handle/11682/11923). Bachelor thesis. University of Stuttgart. [slides](https://garrafao.github.io/publications/211201-optimierung-wugs.pdf)

### Usage

We recommend run the code within a [Anaconda virtual environment](https://docs.anaconda.com/) with Python 3.10.8. Install the required packages running `pip install -r requirements.txt`.

Please run the test script with

	python src/test.py


BibTex
--------

```
@inproceedings{Schlechtweg2021dwug,
 title = {{DWUG}: A large Resource of Diachronic Word Usage Graphs in Four Languages},
 author = {Schlechtweg, Dominik  and Tahmasebi, Nina  and Hengchen, Simon  and Dubossarsky, Haim  and McGillivray, Barbara},
 booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
 publisher = {Association for Computational Linguistics},
 address = {Online and Punta Cana, Dominican Republic},
 pages = {7079--7091},
 url = {https://aclanthology.org/2021.emnlp-main.567},
 year = {2021}
}
```
```
@phdthesis{Schlechtweg2023measurement,
  author  = {Schlechtweg, Dominik},
  title   = {Human and Computational Measurement of Lexical Semantic Change},
  school  = {University of Stuttgart},
  address =  {Stuttgart, Germany},
  year    = {2023},
  url = {http://dx.doi.org/10.18419/opus-12833},
  slides = {https://garrafao.github.io/publications/220324-thesis-slides.pdf}
}
```
```
@mastersthesis{Tunc2021OptimierungWUGs,
author = {Benjamin Tunc},
year = {2021}, 
title = {{Optimierung von Clustering von Wortverwendungsgraphen}},
type = {Bachelor thesis},
school = {University of Stuttgart},
slides = {https://garrafao.github.io/publications/211201-optimierung-wugs.pdf},
url = {https://elib.uni-stuttgart.de/handle/11682/11923}
}
```


