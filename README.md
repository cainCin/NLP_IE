# Applications of NLP approaches in Information Extraction

## Scopes
|-|Descriptions|
|:--|:--|
|Dataset| Invoice|
|Encoder| BoW, Word2Vec, FastText, BERT|
|Classifier| SVM, CNN|

## Benchmark

|Encoder + Classifier\ Dataset| Invoice |
|:--------|:--:|
|Baseline (BoW + SVM)| 0.74 |
| (Word2Vec + SVM)| - |
| (FastText + SVM)| - |
| (BERT + SVM)| 0.98 |


## EXPERIMENTS

#### BoW + SVM

ACCURACY REPORT

||precision|recall|f1-score|support|
|:--|:-:|:-:|:-:|:-:|
|amount|1.00|0.63|0.77|464|
|date|0.95|0.82|0.88|339|
|name|0.64|0.99|0.78|953|
|quantity|1.00|0.01|0.02|131|
|type|0.00|0.00|0.00|159|
|accuracy|||0.74|2046|
|macro avg|0.72|0.49|0.49|2046|
|weighted avg|0.75|0.74|0.68|2046|

|CONFUSION MATRIX|||||
|:-:|:-:|:-:|:-:|:-:|
|292|0|172|0|0|
|0|277|62|0|0|
|0|14|939|0|0|
|1|0|129|1|0|
|0|0|159|0|0|

#### BERT + SVM
||precision|recall|f1-score|support|
|:-|:-:|:-:|:-:|:-:|
|amount|0.99|0.98|0.98|484|
|date|1.00|0.99|1.00|314|
|name|0.98|0.98|0.98|957|
|quantity|0.94|0.94|0.94|125|
|type|0.93|0.98|0.96|166|
|accuracy|||0.98|2046|
|macro avg|0.97|0.98|0.97|2046|
|weighted avg|0.98|0.98|0.98|2046|

CONFUSION MATRIX
||||||
|:-:|:-:|:-:|:-:|:-:|
|472|0|9|2|1|
|0|312|2|0|0|
|0|0|941|5|11|
|4|0|3|118|0|
|0|0|3|0|163|