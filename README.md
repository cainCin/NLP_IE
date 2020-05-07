# Applications of NLP approaches in Information Extraction
--------------------
|Summary ||
|:-|:-|
|Topic| Applications of NLP approaches in Information Extraction|
|Members| Cain|
|Research period| 4/5/2020 - 7/2020|
--------------------
#### Deliverables:
- [Benchmark](#benchmark)
- [Code]()


## Scopes
|-|Descriptions|
|:--|:--|
|Dataset| Invoice|
|Encoder| BoW, Word2Vec, FastText, BERT|
|Classifier| SVM, CNN|

## <a name="benchmark"> Benchmark </a>
|Encoder + Classifier\ Dataset| Invoice |
|:--------|:--:|
|[Baseline (BoW + SVM)](#bowsvm)| 0.6 |
| [Word2Vec + SVM](#w2vsvm)| 0.93 |
| [FastText + SVM](#ftsvm)| 0.94 |
| [BERT + SVM](#bertsvm)| 0.95 |


# EXPERIMENTS
## PRELIMILARY EXPERIMENTS 
### <a name="dataset"> Dataset details </a>
|Dataset| ||
|:-|:-|:-|
|Invoice| | |
|Trainset | DATAPILE v10. (835 files)|| 
|Testset  | DATAPILE v13. (338 files)||
|FULL| Fields| Notes|
|| - 'account_name' <br> - 'account_number' <br> - 'account_type' <br> - 'amount_excluding_tax' <br> - 'amount_including_tax' <br> - 'bank_name' <br> - 'branch_name'<br> - 'company_address' <br> -'company_department_name' <br> - 'company_fax' <br> - 'company_name' <br> - 'company_tel' <br> - 'company_zipcode' <br> - 'delivery_date' <br> - 'document_number' <br> - 'invoice_number' <br> - 'issued_date' <br> - 'item_line_number' <br> - 'item_name' <br> - 'item_quantity' <br> - 'item_quantity_item_unit' <br> - 'item_total_excluding_tax' <br> - 'item_total_including_tax' <br> - 'item_unit' <br> - 'item_unit_amount' <br> - 'payment_date' <br> - 'tax'| 25 acquired fields from Invoice project|
|PREMILARY CONFIG|||
|Classes| - name <br> - date <br> - number <br> - amount <br> - address <br> - UNKNOWN | Selected with the help of Kelly from QA team for *mostly accquired* in recently projects|

SVM configuration:
- Default SVM from sklearn with class balanced weight enabled



#### <a name="bowsvm"> BoW + SVM </a>
Here are the accuracy report and the confusion matrix for this configuration.
|ACCURACY REPORT (Invoice)|precision|recall|f1-score|support|
|:--|:-:|:-:|:-:|:-:|
|     UNKNOWN|       0.22|      1.00|      0.36|       971|
|     address|       0.71|      0.72|      0.71|       185|
|      amount|       1.00|      0.43|      0.60|      2166|
|        date|       0.97|      0.83|      0.89|       734|
|        name|       1.00|      0.37|      0.54|      2154|
|      number|       0.93|      0.24|      0.38|      1110|
|    accuracy|||                           0.51|      7320|
|   macro avg|       0.80|      0.60|      0.58|      7320|
|weighted avg|       0.87|      0.51|      0.55|      7320|

[[ 967    0    0    0    0    4]
 [  51  134    0    0    0    0]
 [1221    0  939    0    0    6]
 [ 125    0    0  607    2    0]
 [1266   56    1   21  799   11]
 [ 839    0    1    0    2  268]]

-------
#### Discussion:
- The classification rate is not good at all with this configuration (0.6)
- Only date is well recognized based on the mechasism of BoW (the existence of date characters as special symbols)
- It totally fails to catch the quantity and type classes, which could only recognize as the number or characters

####  <a name="w2vsvm"> Word2Vec + SVM </a>

|ACCURACY REPORT (Invoice)|precision|recall|f1-score|support|
|:--|:-:|:-:|:-:|:-:|
|UNKNOWN|       0.94|      0.96|      0.95|       971|
|address|       0.74|      0.97|      0.84|       185|
|amount|        0.99|      0.89|      0.94|      2166|
|date|          0.98|      0.89|      0.93|       734|
|name|          0.98|      0.96|      0.97|      2154|
|number|        0.77|      0.94|      0.85|      1110|
|accuracy|||                          0.93|      7320|
|macro avg|     0.90|      0.94|      0.91|      7320|
|weighted avg|  0.94|      0.93|      0.93|      7320|

|CONFUSION MATRIX|UNKNOWN|address|amount|date|name|number|
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
|UNKNOWN|    937|    0|    4|    1|   26|    3|
|address|      0|  179|    0|    0|    6|    0|
|amount|      14|    0| 1934|    1|   10|  207|
|date|         0|    0|    0|  651|    7|   76|
|name|         4|   55|    0|    7| 2058|   30|
|number|      43|    9|   10|    2|    0| 1046|

#### <a name="ftsvm"> FastText + SVM </a>

|ACCURACY REPORT (Invoice)|precision|recall|f1-score|support|
|:--|:-:|:-:|:-:|:-:|
|     UNKNOWN|       0.97|      0.95|      0.96|       971|
|     address|       0.88|      0.91|      0.90|       185|
|      amount|       0.99|      0.90|      0.94|      2166|
|        date|       0.98|      0.88|      0.93|       734|
|        name|       0.96|      0.98|      0.97|      2154|
|      number|       0.78|      0.96|      0.86|      1110|
|    accuracy|||                           0.94|      7320|
|   macro avg|       0.93|      0.93|      0.93|      7320|
|weighted avg|       0.94|      0.94|      0.94|      7320|

|CONFUSION MATRIX|UNKNOWN|address|amount|date|name|number|
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
|UNKNOWN    | 922|    0|    0|    1|   45|    3|
|address    |   0|  169|    0|    0|   16|    0|
|amount     |   8|    2| 1949|    1|    8|  198|
|date       |   1|    0|    0|  644|   13|   76|
|name       |   1|   21|    0|    3| 2101|   28|
|number     |  18|    0|   11|    6|   11| 1064|

#### <a name="bertsvm"> BERT + SVM </a>

|ACCURACY REPORT (Invoice)|precision|recall|f1-score|support|
|:--|:-:|:-:|:-:|:-:|
|     UNKNOWN|       0.95|      0.98|      0.97|       971|
|     address|       0.91|      0.94|      0.93|       185|
|      amount|       0.97|      0.94|      0.96|      2166|
|        date|       1.00|      0.88|      0.94|       734|
|        name|       0.97|      0.96|      0.96|      2154|
|      number|       0.84|      0.94|      0.89|      1110|
|    accuracy|||                           0.95|      7320|
|   macro avg|       0.94|      0.94|      0.94|      7320|
|weighted avg|       0.95|      0.95|      0.95|      7320|


|CONFUSION MATRIX|UNKNOWN|address|amount|date|name|number|
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
|UNKNOWN    | 951|    0|    0|    0|   16|    4|
|address    |   0|  174|    0|    0|   11|    0|
|amount     |   6|    0| 2043|    0|   21|   96|
|date       |   0|    0|   19|  646|   12|   57|
|name       |  28|   17|    2|    0| 2069|   38|
|number     |  12|    0|   43|    0|   15| 1040|




------
#### Discussion:
- A great improvement with this configuration (95% acc)
- Most of classes are satisfied (> 90% acc)
- There is still a confusion between the amount with the unclassified. Actually, in this preliminary experiments, we extract only 5 common classes, and there is still some unconsidered classes with the same pattern (for ex. tax in the unclassified).

#### VISUALIZATION

|CONFIGURATION||
|:-|:-:|
|ENCODER| Word2Vec|
|CLASSIFIER| SVM|
|CLASS|-"name" <br> - "date" <br> - "type" <br> - "quantity" <br> - "amount" <br> - "tel" <br> - "zipcode" <br> - "address" <br> - "unit" <br> - "number"|

<p>
    <img src="demo.png" alt="【TIS様】Pitch Tokyo請求書 (Aniwo)_0"/>
    <br>
    <em>Fig. 1: Visualiztion of NLP approaches in Invoice data sample </em>
</p>

<p>
    <img src="2018030500875_558404_04_20180305l0015000550_1.png.png" alt="2018030500875_558404_04_20180305l0015000550_1.png"/>
    <br>
    <em>Fig. 2: Visualiztion of NLP approaches in Sumitomo data sample </em>
</p>

<p>
    <img src="◆トーホー.tif.png" alt="◆トーホー.tif"/>
    <br>
    <em>Fig. 3: Visualiztion of NLP approaches in a TA data sample </em>
</p>
