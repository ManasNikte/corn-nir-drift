## Calibration transfer, master m5, preprocessing=default

| slave   | method             |   n_standards |   moisture |     oil |   protein |   starch |
|:--------|:-------------------|--------------:|-----------:|--------:|----------:|---------:|
| mp5     | DS                 |             3 |     0.4339 | 0.4004  |    0.5363 |   1.058  |
| mp5     | DS                 |             5 |     0.2624 | 0.1322  |    0.5345 |   0.9088 |
| mp5     | DS                 |             8 |     0.1792 | 0.1746  |    0.4907 |   0.9869 |
| mp5     | DS                 |            10 |     0.2598 | 0.1652  |    0.5089 |   0.9554 |
| mp5     | DS                 |            15 |     0.1956 | 0.2006  |    0.1935 |   0.4728 |
| mp5     | DS                 |            20 |     0.1442 | 0.1097  |    0.184  |   0.2876 |
| mp5     | DS                 |            30 |     0.134  | 0.1199  |    0.1273 |   0.2923 |
| mp5     | PDS                |             3 |     0.1867 | 0.1293  |    0.272  |   0.5358 |
| mp5     | PDS                |             5 |     0.1715 | 0.1083  |    0.2071 |   0.396  |
| mp5     | PDS                |             8 |     0.1854 | 0.102   |    0.1816 |   0.3386 |
| mp5     | PDS                |            10 |     0.1816 | 0.1038  |    0.1811 |   0.3441 |
| mp5     | PDS                |            15 |     0.1609 | 0.1048  |    0.1983 |   0.3262 |
| mp5     | PDS                |            20 |     0.1617 | 0.1011  |    0.183  |   0.3245 |
| mp5     | PDS                |            30 |     0.1566 | 0.1033  |    0.1836 |   0.3251 |
| mp5     | SBC                |             3 |     0.6149 | 0.1266  |    0.3787 |   2.055  |
| mp5     | SBC                |             5 |     0.2867 | 0.1225  |    0.1919 |   2.19   |
| mp5     | SBC                |             8 |     0.2469 | 0.1058  |    0.2307 |   0.4806 |
| mp5     | SBC                |            10 |     0.2378 | 0.1139  |    0.2143 |   0.5025 |
| mp5     | SBC                |            15 |     0.2214 | 0.1013  |    0.1839 |   0.3128 |
| mp5     | SBC                |            20 |     0.2075 | 0.09385 |    0.1752 |   0.3099 |
| mp5     | SBC                |            30 |     0.2109 | 0.08691 |    0.1751 |   0.3081 |
| mp5     | master (reference) |             0 |     0.1339 | 0.05141 |    0.1092 |   0.1254 |
| mp5     | none               |             0 |     1.72   | 0.6338  |    1.053  |   0.9812 |
| mp6     | DS                 |             3 |     0.4275 | 0.2786  |    0.5026 |   0.9134 |
| mp6     | DS                 |             5 |     0.2412 | 0.1672  |    0.5412 |   0.8912 |
| mp6     | DS                 |             8 |     0.2293 | 0.2027  |    0.3537 |   0.6844 |
| mp6     | DS                 |            10 |     0.3358 | 0.1923  |    0.4225 |   0.7062 |
| mp6     | DS                 |            15 |     0.194  | 0.1881  |    0.2541 |   0.3818 |
| mp6     | DS                 |            20 |     0.149  | 0.1072  |    0.1581 |   0.2902 |
| mp6     | DS                 |            30 |     0.1368 | 0.1056  |    0.1143 |   0.2836 |
| mp6     | PDS                |             3 |     0.2519 | 0.1309  |    0.2583 |   0.586  |
| mp6     | PDS                |             5 |     0.2284 | 0.1098  |    0.1831 |   0.4161 |
| mp6     | PDS                |             8 |     0.2704 | 0.1061  |    0.1529 |   0.371  |
| mp6     | PDS                |            10 |     0.2475 | 0.1064  |    0.1535 |   0.3695 |
| mp6     | PDS                |            15 |     0.2238 | 0.1057  |    0.1713 |   0.3416 |
| mp6     | PDS                |            20 |     0.221  | 0.1048  |    0.1644 |   0.3292 |
| mp6     | PDS                |            30 |     0.2122 | 0.1066  |    0.169  |   0.3303 |
| mp6     | SBC                |             3 |     0.4681 | 0.1047  |    0.3561 |   1.703  |
| mp6     | SBC                |             5 |     0.2877 | 0.1208  |    0.1864 |   1.133  |
| mp6     | SBC                |             8 |     0.2887 | 0.1269  |    0.2089 |   0.461  |
| mp6     | SBC                |            10 |     0.2716 | 0.1445  |    0.1965 |   0.4798 |
| mp6     | SBC                |            15 |     0.2657 | 0.1158  |    0.1687 |   0.3007 |
| mp6     | SBC                |            20 |     0.2469 | 0.09878 |    0.1629 |   0.2939 |
| mp6     | SBC                |            30 |     0.2529 | 0.09141 |    0.1644 |   0.2965 |
| mp6     | master (reference) |             0 |     0.1339 | 0.05141 |    0.1092 |   0.1254 |
| mp6     | none               |             0 |     2.389  | 0.398   |    1.532  |   1.42   |

## Fraction of test spectra flagged by drift detector

| slave   | preprocess          | method   |   n_standards |   flagged |
|:--------|:--------------------|:---------|--------------:|----------:|
| mp5     | base:poly+snv+sg-d1 | none     |             0 |      1    |
| mp5     | base:poly+snv+sg-d1 | PDS      |             3 |      0.15 |
| mp5     | base:poly+snv+sg-d1 | DS       |             3 |      0.7  |
| mp5     | base:poly+snv+sg-d1 | PDS      |             5 |      0.05 |
| mp5     | base:poly+snv+sg-d1 | DS       |             5 |      0    |
| mp5     | base:poly+snv+sg-d1 | PDS      |             8 |      0.05 |
| mp5     | base:poly+snv+sg-d1 | DS       |             8 |      0    |
| mp5     | base:poly+snv+sg-d1 | PDS      |            10 |      0    |
| mp5     | base:poly+snv+sg-d1 | DS       |            10 |      0.05 |
| mp5     | base:poly+snv+sg-d1 | PDS      |            15 |      0    |
| mp5     | base:poly+snv+sg-d1 | DS       |            15 |      0    |
| mp5     | base:poly+snv+sg-d1 | PDS      |            20 |      0    |
| mp5     | base:poly+snv+sg-d1 | DS       |            20 |      0    |
| mp5     | base:poly+snv+sg-d1 | PDS      |            30 |      0    |
| mp5     | base:poly+snv+sg-d1 | DS       |            30 |      0    |
| mp6     | base:poly+snv+sg-d1 | none     |             0 |      1    |
| mp6     | base:poly+snv+sg-d1 | PDS      |             3 |      0.15 |
| mp6     | base:poly+snv+sg-d1 | DS       |             3 |      0.4  |
| mp6     | base:poly+snv+sg-d1 | PDS      |             5 |      0    |
| mp6     | base:poly+snv+sg-d1 | DS       |             5 |      0.05 |
| mp6     | base:poly+snv+sg-d1 | PDS      |             8 |      0.05 |
| mp6     | base:poly+snv+sg-d1 | DS       |             8 |      0.05 |
| mp6     | base:poly+snv+sg-d1 | PDS      |            10 |      0    |
| mp6     | base:poly+snv+sg-d1 | DS       |            10 |      0.15 |
| mp6     | base:poly+snv+sg-d1 | PDS      |            15 |      0    |
| mp6     | base:poly+snv+sg-d1 | DS       |            15 |      0    |
| mp6     | base:poly+snv+sg-d1 | PDS      |            20 |      0    |
| mp6     | base:poly+snv+sg-d1 | DS       |            20 |      0    |
| mp6     | base:poly+snv+sg-d1 | PDS      |            30 |      0    |
| mp6     | base:poly+snv+sg-d1 | DS       |            30 |      0    |
| m5      | base:poly+snv+sg-d1 | none     |             0 |      0    |
