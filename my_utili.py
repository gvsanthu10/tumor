import numpy as np


positive = np.array([[0.015, 0.015, 0.015, 0.015, 0.06 , 0.06 , 0.76 , 0.25 , 0.01 ,
        0.56 , 0.46 , 0.11 , 0.36 , 0.31 , 0.96 , 0.96 , 0.36 , 0.21 ,
        0.75 , 0.01 , 0.01 , 0.01 , 0.01 ],
       [0.085, 0.085, 0.085, 0.085, 0.56 , 0.56 , 0.46 , 0.7  , 0.01 ,
        0.81 , 0.86 , 0.985, 0.96 , 0.86 , 0.26 , 0.76 , 0.96 , 0.91 ,
        0.9  , 0.01 , 1.01 , 0.035, 0.05 ],
       [0.61 , 0.11 , 0.085, 0.035, 0.085, 0.085, 0.035, 0.035, 1.01 ,
        0.06 , 0.01 , 0.06 , 0.01 , 0.01 , 0.01 , 0.01 , 0.06 , 0.46 ,
        0.01 , 1.01 , 1.01 , 0.96 , 1.01 ],
       [0.91 , 0.81 , 0.06 , 0.03 , 0.26 , 0.61 , 0.06 , 0.06 , 0.96 ,
        0.01 , 0.02 , 0.06 , 0.01 , 0.06 , 0.01 , 0.01 , 0.01 , 0.11 ,
        0.06 , 1.01 , 0.01 , 0.96 , 0.45 ],
       [0.26 , 0.56 , 0.96 , 0.985, 0.56 , 0.31 , 0.96 , 0.96 , 0.16 ,
        0.56 , 0.06 , 0.31 , 0.06 , 0.41 , 0.01 , 0.01 , 0.01 , 0.96 ,
        0.65 , 0.06 , 0.31 , 0.31 , 0.66 ],
       [0.46 , 0.16 , 0.31 , 0.21 , 0.06 , 0.06 , 0.01 , 0.01 , 0.06 ,
        0.01 , 0.01 , 0.085, 0.01 , 0.03 , 0.01 , 0.01 , 0.01 , 0.01 ,
        0.11 , 0.05 , 0.125, 0.125, 0.05 ],
       [0.035, 0.91 , 0.21 , 0.91 , 0.91 , 0.61 , 0.26 , 0.26 , 0.21 ,
        0.46 , 0.01 , 0.11 , 0.06 , 0.06 , 0.01 , 0.01 , 0.01 , 0.46 ,
        0.26 , 0.01 , 0.01 , 0.01 , 0.01 ],
       [0.21 , 0.71 , 0.08 , 0.06 , 0.46 , 0.01 , 0.11 , 0.11 , 0.21 ,
        0.085, 0.01 , 0.06 , 0.01 , 0.06 , 0.035, 0.035, 0.01 , 0.01 ,
        0.95 , 0.36 , 0.26 , 0.825, 0.16 ],
       [0.02 , 0.02 , 0.02 , 0.02 , 0.01 , 0.085, 0.76 , 0.76 , 0.01 ,
        0.01 , 0.085, 0.035, 0.01 , 0.26 , 0.935, 0.36 , 0.01 , 0.01 ,
        0.01 , 0.01 , 0.01 , 0.01 , 0.05 ],
       [0.16 , 0.16 , 0.21 , 0.21 , 0.01 , 0.01 , 0.91 , 0.91 , 0.46 ,
        0.46 , 0.46 , 0.06 , 0.01 , 0.01 , 0.61 , 0.06 , 0.01 , 0.01 ,
        0.01 , 0.035, 0.01 , 0.01 , 0.01 ],
       [0.06 , 0.16 , 0.11 , 0.16 , 0.31 , 0.46 , 0.06 , 0.2  , 0.035,
        0.31 , 0.56 , 0.11 , 0.56 , 0.41 , 0.01 , 0.01 , 0.01 , 0.085,
        0.125, 0.02 , 0.96 , 1.01 , 0.05 ],
       [0.02 , 0.02 , 0.02 , 0.02 , 0.36 , 0.11 , 0.01 , 0.01 , 0.01 ,
        0.01 , 0.085, 0.02 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 ,
        0.01 , 0.01 , 0.15 , 0.985, 0.05 ],
       [0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.96 ,
        0.36 , 0.01 , 0.02 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 ,
        0.01 , 0.01 , 0.01 , 0.01 , 0.01 ],
       [0.1  , 0.5  , 0.1  , 0.65 , 0.8  , 0.16 , 0.1  , 0.8  , 0.1  ,
        0.21 , 0.45 , 0.16 , 0.16 , 0.17 , 0.1  , 0.1  , 0.1  , 0.1  ,
        0.55 , 0.1  , 0.3  , 0.35 , 0.95 ],
       [0.01 , 0.01 , 0.01 , 0.01 , 0.39 , 0.01 , 0.01 , 0.01 , 0.01 ,
        0.01 , 0.01 , 0.01 , 0.01 , 0.41 , 0.01 , 0.01 , 0.01 , 0.01 ,
        0.01 , 0.01 , 0.01 , 0.01 , 0.03 ]])

negative = np.array([[0.99371463, 0.99371463, 0.99371463, 0.99371463, 0.9749774 ,
        0.9749774 , 0.70647319, 0.8978308 , 0.99580755, 0.77878528,
        0.81626235, 0.95436743, 0.8546201 , 0.87412922, 0.6376838 ,
        0.6376838 , 0.8546201 , 0.91380799, 0.71000513, 0.99580755,
        0.99580755, 0.99580755, 0.99580755],
       [0.98155844, 0.98155844, 0.98155844, 0.98155844, 0.88166211,
        0.88166211, 0.90224751, 0.85324165, 0.9978215 , 0.83123792,
        0.82133125, 0.79682439, 0.80169606, 0.82133125, 0.94413095,
        0.84120398, 0.80169606, 0.81148396, 0.81344867, 0.9978215 ,
        0.79196756, 0.99238563, 0.98913124],
       [0.7035766 , 0.94270545, 0.95557853, 0.98158659, 0.95557853,
        0.95557853, 0.98158659, 0.98158659, 0.5374156 , 0.96853891,
        0.99472156, 0.96853891, 0.99472156, 0.99472156, 0.99472156,
        0.99472156, 0.96853891, 0.7716487 , 0.99472156, 0.5374156 ,
        0.5374156 , 0.55696354, 0.5374156 ],
       [0.59393091, 0.63340958, 0.9699872 , 0.98493644, 0.87324707,
        0.71617751, 0.9699872 , 0.9699872 , 0.5746679 , 0.99496611,
        0.98994493, 0.9699872 , 0.99496611, 0.9699872 , 0.99496611,
        0.99496611, 0.99496611, 0.94532584, 0.9699872 , 0.55572245,
        0.99496611, 0.5746679 , 0.78605003],
       [0.94321701, 0.87976042, 0.79858872, 0.79364591, 0.87976042,
        0.93248747, 0.79858872, 0.79858872, 0.96486022, 0.87976042,
        0.98674893, 0.93248747, 0.98674893, 0.91121252, 0.99778535,
        0.99778535, 0.99778535, 0.79858872, 0.86115431, 0.98674893,
        0.93248747, 0.93248747, 0.85909924],
       [0.82726499, 0.93806204, 0.88179341, 0.91911248, 0.97654124,
        0.97654124, 0.99607087, 0.99607087, 0.97654124, 0.99607087,
        0.99607087, 0.96684893, 0.99607087, 0.98823582, 0.99607087,
        0.99607087, 0.99607087, 0.99607087, 0.95720496, 0.9804317 ,
        0.95144178, 0.95144178, 0.9804317 ],
       [0.98488028, 0.6443193 , 0.91100932, 0.6443193 , 0.6443193 ,
        0.75297379, 0.8904322 , 0.8904322 , 0.91100932, 0.81047422,
        0.99566833, 0.95286872, 0.974151  , 0.974151  , 0.99566833,
        0.99566833, 0.99566833, 0.81047422, 0.8904322 , 0.99566833,
        0.99566833, 0.99566833, 0.99566833],
       [0.91335946, 0.7228714 , 0.96653124, 0.97484502, 0.81533392,
        0.99578525, 0.95412731, 0.95412731, 0.91335946, 0.96445835,
        0.99578525, 0.97484502, 0.99578525, 0.97484502, 0.98528732,
        0.98528732, 0.99578525, 0.99578525, 0.63934106, 0.85387657,
        0.89330931, 0.6822067 , 0.93363212],
       [0.98572878, 0.98572878, 0.98572878, 0.98572878, 0.99285157,
        0.94005568, 0.52979837, 0.52979837, 0.99285157, 0.99285157,
        0.94005568, 0.97509268, 0.99285157, 0.82247441, 0.44250642,
        0.75881085, 0.99285157, 0.99285157, 0.99285157, 0.99285157,
        0.99285157, 0.99285157, 0.96451427],
       [0.92047479, 0.92047479, 0.89629879, 0.89629879, 0.99493316,
        0.99493316, 0.5916161 , 0.5916161 , 0.78024469, 0.78024469,
        0.78024469, 0.96979198, 0.99493316, 0.99493316, 0.71447295,
        0.96979198, 0.99493316, 0.99493316, 0.99493316, 0.98232235,
        0.99493316, 0.99493316, 0.99493316],
       [0.97912086, 0.9448118 , 0.96188985, 0.9448118 , 0.89449547,
        0.84555586, 0.97912086, 0.9312595 , 0.98779373, 0.89449547,
        0.81369429, 0.96188985, 0.81369429, 0.86171609, 0.99650485,
        0.99650485, 0.99650485, 0.97048623, 0.95675037, 0.99301581,
        0.69236678, 0.6778892 , 0.98258542],
       [0.98306233, 0.98306233, 0.98306233, 0.98306233, 0.71725639,
        0.90863308, 0.99151308, 0.99151308, 0.99151308, 0.99151308,
        0.92901401, 0.98306233, 0.99151308, 0.99151308, 0.99151308,
        0.99151308, 0.99151308, 0.99151308, 0.99151308, 0.99151308,
        0.87649377, 0.33771032, 0.95792707],
       [0.9886114 , 0.9886114 , 0.9886114 , 0.9886114 , 0.9886114 ,
        0.9886114 , 0.9886114 , 0.9886114 , 0.20410681, 0.63110025,
        0.9886114 , 0.97728802, 0.9886114 , 0.9886114 , 0.9886114 ,
        0.9886114 , 0.9886114 , 0.9886114 , 0.9886114 , 0.9886114 ,
        0.9886114 , 0.9886114 , 0.9886114 ],
       [0.97932122, 0.89876653, 0.97932122, 0.86944972, 0.840619  ,
        0.96701765, 0.97932122, 0.840619  , 0.97932122, 0.95682408,
        0.90864683, 0.96701765, 0.96701765, 0.96497461, 0.97932122,
        0.97932122, 0.97932122, 0.97932122, 0.88894025, 0.97932122,
        0.93861178, 0.92856945, 0.04      ],
       [0.99110164, 0.99110164, 0.99110164, 0.99110164, 0.68243162,
        0.99110164, 0.99110164, 0.99110164, 0.99110164, 0.99110164,
        0.99110164, 0.99110164, 0.99110164, 0.66777654, 0.99110164,
        0.99110164, 0.99110164, 0.99110164, 0.99110164, 0.99110164,
        0.99110164, 0.99110164, 0.97342421]])

final_labels = ['Serous cyst adenoma: ','Serous cyst adenocarcinoma: ','Mucinous cystadenoma: ', 'Mucinous cystadenocarcinoma: ', 'Endometrioid carcinoma: ','Clear cell carcinoma: ',
          'Brenner tumor: ', 'Malignant Brenner tumor: ',  'Mature Teratoma (dermoid cyst): ','Immature teratoma: ', 'Dysgerminoma: ', 'Endodermal sinus tumor', 'Primary ovarian choriocarcinoma: ',
          'Granulosa-stromal cell tumors: ', 'Fibroma or fibrothecoma: ', 'Sclerosing stromal tumor: ','Steroid cell tumor: ','Struma ovarii tumour: ', 'Metastatic tumor(s): ',
          'Functional follicular cyst: ', 'Hemorrhagic cyst: ','Endometrioma: ', 'Tuboovarian abscess: ']


all_features = ['solid', 'solid_necrosis','cystic','unilocular','honeycomb','vegetations',
            'papillary','multi','hypointese','calcification','haemo','dark','fat','diffusion','endo']

def my_function(pos_weights, neg_weights, some_pos_row, labels):
  #Create inverted row
  some_neg_row = 1- some_pos_row 

  #elememtwise multiplicaiton postive row with postivie matrix and same for negative
  pos_multi = np.multiply(pos_weights,some_pos_row)
  neg_multi = np.multiply(neg_weights,some_neg_row)

  #Combine two matixes
  total_sum = pos_multi + neg_multi

  #elementwise sum
  row_wise_sum = np.prod(total_sum, axis=0)

  #normalize
  row_wise_sum = row_wise_sum/row_wise_sum.sum()

  #sort both labels and the row sum
  list1, list2 = (list(t) for t in zip(*sorted(zip(row_wise_sum, labels))))

  result = []
  for i in range(len(row_wise_sum)):
    result.append(str(list2[::-1][i]) + ': ' + str(round(list1[::-1][i],2)))
  return result