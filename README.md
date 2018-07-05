# FreesideEEG

## The Data

### Data Specification

The eeg data must be a file where each line is a record.    
The record is a list of frequency bin values for each channel follwed by an integer label.    
Each record  begins with a `[` and ends with a `]`.    
Each record is on its own line.    


The meaning of the label can be anything.    
The sampling rate can be anything.


### Loading the Data

The data must be loaded into the `TestData` and `TrainingData` folders.    
The folders can be changed in `classic_training.py`    
The data must have there rows where the circular buffer is not filled removed.  
Any number of data files can be added to the test and training folders.

![Data Location](assets/data_location.png?raw=true "Data Location")    


### Specifying data parameters

The file `eeg_data_util.py` has a number of constanst that need to be adjusted if the data changes.    
<pre>
CHANNELS = 4 # The number of eeg channels in the recordings
FREQ_BINS = 120 # The number of freq bins per channel
MAX_STD = 3 # The max stnadard deviation to crop outliers to
SLICE_SIZE = 4 # The number of samples the network will look at at once to make a prediction
</pre>

## Running the Network






### Sample Training Run

![Accuracy and Loss](assets/loss_accuracy.png?raw=true "Accuracy and Loss")    
![Distributions](assets/distributions.png?raw=true "Distributions")    
![Histograms](assets/histograms.png?raw=true "Histograms")    


<pre>
/usr/bin/python3.6 /home/nburn42/FreesideEEG/classic_train.py
Parsing files: 'TrainingData/eegdataTrain.data'
Parsing files: 'TestData/eegdataTest.data'
Training Data size: 343
Test Data size:     14
Training Sample label: 0.0 record len: 1920 data: [-0.2322498903681478, -0.1786831126435639, -0.20682014466010176, -0.20953366384758268, -0.13545340241287673, ... ...
Test Sample     label: 1.0 record len: 1920 data: [-0.5676211257689656, -0.671304643715787, -0.615245739364888, -0.6716233738454067, -0.5326936977519163, ... ...
****************************************
Building basic model
input layer: Tensor("batch_normalization/batchnorm/add_1:0", shape=(?, 1920), dtype=float32)
hidden layer (96000 parameters): Tensor("dropout/cond/Merge:0", shape=(?, 50), dtype=float32)
hidden layer (2500 parameters): Tensor("dropout_1/cond/Merge:0", shape=(?, 50), dtype=float32)
hidden layer (2500 parameters): Tensor("dropout_2/cond/Merge:0", shape=(?, 50), dtype=float32)
hidden layer (2500 parameters): Tensor("dropout_3/cond/Merge:0", shape=(?, 50), dtype=float32)
output layer (150 parameters): Tensor("prediction:0", shape=(?, 3), dtype=float32)
********************
Model has a total of 103650 parameters. Ideally the training side will be larger than the parameters
****************************************
2018-07-04 22:27:35.743070: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-07-04 22:27:35.812743: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:898] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-07-04 22:27:35.813120: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 0 with properties:
name: GeForce GTX 1070 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.683
pciBusID: 0000:01:00.0
totalMemory: 7.93GiB freeMemory: 7.24GiB
2018-07-04 22:27:35.813131: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-07-04 22:27:35.986806: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-07-04 22:27:35.986830: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0
2018-07-04 22:27:35.986835: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N
2018-07-04 22:27:35.987028: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6988 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1070 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
Starting new training
step  10
step  20
step  30
step  40
step  50
step  60
step  70
step  80
step  90
step  100
step  110
step  120
step  130
step  140
step  150
Step Count:150
Training accuracy: 0.48750001192092896 loss: 0.9094280004501343
Test accuracy: 0.5 loss: 1.0531948804855347
training:*p[0.05402339 0.00639413 0.93958247] l2 d[-0.19759796932587953, -0.035129072531184795, 0.0956621198187041]
training: p[0.31716874 0.1980532  0.48477805] l0 d[-0.2512792974294244, -0.1912912681634597, -0.24123835431294877]
training:*p[0.14286602 0.15086056 0.7062734 ] l2 d[-0.24795173569961929, -0.13177178646998622, -0.1535449638978359]
training: p[0.13970149 0.56289417 0.2974043 ] l2 d[-0.2201921511353298, 0.023189114021106644, -0.22075557367288615]
training: p[0.5219007  0.11021066 0.36788866] l2 d[-0.2515777699564953, -0.17790006892497998, -0.24007323788241453]
training:*p[0.02147298 0.29934508 0.679182  ] l2 d[-0.22523033364869335, -0.18204018843682584, -0.22111688498776044]
test:    *p[0.5241851 0.291358  0.1844569] l0 d[-0.31456958484803804, -0.5635545633943384, -0.16768980002967082]
test:    *p[0.50880754 0.26327315 0.22791931] l0 d[0.4338613082024738, -0.02109101585121592, 0.5975482965327873]
test:    *p[0.4603366  0.17184924 0.36781418] l0 d[-0.523074140567978, -0.5339622704921346, -0.023917769188908075]
test:     p[0.39736992 0.2363455  0.36628452] l2 d[0.7986901793891837, 1.77985053364689, 1.2590887375302922]
test:    *p[0.18044385 0.3370831  0.48247308] l2 d[-0.30649182642714334, -0.4234735925820885, -0.38992290785231026]
test:    *p[0.28014714 0.37533247 0.3445204 ] l1 d[-0.5637256455251425, -0.6057433339858748, -0.6217417155597692]
step  160
step  170
step  180
step  190
step  200
step  210
step  220
step  230
step  240
step  250
step  260
step  270
step  280
step  290
step  300
Step Count:300
Training accuracy: 0.8500000238418579 loss: 0.425253301858902
Test accuracy: 0.5714285969734192 loss: 1.067853569984436
training:*p[0.05932676 0.89509475 0.04557844] l1 d[-0.24744868209531687, -0.17328155063000616, -0.21995428682360793]
training:*p[0.024711   0.02159331 0.95369565] l2 d[-0.22514354351011973, 0.06275810455635147, 0.17963887320017224]
training: p[0.32562697 0.18735321 0.48701984] l1 d[-0.20283875440909746, -0.14923045428878592, -0.1926494259293545]
training: p[0.76514286 0.1235649  0.1112922 ] l2 d[-0.23686312305487697, -0.15285034058333333, -0.18495051438951118]
training:*p[0.9204964  0.07052088 0.00898276] l0 d[-0.24361908164511148, -0.1894674627324903, -0.24139038932850648]
training:*p[0.01869571 0.8710228  0.11028152] l1 d[-0.2511892165187081, -0.1902778740846399, -0.23185931573173532]
test:     p[0.6158395  0.24475268 0.13940783] l2 d[-0.1832697417517035, -0.14178561866246342, -0.49388915077531537]
test:     p[0.01864147 0.02025828 0.9611003 ] l1 d[0.0037404803600481784, -0.37365493526960414, -0.4186983109199274]
test:     p[0.71741927 0.256434   0.02614667] l1 d[-0.5637256455251425, -0.6057433339858748, -0.6217417155597692]
test:     p[0.10329977 0.14947994 0.74722034] l0 d[-0.523074140567978, -0.5339622704921346, -0.023917769188908075]
test:    *p[0.41076687 0.551082   0.03815112] l1 d[-0.5676211257689656, -0.671304643715787, -0.615245739364888]
test:    *p[0.24280891 0.71242666 0.0447645 ] l1 d[-0.5668681508953329, -0.18530633373444605, 0.183856400326364]
step  310
step  320
step  330
step  340
step  350
step  360
step  370
step  380

...
...
...

step  2920
step  2930
step  2940
step  2950
step  2960
step  2970
step  2980
step  2990
step  3000
Step Count:3000
Training accuracy: 1.0 loss: 0.008985666558146477
Test accuracy: 0.5714285969734192 loss: 4.810230255126953
training:*p[1.7336626e-04 9.9970990e-01 1.1664485e-04] l1 d[-0.20283875440909746, -0.14923045428878592, -0.1926494259293545]
training:*p[3.4963592e-05 9.9988985e-01 7.5271630e-05] l1 d[-0.23911303898838424, -0.14223005822832563, -0.15721549691436126]
training:*p[6.612328e-03 9.926272e-01 7.604554e-04] l1 d[-0.038982289549789474, -0.1403942971177852, -0.19775229866235355]
training:*p[9.9932408e-01 5.4201396e-04 1.3392176e-04] l0 d[-0.20207017843727512, -0.15474900249954648, -0.23493217968192953]
training:*p[2.2022543e-05 9.9997711e-01 8.8578764e-07] l1 d[-0.12105908330950413, -0.0512066386927472, 0.003257871941593581]
training:*p[9.9999976e-01 4.1296193e-09 1.9875277e-07] l0 d[-0.23652872960343355, -0.17345264106036107, -0.22093651938671288]
test:     p[1.2222394e-04 1.6748216e-05 9.9986100e-01] l1 d[0.0037404803600481784, -0.37365493526960414, -0.4186983109199274]
test:    *p[1.0000000e+00 4.3750292e-09 3.8584838e-09] l0 d[0.4338613082024738, -0.02109101585121592, 0.5975482965327873]
test:     p[9.154158e-12 3.723779e-08 1.000000e+00] l1 d[-0.24808999617941718, 0.2538331613907019, -0.5774066762655484]
test:     p[0.9961295  0.00270229 0.00116813] l1 d[-0.5676211257689656, -0.671304643715787, -0.615245739364888]
test:     p[1.1779523e-05 6.2716043e-07 9.9998760e-01] l1 d[-0.5637256455251425, -0.6057433339858748, -0.6217417155597692]
test:    *p[6.2105187e-07 3.4886758e-05 9.9996448e-01] l2 d[-0.5316670940947386, -0.6161653143955358, -0.5846998773216535]
Finished training at step 3000.   
</pre>