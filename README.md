[Get Data](https://1drv.ms/u/s!Aq8XcGokRpklgYBTtuad5joEc1ziLA?e=88EwJd)  
[Get Google Word2Vec](https://1drv.ms/u/s!Aq8XcGokRpklgYBc6Llof2bTL5iKhw?e=YGPofe)  
Randomly pick 10% for validation. 
For the remaining 90%, we randomly pick such a subset for training:  
a. # of positive samples = # of negative samples  
b. as large as possible  
Totally, validation / (validation+training) = 12.9%  
Result of Validation :  
Precision: 69%  
Recall: 82.22%  
Accuracy: 79.74%  
F1 score: 0.7503  
![image](https://raw.githubusercontent.com/xyw6/Siamese/master/result/loss.jpg)  
![image](https://raw.githubusercontent.com/xyw6/Siamese/master/result/recall.jpg)  
![image](https://raw.githubusercontent.com/xyw6/Siamese/master/result/f1.jpg)