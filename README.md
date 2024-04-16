# nn-classifier-with-MNIST(LeNet5/MLP)

LeNet5 Parameters
Convolutional Layer = (number input channels * number output channels * kernel size)+ number output channels
Fully Connected Layer = (number of input units * number of output units) + number of output units

first layer: -----------> (1*6*5*5)+6 = 156
second layer: pool------> 0
third layer: -----------> (6*16*5*5)+16 = 2416
fourth layer: pool------> 0
fifth layer:------------> (16*120*5*5)+120 = 48120
sixth layer: -----------> (120*84*1*1)+84 = 10164
seventh layer: ---------> (84*10*1*1)+10 = 850

Total= 61,706

MLP Parameters: 
(input size* output size)+output size
first layer: -----------> (32*32*64)+64 = 65536
second layer: pool------> (64*32)+32 = 2080
third layer: -----------> (32*32)+10 = 1034

Total = 68,650

![lenet-lose](https://github.com/masume-r/nn-classifier-with-MNIST/assets/167098630/a182f4e0-1fe8-460f-a72c-4cfb0802d2e4)
![lenet-acc](https://github.com/masume-r/nn-classifier-with-MNIST/assets/167098630/6984df0f-9ae1-4153-b8b2-225f05371a65)
![mlp-loss](https://github.com/masume-r/nn-classifier-with-MNIST/assets/167098630/8cb61a91-186e-4d5b-9abe-dfc6342deb81)
![mlp-acc](https://github.com/masume-r/nn-classifier-with-MNIST/assets/167098630/568c59fe-900e-445f-903d-837865120572)


#Compare the predictive performances of LeNet-5 and your custom MLP:

regularization techniques:
LeNet 5과 custom MLP는 둘 디 좋은 성능을 보여줍니다.
Mlp와 LeNet5 Accuracy plot에서 보시다시피 test 데이터보다는 training 데이터에서 봏은 accuracy를 가지고 있습니다. Lenet5은 test data에서 조금 더 좋은 accuracy가 있습니다.
그리고 loss plot에서도 lenet5은 더 작은 값을 있습니다.

Lenet5 accuracy:
epoch1에서------> Train data:0.90% /Testdata: 0.98%에서
epoch10에서-----> Train data:100%  /Testdata: 0.99%  
실제 accuracy는 ~99%이라고 합니다

Mlp accuracy:
epoch1에서-----> Train data:0.99% / Testdata: 0.95%
epoch10에서----> Train data:99%   /Testdata: 0.98%  


#Regularization
Regularization Dropput batch normalization 2개를 해봤습니다. Accuracy는 99%이라서 Regularization을 해도 99%을 기지고 있습니다. 
![lenet5-regularization techniques-acc](https://github.com/masume-r/nn-classifier-with-MNIST/assets/167098630/d8fbbd2d-0757-4d80-9a0a-ad968f8e37d4)
![Lenet5regularization techniques-loss](https://github.com/masume-r/nn-classifier-with-MNIST/assets/167098630/f878b4fb-b9ed-454c-b5bb-5e7208e77ccc)

 
