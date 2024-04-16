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


 Compare the predictive performances of LeNet-5 and your custom MLP:
![image](https://github.com/masume-r/nn-classifier-with-MNIST/assets/167098630/e69ce89d-c88f-407d-8597-e85fdcfc3bcc)

 
