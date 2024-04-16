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
![image](https://github.com/masume-r/nn-classifier-with-MNIST/assets/167098630/b77a2cbc-836e-4dca-8d27-7ca1cd2895d0)
![image](https://github.com/masume-r/nn-classifier-with-MNIST/assets/167098630/cf744db7-edee-4d60-af4f-75eff21432c4)
