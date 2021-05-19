# 1 Introduction  

Current approaches to object recognition make essential use of machine learning methods.  
**객체 인식에 대한 현재 접근 방식은 기계 학습 방법을 필수적으로 사용한다.**  
To improve their performance, we can collect larger datasets, learn more powerful models, and use better techniques for preventing overfitting.  
**퍼포먼스를 향상시키기 위해서는 우리는 더 많은 데이터세트를 수집하고 파워풀한 모델을 학습 시키고 오버피팅을 방지하는 더 좋은 기술을 사용해야 한다.**  
Until recently, datasets of labeled images were relatively small - on the order of tens of thousands of images.  
**최근까지 레이블링된 이미지 데이터 세트들은 수만개의 이미지정도로 상대적으로 작다.**  
Simple recognition tasks can be solved quite well with datasets of the size, especially if they are augmented with label-preserving transformations.  
**단순한 인식 작업들은 특히 레이블 보존 변환으로 보강된 사이즈의 데이터 세트들로도 아주 잘 해결할 수 있다.**  
For example, the current best error rate on the MNIST digit-recognition task (<0.3%) approaches human performance.  
**예를 들어 MNIST 숫자 인식 작업의 최적의 에러율은 0.3% 아래로 사람의 퍼포먼스에 도달했다.**  
But objects in realistic setting exhibit considerable variability, so to learn to recognize them it is necessary to use much larger training sets.  
**하지만 현실적인 설정은 상당한 가변성을 나타내므로 이를 인식하는 것을 배우는 데에는 훨씬 많은 학습 세트가 사용되는 것은 필연적이었다.**  
And indeed, the shortcomings of small image datasets have been widely recognized, but it has only recently become possible to collect labeled datasets with millions of images.  
**실제로 작은 이미지 데이터 세트의 단점은 널리 인식되어왔지만 최근에서야 수백만장의 이미지를 수집할 수 있게 됬다.**  
The new larger datasets include LabelMe, which consists of hundreds of thousands of fully-segmented images, and ImageNet, which consists of over 15 million labeled high-resolution images in over 22,000 categories.  
**새로운 대규모 데이터 세트에는 수십만장의 완전분할된 이미지로 구성된 LabelMe와 22,000 이상의 카테고리에 있는 1,500만장의 라벨링된 고해상도 이미지로 구성된 ImageNet이 있다.**  

To learn about thousands of objects from millions of images, we need a model with a large learning capacity.  
**수백만장의 이미지로부터 오는 수천개의 객체를 학습하기 위해 우리는 더 큰 학습 수용성을 지닌 모델이 필요하다.**  
However, the immense complexity of the object recognition task means that this problem cannot be specified even by a dataset as large as ImageNet, so our model should also have lots of prior knowledge to compensate for all the data we don't have.  
**그러나 객체 인식 작업의 엄청난 복잡성은 이 문제는 ImageNet과 같은 큰 데이터 세트로도 정의될 수 없음을 의미하므로, 우리의 모델은 우리가 가지고 있지 않은 데이터들에 대해서도 보상할 수 있을 만큼 많은 사전지식을 가져야 한다.**  
Convolution neural networks consitute one such class of models.  
**컨볼루션 신경망은 이러한 모델 클래스중 하나를 구성한다.**  
Their capacity can be controlled by varying their depth and breadth, and they also make strong and mostly correct assumptions about the nature of images (namely, stationarity of statistics and locality of pixel dependencies).  
**그것들의 능력은 깊이나 폭을 다양하게 제어할 수 있으며 그들은 이미지의 특성(즉, 통계의 정상성 및 픽셀 종속성의 지역성)에 대한 정확하고 강한 가정을 한다.**  
Thus, compared to standard feedforward neural networks with similarly-size layers, CNNs have much fewer connections and parameters and so they are easier to train, while their theoretically-best performance is likely to be only slightly worse.  
**그러므로 비슷한 레이어 층의 standard feedforward 신경망과 비교해보았을 때, CNN은 훨신 더 적은 연결과 파라미터를 가지고 있기 때문에 학습하기가 더 쉽고 이론적으로 최고의 성능은 약간 더 나빠질 수 있다.**  

Despite the attractive qualities of CNNs, and despite the relative efficiency of their local architecture, they have still been prohibitively expensive to apply in large scale to high-resolution images.  
**CNN의 매력적인 퀄리티와 아키텍처의 효율성에도 불구하고 그들은 방대한 양의 고해상도 이미지에 적용하기에는 비용이 매우 컸다.**  
Luckily, current GPUs, paired with a highly-optimized implementation of 2D convolution, are powerful enough to facilitate the training of interestingly-large CNNs, and recent datasets such as ImageNet contain enough labeled examples to train such models without severe overfitting.  
**운좋게도, 2차원 합성곱에 최적으로 구현된 GPU는 거대한 CNN을 훈련시키기에 충분히 강력하고, ImageNet과 같은 최근의 데이터 세트들은 이와 같은 모델들은 심각한 오버피팅 없이 학습시키기에 충분한 라벨을 포함하고 있다.**  

The specific contributions of this paper are as follows: we trained one of the largest convolutional neural networks to data on the subsets of ImageNet used in the ILSVRC-2010 and ILSVRC-2012 competitions and achieved by far the best results ever reported on these datasets.  
**이 논문의 주된 기여는 다음과 같다 : 우리는 ILSVRC-2010, ILSVRC-2012에 사용된 ImageNet의 서브셋 데이터들을 현재까지 거대한 신경망 중 하나를 학습시켰다. 그리고 이 데이터 세트들로 보고된 것중 단연코 최고의 성과를 달성했다.**  
We wrote a highly-optimized GPU implementation of 2D convolution and all the other operations inherent in training convolutional neural networks, which we make available publicly.  
**우리는 2D 컨볼루션의 최적화된 GPU 구현과 컨볼루션 신경망 학습에 내재된 다른 모든 작업들을 작성하여 공개적으로 제공했다.**  
Our network contains a number of new and unusual features which improve its performance and reduce its training time, which are detailed in Section 3.  
**우리의 신경망은 섹션 3에서 퍼포먼스를 향상시키고 학습시간을 감소시킨 다수의 새롭고 일반적이지 않은 특징들을 포함하고 있다.**  
The size of our network made overfitting a significant problem, even with 1.2 million labeled training examples, so we used several effective techniques for preventing overfitting, which are described in Section 4.  
**우리의 신경망의 사이즈와 120만개의 라벨링된 학습 예제는 오버피팅이라는 거대한 문제를 야기시킨다. 그래서 섹션 4에서 우리는 오버피팅을 방지하기 위한 여러 효과적인 테크닉들을 사용한 내용을 다룬다.**  
Our final network contains five convolution and three fully-connected layers, and this depth seems to be important: we found that removing any convolutional layer (each of which contains no more than 1% of model's parameters) resulted in inferior performance.  
**우리의 최종 신경망은 5개의 컨볼루션 레이어와 3개의 FC 레이어를 포함하고 있다. 그리고 이 깊이를 중요하게 본다: 우리는 아무 신경망이나 제거하면 더욱 못한 결과를 낸다는 사실을 발견했다. (각각은 모델 매개 변수의 1 % 이하를 포함)**  

In the end, the network's size is limited mainly by the amount of memory available on current GPUs and by the amount of training time that we are willing to tolerate.  
**마지막으로 네트워크 사이즈는 현재 사용되고 있는 GPU의 메모리 양과 우리가 허용할 학습시간의 양에 의해 제한된다.**  
Our network takes between five and six days to train on two GTX 580 3GB GPUs.  
**우리의 신경은 5~6일동안 두대의 GTX 580 3GB GPU로 학습을 했다.**  
All of our experiments suggest that our results can be improved simply by waiting for faster GPUs and bigger datasets to become available.  
**우리의 모든 실험은 우리의 결과를 향상시키는데에는 더 빠른 GPU들과 큰 데이터셋을 기다리는 것이라고 제안했다.**

# 2 The Dataset  
ImageNet is dataset of over 15 million labeled high-resolution images belonging to roughly 22,000 categories.  
**ImageNet은 약 22,000개 카테고리에 속하는 1,500만개 이상의 레이블이 지정된 고해상도 이미지의 데이터 세트이다.**  
The images were collected from the web and labeled by human labelers using Amazon's Mechanical Turk crowd-sourcing tool.    
**이미지들은 웹에서 수집되었고, Amazon의 Machanical Turk crowd-sourcing tool을 사용하는 사람들에 의해 라벨링되었다.**  
Starting in 2010, as part of the Pascal Visual Object Challenge, an annual competition called the ImageNet Large-Scale Visual Recognition Challenge(ILSVRC) has been held.    
**2010년부터 Pascal Visual Object Challenge의 일환으로 ILSVRC라고 불리는 대회가 열렸다.**  
ILSVRC uses a subset of ImageNet with roughly 1000 images in each of 1000 categories.  
**ILSVRC는 1000개의 카테고리안에 약 1000개의 이미지가 있는 ImageNet 서브셋이 사용되었다.**  
In all, there are roughly 1.2 million training images, 50,000 validation images, and 150,000 testing images.  
**전체적으로, 약 120만장의 학습 이미지와 5만장의 검증 이미지와 15만장의 테스트 이미지가 있다.**  
ILSVRC-2010 is the only version of ILSVRC for which the test set labels are available, so this is the version on which we performed most of our experiments.  
**ILSVRC-2010은 테스트 세트 레이블을 사용할 수 있는 유일한 대회였으므로, 우리는 대부분의 실험을 수행했다.**  
Since we also entered our model in the ILSVRC-2012 competition, in Section 6 we report our results on this version of the dataset as well, for which test set labels are unavailable.  
   
# 3 The Architecture
The architecture of our network is summarized in Figure 2.  
**우리의 신경망 구조는 Firegure 2에 요약되어 있다.**  
It contains eight learned layers - five convolutional and three fully-connected.  
**그것은 8개의 학습된 레이어와 5개의 컨볼루션 레이어, 3개의 FC 레이어들을 포함하고 있다.**  
Below, we describe some of the novel or unusual features of our network's architecture.  
**우리가 묘사한 특정 개요나 우리의 신경망 구조의 일반적이지 않은 특징 안에 포함되어 있다.**  
Sections 3.1-3.4 are sorted according to our estimation of their importance, with the most important first.  
**섹션 3.1~3.4 에서는 중요성에 대한 우리의 평가에 따라 정렬되며, 가장 중요한 것이 먼저이다.**  

## 3.1 ReLU Nonlinearity
The standard way to model a neuron's output f as a function of its input x is with f(x) = tanh(x) or f(x) = (1+e^-x)^1.  
**뉴런을 설계하는 일반적인 방법은 함수 f에 대해 인풋을 x로 정의하고 f(x)는 하이퍼볼릭 탄젠트나 시그모이드 함수로 정의하는 것이다.**  
In terms of training time with gradient descent, these saturating nonlinearities are much slower than the non-saturating nonlinearity f(x) = max(0, x).  
**경사하강법 학습 측면에서, saturating nonlinearities 는 non-saturating nonlinearity (max(0, x) 함수와 같은) 보다 훨씬 느리다.** https://m.blog.naver.com/PostView.nhn?blogId=smrl7460&logNo=221152996685&proxyReferer=https:%2F%2Fwww.google.com%2F  
Following Nair and Hinton, we refer to neurons with this nonlinearity as Rectified Linear Units (ReLUs).  
**Nair와 Hinton에 이어, 이 비선형성을 가진 뉴런을 ReLU라고 한다.**  
Deep convolutional neural networks with ReLUs train several times faster than their equivalents with tanh units.
**ReLU를 사용하는 심층 컨볼루션 신경망은 하이퍼볼릭 탄젠트를 사용하는 신경망보다 몇배 더 빠르게 학습한다.**  
This is demonstrated in Figure 1, which shows the number of iterations required to reach 25% training error on the CIFAR-10 dataset for a particular four-layer convolutional notwork.  
**이것은 Figure 1에서 시연되었으며, 특정 4개의 컨볼루션 레이어로 CIFAR-10 데이터 세트의 학습 오류율이 25%에 도달하기 까지의 반복횟수를 보여준다.**  
This plot shows that we would not have been able to experiment with such large neural networks for this work if we had used traditional saturating neuron models.  
**이 그래프는 우리가 전통적인 saturating 신경 모델을 사용한다면 방대한 신경망으로 이러한 실험을 할 필요가 없다는 것을 보여준다.**  

We are not the first to consider alternatives to traditional neuron models in CNNs.  
**CNN에서 전통적인 신경 모델을 대체를 고려한 것이 우리가 처음은 아니다.**  
For example, Jarrett et al claim that the nonlinearity f(x) = |tanh(x)| works particularly well with their type of contrast normalization followed by local average pooling on the Caltech-101 dataset.  
**예를 들면 Jarrett et al이 비선형 하이퍼볼릭 탄젠트의 스칼라 함수가 Caltech-101 데이터 세트의 로컬 평균 풀링이 뒤따르는 대비 정규화 유형에 잘 작동한다고 주장했다.**  
However, on this dataset the primary concern is preventing overfitting, so the effect they are observing is different from the accelerated ability to fit the training set which we report when using ReLUs.  
Faster learning has a great influence on the performance of large models trained on large datasets.  

## 3.2 Training on Multiple GPUs

# Reducing Overffitting
# Details of learning
# Results