---
layout: tutorial
tags: [Tutorial-alt]
comments: true
title: Redes Neurais Feedforward Densas
subtitle: "Implemente modelos básicos de Deep Learning usando R e H2O."
date: 2017-05-15
author: "Matheus Facure"
header-img: "img/dark-ann.jpg"
---

<div class="row">
<ul class="nav nav-tabs navbar-right">
    <li><a href="/2017/05/15/deep-ff-ann/">TensorFlow</a></li>
    <li><a href="/2017/05/15/deep-ff-ann-pytorch/">PyTorch</a></li>
    <li class="active"><a href="#">H2O (R)</a></li>
</ul>
</div>

## Pré-requisitos

Esta é uma versão alternativa do tutorial em TensorFlow. Assim sendo, vou pressupor que você já está familiarizado com o tutorial padrão, o que tornará este tutorial muito mais rápido e direto. Não explicarei o passo a passo, nem cada linha de código. Em vez disso, espero que você consiga entender apenas traçando os paralelos entre este tutorial e o padrão, em TensorFlow.

## Introdução 

Vamos usar a linguagem R de programação estatistifica e a biblioteca [H2O](https://www.h2o.ai/) para Deep Learning. Você verá que essa biblioteca torna a construção e treinamento de redes neurais muito mais simples do que o TensorFlow. Por outro lado, ela é extremamente abstrata e retira um pouco do controle que temos sobre a construção de modelos de Deep Learning. Eu pessoalmente prefiro o TensorFlow, mas coloco esse tutorial aqui para que você possa escolher por conta própria qual delas prefere.

## Construindo uma rede neural *feedforward* densa

Baixe os dados **de treino** da base MINIST no [Kaggle](https://www.kaggle.com/shuvayan/deep-learning-using-h2o-in-r/data). O código a seguir muda o diretório do R para a pasta de Downloads, onde está a base de dados que você acabou de baixar. Em seguida, lemos a base de dados e convertemos as variáveis dependentes para a codificação *one-hot*. 

```r
setwd("~/Downloads/") # muda o diretório para a pasta de Downloads  
dataset = read.csv("train.csv") # lê a base de dados
dataset$label = as.factor(dataset$label) # formata a variável dependente
```

Em seguida, precisamos separar a base de dados em sets de treino e de teste. Fazemos isso com a biblioteca `caTools`, como demonstrado a seguir.

```r
library(caTools) # ativa a biblioteca
split = sample.split(dataset$label, SplitRatio = 0.8) # 80% para treino
train = subset(dataset, split==TRUE)
test = subset(dataset, split==FALSE)
```

Antes de prosseguir, vamos conferir se os dados foram lidos e processados corretamente. Para isso, vamos ver graficamente algumas imagens dos dígitos MNIST junto com suas respectivas classes.

```r
par(mfrow = c(10,10), mai = c(0,0,0,0))
for(i in 1:100){
    y = as.matrix(train[i, 2:785])
    dim(y) = c(28, 28)
    image( y[,nrow(y):1], axes = FALSE, col = gray(255:0 / 255))
    text( 0.2, 0, train[i,1], cex = 1.5, col = 2, pos = c(3,4))
}
```

<img class="img-responsive center-block thumbnail" src="/img/tutorial/mnist-h2o.png" alt="mnist-digits" />

Finalmente, vamos ativar a biblioteca [H2O](https://www.h2o.ai/) para *Deep Learning*. Com a linha `h2o.init(nthreads = -1)` , vamos configurar a biblioteca para usar todos os núcleos do nosso computador, realizando assim computações em paralelo que aceleram o treinamento de redes neurais. 
Em seguida, em uma linha, construímos uma rede neural com duas camadas ocultas, 512 neurônios por camada e com a função de ativação ReLU.

```r
clf = h2o.deeplearning(y="label", # nome da coluna da var. independente
                       training_frame = as.h2o(train), 
                       activation = 'Rectifier',
                       hidden = c(512, 512),
                       epochs = 2,
                       train_samples_per_iteration = 128)
```
 
Treinamos nosso modelo com mini-lotes de 128 imagens e com duas passadas inteiras pela base de dados (duas épocas de treinamento). Repare como precisamos converter os dados para formato que o H2O entende. Para isso, usamos `as.h2o(...)`. 
Estamos agora prontos para realizar previsões e avaliar nosso modelo com o set de teste.

```r
y_hat = h2o.predict(clf, newdata=as.h2o(test)) # faz previsões   
y_hat = as.data.frame(y_hat) # converte do formato H2O para DataFrame
y_hat = y_hat$predict # extrai a coluna de previsões
print(mean(y_hat == test$label)) # mostra a acurácia no set de teste
```
{% highlight bash %}
[1] 0.9603524
{% endhighlight %}
