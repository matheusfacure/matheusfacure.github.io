---
layout: tutorial
tags: [Tutorial-alt]
comments: true
title: Redes Neurais Feedforward Densas
subtitle: "Implemente modelos básicos de Deep Learning usando PyTorch."
date: 2017-05-15
author: "Matheus Facure"
header-img: "img/dark-ann.jpg"
---

<div class="row">
<ul class="nav nav-tabs navbar-right">
    <li><a href="/2017/05/15/deep-ff-ann/">TensorFlow</a></li>
    <li class="active"><a href="#">PyTorch</a></li>
    <li><a href="/2017/05/15/deep-ff-ann-h2o/">H2O (R)</a></li>
</ul>
</div>

## Pré-requisitos

Esta é uma versão alternativa do tutorial em TensorFlow. Assim sendo, vou pressupor que você já está familiarizado com o tutorial padrão, o que tornará este tutorial muito mais rápido e direto. Não explicarei o passo a passo, nem cada linha de código. Em vez disso, espero que você consiga entender apenas traçando os paralelos entre este tutorial e o tutorial padrão, em TensorFlow. Além disso, a biblioteca aqui considerada é pensada para programação orientada a objetos, então é bom que você esteja familiarizado pelo menos com os [conceitos de OOP](https://www.tutorialspoint.com/python/python_classes_objects.htm).

## Introdução 

<img src="/img/tutorial/pytorch.jpg" alt="tf" class="img-responsive thumbnail pull-right" style="margin-left:3%; width: 30%;">

Vamos usar a biblioteca [PyTorch](http://pytorch.org/) para construção de modelos de *Deep Learning* de forma dinâmica. Diferentemente do TensorFlow, onde construíamos um grafo simbólico e estático, com PyTorch a construção de redes neurais pode ser feita de forma dinâmica. Isso a torna muito mais próxima da forma como a linguagem Python é pensada e faz com que a construção de modelos seja mais intuitiva. Você pode pensar no PyTorch como uma versão eficiente de Numpy, com suporte para GPU e várias funcionalidades auxiliares para *Deep Learning*. No momento desta escrita, pessoalmente, ainda prefiro o TensorFlow, por ser mais eficiente, contar com melhores documentações e com uma comunidade mais ativa. Talvez isso mude conforme o PyTorch se desenvolva. Vale ressaltar que essa biblioteca ainda está em faze de testes. 

## Construindo uma rede neural *feedforward* densa

Antes de iniciar esse tutorial, vamos importar a biblioteca PyTorch e algumas funcionalidades para facilitar na construção do nosso modelo. Também vamos baixar e salvar os dados em uma nova pasta usando o TensorFlow (não se preocupe, só usaremos o TensorFlow para obter os dados).

{% highlight python %}
import torch # para Deep Learning
import torch.autograd as autograd # para autodiferenciação
import torch.nn as nn # para montar redes neurais
import torch.nn.functional as F # funções do Torch
import torch.optim as optim # para otimização com GDE

# carrega os dados MNIST
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("tmp/", one_hot=False)
{% endhighlight %}

Para construir uma rede neural no PyTorch da forma tradicional, criamos uma classe que herda de `torch.nn.Module` e reimplementamos o método `forward(...)`. Ao inicializar a classe, utilizamos `nn.Linear()` para definir o fluxo dos dados na rede neural. `nn.Linear()` cria e mantém variáveis \\(\pmb{W}\\) e \\(\pmb{b}\\) e define uma transformação linear \\(Ax+b\\). Poderemos invocar essa instância no *forward-pass* e não precisaremos nos preocupar com a criação e inicialização dos parâmetros da rede neural. Ao inicializar `nn.Linear()`, passamos, respectivamente, o número de dimensões (colunas) da entrada e do resultado da operação linear.

{% highlight python %}
class Net(nn.Module):

  def __init__(self, n_l1=512, n_l2=512, n_inputs=28*28, n_outputs=10):
    '''Define a arquitetura da rede'''
    super(Net, self).__init__()
    self.in_to_hl1 = nn.Linear(n_inputs, n_l1)
    self.hl1_to_hl2 = nn.Linear(n_l1, n_l2)
    self.hl2_to_out = nn.Linear(n_l2, n_outputs)

  def forward(self, x):
    '''implementa o forward pass'''
    x = F.relu(self.in_to_hl1(x)) # transformação linear, seguida de ReLU
    x = F.relu(self.hl1_to_hl2(x)) # transformação linear, seguida de ReLU
    x = self.hl2_to_out(x) # transformação linear
    return x # retorna os logits
{% endhighlight %}

O método `.forward()` deve aceitar e produzir variáveis Torch (`torch.autograd.Variable()`), de forma que se possa computar as derivadas de maneira automática para o *backpropagation*. No `.forward()` que implementamos, temos duas camadas ocultas com ativação ReLU e uma camada linear, que converte o *output* da última camada oculta em *logits*. Devido a diferenciação automática, basta implementar o método `.forward()` que teremos o *backward-pass* de graça. 

Abaixo, criamos uma instância da classe recém definida. Se tivermos uma GPU, podemos acelerar as computações movendo os parâmetros da rede para a GPU. Isso é feito com o método `.cuda()`, que vem na herança da classe `nn.Module`. Também podemos utilizar `print(net)` para ver a arquitetura definida durante a inicialização.

{% highlight python %}
net = Net() # cria uma rede neural artificial
# net.cuda() # para GPU
print(net)

criterion = nn.CrossEntropyLoss() # define o custo de entropia cruzada
optimizer = optim.Adam(net.parameters(), lr = 0.01) # cria o otimizador
{% endhighlight %}

```
Net (
  (in_to_hl1): Linear (784 -> 512)
  (hl1_to_hl2): Linear (512 -> 512)
  (hl2_to_out): Linear (512 -> 10)
)
```

Também precisamos definir a função objetivo e o otimizador. O primeiro será a o custo de entropia cruzada. Essa função requer como entrada um tensor 2D, no formato `[n_batch_n_class]` e com os *logits* (escores de probabilidade não normalizada), e um tensor 1D, com os indicies das classes. Como otimizador, utilizaremos a variação Adam de gradiente descendente estocástico, com uma taxa de aprendizado de \\(0,01\\).

## Treinando a rede neural

Com a rede construída, estamos prontos para entrar no loop de treinamento. A parte mais complicada desse processo é converter os dados em um formato aceitável pelo *backpropagation* do PyTorch. Inicialmente, nossas variáveis são *arrays* Numpy. Precisamos primeiro converter os dados em tensores Torch com a função `torch.from_numpy()`. Em seguida, precisamos garantir que esse tensor seja de tipo `floats32`, o que é feito com o método `.float()`. As classes precisam ser tensores Torch de tipo `int64`, que podem ser criados com a função `torch.LongTensor()`. Essa função aceita uma lista de inteiros, por isso convertemos o *array* Numpy com `.tolist()`. Se quisermos utilizar a GPU, precisamos utilizar o método `.cuda()` nos tensores. Por fim, envelopamos os dados em variáveis para autodiferenciação com `autograd.Variable()`.

Para realizar uma iteração de treino, precisamos antes zerar o acumulador de gradientes associados a cada variável. Isso é feito com o método `.zero_grad()` do otimizador. Em seguida, computamos o custo, realizamos o *backward-pass* e atualizamos os parâmetros da rede com o método `.step()` do otimizador. 

{% highlight python %}
n_iter = 4000 # iterações de treino
batch_size = 128 # tamanho do mini-lote
for step in range(n_iter+1):

	# cria os mini-lotes
	x_batch, y_batch = data.train.next_batch(batch_size)
	# converte os dados de Numpy para Tensores Torch
	x_batch, y_batch = torch.from_numpy(x_batch).float(), torch.LongTensor(y_batch.tolist())
	# x_batch, y_batch = x_batch.cuda(), y_batch.cuda() # para GPU
	# Envelopa tensores em classe Variable para autodiferenciação
	x_batch, y_batch = autograd.Variable(x_batch), autograd.Variable(y_batch)

	# zera os gradientes computados anteriormente
	optimizer.zero_grad()

	# realiza uma operação de treino
	logits = net(x_batch) # forward pass
	loss = criterion(logits, y_batch) # calcula o custo
	loss.backward() # backward pass
	optimizer.step() # atualiza os parâmetros

	# mostra métricas de treino a cada 1000 iterações
	if step % 1000 == 0:

		# monta o lote de validação
		x_valid, y_valid = data.validation.next_batch(512)
		x_valid, y_valid = torch.from_numpy(x_valid), torch.LongTensor(y_valid.tolist())
		# x_valid = x_valid.cuda() # para GPU
		
		logits = net(autograd.Variable(x_valid))
		# logits = logits.cpu() # tráz de volta para a CPU
		_, y_hat = torch.max(logits.data, dim=1)
		acc = 100 * (y_hat.view(-1,) == y_valid).sum() / len(y_valid)
		print('\nCusto de treino na iteração %d: %.2f' % (step, loss.data[0]))
		print('Erro de validação na iteração %d: %.2f%%' % (step, acc))
{% endhighlight %}
```
Custo de treino na iteração 0: 2.30
Erro de validação na iteração 0: 22.85%

Custo de treino na iteração 1000: 0.08
Erro de validação na iteração 1000: 95.51%

Custo de treino na iteração 2000: 0.17
Erro de validação na iteração 2000: 95.51%

Custo de treino na iteração 3000: 0.04
Erro de validação na iteração 3000: 97.46%

Custo de treino na iteração 4000: 0.06
Erro de validação na iteração 4000: 98.44%

```
A cada 1000 iterações de treino, mostramos algumas métricas de desempenho, relativas a dados de validação. Novamente, precisamos passar pelo processo de conversão de dados. A diferença é que aqui não precisamos converter as classes para variáveis de autodiferenciação, já que não vamos passá-las para a função custo. Nos *logits* produzidos pela rede, vamos utilizar `torch.max` que retorna o valor máximo do tensor (descartamos isso) e o indicie do valor máximo, isto é, o \\(argmax\\). Devemos lembrar de especificar a segunda dimensão do tensor, para que obter os máximos relativos a esse eixo (as colunas da matriz `[n_amostras, logits]`). Além disso, `torch.max` requer tensores como argumento, mas `y_hat`, retornado pela rede, é uma variável de autodiferenciação. Para pegar o tensor relativo a essa variável utilizamos o método `.data`.

Por fim, para calcular a acurácia precisamos que ambos os tensores sejam 1D, então reformatamos `y_hat` com `.view()` (equivalente ao `.reshape()` do Numpy). Então mostramos a acurácia de validação e o custo de treino. Podemos ver que os resultados são similares aos obtidos com o TensorFlow.