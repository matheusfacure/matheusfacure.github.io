---
layout: tutorial
tags: [Tutorial-Alt]
comments: true
title: Máquinas de Boltzmann Restritas
subtitle: "Um modelo gerador gráfico, não supervisionado, probabilístico e baseado em energia."
date: 2017-07-30
true-dt: 2017-07-30
author: "Matheus Facure"
---

<div class="row">
<ul class="nav nav-tabs navbar-left">
    <li><a href="/2017/07/30/RBM/">TensorFlow</a></li>
    <li class="active"><a href="#">PyTorch</a></li>
</ul>
</div>

## Pré-requisitos

Esta é uma versão alternativa do tutorial em TensorFlow. Assim sendo, vou pressupor que você já está familiarizado com o tutorial padrão, o que tornará este tutorial muito mais rápido e direto. Não explicarei o passo a passo, nem cada linha de código. Em vez disso, espero que você consiga entender apenas traçando os paralelos entre este tutorial e o tutorial padrão, em TensorFlow. Além disso, a biblioteca aqui considerada é pensada para programação orientada a objetos, então é bom que você esteja familiarizado pelo menos com os [conceitos de OOP](https://www.tutorialspoint.com/python/python_classes_objects.htm).

## Conteúdo
1. [Introdução](#intro) 
2. [Implementação](#imple)
3. [Referências](#ref)


<a name="intro"></a>
## Introdução

Neste tutorial implementaremos uma Máquina de Boltzmann Restrita usando PyTorch. Será uma excelente oportunidade para comparar essa biblioteca com o TensorFlow e poderemos ver algumas vantagens e desvantagens óbvias de cada uma delas. Eu fiz essa implementação ser extremamente parecida com a do [tutorial em TensorFlow](/2017/07/30/RBM/), então tente traçar os devidos paralelos entre o que desenvolveremos aqui e o que vimos no tutorial padrão. Sem mais enrolações, vamos à implementação!

<a name="imple"></a>
## Implementação

Começamos importando algumas bibliotecas e funcionalidades do PyTorch para redes neurais. Também vamos usar os dados MNIST, que podem ser facilmente  adquiridos do TensorFlow.

{% highlight python %}
import torch # para Deep Learning
import torch.autograd as autograd # para autodiferenciação
import torch.nn as nn # para montar redes neurais
import torch.nn.functional as F # funções do Torch
import torch.optim as optim # para otimização com GDE
import numpy as np # para
# carrega os dados MNIST
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("tmp/", one_hot=False)
data = np.random.permutation(data.train.images) # só precisamos das imagens aqui
{% endhighlight %}

Vamos criar uma classe que implementa a Máquina de Boltzmann Restrita a partir do módulo de redes neurais do PyTorch. Na inicialização da classe, vamos inicializar as super classes e definir os parâmetros do modelo. Também vamos armazenar a quantidade de amostragem de Gibbs que usaremos para *Contrastive Divergence*. Note que os parâmetros precisam ser criados com `nn.Parameter(...)` e não como simples variáveis Torch. Isso faz com que possamos utilizar as funcionalidades herdadas de `nn.Module`, nesse caso a capacidade de gerar facilmente uma lista com todos os parâmetros do modelo e mover esses parâmetros para a GPU.

{% highlight python %}
class RBM(nn.Module):
    def __init__(self, nv=28*28, nh=512, cd_steps=1):
        super(RBM, self).__init__()
        # inicializa os parâmetros da MBR
        self.W = nn.Parameter(torch.randn(nv, nh) * 0.01)
        self.bv = nn.Parameter(torch.zeros(nv))
        self.bh = nn.Parameter(torch.zeros(nh))
        self.cd_steps = cd_steps # define a forma de Contrastive Divergence
{% endhighlight %}

Em seguida, criamos um método para retirar amostras segundo uma distribuição Bernoulli, definida por um vetor de probabilidades `p`.  Se formos utilizar a GPU para o treinamento, precisamos lembrar de mover o tensor aleatório para a GPU após sua inicialização. Isso pode ser feito com o método `.cuda()`. Também vamos definir um método que computa a energia média da MBR, dado um mini-lote  de estados visíveis `v`.

{% highlight python %}
    def bernoulli(self, p):
        # return F.relu(torch.sign(p - autograd.Variable(torch.rand(p.size()).cuda())))     
        return F.relu(torch.sign(p - autograd.Variable(torch.rand(p.size()))))     
        
    def energy(self, v):
        b_term = v.mv(self.bv)
        linear_tranform = F.linear(v, self.W.t(), self.bh)
        h_term = linear_tranform.exp().add(1).log().sum(1)
        return (-h_term -b_term).mean()
{% endhighlight %}

Depois, definimos dois métodos auxiliares `sample_h` e `sample_v`. Eles retiram amostras Bernoulli do estado oculto dado o estado visível e do estado visível dado o estado oculto, respectivamente. 

{% highlight python %}
    def sample_h(self, v):
        ph_given_v = torch.sigmoid(F.linear(v, self.W.t(), self.bh))
        return self.bernoulli(ph_given_v)
    
    def sample_v(self, h):
        pv_given_h = torch.sigmoid(F.linear(h, self.W, self.bv))
        return self.bernoulli(pv_given_h)
{% endhighlight %}

Até aqui, tudo está extremamente similar à implementação em TensorFlow. Agora que definiremos as iterações de *Contrastive Divergence* com amostragem de Gibbs alternada é que veremos a dramática diferença entre as duas bibliotecas de *Deep Learning*.  No PyTorch não precisamos lidar com um loop complicado que define nós em um grafo estático. Em vez disso, podemos usar a própria estrutura de `for loop` do Python. Assim, a definição de um *forward-pass* fica muito mais elegante e fácil de entender. Primeiro, inicializamos o estado visível após `k` iterações de Gibbs alternada para começar com o estado visível inicial. Em seguida, entramos no loop que realiza as iterações de *Contrastive Divergence*, com sucessivas propagações do estado visível para cima e para baixo na MBR. Então retornamos o estado visível inicial `v` e o estado visível após as `k` amostragens de Gibbs, `vk`. Para que o PyTorch não realize *backpropagation* através das cadeias de Markov (iterações de amostragem Gibbs), utilizamos o método `.detach()` para quebrar a corrente que computa os gradientes.

{% highlight python %}
    def forward(self, v):
        vk = v.clone() # inicializa vk
        # realiza k passos de amostragem de Gibbs
        for step in range(self.cd_steps): 
            hk = self.sample_h(vk)
            vk = self.sample_v(hk)
        
        return v, vk.detach()
{% endhighlight %}

Isso finaliza a classe que implemente a Máquina de Boltzmann Restrita. Vamos criar uma instância dela com `RBM()`. Se quisermos utilizar a GPU para o treinamento, precisamos mover os parâmetros da rede para a GPU com o método `.cuda()` da instância. Para o treinamento, vamos utilizar a variação Adam de gradiente descendente estocástico, com uma taxa de aprendizado de 0,001.

{% highlight python %}
rbm = RBM()
# rbm.cuda() # move os parâmetros da rede para a GPU
optimizer = optim.Adam(rbm.parameters(), 0.001)
{% endhighlight %}

O treinamento procede então na forma usual do PyTorch. A cada iteração de treino coletamos um mini-lote de dados, convertemos ele para um tensor Troch e o movemos para a GPU se quisermos acelerar o treinamento. Como os dados da MBR devem ser vetores binários,  precisamos arrendondar os pixeis, que variam de 0 a 1. Com essa amostra, vamos realizar *Contrastive Divergence* com `k` passos de amostragens de Gibbs para obter uma amostra do estado visível após essas iterações. O custo então é definido em termos da \\(\log\\) verossimilhança, que é a diferença entre as energias da MBR quando uma amostra de dados está nas unidades visíveis e quando amostras geradas pelo modelo após `k` iterações de amostragem Gibbs. Esse custo faz sentido intuitivo. Ele captura a noção de que, idealmente, os dados, isto é, `v`, e as amostras geradas pelo modelo, isto é, `vk`, devem ter a mesma energia e, logo, a mesma probabilidade sob a distribuição definida pelo modelo. 

{% highlight python %}
batch_size = 64 # tamanho do mini-lote
epochs = 25 # qtd de épocas de treinamento
for epoch in range(epochs):
    losses = []
    # loop de treinamento
    for i in range(0, len(data)-batch_size, batch_size):
        # cria os mini-lotes
        x_batch = data[i:i+batch_size]
        x_batch = torch.from_numpy(x_batch).float()
        # x_batch = x_batch.cuda()
        x_batch = autograd.Variable(x_batch).bernoulli()

        optimizer.zero_grad() # zera os gradientes computados anteriormente
        v, vk = rbm(x_batch) # realiza o forward-pass (CD com amostragens de Gibbs)
        loss = rbm.energy(v) - rbm.energy(vk) # computa o custo
        losses.append(loss.data[0])

        loss.backward() # realiza o backward-pass
        optimizer.step() # atualiza os parâmetros
    
    print('Custo na época %d: ' % epoch, np.mean(losses))
    if epoch % 5 == 0 and epoch > 0: # a cada 5 épocas
        rbm.cd_steps += 2 # aumenta os as iterações em CD
        print('Alterando para CD%d...' % rbm.cd_steps)
{% endhighlight %}

Acima podemos ver uma outra vantagem do PyTorch sobre o TensorFlow. Como não estamos lidando com o grafo estático, podemos alterar a quantidade de amostragens de Gibbs durante o treinamento. Se quiséssemos fazer isso no TensorFlow, precisaríamos definir um novo gráfico, o que envolveria copiar os parâmetros treinados na execução do grafo anterior para o novo, ou seja, daria muito trabalho. No PyTorch, podemos simplesmente aletrar a varável que define a quantidade de iterações no loop que define o *Contrastive Divergence*.

Mas então qual é a vantagem do TensorFlow, nesse caso? Se o PyTorch oferece maior flexibilidade, o TensorFlow fornece maior velocidade e melhor integração com a GPU. O treinamento da MBR procede muito mais rápido no TensorFlow do que no PyTorch. Além disso, na minha GPU (GTX 1060), o PyTorch só usava 40% da capacidade computacional, enquanto que o TensorFlow batia os 90%. Não sou especialista em otimização computacional, mas chuto que isso aconteça porque o PyTorch volta para o Python muitas vezes no loop de treinamento, ao passo que o TensorFlow executa o *forward-pass*, o *backward-pass* e a atualização dos parâmetros tudo em baixo nível, voltando ao Python apenas para pegar novos mini-lotes. 

<a name="ref"></a>
## Referências

Este tutorial é amplamente baseado no do [deeplearning.net](http://deeplearning.net/tutorial/rbm.html) e é praticamente uma tradução de Thano para TensorFlow. Para um maior entendimento teórico sobre MBR, sugiro a [parte 1](https://www.youtube.com/playlist?list=PLnnr1O8OWc6br8B9iXYFkVJcMc9OnjoZS) e a [parte 2](https://www.youtube.com/playlist?list=PLnnr1O8OWc6bh5CYcqrAjfyzPH3QV745M) da série de vídeos de Geoffrey Hinton sobre Máquinas de Boltzmann Restritas. Por fim, recomendo a parte 3 do livro [*Deep Learning*](http://www.deeplearningbook.org/), sobre as fronteiras de pesquisa em *Deep Learning*. O código deste tutorial está no [meu GitHub](https://github.com/matheusfacure/Tutoriais-de-AM/blob/master/Redes%20Neurais%20Artificiais/RBM_torch.ipynb).

***