---
layout: tutorial
tags: [Tutorial-alt]
comments: true
title: <em>Thompson Sampling</em>
subtitle: "Um algoritmo de aprendizagem por reforço para testes eficientes"
date: 2017-03-04
true-dt: 2017-08-26
author: "Matheus Facure"
header-img: "img/dark-ann.jpg"
---

<div class="row">
<ul class="nav nav-tabs navbar-left">
    <li><a href="/2017/03/04/bernoulli-bandits-thompson">Python</a></li>
    <li class="active"><a href="#">Scala</a></li>
</ul>
</div>

## Pré-requisitos

Esta é uma versão alternativa do tutorial em Python. Assim sendo, vou pressupor que você já está familiarizado com o tutorial padrão, o que tornará este muito mais rápido e direto. Além disso, este tutorial será uma versão desafiadora que usa princípios de programação funcional, ou seja, não haverá redefinição de variáveis e estruturas de loops serão substituídas por recursão. Assim, é fundamental estar familiarizado com funções recursivas. Vou usar `sbt` para gerenciar pacotes no Scala. Caso não tenha familiaridade com essa ferramenta, sugiro que leia [este tutorial](https://github.com/shekhargulati/52-technologies-in-2016/blob/master/02-sbt/README.md) antes de prosseguir

## *Thompson Sampling*

Antes de qualquer coisa, crie um novo diretório e, dentro dele, crie uma arquivo chamado "build.sbt". Copie e cole as seguintes linhas para esse arquivo
```
version := "0.1.0"
scalaVersion := "2.11.6"
libraryDependencies += "org.scalanlp" % "breeze_2.11" % "0.11.2"
```

Então, no início do seu script, importe estas dependências:

{% highlight scala %}
import breeze.stats.distributions.Beta
import util.Random
{% endhighlight %}


Isso tomará conta de importar o pacote de computação numérica [Breeze](https://github.com/scalanlp/breeze). Nós só vamos utilizá-lo para retirar amostrar aleatórias de uma distribuição beta. Não é preciso familiaridade com o pacote para entender este tutorial. Assim como no tutorial em Python, vamos começar criando uma classe de modelo falso que simula classificações com uma acurácia passada como parâmetro.

{% highlight scala %}
class fakeModel(accuracy: Double) {
    val rnd = new scala.util.Random // cria gerador aleatorio

    def score: Int = if (rnd.nextDouble < accuracy) 1 else 0 // checa acerto

}
{% endhighlight %}

Para termos algo com o que comparar, vamos implementar uma classe `randomSelection` que realiza a política de seleção aleatória. No momento da criação, essa classe terá como parâmetros uma sequência de modelos falsos e o número de experiências para realizar. É no método `play` que acontece a seleção aleatória de fato. Dentro desse método vamos definir uma função recursiva que implementa as iterações do experimento. O caso base dessa função é quando o contador `i` excede o número de experimentos. Nesse caso a iteração termina, mostramos quanto cada modelo acertou, quantas vezes cada modelo foi escolhido e retornamos a acurácia **do experimento como um todo**. No caso recursivo da função, selecionamos um modelo aleatoriamente e simulamos uma classificação. Entramos então na invocação seguinte da função com a contagem de iterações incrementada (`i+1`) e atualizando a quantidade de acertos do experimento (` rights + isRight`), a quantidade de acertos do modelo selecionado (`ightCount.updated(chosenMod, rightCount(chosenMod) + isRight)`) e a quantidade de vezes que cada modelo foi selecionado (`numSelect.updated(chosenMod, numSelect(chosenMod) + 1)`).

{% highlight scala %}
class randomSelection(models: Seq[fakeModel], nExper: Int) {
    // Implementa uma política de escolha aleatória

    // parm: models: uma lista de fakeModels
    
    // parm: nExper: o a qtd de experimentos para rodar
    
    val rnd = new scala.util.Random // cria gerador de números aleatorio


    def play = {
        // Itera pelos números de experimentos

        // retorna: a acurácia da política nos experimentos

        def loop(i: Int, rights: Int,
                 rightCount: Vector[Int],
                 numSelect: Vector[Int]): Int = {
            
            // caso base da recursão termina o loop

            if (i > nExper){
                
                println("Acertos de cada modelo: " + rightCount)
                println("Escolhas por modelo: " + numSelect)
                rights // retorna o número de acertos


            } else {

                val chosenMod: Int =  rnd.nextInt(models.length) // escolhe um modelo

                val isRight: Int = models(chosenMod).score // checa se o modelo acerta
                
                // vai para a próxima iteração incrementando o número de acertos

                loop(i + 1, rights + isRight,
                    rightCount.updated(chosenMod, rightCount(chosenMod) + isRight),
                    numSelect.updated(chosenMod, numSelect(chosenMod) + 1)) 
            }
        }

        // inicia vetores de contagem

        val num_rigths: Vector[Int] = Vector.fill(models.length)(0)
        val num_select: Vector[Int] = Vector.fill(models.length)(0)

        // retorna a acurácia nos experimentos

        100.0 * loop(0, 0, num_rigths, num_select) / nExper 

    }
}
{% endhighlight %}

No método `play`, iniciamos as contagens de seleção e de acertos como vetores preenchidos com zeros. Também inicializamos os contadores `i` e a quantidade de acertos do experimentos ambas em zero. Por fim, retornamos a acurácia do experimento, isto é, o número de acertos, retornado pela função recursiva, dividido pela quantidade de experimentos. 

A classe `thompsonSampling` será bastante similar. Uma diferença é que, além de vetores para armazenar a quantidade de acertos e de seleções, também precisaremos de um vetor para armazenar a quantidade de erros. Isso será utilizado no momento de retirar amostras betas para cada modelo. Além disso, em vez de selecionar os modelos aleatoriamente, escolheremos aquele com maior amostra beta.

{% highlight scala %}
lass thompsonSampling(models: Seq[fakeModel], nExper: Int) {
    // Implementa uma política de Thompson Sampling para escolha

    // parm: models: uma lista de fakeModels

    // parm: nExper: o a qtd de experimentos para rodar


    def play = {
        // define o loop de iteração pelos números de experimentos

        def loop(i: Int, rights: Int,
                rightCount: Vector[Int],
                wrongCount: Vector[Int],
                numSelect: Vector[Int]): Int = {
            
            // caso base da recursão termina o loop

            if (i >= nExper){
                println("Acertos de cada modelo: " + rightCount)
                println("Escolhas por modelo: " + numSelect)
                rights // termina a iteração retornando o número de acertos

            
            } else {

                // cria sequencia com amostras beta para cada modelo

                val modBetas: Seq[Double] = Range(0, models.length).map{

                    i => new Beta(rightCount(i)+1, wrongCount(i)+1).draw
                }
                
                // argmax da sequencia criada acima para escolher o modelo

                val chosenMod: Int = modBetas.view.zipWithIndex.maxBy(_._1)._2 

                // verifica se o modelo escolhido acerta

                val isRight: Int = models(chosenMod).score

                // vai para o próximo experimento, atualizando os vetores de contagem e acertos

                if (isRight == 1) loop(i + 1, rights + isRight,
                                         rightCount.updated(chosenMod, rightCount(chosenMod) + 1), 
                                         wrongCount,
                                         numSelect.updated(chosenMod, numSelect(chosenMod) + 1))
                
                else loop(i + 1, rights + isRight,
                                         rightCount, 
                                         wrongCount.updated(chosenMod, wrongCount(chosenMod) + 1),
                                         numSelect.updated(chosenMod, numSelect(chosenMod) + 1))
            }
        }
        // inicializa os vetores de contagens

        val num_ones: Vector[Int] = Vector.fill(models.length)(0)
        val num_zeros: Vector[Int] = Vector.fill(models.length)(0)
        val num_select: Vector[Int] = Vector.fill(models.length)(0)

        // itera pelos experimentos

        100.0 * loop(0, 0, num_ones, num_zeros, num_select) / nExper 
    }
}
{% endhighlight %}

Por fim, criamos o objeto principal do nosso script para comparar as abordagens de seleção aleatória e *Thompson Sampling*. 

{% highlight scala %}
object RL extends App {

    val m1 = new fakeModel(0.62)
    val m2 = new fakeModel(0.6)
    val rs = new randomSelection(Seq(m1, m2), 20000)
    val ts = new thompsonSampling(Seq(m1, m2), 20000)
    println(rs.play)  
    println(ts.play) 

}
{% endhighlight %}

Quando rodamos esse script o resultado é algo como o seguinte.

```
Acertos de cada modelo: Vector(6175, 5967)
Escolhas por modelo: Vector(9944, 10057)
60.71
Acertos de cada modelo: Vector(12121, 297)
Escolhas por modelo: Vector(19490, 510)
62.09
```

Podemos ver como *Thompson Sampling* rapidamente para de experimentar o modelo não ótimo. Isso faz com que o experimento com esse algoritmo tenha, como um todo, uma acurácia bem próxima da do modelo ótimo, já que o explora quase exclusivamente. Por outro lado, na seleção aleatória iremos utilizar o modelo não ótimo para aproximadamente 50% das classificações, incorrendo num custo que poderia ser evitado com *Thompson Sampling*.

***
