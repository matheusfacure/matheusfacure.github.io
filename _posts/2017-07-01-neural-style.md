---
layout: post
comments: true
title: Transferência de Estilo
subtitle: "Explorando o conceito de custo perceptual para criar imagens artísticas."
date: 2017-07-06
tags: [Post]
author: "Matheus Facure"
header-img: /img/fundo_main.png
modal-id: 11
thumbnail: /img/portfolio/style-transfer/thumbnail.jpg
description: Uma forma de entender arte é como a percepção que o artista tem do mundo. Assim, arte se manifesta na sobreposição de um estilo, particular a cada autor, e de um fenômeno da realidade, tais como uma paisagem, um copo d’água ou um sonho. Arte pode então ser uma espécie de filtro ou lente, através da qual o artista nos apresenta o mundo. Com um pouco de inteligência artificial e matemática, é possível explorar essa ideia e transferir estilos de imagens para imagens, fazendo com que fotos de viagem se assemelhem a quadros ou tornando alguns personagens ainda mais assustadores. O algoritmo de transferência de estilo apresentado aqui é um dos melhores exemplos que há das possíveis integrações entre aprendizado de máquina e as ciências humanas!
---

## Conteúdo
1. [Intuição](#intuicao) 
2. [Modelo Matemático](#mat)
4. [Resultados](#result)
5. [Referências](#ref)

<a name="intuicao"></a>
## Intuição

Em transferência de estilo, o objetivo é pegar uma imagem qualquer - que vamos chamar de **imagem de conteúdo** - e re-renderizá-la com o estilo de uma outra imagem - que vamos chamar de **imagem de estilo** -, geralmente uma pintura famosa. Essa é uma tarefa extremamente difícil. Se sequer sabemos muito bem como definir *estilo*, como então  seremos capazes de transferi-lo de imagem para imagem e de maneira programática? Naturalmente, a resposta está em inteligência artificial, mas isso não quer dizer muito. Aliás, deixe-me antes contar uma coisa: IA hoje não passa de técnicas de maximização e minimização. Então, ao tentar resolver transferência de estilo com IA, teremos que, de algum jeito, formular essa tarefa como um problema de otimização.

O processo de otimização envolve dois componentes: (1) uma métrica para otimizar, geralmente algo que mede um custo, que chamamos de **função objetivo**; (2) uma forma de otimizar a métrica antes definida. Mesmo sabendo só isso sobre otimização, você já deve ter percebido o tanto de criatividade necessária para formular o problema de transferência de estilo utilizando apenas essas duas etapas. Pois bem, eis brilhante ideia.

Em 2015, alguns cientistas muito inteligentes propuseram um método para separar e recombinar o estilo e o conteúdo de uma imagem. Esse método parte de três imagens: a de estilo, a de conteúdo e uma nova imagem que será gerada. Em seguida, ele pega a imagem gerada e de conteúdo e **maximiza a similaridade entre os conteúdos delas extraídos**. Ao mesmo tempo, o método **maximiza a similaridade entre os estilos extraídos** da imagem gerada e da imagem de estilo. Assim, a chave para entender como funciona a transferência de estilo está em entender como separar estilo de conteúdo. Mas antes de entrar nesses detalhes, precisamos falar um pouco sobre um tipo especial de rede neural artificial.

Redes Neurais Convolucionais (RNC, daqui em diante) são uma classe de redes neurais que consegue exceder a performance humana na tarefa de reconhecimento de imagem. RNC são feitas empilhando camadas de pequenas unidades de processamento, chamadas filtros. Essas camadas captam **estímulos visuais de forma hierárquica** e os filtros de cada camada conseguem se especializar em detectar algum padrão da imagem. Filtros em camadas mais profundas da RNC detectam padrões mais complexos. Mais ainda, podemos pensar nas camadas de filtros como **representações internas abstratas** que a RNC aprende ([falo mais sobre isso em outro post](https://matheusfacure.github.io/2017/05/09/deepdream/)).

Voltando ao problema de transferência de estilo, descobriu-se que as camadas das RNCs codificavam imagens em representações internas que continham carga semântica, onde estilo e conteúdo eram separáveis! Com isso foi possível desenvolver o algoritmo de [Transferência de Estilo Neural](https://arxiv.org/abs/1508.06576). Intuitivamente, os passos do algoritmo são os seguintes:

1. Inicie uma imagem com valores aleatórios
2. Use uma RNC treinada e, a partir de sua representação interna, extraia o conteúdo da nova imagem e da imagem de conteúdo.
3. Com a representação interna da mesma RNC, extraia o estilo da nova imagem e da imagem de estilo.
4. Defina o objetivo e ser minimizado como a diferença entre o estilo da imagem gerada e o da imagem de estilo **mais** a diferença entre o conteúdo da imagem gerada e o da imagem de conteúdo.
5. Iterativamente, atualize a imagem gerada de forma a minimizar o objetivo definido em 4.

Vamos agora formalizar esses passos com um pouco de matemática. Se você não se sente confortável com matemática, eu sugiro se ater à explicação intuitiva que já dei e [pular para os resultados](#result).

<a name="mat"></a>
## Modelo Matemático

Antes de ver a matemática, vamos analisar visualmente a construção do objetivo a ser minimizado.

<figure class="figure center-block">
  <img class="img-responsive center-block thumbnail" src="/img/portfolio/style-transfer/perceptual-loss.png" alt="perceptual-loss" />
  <figcaption class="figure-caption text-center">Adaptada do artigo <a href="https://arxiv.org/abs/1603.08155"><em>Perceptual Losses for Real-Time Style Transfer and Super-Resolution</em></a>.</figcaption>
</figure>

No esquema acima, o tracejado delimita uma rede neural treinada para reconhecimento de imagem. Nesse caso, trata-se da rede VGG-16. Os quadrados na entrada da rede representam, de cima para baixo, as imagens de estilo, a imagem gerada e a imagem de conteúdo. Vamos alimentar essa RNC com essas três imagens. As setas apontando para cima denotam o processo de colher as representações de estilo da imagem gerada e de estilo. Note como essa representação é colhida a partir da atividade neural de várias camadas da rede. Analogamente, as setas para baixo representam o processo de colher a representação de conteúdo a partir da atividade neural em uma camada intermediária da rede.

Mais formalmente, seja \\(\pmb{p}\\) e \\(\pmb{x}\\) a imagem de conteúdo e a imagem gerada, respectivamente, e seja \\(\pmb{P^l}\\) e \\(\pmb{F^l}\\) as respectivas representações internas na camada \\(l\\) de uma RNC treinada. Então podemos definir a diferença de conteúdo ou **custo de conteúdo** como 

$$\mathcal{L}_{cont}(\pmb{p}, \pmb{x}, l)=\frac{1}{1} \sum_{i, j}(P_{i,j}^l - F_{i,j}^l)^2 $$

Em que essa soma é definida nas duas direções, \\(i\\) e \\(j\\), da estrutura de grade que são as representações internas de imagens em camadas convolucionais. Assim, podemos ver que o **custo de conteúdo a ser minimizado é a distância euclidiana** entre as representações internas da RNC
quando alimentada com a imagem de conteúdo e com a imagem gerada. A otimização consiste em atualizar \\(\pmb{x}\\) de forma a diminuir a diferença acima definida. Isso é feito com [gradiente descendente](https://matheusfacure.github.io/2017/02/20/MQO-Gradiente-Descendente/), por meio de [*backpropagation*](https://matheusfacure.github.io/2017/03/10/backprop/). Um último detalhe é que normalmente se define a diferença acima utilizando várias camadas da RNC. No artigo original, tínhamos \\(l=\\{conv1\\_1, conv2\\_1, conv3\\_1, conv4\\_1, conv5\\_1\\}\\) e a RNN considerada era a *VGG-Network*.

O custo (ou diferença) de estilo é um pouco mais complexo. Em vez de comparar diretamente a atividade nos filtros das camadas da RNC, primeiro computamos a matriz de correlação desses filtros (também chamada de matriz Gram):

$$G^l_{i,j}=\sum_{k}F^l_{i,k} \cdot F^l_{j,k}$$

Adicionando às definições prévias, seja \\(\pmb{a}\\) a imagem de estilo e seja \\(A^l_{i,j}\\) representação de estilo como definida pela matriz de correlações, podemos definir o custo de estilo como sendo:

$$\mathcal{L}_{cont}(\pmb{a}, \pmb{x})= \sum_{i, j}(G_{i,j}^l - A_{i,j}^l)^2 $$

Intuitivamente, esse segundo componente de custo leva em consideração quais unidades de processamento da rede neural tendem a co-ativar quando se deparam com as imagens em questão. Assim, conseguimos captar a diferença de estilo como sendo texturas que aparecem conjuntamente, em diferentes partes da imagem. Novamente, minimizamos esse custo com gradiente descendente por meio de *backpropagation*. Mais uma vez, esse custo pode ser construído para várias camadas da RNC.

O conceito chave que se deve entender aqui é o de **custo perceptual**. Note como os objetivos acima definidos não são em termos das imagens em si, mas de suas representações neurais. Essa ideia é extremamente poderosa e pode ser estendida para outras aplicações além de transferência de estilo, como, por exemplo, super resolução e modelos geradores.

<a name="result"></a>
## Resultados

**AVISO: Contém imagens fortes!!!**

A foto abaixo mostra o resultado de juntar o estilo de uma pintura de Van Gogh com uma foto que tirei no jardim botânico de Buenos Aires.

<img class="img-responsive center-block thumbnail" src="/img/portfolio/style-transfer/st1.jpg" alt="perceptual-loss"/>

Mas não precisamos nos restringir a uma única imagem de estilo. Da forma como definimos o problema de otimização, colher o estilo várias imagens artísticas é tão simples como colar duas pinturas lado a lado antes de passá-las pela rede neural.

<div id="myCarousel1" class="carousel slide" data-ride="carousel">
  <!-- Indicators -->
  <ol class="carousel-indicators">
    <li data-target="#myCarousel1" data-slide-to="0" class="active"></li>
    <li data-target="#myCarousel1" data-slide-to="1"></li>
  </ol>

  <!-- Wrapper for slides -->
  <div class="carousel-inner">
    <div class="item active">
      <img src="/img/portfolio/style-transfer/teatro_in.jpg" alt="in">
      <div class="carousel-caption">
        <h4>Entradas</h4>
      </div>
    </div>

    <div class="item">
      <img src="/img/portfolio/style-transfer/teatro_out.jpg" alt="out">
      <div class="carousel-caption">
        <h4>Saídas</h4>
      </div>
    </div>

  </div>

  <!-- Left and right controls -->
  <a class="left carousel-control" href="#myCarousel1" data-slide="prev">
    <span class="fa fa-angle-left"></span>
  </a>
  <a class="right carousel-control" href="#myCarousel1" data-slide="next">
    <span class="fa fa-angle-right"></span>
  </a>
</div>

Acima, nas entradas, temos a imagem de conteúdo no centro e as de estilo dos lados. Nas saídas, a imagem da esquerda contém apenas o estilo do quadro cubista. Analogamente, a imagem da direita contém apenas o estilo do quadro de Kandinsky. No meio, temos a imagem que mescla o estilo de ambos os quadros e o aplica na imagem de conteúdo. 

Além de criar imagens artísticas, podemos usar transferência de estilo para deixar uma imagem mais assustadora. Foi isso que fiz abaixo, utilizando o estilo macabro de uma imagem de terror.


<div id="myCarousel2" class="carousel slide center-block" data-ride="carousel" style="width: 60%;">
  <!-- Indicators -->
  <ol class="carousel-indicators">
    <li data-target="#myCarousel2" data-slide-to="0" class="active"></li>
    <li data-target="#myCarousel2" data-slide-to="1"></li>
    <li data-target="#myCarousel2" data-slide-to="2"></li>
  </ol>

  <!-- Wrapper for slides -->
  <div class="carousel-inner">
    <div class="item active">
      <img src="/img/portfolio/style-transfer/terror1.jpg" alt="terror1">
      <div class="carousel-caption">
      <h4>Conteúdo</h4>
      </div>
    </div>

    <div class="item">
      <img src="/img/portfolio/style-transfer/terror2.jpg" alt="terror2">
      <div class="carousel-caption">
      <h4>Estilo</h4>
      </div>
    </div>

    <div class="item">
      <img src="/img/portfolio/style-transfer/terror3.jpg" alt="terror3">
      <div class="carousel-caption">
      <h4>Transformada</h4>
      </div>
    </div>

  </div>

  <!-- Left and right controls -->
  <a class="left carousel-control" href="#myCarousel2" data-slide="prev">
    <span class="fa fa-angle-left"></span>
  </a>
  <a class="right carousel-control" href="#myCarousel2" data-slide="next">
    <i class="fa fa-angle-right"></i>
  </a>
</div>


Finalmente, vamos ver se esse algoritmo consegue me ajudar a desenhar melhor. Abaixo, eu fiz alguns rabiscos no computador e usei transferência de estilo para transformá-lo em uma obra de arte. Podemos ver que o resultado é satisfatório, mas acho que a transferência de estilo não consegue fazer milagres...

<img class="img-responsive center-block thumbnail" src="/img/portfolio/style-transfer/desenho.jpg" alt="desenho"/>

<a name="ref"></a>
## Referências

Transferência de estilo se popularizou tanto na academia quanto na indústria. Hoje, existem inúmeros sites e apps disponíveis para aplicar o algoritmo. Até onde sei, o site mais conhecido para isso é o [DeepArt](https://deepart.io/). Como o algoritmo de transferência de estilo é computacionalmente muito intenso, o site dá um prazo longo para processar suas imagens. Apenas para se ter uma ideia, já faz uns 2 meses desde que mandei minhas fotos e até agora não obtive resposta. Felizmente, achei uma [implementação do algoritmo em Python e TensorFlow](https://github.com/anishathalye/neural-style). Então repliquei essa implementação, simplificando algumas coisas e documentando em português para aqueles que quiserem gerar suas próprias transferências. O código está disponível do [meu GitHub](https://github.com/matheusfacure/neural-art), com as devidas referências à implementação original. Note que **não recomendo rodar o algoritmo caso não tenha um GPU**!

O artigo seminal de transferência de estilo se chama [*A Neural Algorithm of Artistic Style (Leon A. Gatys, Alexander S. Ecker, Matthias Bethge*, 2015)](https://arxiv.org/abs/1508.06576). Recentemente, Yongcheng Jing *et al* fizeram uma revisão das principais técnicas desenvolvidas nos últimos anos para transferência de estilo. O artigo deles foi publicado em maio deste ano, com o nome de [*Neural Style Transfer: A Review*](https://arxiv.org/abs/1705.04058). Finalmente, Justin Johnson, Alexandre Alahi, Li Fei-Fei tratam do custo perceptual e uma versão rápida de transferência de estilo em [*Perceptual Losses for Real-Time Style Transfer and Super-Resolution*
](https://arxiv.org/abs/1603.08155).
