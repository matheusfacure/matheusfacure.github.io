---
layout: tutorial
tags: [Tutorial]
comments: true
title: Receita De Aprendizado de Máquina Básico - R
subtitle: "Um aprofundamento teórico sobre a estimação da Regressão Linear"
date: 2017-02-16
true-dt: 2019-01-15
author: "Matheus Facure"
header-img: "img/dark-ann.jpg"
---

## Conteúdo
1. [Introdução](#intro)
2. [Causalidade e Ceteris Paribus](#ceterisparibus)
3. [Intuição](#intuicao)
4. [Fundamentos Matemáticos](#fundamentos)
5. [Parcialização e o Teorema Frisch-Waugh-Lovell](#fwl)
6. [Implementação](#implementacao)
7. [Referências](#ref)

## Pré-requisitos

<a name="1"></a>
## 1 - Carregando os Dados

Os dados estão disponíveis para download neste link: http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29
Os dados são automaticamente baixados na minha pasta Downloads. O comando setwd navega o R até esta pasta. (No Windows o caminho pode ser diferente). 

Essa base contém dados referêntes a um problema de diagnóstico de câncer de mama e esse será o problema que usaremos como exemplo.

```R
setwd("~/Downloads/")
```

Nossos dados são um arquivo de texto onde as entradas são separadas por vírgular. Por isso, colocaremos `sep=","`. Além disso, as variáveis não estão nomeadas e a primeira linha do arquivo já contém amostra de dados. Por isso, colocaremos `header=F`. Por fim, os valores faltantes (missing) estão codificados como uma string "?".

```R
dados  <- read.table("breast-cancer-wisconsin.data", sep=",", header=F, na.strings = c("?"))
```

Como os nossos dados não estão originalmente com nomes nas colunas, nós vamos colocar esses nomes. Os nomes das variáveis pode ser encontrado na documentação dos dados, no link acima. Abaixo, criamos um vetor de strings com os nomes de todas as colunas.

```R
nomes <- c("Sample_code_number",
           "Clump_Thickness",   
           "Uniformity_of_Cell_Size",  
           "Uniformity_of_Cell_Shape",  
           "Marginal_Adhesion",        
           "Single_Epithelial_Cell_Size",
           "Bare_Nuclei",                 
           "Bland_Chromatin",          
           "Normal_Nucleoli",          
           "Mitoses",                    
           "Class")
```

Também vamos criar uma variável que guarda o nome da variável que queremos prever (também chamada de variável Y, target ou dependente). Nesse caso, a variável dependente é `Class`. Ela será o diagnóstico do tumor: 2 se o tumor for benigno e 4 se ele for maligno.

```R
target <- "Class"
```

Por fim, vamos armazenar também o nome de todas as variáveis que vamos usar para prever a classe do tumor. Essas variáveis são chamadas de variáveis independentes (também conhecidas como variáveis X, explicativas, preditivas ou features).

```R
features <- c("Clump_Thickness", "Uniformity_of_Cell_Size",
              "Uniformity_of_Cell_Shape", "Marginal_Adhesion", "Single_Epithelial_Cell_Size",
              "Bare_Nuclei", "Bland_Chromatin", "Normal_Nucleoli", "Mitoses")
```

Tendo salvo os nomes das nossas colunas, vamos agora agora nomeá-las com a função `names` (não confundir com a variável `nomes` que acabamos de criar) 

```R
names(dados) <- nomes
```


Para o nosso problema ficar mais intuitivo, vamos converter a nossa variável Y. Agora ela será 1 quando o tumor for malígno e 0 quando ele for benígno.

```R
dados[, target] <- as.numeric(dados[, target] == 4)
```

<a name="2"></a>
## 2 - Separando os Dados

Logo após ler os dados, o próximo passo envolve separar a base de dados em 3: treino, validação e teste.  

	* Treino: Os dados de treino serão usados para treinar (ou ajustar, ou estimar) o modelo de aprendizado de máquina e qualquer passo de pré-processamento.
	* Validação: Os dados de validação serão utilizados comparar vários modelos e escolher aquele com a melhor performance. Os dados de validação são usados quando queremos ter uma estimativa de como nosso modelo acerta (ou erra) suas previsões em uma base de dados diferente daquele em que ele foi treinado
	* Teste: Os dados de teste serão utilizados para verificar nossa estimativa final de acerto. Ele é parecido com a base de validação, mas pode se observado apenas uma vez, no final do projeto quando nosso modelo final já tiver sido escolhido.

```R
# o seed garante consistência na aleatoridade
set.seed(432) 

# vamos usar 20 % dos dados para teste. O código
# abaixo retira uma amostra aleatória de linhas
# de forma que essa amostra tenha 80% das linhas
# (0.8 * 32561 = 559 linhas)
id <- sample(1:nrow(dados), nrow(dados)*0.8)

# desses 80% de linhas escolhidas, nós vamos pegar 60%
# dos dados originais e chamar de linhas de treino
# (0.6 * 699 = 419 linhas)
id.treino <- sample(id, nrow(dados)*0.6)

# as linhas que não estaão no teste (não estão em id)
# e nem no treino serão as linhas de validação
id.val <- id[!(id %in% id.treino)]

# agora vamos usar essas linhas para criar os 3 datasets
dados.tr <- dados[id.treino,]
dados.val <- dados[id.val,]
dados.test <- dados[-id,]
```
<a name="3"></a>
## 3 - Análise Exploratória

Análise exloratória serve para entendermos um pouco sobre os nossos dados. Algums coisas importantes para se descobrir nessa análise são:

	* A escala de cada variável (média, mínimo, máximo e quantis)
	* Quais variáveis são categóricas e quais são numéricas
	* Há valores faltantes (NAs) em alguma variável

Essa análise deve ser sempre feita na base de treino.

```R
summary(dados.tr)
```
```
Sample_code_number Clump_Thickness  Uniformity_of_Cell_Size Uniformity_of_Cell_Shape Marginal_Adhesion
 Min.   :   95719   Min.   : 1.000   Min.   : 1.000          Min.   : 1.000           Min.   : 1.000   
 1st Qu.:  862838   1st Qu.: 2.000   1st Qu.: 1.000          1st Qu.: 1.000           1st Qu.: 1.000   
 Median : 1174057   Median : 4.000   Median : 1.000          Median : 2.000           Median : 1.000   
 Mean   : 1083825   Mean   : 4.499   Mean   : 3.232          Mean   : 3.267           Mean   : 2.857   
 3rd Qu.: 1241134   3rd Qu.: 6.000   3rd Qu.: 5.000          3rd Qu.: 5.000           3rd Qu.: 4.000   
 Max.   :13454352   Max.   :10.000   Max.   :10.000          Max.   :10.000           Max.   :10.000   
                                                                                                       
 Single_Epithelial_Cell_Size  Bare_Nuclei     Bland_Chromatin  Normal_Nucleoli     Mitoses           Class       
 Min.   : 1.000              Min.   : 1.000   Min.   : 1.000   Min.   : 1.000   Min.   : 1.000   Min.   :0.0000  
 1st Qu.: 2.000              1st Qu.: 1.000   1st Qu.: 2.000   1st Qu.: 1.000   1st Qu.: 1.000   1st Qu.:0.0000  
 Median : 2.000              Median : 1.000   Median : 3.000   Median : 1.000   Median : 1.000   Median :0.0000  
 Mean   : 3.282              Mean   : 3.631   Mean   : 3.484   Mean   : 2.959   Mean   : 1.714   Mean   :0.3652  
 3rd Qu.: 4.000              3rd Qu.: 7.000   3rd Qu.: 5.000   3rd Qu.: 4.000   3rd Qu.: 1.000   3rd Qu.:1.0000  
 Max.   :10.000              Max.   :10.000   Max.   :10.000   Max.   :10.000   Max.   :10.000   Max.   :1.0000  
                             NA's   :10       
```

Com a função summary, podemos responder todas as perguntas acima. 1) Todas as nossas variáveis variam de 1 a 10. 2) Não temos variáveis categóricas, apenas contínuas. 3) A variável Bare_Nuclei contem valores faltantes (NAs). Também podemos ver mais agumas coisas interessantes, como por exemplo, na variável `Class`, a média é 0.365, indicando que 36.5% dos tumores analisados na nossa base de treino são malígnos.

<a name="4"></a>
4 - Pre-Processamento

Antes de passar nossos dados por um algorítmo de aprendizado de máquina é preciso fazer alguns pré-processamentos. Alguns algorítmos funcionam melhor quando os dados estão todos centrados e escalonados (com média 0 e desvio padrão 1). Além disso, precisamos tratar as variáveis. Faltantes de alguma forma. Aqui, vamos simplesmente imputar todos os dados faltates com a médiana.

Esses passos de pré-processamento envolvem uma estimação e toda estimação deve ser feita no dataset de treino. A função `preProcess` treina um pre-processador que, nesse caso, vai ser utilizado para normalizar as variáveis e imputar valores faltantes com a mediana. 


```R
# Utilizaremos o pacote caret para isso. 
# Caso você não o tenha instalado, tire o comentário da linha abaixo.
# install.packages("caret")
library(caret)

preProcValues <- preProcess(dados.tr[, features], method = c("center", "scale", "medianImpute"))
```

Uma vez que esse pre-processador está treinado, nós podemos aplicá-lo nos ddos de treino, validação e teste. Fazemos isso com a função predict e passando tanto o pre-processador quanto os novos dados.

```R
dados.tr = predict(preProcValues, newdata = dados.tr)
dados.val = predict(preProcValues, newdata = dados.val)
dados.test = predict(preProcValues, newdata = dados.test)
```

Rodando a função `summary` novamente podemos ver o efeito do nosso pré-processamento. Todas as variáveis tratadas tem agora média zero. Além disso, a variável `Bare_Nuclei` não tem mais nenhum valor faltante.

```R
summary(dados.tr)
```
```
Sample_code_number Clump_Thickness   Uniformity_of_Cell_Size Uniformity_of_Cell_Shape Marginal_Adhesion
 Min.   :   95719   Min.   :-1.1904   Min.   :-0.7163         Min.   :-0.7586          Min.   :-0.6363  
 1st Qu.:  862838   1st Qu.:-0.8502   1st Qu.:-0.7163         1st Qu.:-0.7586          1st Qu.:-0.6363  
 Median : 1174057   Median :-0.1697   Median :-0.7163         Median :-0.4240          Median :-0.6363  
 Mean   : 1083825   Mean   : 0.0000   Mean   : 0.0000         Mean   : 0.0000          Mean   : 0.0000  
 3rd Qu.: 1241134   3rd Qu.: 0.5108   3rd Qu.: 0.5677         3rd Qu.: 0.5797          3rd Qu.: 0.3918  
 Max.   :13454352   Max.   : 1.8717   Max.   : 2.1727         Max.   : 2.2526          Max.   : 2.4479  
 
 Single_Epithelial_Cell_Size  Bare_Nuclei       Bland_Chromatin   Normal_Nucleoli      Mitoses      
 Min.   :-0.9957             Min.   :-0.71156   Min.   :-0.9984   Min.   :-0.6227   Min.   :-0.375  
 1st Qu.:-0.5593             1st Qu.:-0.71156   1st Qu.:-0.5965   1st Qu.:-0.6227   1st Qu.:-0.375  
 Median :-0.5593             Median :-0.71156   Median :-0.1947   Median :-0.6227   Median :-0.375  
 Mean   : 0.0000             Mean   :-0.01698   Mean   : 0.0000   Mean   : 0.0000   Mean   : 0.000  
 3rd Qu.: 0.3135             3rd Qu.: 0.77603   3rd Qu.: 0.6090   3rd Qu.: 0.3307   3rd Qu.:-0.375  
 Max.   : 2.9320             Max.   : 1.72268   Max.   : 2.6182   Max.   : 2.2376   Max.   : 4.355  
 
     Class       
 Min.   :0.0000  
 1st Qu.:0.0000  
 Median :0.0000  
 Mean   :0.3652  
 3rd Qu.:1.0000  
 Max.   :1.0000  
```

<a name="5"></a>
## 5 - Treinando um Modelo de ML

Com os dados pre-processados, estamos prontos para treinar nosso algorítmo de aprendizado de máquina. Nesse caso, vamos utilizar uma floresta aleatória. A floresta aleatória treina várias [árvores de decisão](http://www.r2d3.us/uma-introducao-visual-ao-aprendizado-de-maquina-1/) em amostras aleatórias dos dados e depois combina as previsões de todas essas árvores em uma previsão final.

```R
# install.packages("randomForest")
library(randomForest)
```

Nosso modelo vai ser ajutado para um problema de classificação, isto é, prever uma variável binária (ou categórica). No nosso caso, estamos tentando prever se um tomor é malígno (Class = 1) ou não (Class = 0). Para isso, vamos usar variás variáveis sobre o tumor, como `Clump_Thickness` e `Uniformity_of_Cell_Size`. Nós representamos o que queremos prever (variável target) e o que vamos usar para prever (features) com uma fórmula. A variável target fica a esqueda do "~". Além disso, como estamos lidando com um problema de classificação, vamos dizer que a variável target é categórica com "as.factor(...)". As variáveis preditivas (features) vao do lado direito do "~" e são separadas por um "+". O "+" aqui NÃO significa que vamos somar as variáveis. Ele quer dizer que vamos INCLUIR essas variáveis no modelo.

```R
formula <- paste0("as.factor(", target, ")~", paste0(features, collapse="+"))
```

Finalmente, famos treinar nossa floresta aleatória. O primeiro argumento para o algorítmo é a fórmula que criamos acima. Em segundo, vem os dados onde vamos treinar esse modelo. Nós sempre treinamos nossos modelos nos dados de treino! Os dados de validação e teste são apenas utilizados para verificar a qualidade do modelo, isto é, como o modelo se sai prevendo novos dados, diferentes daqueles que ele já viu durante o treinamento.

Os próximos argumentos são os híper-parâmetros. Híper-parâmetros ajustam a complexidade do nosso modelo. Se um modelo for muito complexo ele conseguirá ajustar perfeitamente os dados de treino, isto é, suas previsoes nos dados de treino serão perfeitas e ele não errará nada nessa base de dados. No entanto, é comum que um modelo que seja perfeito na base de treino sofra com overfitting, isto é, o modelo ajusta muito bem a base de treino mas não generaliza a boa performance para novos dados, dados que não foram utilizados para treinar o modelo. Por outro lado, se um modelo for muito simples a sua performance será ruim na base de treino, pois ele não consegue achar os padrões nos dados. Assim, quando o modelo for fazer previsões numa nova base de dados também não conseguirá explorar os padrões e terá uma performance ruim. O grande desafio então é encontrar um modelo nem muito complexo e nem muito simples.

Para começar, vamos treinar um modelo relativamente complexo. Será um modelo com apenas 5 árvores (`ntree=5`), mas cada árvore poderá crescer até que cada amostra seja classificada corretamente. Isso é feito com `maxnodes=NULL`, que diz que não há limite para o crescimento de cada árvore. Por fim, o hiper-parâmetro `mntry=9` diz que cada árvore pode usar todas as nove variáveis para ajustar suas previsões.

```R
set.seed(432) 
model <- randomForest(as.formula(formula),
                      data=dados.tr,
                      ntree=5,
                      mntry=9,
                      maxnodes=NULL)
```

<a name="6"></a>
## 6 - Fazendo previsões

Agora que temos um modelo treinado, vamos fazer previsões com eles. Passamos o modelo e os dados para a função `predict`, que faz as previsões.

```
pred.train = predict(model, newdata=dados.tr)
pred.val = predict(model, newdata=dados.val)
```

Podemos ver as 5 primeiras previsões do nosso modelo com o comando `head`
```R
head(pred.train)
```
```
322 367 655  60 199 274 
  0   1   0   1   0   1 
Levels: 0 1
```
Podemos ver que nosso modelo prevê que o cliente da linha 322 tem um tumor benigno. Já o cliente da linha 367 teria um tumor malígno.


<a name="7"></a>
7 - Vendo a performance

Com as nossas previsões em mãos, vamos ver a taxa de acerto, ou acurácia do nosso modelo. A acuráveis é simplesmente a quantidade de previsões corretas dividida pelo tamano da base de dados. Aqui, para conseguir esse número vamos simplesmente tirar a média dos acertos (previsto == observado)

```R
mean(pred.train == dados.tr[,target])
```
```
0.9976134
```
```R
mean(pred.val == dados.val[,target])
```
```
0.9571429
```
Como era de se esperar com um modelo complexo, a nossa performance na base de treino é quase perfeita. Temos uma acurácia de 0.997, o que quer dizer que praticamente todas as nossas previsões estão corretas. Por outro lado, nossas previsões em dados de validação, que não foram utilizados para treinar o modelo, é bem mais baixa: 0.957. Isso quer dizer que estamos errando mais do que 4% das nossas previsões. Nós não vamos observar ainda a performance de teste pois testaremos novos modelos. A base de teste só pode ser olhada quando tivermos selecionado nosso modelo final.

<a name="8"></a>
## 8 - Ajustando a Complexidade

Nosso primeiro modelo provavelmente está sofrendo com overfitting, isto é, boa performance de treino mas não tão boa performance em novos dados. Vamos tentar corrigir isso ajustando a complexidade do nosso modelo. Para fazer isso, vamos treinar vários modelos com complexidade (ou hiper-parâmetros) diferentes. Seleção de modelos é ainda um problema aberto. Existem várias formas de selecionar modelos sendo estudadas, mas uma que funciona bem na prática é definir um espaço de hiper-parâmetros e ir tentando combinações
aleatórias desses hiper-parâmetros (Random Search).

Primeiro definimos todos os possíveis hiper-parâmetros. Por exemplo, para o hiper-parâmetros mtry testaremos 2,3,4,5,8,10,11 e assimo por diante

```R
mtry <- c(2,3,4,5,8,9)
ntree <- c(50, 100, 150)
nodesize <- c(1, 2, 5, 10)
maxnodes <- c(2, 3, 5, 6, 10)
```
Em seguida definimos o tanto de modelos que testaremos. Vamos testar 20 modelos.

```
n.try = 20
```
Depois criamos um dataframe de 20 linhas em que cada linha é uma combinação dos hiper-parâmetros. Definidos acima e cada coluna é corresponde a um hiper-parâmetro. Para isso, a função sample vai retirar 20 amostras (com substituição) do espaço de hiper-parâmetro que definimos acima. Fazemos isso para cada hiper-parâmetro.

```R
try.df <- data.frame(mtry     = sample(mtry, n.try, replace=T),
                     ntree    = sample(ntree, n.try, replace=T), 
                     nodesize = sample(nodesize, n.try, replace=T),
                     maxnodes = sample(maxnodes, n.try, replace=T))
```

Por último, criamos uma coluna vazia nesse dataframe. Essa coluna será preenchida com a performance de cada modelo que testaremos. Nós olharemos apenas a performance em dados diferentes dos usador para treinar, isto é, nos dados de validação. Aliás, esse é justamente o papel dos dados de validação: servir de base de comparação entre modelos.

```R
try.df$performance.val <- NA
```

Vamos ver como é esse data frame. Abaixo podemos ver que cada linha corresponde a uma combinação aleatória de hiper-parâmetros. A última coluna corresponde a performance de validação, que será preenchida a seguir.
```R
head(try.df)
```
```
  mtry ntree nodesize maxnodes performance.val
1    2    50        1        5              NA
2    3    50       10        3              NA
3    3    50        1       10              NA
4    9    50        1        6              NA
5    2   150       10        6              NA
6    3    50        5       10              NA
```

Para testar vários modelos, vamos definir um loop que itera pelas linhas do dataframe acima.

```R
set.seed(432) # para ter consistência nas aleatoriedades
for (linha in 1:nrow(try.df)) {
  
  # treinamos um modelo com os hiper-parâmetros da linha 
  # da iteração atual. Note como estamos treinando nos dados
  # de treino: dados.tr
  model.iter <- randomForest(as.formula(formula),
                             data=dados.tr,
                             ntree=try.df[linha, "ntree"],
                             mntry=try.df[linha, "mtry"],
                             nodesize=try.df[linha, "nodesize"],
                             maxnodes=try.df[linha, "maxnodes"])
  
  # fazemos previsões nos dados de validação: dados.val
  pred.val.iter <- predict(model.iter, newdata=dados.val)
  
  # computamos a acurácia para as nossas previsões nos 
  # dados de validação
  acc.val.iter <- mean(pred.val.iter == dados.val[,target])
  
  # salvamos esse resultado na coluna de performance
  try.df[linha, "performance.val"] <- acc.val.iter
}
```

Vendo mais uma vez nosso dataset `try.df`, podemos observar que a coluna de performance está preenchida com a acurávia de validação para todos os modelos que treinamos.

```R
head(try.df)
```
```
 mtry ntree nodesize maxnodes performance.val
1    4   150        2        5       0.9642857
2    2   100        1       10       0.9714286
3    4    50        2        3       0.9571429
4    9   100       10        3       0.9642857
5    9    50        2        2       0.9642857
6    8   100        2        5       0.9714286
```

Agora que testamos um monte de modelos, temos que escolher o que achamos melhor. Um critério bem simples é pegar aquele que teve menos erros na base de validação. Pode haver mais de um. Para achar esses modelos vamos ver aqueles que tiveram acruácia igual a acruácia máxima encontrada.

```R
best.models <- which(try.df$performance.val == max(try.df$performance.val))

try.df[best.models, ]
```
```
mtry ntree nodesize maxnodes performance.val
8     5    50       10       10       0.9785714
17    8   100        2        5       0.9785714
20    8    50        5        5       0.9785714
```

Temos 4 modelos empatados segundo esse critério de acurácia. Vamos simplesemnte pegar o primeiro.

<a name="9"></a>
## 9 - Modelo Final

Depois de termos testado vários modelo precisamos criar o nosso modelo final, aquele que de fato usaremos para fazer novas previsões.

Já temos nosso modelo selecionado, então não precisamos mais reservar os nossos dados de validação. Vamos então criar uma base final de treino que contém os dados de treino e validação. Para isso, usamos a função `rbind`, que pode ser usada para colar as linhas de dois dataframes.

```R
final.train <- rbind(dados.tr, dados.val)
```

Por fim, vamos treinar um modelo com a mesma complexidade (mesmos hiper-parâmetros) do melhor modelo, selecionado no passo anterior.

```R
final.model <- randomForest(as.formula(formula),
                            data=final.train,
                            ntree=150,
                            mntry=4,
                            nodesize=2,
                            maxnodes=5)
```

<a name="10"></a>
## 10 - Performance Final

Depois de termos testados todos esses modelos e escolhido o que tem a melhor performance de previsão, podemos finalmente ver a nossa performance nos dados de teste.

```R
pred.test <- predict(final.model, newdata=dados.test)
mean(pred.test == dados.test[,target])
```
```
0.9357143
```

A nossa performance final foi pior do que a performance de validação do nosso melhor modelo (0.9785714). Isso acontece. Pode ser que a nossa seleção de hiper-parâmetros tenha levado a um overfitt também nos dados de validação. De qualquer forma, esse é nossa estimativa final de performance e é a que devemos reportar como a esperada.

<a name="11"></a>
## 11 - Prevendo Uma Nova Amostra

Temos nosso modelo treinado. Vamos ver como utilizá-lo na prática. Imagine que um novo paciente chegou e traz consigo as seguintes informações, obtidas por exames médicos

```R
new.sample <- data.frame(Clump_Thickness = 8,
                         Uniformity_of_Cell_Size = NA,
                         Uniformity_of_Cell_Shape = 1.0,
                         Marginal_Adhesion = 8.0,
                         Single_Epithelial_Cell_Size = 2.0,
                         Bare_Nuclei = 10.0,
                         Bland_Chromatin = 3.0,
                         Normal_Nucleoli = NA,
                         Mitoses = 1.0) 
```

Esse novo paciente não tem os dados dos exames de Normal_Nucleoli nem de Uniformity_of_Cell_Size, por isso vamos colocar como um valor faltante. 

Antes de prever qual a probabilidade deste paciente ter um tumor malígno, precisamos passar os dados dele pelo pre-processador. Isso lidará com os valores faltantes de maneira correta. Também colocará os dados na escala normalizada em que o modelo foi treinado.

```R
new.sample.process <- predict(preProcValues, newdata = new.sample)
new.sample.process
```
```
  Clump_Thickness Uniformity_of_Cell_Size Uniformity_of_Cell_Shape Marginal_Adhesion Single_Epithelial_Cell_Size
1        1.191257              -0.7163209               -0.7585897          1.762514                  -0.5593119

  Bare_Nuclei Bland_Chromatin Normal_Nucleoli    Mitoses
1    1.722679      -0.1946855      -0.6227385 -0.3750033

```

Finalmente, podemos fazer nossas previsões para esse novo paciente.

```R
predict(final.model, newdata = new.sample.process)
```
```
1 
1 
Levels: 0 1
```

```R
predict(final.model, newdata = new.sample.process, type = "prob")
```
```
     0    1
1 0.34 0.66
attr(,"class")
[1] "matrix" "votes" 
```

Más notícias: Nosso modelo prevê que o tumor deste paciente é um tumor malígno (Class=1). Mas isso não é tudo. Nosso modelo diz que a probabilidade deste paciente ter um tumor malígno é de 0.66. Talvez seja uma boa ideia o paciente fazer uns testes a mais para termos mais certeza.



