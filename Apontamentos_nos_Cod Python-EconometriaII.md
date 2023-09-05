Apontamentos dos Códigos Python usados nas aulas 1 a 4 de Econometria II

O objetivo desses apontamentos é um esforço de tentar entender
estritamente o que é a linguagem Python como funciona seus principais
pacotes e funções voltadas a análise das ciências de dados, com foco na
econometria, conforme objetivos apresentados na própria ementa da
disciplina Econometria II, porém sem focar nas teorias econométricas e
na parte dos resultados, visto que essa parte se faz necessariamente
presente na lista de exercícios da disciplina.

1\. Pyton e suas IDEs

** Python** é uma *linguagem de programação* de alto nível,
*interpretada* e de *propósito geral*. Uma *linguagem de programação* é
uma forma padronizada de comunicação entre humanos e máquinas,
permitindo que os desenvolvedores expressem algoritmos e instruções que
serão executadas por um computador, *é linguagem interpretada*, pois,
permite que você escreva e execute código diretamente, sem a necessidade
de um processo de compilação prévio. Isso oferece vantagens em termos de
agilidade no desenvolvimento e facilidade de uso, especialmente para
tarefas como scripting, prototipação e análise de dados.

As linguagens de programação fornecem uma sintaxe e uma semântica
específica, que definem a estrutura e o significado das instruções. As
linguagens de programação podem ser classificadas em diferentes níveis
de abstração, variando desde linguagens de baixo nível, como a linguagem
Assembly, que estão mais próximas da linguagem de máquina, até
linguagens de alto nível, como Python, Java, R, C++, entre outras, que
fornecem abstrações mais poderosas e facilitam o desenvolvimento de
software complexo.

De *propósito geral,* pois é amplamente usada em uma variedade de
domínios, desde desenvolvimento web e científico até automação de
tarefas e inteligência artificial. Python se destaca por sua sintaxe
clara e legível, além de possuir uma vasta comunidade de desenvolvedores
e bibliotecas disponíveis.

**Jupyter** **Notebook** é uma aplicação web (ou extensão do VS Code )
de código aberto que permite criar e compartilhar documentos que
contenham código executável, visualizações, texto explicativo e outros
elementos. Ele é amplamente utilizado no campo da ciência de dados e
análise exploratória, *permitindo aos usuários escrever, testar e
documentar seu código Python de forma interativa, *sendo esse seu grande
diferencial. Jupyter Notebook é organizado em células, onde cada célula
pode conter código, texto formatado, equações matemáticas,
visualizações, entre outros.

**VS** **Code** (Visual Studio Code) é um editor de código-fonte
desenvolvido pela Microsoft. Ele é altamente popular entre
desenvolvedores de software devido à sua interface intuitiva, recursos
avançados de edição, depuração integrada, suporte a extensões e
integração com controle de versão. O VS Code é uma ferramenta poderosa
para escrever, depurar e executar código Python, além de suportar várias
outras linguagens de programação.

1.1Unindo os três

Assim com não é necessário um capacete e uma cotoveleira para conseguir
andar sobre um skate, é também possível usar o Python apenas com o
terminal, todavia você ficará limitado a manobras menos "profissionais"
não conseguindo utilizar a capacidade máxima dessa ferramenta. O Python
é a *linguagem de programação em si*, enquanto Jupyter Notebook e o VS
Code são ambientes de desenvolvimento (*IDE -- Integrated Development
Environment*) utilizados para escrever e executar o código Python, além
de suportarem outras linguagens. O Jupyter Notebook é um ambiente
interativo e colaborativo que permite a criação de documentos com código
executável, enquanto o VS Code é um editor de código tradicional, porém
com recursos avançados de edição e depuração, e extensões, incluindo o
próprio Jupyter Notebook, assim ambos os ambientes são amplamente
utilizados por desenvolvedores Python.

1.2 Código de progamação.

Um código em Python é uma sequência de instruções escritas na linguagem
de programação Python. Essa sequência de instruções, vira um roteiro,
que literalmente é chamada de **Script**, que quando salvo como arquivo,
contém um conjunto de instruções escritas em uma linguagem de
programação, como Python. Assim os scripts são geralmente usados para
automatizar tarefas ou executar um conjunto de comandos em sequência,
usando alguma linguagem de programação com Python, C++, R, etc.

ATENÇÃO! A primeira coisa ao monta o seu roteiro de códigos é iniciar
preparando o Software Python a buscar os pacotes necessários para a
execução do programa. Seria análogo a um *cheklist* de tudo que se
precisa ter em mãos antes de começar uma tarefa. Ao desenvolver um
script em Python, se faz necessário apresentar (importar) os pacotes
relevantes para ter acesso às funcionalidades que eles oferecem.
Portanto a importação de pacotes permite que você use as classes,
funções e variáveis definidas nesses pacotes em seu próprio código. Sem
a importação adequada, seu *programa não terá conhecimento dessas
funcionalidades* e você receberá *erros ao tentar usá-las*.

Todavia há um outro problema, não será possível importar alguns pacotes
no seu programa Python, e isso acontece pois *a grande maioria dos
pacotes estão em falta*, o que faz necessário a instalação desses
pacotes para que assim se possa importá-los em seu roteiro.

O Python não contém todos os pacotes existentes, pelo motivo de haverem
muitos, como também por abrangem diversas categorias de processamentos
distintas das quais não se pode determinar se serão úteis para todos os
usuários. Se fossem incluídos muitos pacotes sobrecarregaria alguns
computadores, bem como poderia deixar o programa mais lento.

De acordo com o Python Package Index (PyPI), que é um repositório
central de pacotes Python, existem atualmente 198.826 pacotes
disponíveis. Esses pacotes cobrem uma ampla gama de tarefas, incluindo
**ciência de dados**, aprendizado de máquina, processamento de linguagem
natural, web development, sistemas operacionais, jogos e muito mais! Só
na nossa categoria de estudo, que seria Ciências de dados, em alguns
casos Aprendizado de máquina, existem atualmente 44.932 pacotes
disponíveis! Para importar pacotes, você pode usar a declaração *import*
seguida do nome do pacote.

Caso haja um erro é possível que o pacote não esteja instalado. Para
instalar o pacote usando a IDE do Vscode na "caixinha de texto" do
Jupyter Notebook basta escrever o comando **!pip install** e o **nome do
pacote **e pressione Ctrl + Enter, para executar a instalação. Pelo
terminal não há a necessidade do sinal de escrever o sinal de exclamação
ficando apenas **pip install nome_do_pacote** e para executar a
instalação apenas pressione Enter.

2\. Os Códigos

Os códigos são os mesmos usados nas aulas, apenas estão com uma
brevíssima explicação de cada pacote/função por meio do *dicionário do
Vscode* (ao deixar o cursor ou a mãozinha em cima de cada função), como
também por pesquisas externas e comentários próprios mediante o processo
de estudo, além é claro das explicações já existentes no material das
aulas.

Pode-se copiar o código e colar diretamente na "caixinha" de texto do
Jupyter Notebook para rodar com os mesmos resultados das aulas. Dica:
(*Pode ser interessante rodar linha por linha, para assim ver a
construção do programa, em vez de rodar o texto de uma só vez, para isso
basta posicionar o cursor na primeira letra da primeira linha e ir
apertando o botão F10 no* *teclado*.)

2.1 Regressão Linear

Este é o roteiro (script) 01 da aula 01, aonde é possível se criar uma
regressão linear, onde se faz necessária primeiramente a importação dos
pacotes. Os pacotes que serão necessários são; Numpy, que é um pacote de
Álgebra linear e Estatística; Matplotlib, que cria visualizações
estáticas; e animadas e interativas de dados. Existem outras formas de
se fazer uma regressão linear, essa não é a forma mais prática, todavia
essa certamente é uma das formas mais didáticas, por possuir mais
detalhes em sua execução.

Segue o código:

``` python
import numpy as np # Essa importação diz que vamos usar numpy toda vez
que escrevermos np

import matplotlib.pyplot as plt # Essa importação diz que vamos usar
matplotlib, porém não todo o pacote.

# Mas tão somente a parte que gera os gráficos, e para isso existe um Submódulo chamado .pyplot
# Assim fica "matplotlib.pyplot"
# Isso vai acontecer toda vez que se ecrever plt
# Deve-se notar que escrever np e plt é só por comodidade pois facilita a escrita do roteiro.
# Alturas fictícias dos pais e dos filhos (em polegadas)

alturas_pais = np.array([72.3, 71.2, 70.2, 69.3, 68.3, 67.3, 66.2,
65.5, 64.5])

alturas_filhos = np.array([72, 69.7, 69.5, 69, 68.1, 67.2, 67.1, 66.5,
65.8])

# .array é um Submódulo que gera matrizes com os números adicionados em parênteses
# Após esse código alturas_pais é uma matriz, assim como alturas_filhos
# Uma matriz 9x1 correspondente aos valores que foram adicionados
# Realizar regressão linear usando o NumPy

coeficientes = np.polyfit(alturas_pais, alturas_filhos, 1)

funcao_regressao = np.poly1d(coeficientes)

#.polyfit é um submódulo do pacote numpy

#.polyfit calcula os coeficientes do polinômio que minimizam o erro
quadrático

# Plotar os dados e a linha de regressão

plt.scatter(alturas_pais, alturas_filhos, label='Dados')#.scatter é usado para criar um gráfico de dispersão

plt.plot(alturas_pais, funcao_regressao(alturas_pais), color='red',
label='Regressão Linear') #.plot é usado para criar um gráfico de linhas.

plt.xlabel('Altura dos Pais (polegadas)')# .xlabel é o título da
variável x

plt.ylabel('Altura dos Filhos (polegadas)')# .xlabel é o título da
variável y

plt.title('Regressão Linear de Altura dos Filhos vs. Altura dos Pais (Exemplo de Galton)')# .title é o título do gráfico

plt.legend()# .legend é para gerar a legenda do gráfico

plt.grid(True)# .grid é para gerar as grades do gráfico

plt.show() #.show é usado para mostrar as figuras do gráfico

# Coeficientes da regressão linear

print("Coeficiente angular (inclinação):", coeficientes[0])

print("Coeficiente linear (intercepto):", coeficientes[1])# print Imprime os valores em um stream

# Nota-se que a stream em questão nada mais é np.polyfit(alturas_pais, alturas_filhos)

# E que o intercepto advém de np.polyfit(alturas_pais,alturas_filhos,** 1**)
```

\_\_\_\_

2.2 Regressão Linear

Este é o script 02 da aula 02, aonde é possível se criar uma regressão
linear com dados reais com o Python. Essa aplicação utilizar a base de
dados \"cattaneo2.dta\" (Cattaneo, Journal of Econometrics, 2010). A
base de dados estuda o efeito do fumo durante a gravidez no peso do bebê
ao nascer. Para essa aplicação se utiliza como variável de dependente
(y) o peso do bebê ao nascer (*bweight*) e como variável de independente
(x) o fato da mãe fumar (*mbsmoke*).

Os pacotes necessários para a análise dos dados são: *requests*,
*pyreadstat*, *pandas*, *matplotlib*, s*tatmodels, seaborn *e* scipy*. O
pacote *requests* é utilizado para realizar o download do arquivo, o
pyreadstat para ler o *arquivo.dta*, o pandas para manipular os dados, o
*matplotlib* e o *seaborn* para plotar os gráficos e o *scipy* para
realizar o *teste t*. O pacote *statsmodels* é utilizado para realizar a
regressão linear (diferentemente do código do script anterior que usou
apenas o *numpy*).

Note que dessa vez o Script vai importar alguns pacotes usando a palavra
"from \<nome do pacote\> import" em vez de "import \<nome do pacote\>
as". Existem três vantagens em usar a forma *from* em vez de *import* em
Python:

I. Redução de Digitação: Quando você usa a sintaxe *"from scipy.stats
import ttest_ind"*, ao invés de *"import scipy.stats.ttest_ind as
ttest", *por exemplo,* *você pode usar a função *ttest_ind *diretamente
no seu código, sem ter que digitar o nome do módulo *scipy.stats* toda
vez que precisar usá-la. Isso torna o código mais conciso e legível.

II. Evitar Conflitos de Nomes: Se você importar apenas a função
específica usando "from scipy.stats import ttest_ind", você evita
importar todo o módulo scipy.stats e ter que usar o nome completo da
função sempre que usá-la. Isso é útil quando há outras funções ou
classes com o mesmo nome em diferentes módulos.

III\. Melhorar a Legibilidade: Ao usar from scipy.stats import
ttest_ind, você torna o código mais legível, pois fica claro quais
funções estão sendo importadas e usadas em seu código. Isso facilita a
compreensão do código tanto para quem escreve quanto para outros
desenvolvedores que possam trabalhar no mesmo projeto.

Segue o código:
``` python
import requests

import pyreadstat

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.stats.diagnostic import het_white

import statsmodels.api as sm

import numpy as np

# URL do arquivo no GitHub

url = "https://github.com/Daniel-Uhr/data/raw/main/cattaneo2.dta"

# Realizar o download do arquivo

response = requests.get(url)

# Salvar o arquivo localmente

with open('cattaneo2.dta', 'wb') as f: f.write(response.content) # with open(\...) as f:: Isso abre um contexto de gerenciamento de arquivo usando a declaração with, garantindo que o arquivo seja fechado automaticamente após o uso.
# Carregar os dados do arquivo .dta para um DataFrame

dados, metadata = pyreadstat.read_dta('cattaneo2.dta')

# pyreadstat: É uma biblioteca em Python para ler arquivos no formato DTA.

# read_dta(\...): É uma função da biblioteca pyreadstat que lê um arquivo DTA e retorna os dados e metadados contidos nele.

# Separe as variáveis independentes (X) e a variável dependente (Y)

X = dados[['mbsmoke', 'mage']]# Variável independente (fumanteou não fumante) e idade da mãe

Y = dados['bweight'] # Variável dependente (peso do bebê)

# Adicione uma constante para estimar o intercepto na regressão

X = sm.add_constant(X)

#sm.add_constant() do módulo statsmodels.api adiciona uma coluna de 1s à matriz de recursos X, permitindo que você ajuste um modelo de regressão linear com intercepto.

# Crie o modelo de regressão linear

model = sm.OLS(Y, X)

# Ajuste o modelo aos dados

results1 = model.fit()

# Imprima os resultados da regressão

print(results1.summary())
```

\-\-\-\-

2.3 Teste de Heterocedasticidade de White.

Este é o script 03 da aula 02, na qual se faz o teste de
Heterocedasticidade de White.

Onde:

-   H0: Homocedasticidade está presente (resíduos são igualmente
    dispersos)
-   HA: Heterocedasticidade está presente (resíduos não são igualmente
    dispersos)

Segue o código:
``` python

#Deve-se notar que a importaç*ão* necess*ária foi* feita no script anterior

#Realizando o teste de White

white_test = het_white(results1.resid, results1.model.exog)

#Defindo os t*ítulos* a serem usados na saída do teste de White

labels = ['Estatística de teste', 'Valor p da estatística de teste', 'Teste F', 'Valor p do teste F']

#Para imprimir os resultados do teste de White

print(dict(zip(labels, white_test)))
```

\-\--

**2.4 OLS considerando erros heterocedásticos (Erro-Padrão Robusto)**

Este é o script 04 da aula 02, na qual se faz o OLS considerando erros
heterocedásticos (HC2).

Segue o código:
``` python

\# Deve-se notar que a importação necessária foi feita no script
anterior

\# Crie o modelo de regressão linear

model = sm.OLS(Y, X)

\# Ajuste o modelo aos dados

results2 = model.fit(cov_type=\'HC2\')

\# Imprima os resultados da regressão

print(results2.summary())
```

\-\-\--


**2.5 Aplicação do FGLS (Feasible Generalized Least Squares)**

Este é o script 05 da aula 02, na qual FGLS é a abreviação para
*Feasible Generalized Least Squares,* O FGLS é um método estatístico
utilizado para estimar os parâmetros de um modelo de regressão quando os
pressupostos dos mínimos quadrados ordinários (OLS) não são satisfeitos,
como neste caso em que há existência de Heterocedasticidade.

**Segue o código:**
``` python

\# Deve-se notar que a importação necessária foi feita no script
anterior

\# Crie um modelo de regressão linear usando FGLS

model = sm.GLS(Y, X)

\# Ajuste o modelo aos dados

results = model.fit()

\# Imprima os resultados da regressão

print(results.summary())
```

\-\--

**2.6 Analise descrita de uma pesquisa imobiliária em Manchester (dados
em Excel)**

Este é o script 01 da aula 03, se analisa um trecho de dados de uma
pesquisa imobiliária em Manchester, Reino Unido, que compara A renda
familiar é medida em mil libras, e se possui casa.

Segue o código:
``` python
#Importaç*ão dos pacotes necessários*

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import scipy as sp

import scipy.stats

import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression #Realiza a regressão
logística

\# URL do arquivo no GitHub

url =
\"https://github.com/Daniel-Uhr/data/raw/main/Basic_Econometrics_practice_data.xlsx\"

\# Carregar os dados do arquivo excel para um DataFrame e mostrar as
primeiras linhas

dados = pd.read_excel(url, sheet_name = \'HouseOwn_Quali_Resp\')

print(dados.head())
```

\-\--

**2.6 Estimativa do OLS ignorando a heteroscedasticidade (traçando a
reta de regressão)**

Este é o script 02 da aula 03, Realiza uma estimativa OLS rápida
ignorando a heteroscedasticidade neste momento, como também traçando a
reta de regressão.

Segue o código:
``` python

X = dados\[\'Annual_Family_Income\'\]

Y = dados\[\'Own_House\'\]

X = sm.add_constant(X) \# para adicionar a constante

model = sm.OLS(Y, X).fit()

print_model = model.summary()

print(print_model)
```
- - -

**2.7 Traçando a reta de regressão**

Este é o script 03 da aula 03, traçando a reta de regressão.

Segue o código:
``` python

fig, ax = plt.subplots(figsize = (14, 7))

#Fig *é uma figura de uma área na qual as plotagens são exibidas, que
pode conter um ou mais eixos.*

*#Este exemplo contém um eixo x e um eixo y.*

#figsize *é para atribuir *o tamanho do gráfico, que nest eexemplo ficou
14x7 polegads*.*

ax.scatter(dados\[\'Annual_Family_Income\'\], dados\[\'Own_House\'\]) #A
função *scatter* plota pontos de dados em um gráfico conforme os eixos
descriminados.

ax.plot(dados\[\'Annual_Family_Income\'\], model.fittedvalues, color =
\'Green\') #P*lot, *plota uma linha em um gráfico.

\#*model.fittedvalues é* um array (matriz) com os valores ajustados do
modelo.

ax.grid(True)# True ou Flase: \# grid atribuiu o n*ão uma *grade

ax.set_xlabel(\'Renda\') #A função *set\_xlabel* define o rótulo do eixo
x de um gráfico.

ax.set_ylabel(\'Possuir um apartamento = 1, Não Possui = 0\') #A função
*set_ylabel()* #define o rótulo do eixo y de um gráfico.

plt.show() #Mostra o gr*áfico plotado.*
```
\-\--

**2.8 Estimando os erros padrão robustos para heterocedasticidade**

Este é o script 04 da aula 03, Apresenta os erros padrão robustos para heterocedasticidade (Eicker-Huber-White robust standard errors)

Segue o código:
``` python

\# "Model.HC2_se" Mostra os erros padrão robustos para
heterocedasticidade (Eicker-Huber-White robust standard errors)

print(model.HC2_se) \# print imprime os resultados

\-\--

**2.8 Estimando o OLS para heterocedasticidade com sumário **

Este é o script 05 da aula 03, Caso necessite verificar o sumário da
regressão completa.

Segue o código:

\# Ajuste o modelo aos dados com erros robustos

results2 = sm.OLS(Y, X).fit(cov_type=\'HC2\')

\# Imprima os resultados da regressão com erros robustos

print(results2.summary())
```

**2.8 Visualizando graficamente a função logística **

Este é o script 06 da aula 03, Neste exemplo o professor quer nos
apresentar o conceito do modelo de logit, ao mesmo tempo que usar o
python como ferramenta em sua aula para podermos visualizar, e assim
melhor assimilar os conceitos e aforma de uma função logística. Neste
exemplo ele condiciona o Beta1 = 2, e Beta2 = 3. Considerando o
intervalo de valores para x de -3 a 2.

Segue o código;
``` python

#Lembrando que uma vez importados os pacotes, durante esse roteiro
(script) não é necessário importá-los novamente.

beta1, beta2 = 2, 3 \# Aqui se declara duas variáveis, beta1 e beta2, e
atribui a elas os valores 2 e 3, respectivamente.

X = np.linspace(-3, 2) #Aqui se declara que X é um vetor de 50
elementos, igualmente espaçados entre -3 e 2.

P = 1/(1 + np.exp(-beta1-beta2\*X))#O submódulo .linspace é uma função
que retorna uma matriz de números espaçados de forma uniforme

#-\>em um intervalo especificado. Neste caso o intervalo é de -3 a 2, e
o número de elementos é 50.

fig, ax = plt.subplots(figsize = (9, 5)) #Fig é uma figura de uma área
na qual as plotagens são exibidas, que pode conter um ou mais eixos.

#Este exemplo contém um eixo x e um eixo y.

#figsize é para atribuir o tamanho do gráfico, que nest eexemplo ficou
12x7 polegads.

ax.plot(X, P)#Aqui se declara que o eixo x é X e o eixo y é P.

ax.set_title(r\'\$P = 1/(1+e\^{-Z})\$\')#Aqui se declara o título do
gráfico.

ax.grid()#Aqui se declara que o gráfico terá uma grade.

**2.8 Visualizando graficamente a função logística **

Este é o scripts 07 e 8 da aula 03, Assumindo o modelo não linear
(Logit). Imprima os resultados da estimativa.

*Y = dados\[\[\'Own_House\'\]\]#Atribundo a Y os dados de "com casas"*

X = dados\[\[\'Annual_Family_Income\'\]\] *#Atribundo a X os dados de
"Renda familiar Anual"*

X = sm.add_constant(X)#A função *add_constant()* adiciona uma coluna de
constantes, ou seja, uma coluna de 1s, a um dataframe.

log_reg = sm.Logit(Y, X).fit()
```
