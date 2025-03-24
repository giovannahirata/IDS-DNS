Malicious DNS and Attacks (BCCC-CIC-Bell-DNS-2024)

Unveiling malicious DNS behavior profiling and generating benchmark dataset through application layer traffic analysis

Resumo
O Sistema de Nomes de Domínio (DNS) é um alvo frequente de ataques cibernéticos, tornando essencial o monitoramento e a análise de suas atividades para detectar comportamentos maliciosos. Este estudo propõe uma abordagem inovadora de perfilamento comportamental DNS que enfrenta desafios como táticas de evasão, variabilidade de conteúdo, detecção de intenções maliciosas, ofuscação de URLs, ataques de baixa intensidade e a manutenção da precisão diante de comportamentos normais diversos.
O modelo utiliza comportamentos únicos de características e suas correlações, combinando um novo algoritmo de seleção de características, uma metodologia de extração de padrões e uma arquitetura de rede neural para a criação precisa de perfis.
Além disso, o estudo apresenta o ALFlowLyzer, um analisador de fluxo de rede na camada de aplicação, e o novo conjunto de dados BCCC-CIC-Bell-DNS-2024, que supera as limitações dos datasets DNS públicos existentes. Os resultados experimentais comprovam a eficácia do modelo na detecção e no perfilamento de diversas atividades DNS.
Introdução
“Além da criação do conjunto de dados, esta pesquisa propõe um modelo de criação de perfil de comportamento DNS para detectar diferentes atividades DNS.”

Principais inovações para o perfil de comportamento do DNS contribuídas pelo artigo:
introdução de um novo algoritmo de seleção de features baseados em grafos;
introdução de um algoritmo de cálculo de similaridade de comportamento;
introdução do DNS Flow como um componente vital na análise de DNS;
criação de um novo e sofisticado sistema de criação de perfis comportamentais por meio de extração de padrões e arquitetura de rede neural;
implementação do ALFlowLyzer, um analisador de fluxo de rede da camada de aplicação com mais de 120 features de DNS;
introdução de um novo conjunto de dados DNS malicioso, chamado BCCC-CIC-Bell-DNS-2024, com uma abordagem baseada em fluxo de DNS.

Estrutura da parte restante do artigo:
Seção 2 fornece uma visão geral de pesquisas anteriores sobre análise de tráfego DNS malicioso;
Seção 3 apresenta uma explicação detalhada do modelo de criação de perfil proposto;
Seção 4 apresenta o ALFlowLyzer;
Seção 5 explora os conjuntos de dados disponíveis e discute a decisão de integrar dois conjuntos de dados, juntamente com uma explicação das características do novo conjunto de dados;
Seção 6 apresenta a configuração experimental e os resultados da aplicação do modelo proposto ao conjunto de dados recém-introduzido;
Seção 7 analisa e discute esses resultados, enfatizando as principais descobertas;
Seção 8 conclui o artigo e sugere possíveis direções para pesquisas futuras.

Trabalhos Relacionados
Visão geral de pesquisas anteriores sobre análise de tráfego DNS malicioso. 
Os trabalhos anteriores são categorizados com base em suas metodologias primárias: métodos baseados em regras, em aprendizado e em perfis.
A conclusão destaca as limitações de estudos de pesquisas anteriores e identifica os desafios que este trabalho (artigo) pretende abordar.

Abordagens baseadas em regras
Métodos baseados em regras analisam features de tráfego DNS por meio da formulação de regras e da comparação de features com assinaturas predefinidas.
(aqui o autor descreve e analisa as metodologias usadas em diferentes pesquisas já realizadas usando esta abordagem de análise de tráfego DNS)

Abordagens baseadas em aprendizagem
Essas abordagens utilizam técnicas de ML e DL para analisar padrões de tráfego DNS e identificar ameaças potenciais.
(aqui o autor descreve e analisa as metodologias usadas em diferentes pesquisas já realizadas usando esta abordagem de análise de tráfego DNS)

Abordagens baseadas em perfil comportamental
A criação de um perfil na análise de tráfego DNS envolve coletar e analsar sistematicamente características distintivas e padrões comportamentais.Inclui a criação de modelos para caracterizar o tráfego DNS normal, permitindo a identificação de anomalias ou atividades maliciosas com base em desvios de normas estabelecidas.
(aqui o autor descreve e analisa as metodologias usadas em diferentes pesquisas já realizadas usando esta abordagem de análise de tráfego DNS)

Síntese
Em resumo, os trabalhos anteriores apresentaram as seguintes limitações:
Necessidade de enormes recursos computacionais;
Baixa precisão e alta taxa de falsos positivos;
Carência de um conjunto de dados abrangente;
Carência de um conjunto abrangente de features;
Restrição a certas atividades maliciosas de DNS;
Capacidade limitada de detectar ataques de zero-day;
Necessidade de atualização regular da blacklist;
Necessidade de conhecimento prévio para atualização das atividades;
Carência de detecção de intrusão em tempo real;
Suscetível a táticas de evasão por invasores com conhecimento do sistema.

A técnica proposta emprega perfil comportamental para modelar tráfego DNS malicioso e benigno com base nos features extraídos dos metadados da camada de aplicação e do fluxo de tráfego de rede.
Esse trabalho focou em abordar os 7 primeiros problemas descritos acima.


Modelo proposto
O trabalho aplicou um modelo inovador de criação de perfil comportamental para análise de tráfego de rede com foco em atividades baseadas em DNS, em que aprimorou a precisão, interpretabilidade e eficiência ao abordar limitações identificadas em metodologias atuais.

A Fig. 1 descreve o procedimento geral, começando com a seleção de features para representação de perfil. Um sistema de extração de padrões captura padrões de comportamento, formando o núcleo do modelo. A etapa final envolve uma estrutura de Rede Neural, agregando perfis, e atribuindo pesos a cada perfil criado para rotular novas instâncias com precisão. Um algoritmo de cálculo de similaridade derivado do processos de seleção de features também foi introduzido para aprimorar a análise de dados.

Seleção de features
Usa um grafo de correlação para mapear e avaliar inter-relações de features.
É criado grafo conexo e ponderado, inicialmente com arestas de peso zero, cujos nós são as features. Então, usa-se o Coeficiente de Correlação de Pearson para atualizar os pesos das arestas e, dessa forma, informar a correlação entre os respectivos pares de features conectados.
As arestas com valores abaixo de um limite predefinido são removidas.
Para identificar conjuntos de features ideais, o grafo é percorrido para encontrar o caminho mais robusto de comprimento (n + 1) entre quaisquer dois nós, onde o caminho contém n características distintas.
Isso é feito para cada tipo de atividade do conjunto de dados: Benign, Malware, Spam, Exfiltration, Phishing.

Criação de Perfis
A criação de um perfil é baseada nas features selecionadas para uma determinada atividade.
Os perfis são construídos por meio do cálculo de intervalos para cada feature selecionada, seguido pela utilização dos valores mapeados dessas features durante a fase de extração de padrões associados.
Posteriormente, todos os perfis gerados para várias atividades são consolidados dentro de uma estrutura de rede neural.

Cálculo de intervalos
Aqui são definidos os intervalos válidos para cada feature em uma atividade específica, ou seja:
a abordagem é identificar os limites dentro dos quais cada feature opera em diferentes atividades
para isso, utiliza-se o modelo Mixture of Gaussians (MoG) para representar as distribuições das features, pois ele captura padrões complexos dos dados.
O MoG modela a densidade f^​(x) como a soma ponderada de várias distribuições Gaussianas;
os parâmetros (pesos, médias, variâncias) são otimizados de forma iterativa através do algoritmo Expectation-Maximization (EM) que executa os processos E-step (calcula a responsabilidade de cada ponto de dado pertencer a um componente Gaussiano) e M-Step (atualiza os pesos, médias e variâncias para maximizar a verossimilhança dos dados observados);
o processo é repetido até a convergência, para, assim, garantir maior precisão dos intervalos de cada feature em uma atividade.

Extração de Padrões
Aqui são identificadas as correlações entre os valores das features em diferentes atividades.
Ao examinar como os valores interagem entre features e como variam conforme a atividade, é possível capturar padrões comportamentais distintos.
O algoritmo FP-Growth foi usado para extrair regras de associação, devido à sua eficácia em lidar com grandes volumes de dados.
O algoritmo Differential Evolution (DE) foi usado para ajustar os parâmetros do FP-Growth, devido à sua capacidade de lidar com espaços de parâmetros contínuos e explorar paisagens complexas e não-lineares. Com a perturbação e recombinação iterativas das soluções, o objetivo foi maximizar a precisão enquanto minimiza taxas de falso positivo para, enfim, encontrar valores ótimos de parâmetros.
Após encontrar os parâmetros ótimos, o FP-Growth é executado novamente para extrair os padrões de cada atividade.
Esses padrões formam o núcleo dos perfis, representando os comportamentos específicos de cada atividade.

Estrutura da Rede Neural
Integra os componentes já mencionados acima para criar um sistema completo de criação de perfis DNS.
Processo de Construção:
Perfis por Atividade: Um ou mais perfis são gerados para cada atividade DNS, dependendo do número de conjuntos de features ótimos identificados.
Transformação em Matrizes: As features selecionadas para cada fluxo DNS são convertidas em uma matriz (com as linhas representando os valores das features) para facilitar o processamento na rede neural.
Normalização: Os valores das features são normalizados em intervalos predefinidos antes do processamento.
Camadas da Rede Neural:
Primeira camada: calcula os intervalos válidos para cada característica com base no input.
Segunda camada: avalia a similaridade do padrão entre o input e os perfis extraídos. O output varia de 0 a 1, indicando a correspondência com os perfis existentes. Cada saída é ponderada para refletir a influência dos diferentes perfis em cada atividade.
Terceira camada: consolida os outputs ponderados de todos os perfis por atividade, gerando um valor contínuo que representa a probabilidade de associação do input a uma atividade específica.
Camada de saída: utiliza a função Softmax para calcular a probabilidade de cada atividade, normalizando os valores para produzir uma distribuição de probabilidade.
Cálculo de Pesos:
ajustar os pesos dos perfis para minimizar a perda de Entropia Cruzada Categórica, adequada para classificação multiclasse.
Processo de treinamento:
Inicialização: os pesos dos perfis são iniciados aleatoriamente.
Propagação direta: calcula-se o output ponderado para cada atividade.
Cálculo da perda: aplica-se a Entropia Cruzada Categórica para medir o erro entre as probabilidades previstas e as reais.
Retropropagação: calcula-se o gradiente do erro em relação aos pesos usando a regra da cadeia.
Atualização dos pesos: usa-se o otimizador Adagrad, que ajusta a taxa de aprendizado de forma adaptativa para cada peso, garantindo ajustes mais eficazes.
Médias Móveis Exponenciais:
Durante o treinamento, são atualizadas as médias móveis dos gradientes para estabilizar a atualização dos pesos, utilizando taxas de decaimento exponencial (β1​ e β2).
Iteração:
O processo de propagação direta, cálculo de perda, retropropagação e atualização dos pesos é repetido por um número fixo de épocas, permitindo a redução progressiva do erro e a melhoria da precisão do modelo.
Adaptabilidade do modelo:
A arquitetura é flexível e pode ser ajustada para diferentes conjuntos de dados DNS. Por exemplo, o número de nós na última camada corresponde ao número de classes no conjunto de dados.
Complexidade estrutural:
O modelo adota uma abordagem híbrida entre uma rede totalmente conectada e arquiteturas mais complexas (ex.: CNNs ou LSTMs).
Apenas os pesos das saídas da segunda camada são ajustados durante a retropropagação.
Cada perfil gerado na segunda camada considera apenas um conjunto específico de características, com conexões ponderadas em 1 para as características relevantes e 0 para as demais.
Similaridade de comportamento
É uma métrica para quantificar a similaridade entre diferentes atividades DNS. A similaridade leva em conta duas atividades a serem comparadas, conjuntos de arestas que representam as correlações entre as features em cada atividade. Por fim, a similaridade é normalizada para um vetor entre -1 e +1.
Funcionamento do Algoritmo:
Começa com o número total de correlações.
A similaridade diminui conforme a diferença absoluta entre as arestas nos gráficos de características aumenta.
Aplicações:
detecção de anomalias e intrusões ao comparar com padrões normais;
criação de perfis: categorizar tráfegos de rede com maior precisão;
análise de ameaças: reconhecimento de comportamentos semelhantes a atividades maliciosas conhecidas.

Implementação
Visão geral do ALFlowLyzer, uma ferramenta inovadora de análise de fluxo de rede na camada de aplicação. Além disso, traz uma abordagem da captura de tráfego de rede e extração de recursos relevantes para o modelo de perfil.

ALFlowLyzer
ALFlowLyzer é uma ferramenta em Python para analisar arquivos PCAP de tráfego de rede e gerar dados em formato CSV. Desse modo, sua funcionalidade contempla:
identificação de fluxos DNS, e extração de features da camada de aplicação
geração de CSV com as informações extraídas, que devem ser rotuladas manualmente para treinar modelos de Machine Learning (ML) ou Deep Learning (DL).
Características:
Suporta vários protocolos da camada de aplicação (o foco aqui é no protocolo DNS).
Extração de um conjunto amplo de atributos (mostrados na Fig. 3 e tabelas 1 e 2).
Validação com diversos conjuntos de dados e tráfegos reais, garantindo precisão na extração de características.
Destaques:
Flexibilidade para futuras inclusões de protocolos.
Capacidade de detectar atividades maliciosas na camada de aplicação.
Documentação detalhada no repositório do GitHub.







Criação de fluxos
A abordagem de criação de fluxos no ALFlowLyzer se diferencia por considerar as camadas de rede e de aplicação.
Diferenças em relação a abordagens tradicionais:
Em vez de focar apenas no nível UDP, adota uma definição de fluxo específica para cada protocolo, garantindo maior precisão na representação do comportamento da camada de aplicação.
Usa o ID de transação do cabeçalho DNS como identificador primário para fluxos DNS, facilitando a detecção de comportamentos anômalos, como Poisoning Attacks.
Critérios de terminação de fluxo:
Duração máxima do fluxo (tempo limite do fluxo).
Tempo máximo de inatividade do fluxo.

Seleção de comportamentos e extração de features
A análise eficaz do comportamento de rede depende da seleção criteriosa dos comportamentos e da extração detalhada das features. O ALFlowLyzer identifica oito categorias principais:
DNS Lexical-based
DNS Statistical-based
DNS Resource Record-based
DNS Third-party-based
Size-based
Delta-length-based
Delta-time-based
Side-based
O ALFlowLyzer extrai 130 features de duas fontes principais:
Estatísticas de fluxo: dados como duração, contagem de pacotes, tamanho do payload.
Meta-dados: informações específicas do protocolo na camada de aplicação.
Features de estatísticas de fluxo
Extraídas por meio de funções estatísticas (mínimo, máximo, média, mediana, moda, variância, desvio-padrão, assimetria, coeficiente de variação). Dividem-se em:
Características Baseadas no Tamanho dos Pacotes: Análise do comprimento dos pacotes recebidos/enviados.
Características Baseadas no Delta de Tempo: Diferenças temporais entre pacotes consecutivos.
O ALFlowLyzer extrai 79 características a partir dessas estatísticas.
Características de Meta-dados
Essas características vêm das informações específicas da camada de aplicação e são divididas em:
Features léxicas de DNS: avalia nomes de domínio gerados por algoritmos maliciosos (ex.: comprimento do domínio, entropia, proporção de vogais e consoantes).
Features estatísticas de DNS: análises e funções estatísticas aplicadas aos registros de resposta.
Features baseadas em TTL: Tempo de vida de um registro DNS (ex.: análise estatística do TTL).
Features baseadas em registros DNS: Extração de informações dos quatro tipos de registros DNS.
Features de terceiros (WHOIS): Informações obtidas de bancos de dados WHOIS (ex.: dados de registro do domínio, localização geográfica).
O ALFlowLyzer extrai 51 características dos meta-dados do tráfego de rede.

Novo conjunto de dados DNS malicioso
Detalhes do novo conjunto de dados BCCC-CIC-Bell-DNS-2024, criado a partir da fusão dos conjuntos anteriores utilizando o ALFlowLyzer.
Conjunto de dados DNS disponíveis
Boss of the SOC Dataset Version 1 (Botsv1)
PUF Dataset (ICCIDS 2018)
Labeled FQDN/IP dataset (Computers & Security)
CIC-Bell-DNS-2021
CIC-Bell-DNS-EXF-2021

Integração dos conjuntos de dados selecionados
A integração do CIC-Bell-DNS-2021 e CIC-Bell-DNS-EXF-2021 foi realizada para superar as limitações de outros conjuntos e ampliar a diversidade do tráfego DNS.
Motivações para integração:
Abrange uma ampla gama de atividades DNS benignas e maliciosas.
Permite uma avaliação mais robusta e sofisticada do modelo.
A estrutura uniforme de ambos os conjuntos facilita a integração.
Etapas de integração:
Limpeza: Remoção de pacotes fora do formato correto e PCAPs inconsistentes.
Aprimoramento: Uso do ALFlowLyzer para converter pacotes em fluxos DNS e padronizar as features.
Rotulagem: Verificação cruzada com as informações originais para atribuir rótulos, consolidando as atividades de Exfiltração em um único rótulo.
O novo conjunto possui mais de 120 features, superando os anteriores (menos de 60).
Estrutura do novo conjunto de dados
O ALFlowLyzer extraiu fluxos do tráfego de rede bruto para gerar arquivos CSV.
Destaques:
Consolidação das atividades de exfiltração em seis subcategorias (áudio, imagem, texto, etc.).
Cada linha no CSV representa um fluxo DNS, não mais um domínio específico.
As características são divididas em Metadados DNS e camada de aplicação.
A integração dos dois conjuntos resultou em um novo conjunto mais abrangente, utilizado para avaliar o modelo de perfilamento.

Resultados experimentais
Detalha os resultados da aplicação do novo conjunto de dados no modelo proposto, abrangendo seleção de características, análise de similaridade de comportamento, criação de perfis e avaliação de desempenho.
Seleção de features
Um algoritmo selecionou as melhores features para cada atividade DNS.
Dois perfis foram criados para cada atividade, com quatro features cada.
Reduziu-se a sobreposição de features para aumentar a distinção.
Apenas as features com peso acima de 0,3 foram mantidas.
Características de terceiros
WHOIS foi excluído devido à alta presença de valores nulos, especialmente em domínios maliciosos.
A remoção garantiu consistência e evitou vieses no conjunto de dados.
Similaridade de comportamento
O algoritmo de similaridade de comportamento utilizou três técnicas de correlação (Pearson, Spearman e KendallTau) para medir a semelhança entre atividades.

Alta similaridade em atividades maliciosas
Detectou-se uma forte semelhança entre as atividades maliciosas (áudio, vídeo, imagem, executável, etc.).
Para simplificar e distinguir melhor, todas essas atividades foram unificadas sob o rótulo "Exfiltration".
Cálculo de intervalos de features
Determinaram-se os intervalos principais para cada feature em cada atividade.
Os gráficos do tipo "violin plot" ilustram a variabilidade dos intervalos, destacando diferenças entre as atividades.

Criação de perfis
Os perfis foram criados a partir das features selecionadas.
Cada nó do grafo representa um intervalo de feature, e as arestas simbolizam as conexões extraídas pelas regras.

Desempenho
A avaliação de desempenho do modelo foi realizada por meio de testes extensivos.
Os resultados detalhados estão na Tabela 5.


Análise e discussão
Esta seção avalia o modelo proposto de criação de perfis comportamentais, introduzindo o conceito de DNS Flow para aprofundar a compreensão das atividades na camada de aplicação. O estudo analisa os comportamentos individuais das características, explora correlações e investiga diferentes cenários de ataque.
Análise das ideias principais
O modelo baseia-se em dois princípios fundamentais:
Comportamento único das features – cada feature apresenta padrões específicos em diferentes atividades.
Correlação entre features – as relações entre features variam conforme a atividade.
A criação de perfis inicia-se com a identificação dos valores das features para cada atividade. Em seguida, aplica-se a mineração de regras de associação para identificar padrões de co-ocorrência. Esse processo combina comportamentos únicos e correlações para construir perfis abrangentes e representativos.
Análise da criação de perfis
A criação eficaz de perfis depende de uma seleção criteriosa de features, com ênfase em:
Intervalos de features – delimitam os valores típicos para cada atividade, permitindo identificar padrões específicos.
Correlação entre features – captura as interdependências entre diferentes features, fortalecendo a precisão do perfil.
Um algoritmo inovador prioriza features altamente correlacionadas, pois features não correlacionadas produzem regras insignificantes, prejudicando a precisão do modelo.
Análise da seleção de features
A seleção de características é essencial para a construção de perfis robustos. O modelo propõe um novo algoritmo que identifica e prioriza características com correlações significativas, permitindo detectar padrões complexos no tráfego DNS.
Os experimentos confirmam que focar em características altamente correlacionadas melhora a precisão e a capacidade de caracterizar cada atividade.
Análise das similaridades comportamentais
A análise das similaridades comportamentais valida a eficácia do modelo:
Atividades Similares – As atividades mais próximas do comportamento benigno são Spam, Phishing e Malware.
Estrutura Compartilhada – Atividades como Áudio, Imagem, Vídeo, Executáveis e Texto compartilham padrões semelhantes.
Esses achados aprimoram a detecção de atividades maliciosas. Apesar de pequenas variações, o algoritmo de correlação se mostra consistente. Contudo, a diferenciação entre Áudio e outras atividades relacionadas à exfiltração permanece um desafio e exige mais pesquisas.
Análise dos perfis criados
Os experimentos demonstram a capacidade do modelo de criar perfis precisos para atividades maliciosas:
Perfil de Exfiltração – destaca-se pelo alto comprimento de nomes de domínio (domain_name_len), um comportamento comum em atividades de exfiltração.
Características auxiliares – a inclusão de features relacionadas ao tamanho e ao tempo melhora a análise e a precisão na detecção de comportamentos.
O modelo identifica de forma consistente padrões únicos para cada tipo de ataque, validando sua eficácia.
Perfilagem de ataques zero-day
O modelo proposto caracteriza eficientemente ataques zero-day com base na análise do tráfego de rede:
Comportamento Distinto – Se nenhuma atividade conhecida corresponder ao novo comportamento, ele é rotulado provisoriamente como "Desconhecido" ou "Ataque Zero-Day" até que dados suficientes permitam a criação de um perfil específico.
Sobreposição de Perfis – Se um novo comportamento compartilha características com múltiplos perfis, são aplicadas técnicas para eliminar padrões comuns e, se o alinhamento persistir, ele é classificado como um possível Ataque Zero-Day.
Essa abordagem combina a definição de um comportamento normal com a capacidade de reconhecer ataques desconhecidos, reforçando a segurança da rede contra ameaças emergentes.
Análise comparativa com trabalhos anteriores
maior precisão
maior diversidade de rótulos
metodologia inovadora

Conclusão e trabalhos futuros
Principais contribuições:
ALFlowLyzer
Algoritmo de seleção de características baseado em grafos
métrica de similaridade comportamental
Os resultados experimentais demonstram que o modelo alcança uma precisão superior a 99% em diferentes cenários de perfilamento, detectando ataques como Exfiltração e Spam por meio da análise de características-chave.


Perspectivas Futuras:
Ampliar a análise para outras atividades de rede, além do DNS.
Validar e aprimorar o modelo com conjuntos de dados mais diversos e de maior escala.
Melhorar a detecção de ameaças emergentes, aprofundando a análise comportamental para fortalecer a segurança das redes em evolução.
