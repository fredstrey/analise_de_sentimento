# Análise de sentimento em comunidades brasileiras de tecnologia no Reddit

Projeto de iniciação científica sobre o sentimento expresso em duas comunidades brasileiras de tecnologia no Reddit:

- **r/brdev**: desenvolvimento de software e carreira em TI.
- **r/datasciencebr**: ciência de dados.

Cada texto recebe um de três rótulos: **negativo**, **neutro** ou **positivo**. A classificação é feita por um modelo de linguagem através da OpenAI Decisions API.

Este documento descreve o que é feito em cada etapa e **por que** cada decisão foi tomada, para que o processo possa ser reproduzido e avaliado.

---

## 1. Estrutura do projeto

```
analise de sentimento/
├── data/                              # dados brutos (não versionar; ver seção 3)
│   ├── brdev/r_brdev_{posts,comments}.jsonl
│   └── datasciencebr/r_datasciencebr_{posts,comments}.jsonl
├── dados_limpos/
│   ├── amostra_sentimento.csv         # amostra final, entrada do classificador
│   ├── relatorio_preparacao.json      # parâmetros e contagens de cada filtro
│   ├── amostra_sentimento_classificado.csv   # gerado pelo classificador
│   └── amostra_sentimento_erros.csv          # amostras que falharam na classificação
├── scripts/
│   ├── preparar_dados.py              # seleção, limpeza e amostragem
│   ├── classificar_sentimento.py      # classificação via Decisions API
│   ├── csv_para_excel.py              # exporta o resultado para .xlsx
│   └── preparar_base_treino.py        # base de treino (confiança > 0,7) + teste separado
├── dados_treino/                      # treino_conf_0.7.csv, teste.csv
├── notebooks/
│   └── comparacao_modelos.ipynb       # TF-IDF+NB, BERTweet-PT, BERTimbau (rodar no Colab)
├── requirements.txt                   # dependências dos scripts locais
└── requirements-notebook.txt          # dependências do notebook (Colab)
```

## 2. Como reproduzir

```powershell
pip install -r requirements.txt

# 1) Preparar os dados (determinístico: mesma seed -> mesma amostra)
python scripts/preparar_dados.py

# 2) Classificar (antes, coloque a chave no arquivo .env: OPENAI_API_KEY=sk-...;
#    use o .env.example como modelo. O .env está no .gitignore e não é versionado.)
python scripts/classificar_sentimento.py dados_limpos/amostra_sentimento.csv

# 3) Reprocessar as amostras que falharam
python scripts/classificar_sentimento.py dados_limpos/amostra_sentimento.csv --retry-erros
```

Para testar sem chave de API, use `--mock`, que simula as respostas e as falhas. Para rodar em poucas amostras, use `--limite N`.

---

## 3. Coleta de dados

**Fonte:** os dados vêm do [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift), um arquivo público de posts e comentários do Reddit, baixados pela ferramenta de download por subreddit em formato JSONL (um objeto JSON por linha).

**Por que o Arctic Shift e não a API oficial do Reddit:**
- A API oficial devolve no máximo cerca de 1.000 itens por listagem e não permite voltar no tempo além disso, então não dá acesso ao histórico completo.
- O Pushshift, a fonte usada tradicionalmente em pesquisa, está restrito a moderadores desde 2023.
- O Arctic Shift tem o histórico completo das comunidades e preserva os campos originais da API do Reddit (`created_utc`, `parent_id`, `author`, `removed_by_category`…).

**Campos usados:**

| Campo | Uso |
|---|---|
| `id` / `name` | Identificador. `t3_` indica post e `t1_` indica comentário. |
| `created_utc` | Data de criação (Unix timestamp, em segundos, UTC) |
| `title`, `selftext` | Texto dos posts |
| `body` | Texto dos comentários |
| `parent_id`, `link_id` | Estrutura da thread |
| `author`, `stickied`, `distinguished`, `removed_by_category` | Filtros de qualidade |

> **Observação sobre `depth`:** o campo `depth` (nível de aninhamento do comentário) só aparece em uma pequena parte dos dados, nos comentários coletados mais recentemente. Por isso **ele não é usado**. O nível 0 é identificado por `parent_id` começar com `t3_`, ou seja, o comentário responde diretamente ao post. Nos registros que têm `depth`, essa regra coincide em 100% dos casos com `depth == 0`.

---

## 4. Seleção e limpeza (`scripts/preparar_dados.py`)

Os filtros são aplicados **separadamente para cada comunidade**, na ordem abaixo. O arquivo `relatorio_preparacao.json` registra quantos itens cada filtro removeu.

### 4.1 Recorte temporal: a partir de 01/01/2022 (UTC)
**Por quê:** o foco é o período recente, de 2022 em diante. Nesse período houve a onda de demissões em empresas de tecnologia (*layoffs*), a popularização de ferramentas de IA generativa e mudanças no mercado de trabalho de TI. O recorte também deixa as duas comunidades comparáveis no mesmo intervalo de tempo: os dados do r/datasciencebr só começam em janeiro de 2022, enquanto o r/brdev existe desde 2016.

### 4.2 Apenas posts e comentários de nível 0
São incluídos:
- os **posts** (título + corpo);
- os **comentários que respondem diretamente ao post** (`parent_id` começa com `t3_`).

**Por quê:**
- Um comentário de nível 0 pode ser entendido sozinho, porque reage ao post. Um comentário aninhado ("concordo", "não é bem assim") depende da mensagem a que responde, que o classificador não vê. Sem esse contexto, o sentimento fica ambíguo e o rótulo perde confiabilidade.
- Respostas aninhadas tendem a ser trocas entre poucos usuários. Elas sobrerrepresentariam discussões longas entre as mesmas pessoas.

### 4.3 Exclusão de conteúdo sem autoria humana ou sem texto
São removidos:
- textos `[deleted]` ou `[removed]`, e posts com `removed_by_category` preenchido;
- mensagens do **AutoModerator** e conteúdo fixado ou de moderação (`stickied`, `distinguished == "moderator"`).

**Por quê:** textos removidos não têm conteúdo para analisar. Mensagens automáticas e avisos da moderação são padronizados e não expressam a opinião dos membros, o que distorceria a distribuição de sentimento.

### 4.4 Normalização do texto
A normalização é **conservadora**: remove ruído de formatação, mas **preserva tudo que pode carregar sentimento**.

| Transformação | Por quê |
|---|---|
| Decodificação de entidades HTML (`&amp;` → `&`) e Unicode NFC | O Reddit armazena o texto com escapes HTML. NFC padroniza acentos que podem ter codificações diferentes. |
| Remoção de caracteres invisíveis (`&#x200B;`, zero-width, controle) | Ruído comum em textos do Reddit. Não tem significado. |
| Remoção de linhas citadas (`> ...`) | São palavras de **outra pessoa**. Manter atribuiria ao autor o sentimento de quem foi citado. |
| Blocos de código → `[código]` | Código não expressa sentimento e consome muitos tokens. O marcador registra que havia código ali. |
| URLs → `[link]`; `[texto](url)` → `texto` | URLs não carregam sentimento. O texto âncora do link é mantido. |
| `u/usuario` → `@usuario` | **Anonimização** dos usuários citados (cuidado ético; ver seção 7). |
| Remoção de marcação markdown (`**`, `__`, `~~`, `#`, tabelas) | É formatação, não conteúdo. |
| Repetições de 6 ou mais caracteres iguais reduzidas a 5 (`kkkkkkkkkk` → `kkkkk`) | Mantém o sinal (risada, ênfase) e evita gastar tokens com variações sem significado. |
| Colapso de espaços e quebras de linha | Padronização. |

**O que não é alterado:** maiúsculas e minúsculas, pontuação (`!!!`, `?`), emojis, gírias, palavrões e erros de digitação. Todos são sinais relevantes de sentimento, e modelos de linguagem lidam bem com eles. Por isso **não** há conversão para minúsculas, remoção de *stopwords* nem *stemming*, técnicas de modelos clássicos (bag-of-words) que aqui só eliminariam informação.

Textos que ficam vazios ou só com marcadores (`[link]`, `[código]`) depois da normalização são descartados.

### 4.5 Filtro de tamanho: entre 5 e 500 tokens
Os tokens são contados com o tokenizer `o200k_base` (biblioteca `tiktoken`).

**Por quê:**
- **Mínimo de 5 tokens:** textos muito curtos ("isso", "kkk", "+1") quase não têm conteúdo para a classificação, e o sentimento deles depende quase inteiramente do contexto.
- **Máximo de 500 tokens:** textos longos costumam misturar vários assuntos e sentimentos (por exemplo, um relato com partes positivas e negativas). Isso torna um rótulo único menos representativo e aumenta o custo.
- Na prática o máximo corta pouco: a mediana dos textos elegíveis é de cerca de **52 tokens** e o percentil 95 fica perto de **260**.

> O tokenizer do modelo usado na classificação não é público. O `o200k_base` é uma aproximação, suficiente para um filtro de tamanho.

### 4.6 Remoção de duplicatas
São removidos itens com o mesmo `id` e textos idênticos após a normalização (comparação sem diferenciar maiúsculas e minúsculas), dentro de cada comunidade.

**Por quê:** posts repetidos, *copypastas* e spam dariam peso extra a um mesmo conteúdo.

### 4.7 Amostragem: até 10.000 itens por comunidade
Dos itens elegíveis, é feita uma **amostragem aleatória simples sem reposição**, com semente fixa (`seed = 42`), de até 10.000 itens por comunidade.

**Por quê:**
- **Mesmo tamanho para as duas comunidades:** o r/brdev tem muito mais atividade que o r/datasciencebr. Um teto igual impede que uma comunidade domine a análise conjunta.
- **Aleatória sobre todo o período:** a amostra cobre todo o intervalo de 2022 até hoje, em vez de um bloco contínuo de datas. A distribuição temporal da amostra acompanha a atividade real de cada comunidade.
- **Semente fixa:** rodar o script de novo com os mesmos dados produz exatamente a mesma amostra (reprodutibilidade).
- Posts e comentários não foram balanceados entre si. A proporção entre eles reflete a proporção real entre os elementos elegíveis.

### 4.8 Saída
O arquivo `dados_limpos/amostra_sentimento.csv` tem uma linha por texto:

| Coluna | Descrição |
|---|---|
| `id` | Identificador do Reddit (`t3_…` post, `t1_…` comentário) |
| `texto` | Texto normalizado (posts: título + corpo) |
| `subreddit` | Comunidade de origem |
| `tipo` | `post` ou `comentario` |
| `data`, `created_utc` | Data de criação (UTC) |
| `post_id` | Post ao qual o item pertence (permite agrupar por thread) |
| `score` | Pontuação (votos) no momento da coleta |
| `n_tokens` | Tamanho do texto em tokens |

---

## 5. Classificação de sentimento (`scripts/classificar_sentimento.py`)

### 5.1 Modelo e formulação
- **API:** [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions) (`POST /v1/decisions`).
- **Modelo:** `gpt-6-luna`, o único suportado por essa API.
- **Tipo de pergunta:** `choice`, com três opções:

| Rótulo | Descrição enviada ao modelo |
|---|---|
| `negativo` | Raiva, frustração, tristeza, crítica, reclamação, desprezo ou sarcasmo negativo |
| `neutro` | Factual, informativo, pergunta neutra, ou sentimento misto/ausente |
| `positivo` | Elogio, gratidão, alegria, entusiasmo, concordância ou apoio |

A instrução pede ao modelo que considere sarcasmo, ironia e gírias de internet, e que use `neutro` para textos informativos ou com sentimento misto sem predominância.

**Por que a Decisions API e não um prompt de texto livre:**
- A resposta é **restrita às opções definidas**. Não há rótulos fora do conjunto nem necessidade de interpretar texto livre.
- A API devolve a **probabilidade de cada classe** e uma medida de **confiança**. Elas são usadas para filtrar e analisar a qualidade dos rótulos (seção 5.3).
- A documentação diz que a API é cerca de 10x mais rápida que a Responses API, e o custo é baixo (só os tokens de entrada são cobrados).

**Por que um texto por chamada (e não vários juntos):** classificar cada texto isoladamente evita que o tom de um comentário influencie a leitura de outro. Também permite que cada falha seja registrada e reprocessada individualmente. O custo extra é desprezível nesse volume.

### 5.2 Robustez da execução
- **Paralelismo controlado:** várias requisições simultâneas, limitadas por um semáforo (`--concorrencia`, padrão 10) para não estourar o limite de requisições da API.
- **Novas tentativas:** até 3 por texto, com intervalo crescente e aleatório entre elas. Quando a API informa o tempo de espera (`Retry-After`), ele é respeitado. Só são repetidos erros transitórios: limite de requisições (429), erros do servidor (5xx), timeout e falhas de rede.
- **Registro de falhas:** quando um texto esgota as tentativas, o id, o texto e o erro vão para `*_erros.csv`. Esses textos podem ser reprocessados depois com `--retry-erros`.
- **Retomada:** cada resultado é gravado no disco assim que chega. Se a execução for interrompida, rodar o mesmo comando continua de onde parou, pulando os ids já classificados.
- **Log completo** em `classificacao.log`.

### 5.3 Saída da classificação
`amostra_sentimento_classificado.csv`: `id, texto, rotulo, confidence, prob_negativo, prob_neutro, prob_positivo`.

Os metadados (subreddit, data, tipo) são recuperados juntando esse arquivo com `amostra_sentimento.csv` pela coluna `id`.

**Por que guardar a confiança e as probabilidades:**
- permitem identificar casos ambíguos (por exemplo, `prob_negativo ≈ prob_neutro`) e encaminhá-los para revisão humana;
- permitem verificar se a confiança do modelo é **calibrada**, isto é, se quando ele diz 0,9 acerta perto de 90%;
- permitem análises além do rótulo, como a média de `prob_negativo` por mês.

---

## 6. Avaliação (planejada)

Rótulos gerados por um modelo precisam ser validados antes de serem usados como verdade. O plano é:

1. **Conjunto de referência anotado por humanos:** 300 a 500 textos sorteados da amostra **sem filtro de confiança** e estratificados por comunidade, rotulados à mão.
2. **Concordância entre anotadores:** dois anotadores independentes, com **kappa de Cohen** calculado entre eles. Esse valor indica o teto de concordância esperado para a tarefa.
3. **Métricas do modelo contra o conjunto de referência:** acurácia, **F1 por classe** e **F1-macro**. O F1-macro é a métrica principal, porque as classes tendem a ser desbalanceadas (no Reddit, `neutro` costuma ser a maioria) e a acurácia sozinha engana nesse caso.
4. **Calibração:** comparar a `confidence` com a taxa de acerto real em faixas de confiança.
5. **Uso do filtro de confiança:** se os rótulos forem usados para treinar outro modelo (*fine-tuning*), os seguintes cenários serão comparados **no mesmo conjunto de referência**:
   - todos os rótulos;
   - só os de alta confiança;
   - só os de alta confiança, com as classes balanceadas;
   - todos os rótulos, com pesos por classe.

   Filtrar só pela alta confiança tende a deixar de fora os casos difíceis (sarcasmo, críticas mistas). Por isso a avaliação é sempre feita na distribuição real.

---

## 7. Limitações e cuidados

- **Viés do classificador:** os rótulos refletem a interpretação de um modelo de linguagem, que pode errar com ironia, gírias regionais e jargão técnico (por exemplo, "matar o processo" não é negativo). A seção 6 serve para medir esses erros.
- **Comentários sem contexto:** mesmo os comentários de nível 0 são classificados sem o texto do post a que respondem.
- **Rótulo único por texto:** textos com sentimentos mistos recebem um só rótulo, normalmente `neutro`.
- **Viés de sobrevivência:** conteúdo removido pela moderação ou apagado pelo autor antes da coleta não aparece nos dados.
- **`score`** reflete o momento da coleta, não o momento da publicação.
- **Tokenizer aproximado:** o filtro de tamanho usa `o200k_base`, não o tokenizer exato do modelo.
- **Completude dos dados:** na versão atual (11/11/2026), o arquivo de comentários do r/brdev vai só até **02/12/2023**, porque o download foi interrompido. Por isso a amostra do brdev tem poucos itens de 2024 em diante, e esses são só posts. **Antes da análise final, os comentários do brdev precisam ser baixados de novo e o `preparar_dados.py` rodado outra vez.** O campo `data_mais_recente_no_arquivo` do relatório permite verificar isso.
- **Ética e privacidade:** os dados são públicos. Mesmo assim:
  - menções a usuários dentro dos textos são anonimizadas (`@usuario`);
  - o nome do autor não é incluído na amostra;
  - os dados brutos (`data/`) não devem ser publicados.

  Resultados devem ser apresentados de forma agregada, e citações literais devem ser evitadas ou parafraseadas, porque podem identificar o autor por busca.
