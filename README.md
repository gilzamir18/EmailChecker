# EmailChecker

Agente de IA para classificação e análise automática de e-mails, construído com a biblioteca [agenticblocks](https://agenticblocks.io) e modelos LLM locais via Ollama.

## Visão Geral

O EmailChecker analisa o conteúdo de e-mails e os classifica em duas categorias:

- **informativo** — e-mails que não exigem nenhuma ação do usuário; o agente resume a informação principal.
- **pedido** — e-mails que requerem ações do usuário; o agente sugere ações claras e concisas a serem tomadas.

A saída segue o formato estruturado:

```
Tipo = [informativo/pedido], Resumo = [resumo], Ações = ["ação1", "ação2", ...]
```

## Versões

O projeto evoluiu em três versões, cada uma adicionando uma camada de sofisticação:

| Arquivo | Descrição |
|---|---|
| [EmailCheckerV1.py](EmailCheckerV1.py) | Agente único: classifica e resume e-mails diretamente. |
| [EmailCheckerV2.py](EmailCheckerV2.py) | Pipeline de dois agentes: `EmailPreparer` reformata o e-mail antes do `EmailCheckerAgent` classificar. |
| [EmailCheckerV3.py](EmailCheckerV3.py) | Adiciona ciclo de reflexão com validação de formato (`check_content`), garantindo saída estruturada. |

## Dependências

- Python 3.12+
- [agenticblocks-io](https://agenticblocks.io) `>= 0.8.1`
- [pandas](https://pandas.pydata.org/)
- [Ollama](https://ollama.com/) com o modelo `mistral-nemo` instalado

## Instalação

```bash
# Criar e ativar o ambiente virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install agenticblocks-io pandas
```

Certifique-se de que o Ollama está em execução e o modelo está disponível:

```bash
ollama pull mistral-nemo
```

## Uso

### Preparar a entrada

O arquivo [inputs.csv](inputs.csv) contém os e-mails a serem analisados. O formato esperado é:

```csv
tipo,entrada,resumo,acoes
pedido,"Texto do e-mail aqui...",resumo_esperado,acoes_esperadas
```

A coluna `entrada` é obrigatória. As demais colunas (`tipo`, `resumo`, `acoes`) são opcionais e servem para referência/avaliação.

### Executar

```bash
# Versão básica (agente único)
python EmailCheckerV1.py

# Versão com pipeline de dois agentes
python EmailCheckerV2.py

# Versão com ciclo de reflexão e validação de formato (recomendada)
python EmailCheckerV3.py
```

## Arquitetura (V3)

```
entrada (e-mail)
       │
       ▼
 email_checker  ──► check_content
       ▲                  │
       │                  │ (inválido, até 10 iterações)
       └──────────────────┘
                  │ (válido)
                  ▼
             saída final
```

O bloco `check_content` valida se a resposta do LLM segue o formato esperado. Caso contrário, o resultado é devolvido ao agente com feedback, formando um ciclo de reflexão de até 10 iterações.

## Segurança

Os arquivos `credentials.json` e `token.json` (credenciais OAuth do Gmail) **nunca devem ser versionados**. Eles já estão listados no [.gitignore](.gitignore).
