# README

Breve instruções para preparar e rodar o projeto.

## Pré-requisitos
- UV

## Instalar o UV
Exemplos de instalação (use o método apropriado para seu sistema):

- macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Ou verifique a melhor opção aqui: https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

## Copiar a pasta FORTH_TRACE_DATASET para a raiz do projeto
No diretório do projeto (raiz), copie a pasta FORTH_TRACE_DATASET.

Verifique que a pasta `FORTH_TRACE_DATASET` está exatamente na raiz do repositório antes de executar o projeto.

## Rodar o projeto
Passos típicos (ajuste conforme o stack do projeto):

1. Instalar dependências:
```bash
uv sync
```

2. Para execução total:
```bash
uv run ./mainActivity.py
```

3. Para execução parcial pode-se optar por executar célula a célula utilizando kernel compatível com jupiter.