# Iniciar Agente

!!! note
    Todo o documento foi traduzido por claude3.5 Sonnet, se você for um falante nativo e achar a tradução problemática, muito obrigado por nos enviar um problema ou uma solicitação pull!

## Requisitos

- Memória GPU: No mínimo 8GB (com quantização), 16GB ou mais é recomendado.
- Uso de disco: 10GB

## Download do Modelo

Você pode obter o modelo através de:

```bash
huggingface-cli download fishaudio/fish-agent-v0.1-3b --local-dir checkpoints/fish-agent-v0.1-3b
```

Coloque-os na pasta 'checkpoints'.

Você também precisará do modelo fish-speech que pode ser baixado seguindo as instruções em [inference](inference.md).

Então haverá 2 pastas em checkpoints.

O `checkpoints/fish-speech-1.4` e `checkpoints/fish-agent-v0.1-3b`

## Preparação do Ambiente

Se você já tem o Fish-speech, pode usar diretamente adicionando a seguinte instrução:
```bash
pip install cachetools
```

!!! nota
    Por favor, use a versão Python abaixo de 3.12 para compilação.

Se você não tem, use os comandos abaixo para construir seu ambiente:

```bash
sudo apt-get install portaudio19-dev

pip install -e .[stable]
```

## Iniciar a Demo do Agente

Para construir o fish-agent, use o comando abaixo na pasta principal:

```bash
python -m tools.api_server --llama-checkpoint-path checkpoints/fish-agent-v0.1-3b/ --mode agent --compile
```

O argumento `--compile` só suporta Python < 3.12, o que aumentará muito a velocidade de geração de tokens.

Não será compilado de uma vez (lembre-se).

Então abra outro terminal e use o comando:

```bash
python -m tools.e2e_webui
```

Isso criará uma WebUI Gradio no dispositivo.

Quando você usar o modelo pela primeira vez, ele irá compilar (se `--compile` estiver True) por um curto período, então aguarde com paciência.

## Gradio Webui
<p align="center">
   <img src="../../assets/figs/agent_gradio.png" width="75%">
</p>

Divirta-se!

## Desempenho

Em nossos testes, um laptop com 4060 mal consegue rodar, ficando muito sobrecarregado, gerando apenas cerca de 8 tokens/s. A 4090 gera cerca de 95 tokens/s com compilação, que é o que recomendamos.

# Sobre o Agente

A demo é uma versão alpha inicial de teste, a velocidade de inferência precisa ser otimizada, e há muitos bugs aguardando correção. Se você encontrou um bug ou quer corrigi-lo, ficaremos muito felizes em receber uma issue ou um pull request.
