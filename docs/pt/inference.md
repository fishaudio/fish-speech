# Inferência

O modelo Fish Audio S2 requer uma grande quantidade de VRAM. Recomendamos o uso de uma GPU com pelo menos 24GB para inferência.

## Baixar Pesos

Primeiro, você precisa baixar os pesos do modelo:

```bash
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

## Inferência por Linha de Comando

!!! note
    Se você planeja deixar o modelo escolher aleatoriamente um timbre de voz, pode pular esta etapa.

### 1. Obter tokens VQ do áudio de referência

```bash
python fish_speech/models/dac/inference.py \
    -i "test.wav" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

Você deve obter um `fake.npy` e um `fake.wav`.

### 2. Gerar tokens Semânticos a partir do texto:

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "O texto que você deseja converter" \
    --prompt-text "Seu texto de referência" \
    --prompt-tokens "fake.npy" \
    # --compile
```

Este comando criará um arquivo `codes_N` no diretório de trabalho, onde N é um número inteiro começando em 0.

!!! note
    Você pode querer usar `--compile` para fundir kernels CUDA para uma inferência mais rápida. No entanto, recomendamos usar nossa otimização de aceleração de inferência sglang.
    Da mesma forma, se você não planeja usar aceleração, pode comentar o parâmetro `--compile`.

!!! info
    Para GPUs que não suportam bf16, você pode precisar usar o parâmetro `--half`.

### 3. Gerar vocais a partir de tokens semânticos:

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

Depois disso, você obterá um arquivo `fake.wav`.

## S3-drama

O S3-drama usa a mesma linha de comando e o mesmo servidor API do S2-Pro; o tipo de modelo é detectado a partir do `config.json` do checkpoint.

!!! warning
    O S3-drama funciona apenas em GPUs NVIDIA. CPU, Apple MPS e AMD ROCm não são suportados, e `--compile` também não.

| GPU | Modo | Memória da GPU |
| --- | --- | --- |
| SM89 ou mais recente (L40, H100, H20) | Especialistas MoE mantidos em FP8, decodificação com CUDA graphs, uma GPU | cerca de 41 GB |
| GPUs mais antigas (V100, A100) | Especialistas MoE desquantizados e distribuídos por várias GPUs | cerca de 70 GB no total |

!!! note
    A V100 e outras GPUs anteriores à SM80 não suportam bfloat16, então o S3 passa a usar float16 nelas (mantendo o fluxo residual em fp32). Esse caminho ainda não foi testado em hardware real; se você obtiver silêncio, ruído ou um erro "non-finite logits", abra uma issue.

### Baixar pesos

```bash
hf download fishaudio/s3-drama-it-fp8 --local-dir checkpoints/s3-drama-fp8
```

Se o diretório do checkpoint não tiver `codec.pth`, será usado o codec do S2-Pro em `checkpoints/s2-pro/codec.pth`.

### Inferência por linha de comando

```bash
python fish_speech/models/text2semantic/inference.py \
    --checkpoint-path checkpoints/s3-drama-fp8 \
    --text "O texto que você deseja converter" \
    --prompt-audio "reference.wav" \
    --prompt-text "Seu texto de referência" \
    --output "output.wav" \
    --device cuda:0
```

`--output` grava diretamente o áudio decodificado. Em GPUs mais antigas, indique vários dispositivos, por exemplo `--device cuda:0,cuda:1,cuda:2,cuda:3`; o último fica com o decodificador de áudio e o codec. Apenas `--device cuda` usa até as quatro primeiras GPUs visíveis.

### Servidor API

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/s3-drama-fp8 \
    --device cuda:0
```

Com Docker, execute a imagem do servidor apontando para o checkpoint do S3. Em placas SM89 ou mais recentes, exponha uma única GPU como abaixo; em placas mais antigas, use `--gpus all`:

```bash
docker run --gpus device=0 \
    -v ./checkpoints:/app/checkpoints \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/s3-drama-fp8 \
    -p 8080:8080 \
    fishaudio/fish-speech:server-cuda
```

Com o Docker Compose, defina as mesmas variáveis; `DEVICE=cuda:0` mantém o S3 em uma GPU, e sem ela são usadas até quatro GPUs:

```bash
LLAMA_CHECKPOINT_PATH=checkpoints/s3-drama-fp8 DEVICE=cuda:0 docker compose --profile server up
```

### Diferenças em relação ao S2-Pro

- Textos longos são divididos em blocos de cerca de 300 bytes (UTF-8) com no máximo três locutores cada, de preferência no fim das frases, e os blocos são gerados um após o outro. O primeiro bloco e o mais recente são reenviados como contexto para manter as vozes consistentes. Por padrão, cada bloco produz no máximo 1024 frames (cerca de 47 segundos).
- Defina `single_chunk` (`--single-chunk` na linha de comando) para gerar o texto inteiro de uma só vez; o limite padrão de 1024 frames passa então a valer para o texto todo, então aumente `max_new_tokens` para textos longos ou use 0 para 2048 frames (cerca de 95 segundos). Em ambos os modos, um `max_new_tokens` maior é limitado a 10.240 frames menos o comprimento do prompt.
- São aceitas as tags de locutor de `<|speaker:0|>` a `<|speaker:15|>`; texto sem tag é o locutor 0. Para dar a cada locutor a sua própria voz, defina o campo `speaker` de cada áudio de referência (`--reference_speaker` em `tools/api_client.py`), que adiciona a tag às transcrições que não começam com uma, ou comece você mesmo a transcrição com a tag desse locutor (por exemplo `<|speaker:1|>`); a ordem das referências não define o locutor.
- `chunk_length` não tem efeito. `repetition_penalty` é aceito por compatibilidade, mas ignorado, assim como no S2-Pro: os dois modelos usam amostragem com reconhecimento de repetição integrada. A amostragem padrão segue o SDK oficial da Fish Audio, como no S2-Pro: temperature 0.7, top-p 0.7, top-k 30.
- Com `streaming` ativado, o áudio é retornado bloco a bloco.
- `FISH_S3_FP8_EXPERTS=0` ou `1` força especialistas desquantizados ou em FP8, e `FISH_S3_CUDA_GRAPH=0` desativa a decodificação com CUDA graphs.

## Inferência WebUI

### Awesome WebUI

A Awesome WebUI é uma interface web moderna baseada em TypeScript, oferecendo funcionalidades mais ricas e uma melhor experiência do usuário.

As imagens Docker do servidor já incluem a WebUI compilada: abra `http://localhost:8080/ui` depois que o contêiner estiver em execução. Os passos abaixo são para executar a partir do código-fonte.

**Construir a WebUI:**

Você precisa ter o Node.js e o npm instalados em seu computador local ou servidor.

1. Entre no diretório `awesome_webui`:
   ```bash
   cd awesome_webui
   ```
2. Instale as dependências:
   ```bash
   npm install
   ```
3. Construa a WebUI:
   ```bash
   npm run build
   ```

**Iniciar o Servidor Backend:**

Após a construção da WebUI, retorne ao diretório raiz do projeto e inicie o servidor API:

```bash
python tools/api_server.py --listen 0.0.0.0:8888 --compile
```

**Acesso:**

Após o servidor ser iniciado, você pode acessá-lo através do navegador no seguinte endereço:
`http://localhost:8888/ui`
