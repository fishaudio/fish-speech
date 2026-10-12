# Inferência

O Fish Audio disponibiliza dois modelos: **S2-Pro** e **Drama 3**. Ambos partilham a mesma linha de comandos e o mesmo servidor API, e o tipo de modelo é lido do `config.json` do diretório de pesos, pelo que cada secção abaixo está organizada como "tarefa → comando de cada modelo".

Conclua primeiro a [Instalação](install.md) e descarregue os [pesos do modelo](install.md#baixar-pesos).

!!! warning "Limites de execução do Drama 3"
    O Drama 3 só funciona em GPUs NVIDIA: CPU, Apple MPS e AMD ROCm não são suportados, tal como `--compile`. Veja os [Requisitos](install.md#requisitos) para a memória GPU.

!!! note
    A V100 e outras GPUs anteriores a SM80 não têm bfloat16, pelo que o Drama 3 passa a float16 (mantendo o fluxo residual em fp32). Este caminho ainda não foi testado em hardware real; se obtiver silêncio, ruído ou um erro "non-finite logits", abra uma issue.

## Inferência por Linha de Comando

### S2-Pro

O S2-Pro usa um pipeline de três passos: "tokens VQ → tokens Semânticos → áudio".

!!! note
    Se quer deixar o modelo escolher aleatoriamente o timbre, pode saltar o passo 1.

**1. Obter tokens VQ do áudio de referência**

```bash
python fish_speech/models/dac/inference.py \
    -i "test.wav" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

Deverá obter um `fake.npy` e um `fake.wav`.

**2. Gerar tokens Semânticos a partir do texto:**

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "O texto que quer converter" \
    --prompt-text "O seu texto de referência" \
    --prompt-tokens "fake.npy" \
    # --compile
```

Este comando cria um ficheiro `codes_N` no diretório de trabalho, onde N é um inteiro a partir de 0.

!!! note
    Pode querer usar `--compile` para fundir kernels CUDA e acelerar a inferência. No entanto, recomendamos a nossa otimização de aceleração de inferência sglang.
    Se não planeia usar aceleração, comente o parâmetro `--compile`.

!!! info
    Em GPUs sem suporte a bf16, pode precisar do parâmetro `--half`.

**3. Gerar voz a partir dos tokens semânticos:**

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

Depois disso obterá um ficheiro `fake.wav`.

### Drama 3

O Drama 3 faz tudo num comando; `--output` escreve diretamente o áudio descodificado:

```bash
python fish_speech/models/text2semantic/inference.py \
    --checkpoint-path checkpoints/drama-3 \
    --text "O texto que quer converter" \
    --prompt-audio "reference.wav" \
    --prompt-text "O seu texto de referência" \
    --output "output.wav" \
    --device cuda:0
```

Em GPUs mais antigas, indique vários dispositivos, por exemplo `--device cuda:0,cuda:1,cuda:2,cuda:3`; o último fica com o descodificador de áudio e o codec. Apenas `--device cuda` usa até às quatro primeiras GPUs visíveis.

## Servidor API

### Iniciar o S2-Pro

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/s2-pro \
    --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
    --decoder-config-name modded_dac_vq
```

### Iniciar o Drama 3

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/drama-3 \
    --device cuda:0
```

!!! note
    O Drama 3 não suporta `--compile`. Em GPUs mais antigas, liste vários dispositivos com `--device`; o último fica com o descodificador de áudio e o codec.

Opções de arranque partilhadas por ambos os modelos:

- `--compile`: ativa a otimização `torch.compile` (apenas S2-Pro)
- `--half`: usa o modo fp16
- `--api-key`: exige autenticação por bearer token
- `--workers`: define o número de processos de trabalho
- `--max-text-length`: limita o comprimento do texto de um pedido; 0 significa sem limite

Verificação de saúde:

```bash
curl -X GET http://127.0.0.1:8080/v1/health
```

A resposta `{"status":"ok"}` indica que o serviço está operacional.

### Endpoints da API

| Método | Caminho | Descrição |
| --- | --- | --- |
| `POST` | `/v1/tts` | Texto para voz; o corpo do pedido é msgpack |
| `GET` | `/v1/tts/defaults` | Valores de amostragem predefinidos do modelo carregado |
| `GET` | `/v1/health` | Verificação de saúde |
| `POST` | `/v1/vqgan/encode` | Áudio de referência → tokens VQ |
| `POST` | `/v1/vqgan/decode` | Tokens VQ → áudio |
| `POST` | `/v1/references/add` | Guarda uma voz de referência; campos `id`, `audio` (ficheiro), `text` |
| `GET` | `/v1/references/list` | Lista as vozes guardadas |
| `POST` | `/v1/references/update` | Renomeia; campos `old_reference_id`, `new_reference_id` |
| `DELETE` | `/v1/references/delete` | Remove; campo `reference_id` |
| `GET` | `/ui` | WebUI (incluída na imagem Docker; tem de ser construída em execução a partir do código) |

Depois de guardar a voz, use-a com `--reference_id` (o campo de áudio precisa do tipo MIME, caso contrário o servidor não o reconhece):

```bash
curl -X POST http://127.0.0.1:8080/v1/references/add \
    -F id=my-speaker \
    -F "audio=@reference.wav;type=audio/wav" \
    -F text="Transcrição do áudio de referência"

python tools/api_client.py --url http://127.0.0.1:8080/v1/tts \
    --text "Hello" --reference_id my-speaker --output demo
```

### Enviar um pedido

O modelo usado depende dos pesos carregados pelo servidor no arranque; não há campo de modelo no pedido.

```bash
python tools/api_client.py \
    --url http://127.0.0.1:8080/v1/tts \
    --text "Hello from Fish Speech" \
    --output demo
```

Parâmetros mais usados:

| Parâmetro | Descrição |
| --- | --- |
| `--reference_audio` / `--reference_text` | Áudio de referência e a respetiva transcrição; ambos precisam do mesmo número de valores |
| `--reference_speaker` | ID do falante (0–15) de cada áudio de referência, escrito como prefixo <code>&lt;&#124;speaker:N&#124;&gt;</code> (o uso principal no Drama 3) |
| `--reference_id` | Usa uma voz de referência guardada, por exemplo `my-speaker` |
| `--format` | Formato de saída: `wav`, `pcm`, `mp3` ou `opus`; predefinição `wav` |
| `--streaming` | Devolve o áudio em fluxo |
| `--single_chunk` | Gera o texto todo de uma vez, sem dividir em blocos (apenas Drama 3) |
| `--chunk_length` | Comprimento do bloco, 200 por predefinição (apenas S2-Pro) |
| `--max_new_tokens` | Máximo de frames por bloco; se vazio usa a predefinição do servidor, 1024; 0 usa o limite do próprio modelo |
| `--temperature` / `--top_p` | Amostragem; se vazio usa as predefinições do modelo (0.7 / 0.7) |
| `--seed` | Fixa a semente aleatória (não serve para fixar o timbre) |
| `--api_key` | Obrigatório quando o servidor usa `--api-key` |
| `--no-play` | Não reproduz o áudio no final (por predefinição reproduz; necessário em máquinas sem dispositivo de áudio) |

!!! note
    `repetition_penalty` existe apenas por compatibilidade; o S2-Pro e o Drama 3 ignoram-no e usam amostragem com perceção de repetição incorporada.

!!! tip
    Para iniciar o servidor com Docker, veja [Configuração do Docker](install.md#configuração-do-docker) (a imagem do servidor já inclui a WebUI: abra `http://localhost:8080/ui` depois de arrancar).

## Diferenças entre modelos

| Item | S2-Pro | Drama 3 |
| --- | --- | --- |
| Hardware | GPU / CPU | apenas NVIDIA GPU, sem `--compile` |
| Texto longo | Dividido por `chunk_length` | Dividido automaticamente em blocos de cerca de 300 bytes (UTF-8) com no máximo três falantes, cortando nos fins de frase sempre que possível; o primeiro bloco e o mais recente são reenviados como contexto |
| Limite por passagem | 1024 frames por lote por padrão; `max_new_tokens=0` gera até o comprimento do contexto | 1024 frames (cerca de 47 segundos) por bloco por padrão, o que com `single_chunk` vale para o texto inteiro; `max_new_tokens=0` significa 600 frames por bloco, ou 2048 (cerca de 95 segundos) com `single_chunk`; sempre limitado a 10.240 frames menos o comprimento do prompt |
| Falantes | A ordem das referências define o papel | As etiquetas <code>&lt;&#124;speaker:0&#124;&gt;</code>–<code>&lt;&#124;speaker:15&#124;&gt;</code> definem o papel; a ordem das referências não importa |
| `chunk_length` | Tem efeito | Não tem efeito |
| Amostragem predefinida | temperature 0.7, top-p 0.7, top-k 30 | Igual |
| `repetition_penalty` | Ignorado | Ignorado |

O Drama 3 tem ainda duas variáveis de ambiente: `FISH_DRAMA3_FP8_EXPERTS=0` ou `1` força especialistas desquantizados ou em FP8, e `FISH_DRAMA3_CUDA_GRAPH=0` desativa a descodificação com CUDA graphs.

## Inferência WebUI

### Awesome WebUI

A Awesome WebUI é uma interface Web moderna construída com TypeScript, com funcionalidades mais ricas e melhor experiência de utilização.

As imagens Docker do servidor já incluem a WebUI construída: abra `http://localhost:8080/ui` depois de o contentor arrancar. Os passos abaixo aplicam-se a execução a partir do código.

**Construir a WebUI:**

Precisa de ter Node.js e npm instalados na sua máquina ou servidor.

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

**Iniciar o servidor backend:**

Depois de construir a WebUI, volte à raiz do projeto e inicie o servidor API (`--compile` é apenas para o S2-Pro; remova-o ao carregar o Drama 3):

```bash
python tools/api_server.py --listen 0.0.0.0:8888 --compile
```

**Acesso:**

Com o servidor a correr, aceda pelo browser a:
`http://localhost:8888/ui`
