# Introdução

<div>
<a target="_blank" href="https://discord.gg/Es5qTB9BcN">
<img alt="Discord" src="https://img.shields.io/discord/1214047546020728892?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square"/>
</a>
<a target="_blank" href="http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=jCKlUP7QgSm9kh95UlBoYv6s1I-Apl1M&authKey=xI5ttVAp3do68IpEYEalwXSYZFdfxZSkah%2BctF5FIMyN2NqAa003vFtLqJyAVRfF&noverify=0&group_code=593946093">
<img alt="QQ" src="https://img.shields.io/badge/QQ Group-%2312B7F5?logo=tencent-qq&logoColor=white&style=flat-square"/>
</a>
<a target="_blank" href="https://hub.docker.com/r/fishaudio/fish-speech">
<img alt="Docker" src="https://img.shields.io/docker/pulls/fishaudio/fish-speech?style=flat-square&logo=docker"/>
</a>
</div>

!!! warning
    Não nos responsabilizamos por qualquer uso ilegal do código-fonte. Consulte as leis locais sobre DMCA (Digital Millennium Copyright Act) e outras leis relevantes em sua região. <br/>
    Este repositório de código e os modelos são distribuídos sob a licença CC-BY-NC-SA-4.0.

<p align="center">
   <img src="../assets/figs/diagrama.png" width="75%">
</p>

## Requisitos

- Memória da GPU: 4GB (para inferência), 8GB (para ajuste fino)
- Sistema: Linux, Windows

## Configuração para Windows

No Windows, usuários avançados podem considerar usar o WSL2 ou Docker para executar o código.

Para Usuários comuns (não-avançados), siga os métodos abaixo para executar o código sem um ambiente Linux (incluindo suporte para `torch.compile`):

<ol>
   <li>Extraia o arquivo compactado do projeto.</li>
   <li>Prepare o ambiente conda:
    <ul>
      <li>Abra o <code>install_env.bat</code> para baixar e iniciar a instalação do miniconda.</li>
      <li>Personalize o download (opcional):
        <ul>
          <li>**Site espelho:** Para usar um site espelho para downloads mais rápidos, defina <code>USE_MIRROR=true</code> no <code>install_env.bat</code> (padrão). Caso contrário, use <code>USE_MIRROR=false</code>.</li>
          <li>**Ambiente compilado:** Para baixar a versão de prévia com o ambiente compilado, defina <code>INSTALL_TYPE=preview</code>. Para a versão estável sem ambiente compilado, use <code>INSTALL_TYPE=stable</code>.</li>
        </ul>
      </li>
      </ul>
   </li>
   <li>Se você escolheu a versão de prévia com ambiente compilado (<code>INSTALL_TYPE=preview</code>), siga para a próxima etapa (opcional):
      <ol>
            <li>Baixe o compilador LLVM usando os seguintes links:
               <ul>
                  <li><a href="https://huggingface.co/fishaudio/fish-speech-1/resolve/main/LLVM-17.0.6-win64.exe?download=true">LLVM-17.0.6 (download do site original)</a></li>
                  <li><a href="https://hf-mirror.com/fishaudio/fish-speech-1/resolve/main/LLVM-17.0.6-win64.exe?download=true">LLVM-17.0.6 (download do site espelho)</a></li>
                  <li>Após baixar o <code>LLVM-17.0.6-win64.exe</code>, clique duas vezes para instalá-lo, escolha um local de instalação apropriado. E durante a instalação, marque a opção <code>Add Path to Current User</code> para adicionar às variáveis de ambiente.</li>
                  <li>Confirme se a instalação foi concluída.</li>
               </ul>
            </li>
            <li>Baixe e instale o pacote Microsoft Visual C++ Redistributable para resolver possíveis problemas de .dll ausentes.
               <ul>
                  <li><a href="https://aka.ms/vs/17/release/vc_redist.x64.exe">Download do MSVC++ 14.40.33810.0</a></li>
               </ul>
            </li>
            <li>Baixe e instale o Visual Studio Community Edition para obter as ferramentas de compilação MSVC++, resolvendo as dependências do arquivo de cabeçalho LLVM.
               <ul>
                  <li><a href="https://visualstudio.microsoft.com/pt-br/downloads/">Download do Visual Studio</a></li>
                  <li>Após instalar o Visual Studio Installer, baixe o Visual Studio Community 2022.</li>
                  <li>Clique no botão <code>Modificar</code>, conforme mostrado abaixo, encontre a opção <code>Desenvolvimento para desktop com C++</code> e marque-a para download.</li>
                  <p align="center">
                     <img src="../assets/figs/VS_1_pt-BR.png" width="75%">
                  </p>
               </ul>
            </li>
            <li>Instale o <a href="https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64">CUDA Toolkit 12</a></li>
      </ol>
   </li>
   <li>Clique duas vezes em <code>start.bat</code> para entrar na página da WebUI de configuração de inferência de treinamento do Fish-Speech.
      <ul>
            <li>(Opcional) Se desejar ir direto para a página de inferência, edite o arquivo <code>API_FLAGS.txt</code> no diretório raiz do projeto e modifique as três primeiras linhas da seguinte forma:
               <pre><code>--infer
# --api
# --listen ...
...</code></pre>
            </li>
            <li>(Opcional) Se preferir iniciar o servidor da API, edite o arquivo <code>API_FLAGS.txt</code> no diretório raiz do projeto e modifique as três primeiras linhas da seguinte forma:
               <pre><code># --infer
--api
--listen ...
...</code></pre>
            </li>
      </ul>
   </li>
   <li>(Opcional) Clique duas vezes em <code>run_cmd.bat</code> para entrar na CLI do conda/python deste projeto.</li>
</ol>

## Configuração para Linux

```bash
# Crie um ambiente virtual python 3.10, você também pode usar virtualenv
conda create -n fish-speech python=3.10
conda activate fish-speech

# Instale o pytorch
pip3 install torch torchvision torchaudio

# Instale o fish-speech
pip3 install -e .[stable]

# Para os Usuário do Ubuntu / Debian: Instale o sox + ffmpeg
apt install libsox-dev ffmpeg
```

## Configuração do Docker

1. Instale o NVIDIA Container Toolkit:

    Para usar a GPU com Docker para treinamento e inferência de modelos, você precisa instalar o NVIDIA Container Toolkit:

    Para usuários Ubuntu:

    ```bash
    # Adicione o repositório remoto
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    # Instale o nvidia-container-toolkit
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    # Reinicie o serviço Docker
    sudo systemctl restart docker
    ```

    Para usuários de outras distribuições Linux, consulte o guia de instalação: [NVIDIA Container Toolkit Install-guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

2. Baixe e execute a imagem fish-speech

    ```shell
    # Baixe a imagem
    docker pull fishaudio/fish-speech:latest-dev
    # Execute a imagem
    docker run -it \
        --name fish-speech \
        --gpus all \
        -p 7860:7860 \
        fishaudio/fish-speech:latest-dev \
        zsh
    # Se precisar usar outra porta, modifique o parâmetro -p para YourPort:7860
    ```

3. Baixe as dependências do modelo

    Certifique-se de estar no terminal do contêiner Docker e, em seguida, baixe os modelos necessários `vqgan` e `llama` do nosso repositório HuggingFace.

    ```bash
    huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4
    ```

4. Configure as variáveis de ambiente e acesse a WebUI

    No terminal do contêiner Docker, digite `export GRADIO_SERVER_NAME="0.0.0.0"` para permitir o acesso externo ao serviço gradio dentro do Docker.
    Em seguida, no terminal do contêiner Docker, digite `python tools/webui.py` para iniciar o serviço WebUI.

    Se estiver usando WSL ou MacOS, acesse [http://localhost:7860](http://localhost:7860) para abrir a interface WebUI.

    Se estiver implantando em um servidor, substitua localhost pelo IP do seu servidor.

## Histórico de Alterações
- 10/09/2024: Fish-Speech atualizado para a versão 1.4, aumentado o tamanho do conjunto de dados, quantizer n_groups 4 -> 8.
- 02/07/2024: Fish-Speech atualizado para a versão 1.2, removido o Decodificador VITS e aprimorado consideravelmente a capacidade de zero-shot.
- 10/05/2024: Fish-Speech atualizado para a versão 1.1, implementado o decodificador VITS para reduzir a WER e melhorar a similaridade de timbre.
- 22/04/2024: Finalizada a versão 1.0 do Fish-Speech, modificados significativamente os modelos VQGAN e LLAMA.
- 28/12/2023: Adicionado suporte para ajuste fino `lora`.
- 27/12/2023: Adicionado suporte para `gradient checkpointing`, `causual sampling` e `flash-attn`.
- 19/12/2023: Atualizada a interface web e a API HTTP.
- 18/12/2023: Atualizada a documentação de ajuste fino e exemplos relacionados.
- 17/12/2023: Atualizado o modelo `text2semantic`, suportando o modo sem fonemas.
- 13/12/2023: Versão beta lançada, incluindo o modelo VQGAN e um modelo de linguagem baseado em LLAMA (suporte apenas a fonemas).

## Agradecimentos

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [Transformers](https://github.com/huggingface/transformers)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
