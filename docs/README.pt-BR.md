<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | [简体中文](README.zh.md) | **Portuguese** | [日本語](README.ja.md) | [한국어](README.ko.md)<br>

<a href="https://www.producthunt.com/posts/fish-speech-1-4?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-fish&#0045;speech&#0045;1&#0045;4" target="_blank">
    <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=488440&theme=light" alt="Fish&#0032;Speech&#0032;1&#0046;4 - Open&#0045;Source&#0032;Multilingual&#0032;Text&#0045;to&#0045;Speech&#0032;with&#0032;Voice&#0032;Cloning | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" />
</a>
<a href="https://trendshift.io/repositories/7014" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/7014" alt="fishaudio%2Ffish-speech | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
</a>
<br>
</div>
<br>

<div align="center">
    <img src="https://count.getloli.com/get/@fish-speech?theme=asoul" /><br>
</div>

<br>

<div align="center">
    <a target="_blank" href="https://discord.gg/Es5qTB9BcN">
        <img alt="Discord" src="https://img.shields.io/discord/1214047546020728892?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square"/>
    </a>
    <a target="_blank" href="https://hub.docker.com/r/fishaudio/fish-speech">
        <img alt="Docker" src="https://img.shields.io/docker/pulls/fishaudio/fish-speech?style=flat-square&logo=docker"/>
    </a>
    <a target="_blank" href="https://huggingface.co/spaces/fishaudio/fish-speech-1">
        <img alt="Huggingface" src="https://img.shields.io/badge/🤗%20-space%20demo-yellow"/>
    </a>
</div>

Este repositório de código é disponibilizado sob a licença Apache 2.0, e o modelo sob a licença CC-BY-NC-SA-4.0. Consulte [LICENSE](../LICENSE) para mais detalhes.

---

## Funcionalidades

1. **TTS Zero-shot & Few-shot**: Insira uma amostra vocal de 10 a 30 segundos para gerar saída de TTS de alta qualidade. **Para diretrizes detalhadas, veja [Melhores Práticas para Clonagem de Voz](https://docs.fish.audio/text-to-speech/voice-clone-best-practices).**

2. **Suporte Multilíngue e Interlingual**: Basta copiar e colar o texto multilíngue na caixa de entrada—não se preocupe com o idioma. Atualmente suporta inglês, japonês, coreano, chinês, francês, alemão, árabe e espanhol.

3. **Sem Dependência de Fonemas**: O modelo tem forte capacidade de generalização e não depende de fonemas para TTS. Ele pode lidar com textos em qualquer script de idioma.

4. **Alta Precisão**: Alcança uma CER (Taxa de Erro de Caracteres) e WER (Taxa de Erro de Palavras) de cerca de 2% para textos de 5 minutos em inglês.

5. **Rápido**: Com a aceleração fish-tech, o fator de tempo real é de aproximadamente 1:5 em um laptop Nvidia RTX 4060 e 1:15 em uma Nvidia RTX 4090.

6. **Inferência WebUI**: Apresenta uma interface de usuário web baseada em Gradio, fácil de usar e compatível com navegadores como Chrome, Firefox e Edge.

7. **Inferência GUI**: Oferece uma interface gráfica PyQt6 que funciona perfeitamente com o servidor API. Suporta Linux, Windows e macOS. [Veja o GUI](https://github.com/AnyaCoder/fish-speech-gui).

8. **Fácil de Implantar**: Configura facilmente um servidor de inferência com suporte nativo para Linux, Windows e macOS, minimizando a perda de velocidade.

## Isenção de Responsabilidade

Não nos responsabilizamos por qualquer uso ilegal do código-fonte. Consulte as leis locais sobre DMCA (Digital Millennium Copyright Act) e outras leis relevantes em sua região.

## Demonstração Online

[Fish Audio](https://fish.audio)

## Início Rápido de Inferência Local

[inference.ipynb](/inference.ipynb)

## Vídeos

#### 1.5 Introdução: [Watch the video on X (Twitter).](https://x.com/FishAudio/status/1864370933496205728)

## Documentação

- [Inglês](https://speech.fish.audio/)
- [Chinês](https://speech.fish.audio/zh/)
- [Japonês](https://speech.fish.audio/ja/)
- [Português (Brasil)](https://speech.fish.audio/pt/)

## Exemplos

- [Inglês](https://speech.fish.audio/samples/)
- [Chinês](https://speech.fish.audio/zh/samples/)
- [Japonês](https://speech.fish.audio/ja/samples/)
- [Português (Brasil)](https://speech.fish.audio/pt/samples/)

## Agradecimentos

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

## Patrocinadores

<div>
  <a href="https://6block.com/">
    <img src="https://avatars.githubusercontent.com/u/60573493" width="100" height="100" alt="6Block Avatar"/>
  </a>
  <br>
  <a href="https://6block.com/">Servidores de processamento de dados fornecidos por 6Block</a>
</div>
