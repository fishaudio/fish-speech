<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | [简体中文](README.zh.md) | **Portuguese** | [日本語](README.ja.md) | [한국어](README.ko.md) <br>

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
    <a target="_blank" href="https://pd.qq.com/s/bwxia254o">
      <img alt="QQ Channel" src="https://img.shields.io/badge/QQ-blue?logo=tencentqq">
    </a>
</div>

Esta base de código é lançada sob a Licença Apache e todos os pesos dos modelos são lançados sob a Licença CC-BY-NC-SA-4.0. Consulte [LICENSE](../LICENSE) para mais detalhes.

Estamos animados em anunciar que mudamos nosso nome para OpenAudio, esta será uma nova série de modelos Text-to-Speech.

Demo disponível em [Fish Audio Playground](https://fish.audio).

Visite o [site OpenAudio](https://openaudio.com) para blog e relatório técnico.

## Recursos
### OpenAudio-S1 (Nova versão do Fish-Speech)

1. Este modelo possui **TODOS OS RECURSOS** que o fish-speech tinha.

2. O OpenAudio S1 suporta uma variedade de marcadores emocionais, de tom e especiais para aprimorar a síntese de fala:
   
      (angry) (sad) (disdainful) (excited) (surprised) (satisfied) (unhappy) (anxious) (hysterical) (delighted) (scared) (worried) (indifferent) (upset) (impatient) (nervous) (guilty) (scornful) (frustrated) (depressed) (panicked) (furious) (empathetic) (embarrassed) (reluctant) (disgusted) (keen) (moved) (proud) (relaxed) (grateful) (confident) (interested) (curious) (confused) (joyful) (disapproving) (negative) (denying) (astonished) (serious) (sarcastic) (conciliative) (comforting) (sincere) (sneering) (hesitating) (yielding) (painful) (awkward) (amused)

   Também suporta marcadores de tom:

   (tom apressado) (gritando) (berrando) (sussurrando) (tom suave)

    Há alguns marcadores especiais que são suportados:

    (rindo) (dando risadinhas) (soluçando) (chorando alto) (suspirando) (ofegando) (gemendo) (multidão rindo) (riso de fundo) (audiência rindo)

    Você também pode usar **Ha,ha,ha** para controlar, há muitos outros casos esperando para serem explorados por você mesmo.

3. O OpenAudio S1 inclui os seguintes tamanhos:
-   **S1 (4B, proprietário):** O modelo de tamanho completo.
-   **S1-mini (0.5B, código aberto):** Uma versão destilada do S1.

    Tanto S1 quanto S1-mini incorporam Aprendizado por Reforço online a partir de Feedback Humano (RLHF).

4. Avaliações

    **Métricas de Avaliação Seed TTS (Inglês, avaliação automática, baseada no OpenAI gpt-4o-transcribe, distância do locutor usando Revai/pyannote-wespeaker-voxceleb-resnet34-LM):**

    -   **S1:**
        -   WER (Taxa de Erro de Palavra): **0.008**
        -   CER (Taxa de Erro de Caractere): **0.004**
        -   Distância: **0.332**
    -   **S1-mini:**
        -   WER (Taxa de Erro de Palavra): **0.011**
        -   CER (Taxa de Erro de Caractere): **0.005**
        -   Distância: **0.380**
    

## Aviso Legal

Não assumimos qualquer responsabilidade por qualquer uso ilegal da base de código. Consulte suas leis locais sobre DMCA e outras leis relacionadas.

## Vídeos

#### A ser continuado.

## Documentos

- [Construir Ambiente](en/install.md)
- [Inferência](en/inference.md)

Deve-se notar que o modelo atual **NÃO SUPORTA AJUSTE FINO**.

## Créditos

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

## Relatório Técnico (V1.4)
```bibtex
@misc{fish-speech-v1.4,
      title={Fish-Speech: Leveraging Large Language Models for Advanced Multilingual Text-to-Speech Synthesis},
      author={Shijia Liao and Yuxuan Wang and Tianyu Li and Yifan Cheng and Ruoyi Zhang and Rongzhi Zhou and Yijin Xing},
      year={2024},
      eprint={2411.01156},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2411.01156},
}
```

