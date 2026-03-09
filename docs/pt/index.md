<div align="center">
<h1>Fish Speech</h1>

[English](../en/) | [简体中文](../zh/) | **Portuguese** | [日本語](../ja/) | [한국어](../ko/) | [العربية](../ar/) <br>

<a href="https://www.producthunt.com/products/fish-speech?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-fish&#0045;audio&#0045;s1" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1023740&theme=light&period=daily&t=1761164814710" alt="Fish&#0032;Audio&#0032;S1 - Expressive&#0032;Voice&#0032;Cloning&#0032;and&#0032;Text&#0045;to&#0045;Speech | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
<a href="https://trendshift.io/repositories/7014" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/7014" alt="fishaudio%2Ffish-speech | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
</a>
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
    <a target="_blank" href="https://pd.qq.com/s/bwxia254o">
      <img alt="QQ Channel" src="https://img.shields.io/badge/QQ-blue?logo=tencentqq">
    </a>
</div>

<div align="center">
    <a target="_blank" href="https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2">
      <img alt="TTS-Arena2 Score" src="https://img.shields.io/badge/TTS_Arena2-Rank_%231-gold?style=flat-square&logo=trophy&logoColor=white">
    </a>
    <a target="_blank" href="https://huggingface.co/spaces/fishaudio/fish-speech-1">
        <img alt="Huggingface" src="https://img.shields.io/badge/🤗%20-space%20demo-yellow"/>
    </a>
    <a target="_blank" href="https://huggingface.co/fishaudio/s2-pro">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
</div>

!!! info "Aviso de Licença"
    Este repositório é lançado sob a **Licença Apache** e todos os pesos do modelo são lançados sob a **Licença CC-BY-NC-SA-4.0**. Consulte [LICENSE](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) para mais detalhes.

!!! warning "Isenção de Responsabilidade Legal"
    Não nos responsabilizamos por qualquer uso ilegal da base de códigos. Consulte as regulamentações locais sobre DMCA e outras leis relacionadas.

## Começar

Esta é a documentação oficial do Fish Speech. Siga as instruções para começar facilmente.

- [Instalação](install.md)
- [Inferência](inference.md)

## Fish Audio S2
**O melhor sistema de texto para fala em código aberto e código fechado**

O Fish Audio S2 é o modelo mais recente desenvolvido pela [Fish Audio](https://fish.audio/), projetado para gerar fala que soe natural, autêntica e emocionalmente rica — não mecânica, monótona ou confinada à leitura em estúdio.

O Fish Audio S2 foca em conversas cotidianas, suportando geração nativa de múltiplos locutores e múltiplos turnos. Também suporta controle por instruções.

A série S2 inclui vários modelos. O modelo de código aberto é o S2-Pro, que é o modelo mais poderoso da série.

Visite o [site da Fish Audio](https://fish.audio/) para uma experiência em tempo real.

### Variantes do Modelo

| Modelo | Tamanho | Disponibilidade | Descrição |
|------|------|-------------|-------------|
| S2-Pro | 4B Parâmetros | [huggingface](https://huggingface.co/fishaudio/s2-pro) | Modelo emblemático completo com a mais alta qualidade e estabilidade |
| S2-Flash | - - - - | [fish.audio](https://fish.audio/) | Nosso modelo de código fechado com maior velocidade e menor latência |

Para mais detalhes sobre os modelos, consulte o relatório técnico.

## Destaques

<img src="../assets/totalability.png" width=200%>

### Controle por Linguagem Natural

O Fish Audio S2 permite que os usuários usem linguagem natural para controlar o desempenho, informações paralinguísticas, emoções e outras características de voz de cada frase, em vez de usar apenas tags curtas para controlar vagamente o desempenho do modelo. Isso aumenta muito a qualidade geral do conteúdo gerado.

### Suporte Multilíngue

O Fish Audio S2 suporta conversão de texto em fala multilíngue de alta qualidade sem a necessidade de fonemas ou pré-processamento específico por idioma. Incluindo:

**Inglês, Chinês, Japonês, Coreano, Árabe, Alemão, Francês...**

**E muito mais!**

A lista está em constante expansão. Verifique a [Fish Audio](https://fish.audio/) para os lançamentos mais recentes.

### Geração Nativa de Múltiplos Locutores

<img src="../assets/chattemplate.png" width=200%>

O Fish Audio S2 permite que os usuários carreguem áudio de referência contendo múltiplos locutores, e o modelo processará as características de cada locutor por meio do token `<|speaker:i|>`. Você pode então controlar o desempenho do modelo por meio de tokens de ID de locutor, alcançando múltiplos locutores em uma única geração. Não há mais necessidade de carregar áudio de referência e gerar fala para cada locutor individualmente.

### Geração de Diálogos em Múltiplos Turnos

Graças à expansão do contexto do modelo, nosso modelo agora pode usar as informações das partes anteriores do diálogo para melhorar a expressividade do conteúdo gerado subsequentemente, aumentando assim a naturalidade do conteúdo.

### Clonagem de Voz Rápida

O Fish Audio S2 suporta clonagem de voz precisa usando amostras de referência curtas (geralmente de 10 a 30 segundos). O modelo pode capturar timbre, estilo de fala e tendência emocional, gerando vozes clonadas realistas e consistentes sem ajuste fino adicional.

---

## Agradecimentos

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

## Relatório Técnico

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
