<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | [简体中文](README.zh.md) | **Portuguese** | [日本語](README.ja.md) | [한국어](README.ko.md) | [العربية](README.ar.md) <br>

<a href="https://www.producthunt.com/products/fish-speech?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-fish&#0045;audio&#0045;s1" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1023740&theme=light&period=daily&t=1761164814710" alt="Fish&#0032;Audio&#0032;S1 - Expressive&#0032;Voice&#0032;Cloning&#0032;and&#0032;Text&#0045;to&#0045;Speech | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
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
    <a target="_blank" href="https://pd.qq.com/s/bwxia254o">
      <img alt="QQ Channel" src="https://img.shields.io/badge/QQ-blue?logo=tencentqq">
    </a>
</div>

<div align="center">
    <a target="_blank" href="https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2">
      <img alt="TTS-Arena2 Score" src="https://img.shields.io/badge/TTS_Arena2-Rank_%231-gold?style=flat-square&logo=trophy&logoColor=white">
    </a>
    <a target="_blank" href="https://huggingface.co/fishaudio/s2-pro">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
</div>

> [!IMPORTANT]
> **Aviso de Licença**
> Este repositório e os pesos de modelo associados são lançados sob a **[FISH AUDIO RESEARCH LICENSE](../LICENSE)**. Consulte [LICENSE](../LICENSE) para obter mais detalhes.

> [!WARNING]
> **Isenção de Responsabilidade Legal**
> Não nos responsabilizamos por qualquer uso ilegal do repositório. Consulte as leis locais sobre DMCA e outras leis relacionadas.

## Comece Aqui

Aqui estão os documentos oficiais do Fish Speech, siga as instruções para começar facilmente.

- [Instalação](https://speech.fish.audio/pt/install/)
- [Inferência](https://speech.fish.audio/pt/inference/)

## Fish Audio S2
**O melhor sistema de conversão de texto em fala entre código aberto e código fechado**

O Fish Audio S2 é o modelo mais recente desenvolvido pela [Fish Audio](https://fish.audio/), projetado para gerar falas que soam naturais, realistas e emocionalmente ricas — não robóticas, não monótonas e não limitadas à narração em estilo de estúdio.

O Fish Audio S2 foca em conversas diárias e diálogos, o que permite a geração nativa de múltiplos falantes e turnos. Também suporta controle por instrução.

A série S2 contém vários modelos, o modelo de código aberto é o S2-Pro, que é o melhor modelo da coleção.

Visite o [site da Fish Audio](https://fish.audio/) para um playground ao vivo.

### Variantes do Modelo

| Modelo | Tamanho | Disponibilidade | Descrição |
|------|------|-------------|-------------|
| S2-Pro | 4B parâmetros | [huggingface](https://huggingface.co/fishaudio/s2-pro) | Modelo carro-chefe completo com máxima qualidade e estabilidade |
| S2-Flash | - - - - | [fish.audio](https://fish.audio/) | Nosso modelo de código fechado com maior velocidade e menor latência |

Mais detalhes do modelo podem ser encontrados no relatório técnico.

## Destaques

<img src="./assets/totalability.png" width=200%>

### Controle de Linguagem Natural

O Fish Audio S2 permite que os usuários usem linguagem natural para controlar a expressão de cada frase, informações paralinguísticas, emoções e outras características de voz. Em vez de simplesmente usar etiquetas curtas para controlar vagamente o desempenho do modelo, isso melhora significativamente a qualidade geral do conteúdo gerado.

### Suporte Multilíngue

O Fish Audio S2 oferece suporte a conversão de texto em fala multilíngue de alta qualidade sem a necessidade de fonemas ou processamento específico de idioma. Incluindo:

**Inglês, Chinês, Japonês, Coreano, Árabe, Alemão, Francês...**

**E MUITO MAIS!**

A lista está em constante expansão, verifique o [Fish Audio](https://fish.audio/) para os lançamentos mais recentes.

### Geração Nativa de Múltiplos Falantes

<img src="./assets/chattemplate.png" width=200%>

O Fish Audio S2 permite que os usuários carreguem áudio de referência com vários falantes; o modelo lidará com as características de cada falante por meio do token `<|speaker:i|>`. Então, você pode controlar o desempenho do modelo com the token de ID do falante, permitindo que uma única geração inclua vários falantes. Você não precisa mais carregar áudios de referência separadamente para cada falante.

### Geração de Múltiplos Turnos

Graças à extensão do contexto do modelo, nosso modelo agora pode usar informações anteriores para melhorar a expressividade e a naturalidade dos conteúdos gerados subsequentemente.

### Clonagem de Voz Rápida

O Fish Audio S2 suporta clonagem de voz precisa usando uma pequena amostra de referência (tipicamente de 10 a 30 segundos). O modelo captura o timbre, o estilo de fala e as tendências emocionais, produzindo vozes clonadas realistas e consistentes sem ajuste fino adicional.

---

## Créditos

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
