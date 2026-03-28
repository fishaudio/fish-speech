<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | [简体中文](docs/README.zh.md) | [Portuguese](docs/README.pt-BR.md) | [日本語](docs/README.ja.md) | [한국어](docs/README.ko.md) | [العربية](docs/README.ar.md) | **Español** <br>

<a href="https://www.producthunt.com/products/fish-speech?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-fish&#0045;audio&#0045;s1" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1023740&theme=light&period=daily&t=1761164814710" alt="Fish&#0032;Audio&#0032;S1 - Clonación&#0032;de&#0032;voz&#0032;expresiva&#0032;y&#0032;texto&#0045;a&#0045;voz | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a> <a href="https://trendshift.io/repositories/7014" target="_blank"> <img src="https://trendshift.io/api/badge/repositories/7014" alt="fishaudio%2Ffish-speech | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/> </a> <br>

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
    <a target="_blank" href="https://huggingface.co/fishaudio/s2-pro">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
    <a target="_blank" href="https://fish.audio/blog/fish-audio-open-sources-s2/">
        <img alt="Fish Audio Blog" src="https://img.shields.io/badge/Blog-Fish_Audio_S2-1f7a8c?style=flat-square&logo=readme&logoColor=white"/>
    </a>
    <a target="_blank" href="https://arxiv.org/abs/2603.08823">
        <img alt="Paper | Informe Técnico" src="https://img.shields.io/badge/Paper-Technical_Report-b31b1b?style=flat-square"/>
    </a>
</div>

> [!IMPORTANT]
> **Aviso de Licencia**
> Este código y los pesos de modelo asociados se publican bajo la **[FISH AUDIO RESEARCH LICENSE](LICENSE)**. Consulta [LICENSE](LICENSE) para más detalles. Se tomarán acciones ante cualquier violación de la licencia.

> [!WARNING]
> **Descargo de Responsabilidad Legal**
> No asumimos ninguna responsabilidad por el uso ilegal de este código. Consulta las leyes locales relacionadas con DMCA y otras normativas aplicables.

## Inicio Rápido

### Para humanos

Aquí tienes la documentación oficial de Fish Audio S2. Sigue las instrucciones para comenzar fácilmente.

* [Instalación](https://speech.fish.audio/install/)
* [Inferencia por línea de comandos](https://speech.fish.audio/inference/#command-line-inference)
* [Inferencia con WebUI](https://speech.fish.audio/inference/#webui-inference)
* [Inferencia en servidor](https://speech.fish.audio/server/)
* [Configuración de Docker](https://speech.fish.audio/install/#docker-setup)

> [!IMPORTANT]
> **Para el servidor SGLang, consulta [SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md).**

### Para agentes LLM

```
Instala y configura Fish-Audio S2 siguiendo las instrucciones aquí: https://speech.fish.audio/install/
```

## Fish Audio S2 Pro

**Sistema de texto a voz (TTS) multilingüe de última generación, redefiniendo los límites de la generación de voz.**

Fish Audio S2 Pro es el modelo multimodal más avanzado desarrollado por Fish Audio. Entrenado con más de **10 millones de horas** de datos de audio que abarcan más de **80 idiomas**, S2 Pro combina una arquitectura **Dual-Autoregressive (Dual-AR)** con alineación mediante aprendizaje por refuerzo (RL) para generar voz extremadamente natural, realista y emocionalmente rica, liderando tanto sistemas open-source como closed-source.

La principal fortaleza de S2 Pro es su soporte para control fino a nivel **sub-palabra (sub-word level)** de prosodia y emoción usando etiquetas en lenguaje natural (por ejemplo `[whisper]`, `[excited]`, `[angry]`), además de soportar de forma nativa generación multi-speaker y conversaciones multi-turno.

Visita el sitio web de Fish Audio para probarlo en vivo, o lee el informe técnico y el blog para más detalles.

### Variantes del modelo

| Modelo | Tamaño        | Disponibilidad                                         | Descripción                                               |
| ------ | ------------- | ------------------------------------------------------ | --------------------------------------------------------- |
| S2-Pro | 4B parámetros | [HuggingFace](https://huggingface.co/fishaudio/s2-pro) | Modelo insignia completo con máxima calidad y estabilidad |

Más detalles pueden encontrarse en el informe técnico.

## Resultados de benchmarks

| Benchmark                                 | Fish Audio S2              |
| ----------------------------------------- | -------------------------- |
| Seed-TTS Eval — WER (Chino)               | **0.54%** (mejor global)   |
| Seed-TTS Eval — WER (Inglés)              | **0.99%** (mejor global)   |
| Audio Turing Test (con instrucciones)     | **0.515** media posterior  |
| EmergentTTS-Eval — Tasa de victoria       | **81.88%** (máximo global) |
| Fish Instruction Benchmark — TAR          | **93.3%**                  |
| Fish Instruction Benchmark — Calidad      | **4.51 / 5.0**             |
| Multilingüe (MiniMax Testset) — Mejor WER | **11 de 24** idiomas       |
| Multilingüe (MiniMax Testset) — Mejor SIM | **17 de 24** idiomas       |

En Seed-TTS Eval, S2 logra el menor WER entre todos los modelos evaluados, incluyendo sistemas cerrados: Qwen3-TTS (0.77/1.24), MiniMax Speech-02 (0.99/1.90), Seed-TTS (1.12/2.25). En el Audio Turing Test, 0.515 supera a Seed-TTS (0.417) en un 24% y a MiniMax-Speech (0.387) en un 33%. En EmergentTTS-Eval, S2 destaca especialmente en paralingüística (91.61%), preguntas (84.41%) y complejidad sintáctica (83.39%).

## Highlights

<img src="./docs/assets/totalability.png" width=200%>

### Control fino inline mediante lenguaje natural

S2 Pro aporta un nivel de “alma” sin precedentes a la voz. Usando sintaxis `[tag]`, puedes insertar instrucciones emocionales con precisión en cualquier parte del texto.

* **Más de 15,000 tags únicos soportados**
* Soporta descripciones libres como `[whisper in small voice]`, `[professional broadcast tone]`, `[pitch up]`

### Arquitectura Dual-Autoregressive (Dual-AR)

* **Slow AR (4B parámetros)**: modela la estructura temporal
* **Fast AR (400M parámetros)**: reconstruye detalles acústicos finos

### Alineación mediante RL

* Usa GRPO
* Señales de recompensa multidimensionales

### Rendimiento extremo en streaming

* RTF: 0.195
* TTFA: ~100 ms
* +3000 tokens/s

### Soporte multilingüe robusto

* Más de 80 idiomas
* Sin necesidad de phonemes específicos

### Generación multi-speaker nativa

<img src="./docs/assets/chattemplate.png" width=200%>

Permite múltiples hablantes usando `<|speaker:i|>` en una sola generación.

### Generación multi-turno

Mantiene contexto para mejorar la naturalidad.

### Clonación de voz rápida

* Solo 10–30 segundos de audio
* Alta fidelidad de timbre y estilo

Para usar con SGLang Server, consulta el README correspondiente.

---

## Créditos

* [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
* [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
* [GPT VITS](https://github.com/innnky/gpt-vits)
* [MQTTS](https://github.com/b04901014/MQTTS)
* [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
* [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
* [Qwen3](https://github.com/QwenLM/Qwen3)

## Informe Técnico

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

@misc{liao2026fishaudios2technical,
      title={Fish Audio S2 Technical Report}, 
      author={Shijia Liao and Yuxuan Wang and Songting Liu and Yifan Cheng and Ruoyi Zhang and Tianyu Li and Shidong Li and Yisheng Zheng and Xingwei Liu and Qingzheng Wang and Zhizhuo Zhou and Jiahua Liu and Xin Chen and Dawei Han},
      year={2026},
      eprint={2603.08823},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.08823}, 
}
```
