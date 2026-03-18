<div align="center">
<h1>Fish Speech</h1>

[English](README.md) | [简体中文](docs/README.zh.md) | [Portuguese](docs/README.pt-BR.md) | [日本語](docs/README.ja.md) | [한국어](docs/README.ko.md) | [العربية](docs/README.ar.md) | **Español** <br>

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
    <a target="_blank" href="https://huggingface.co/fishaudio/s2">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
    <a target="_blank" href="https://fish.audio/blog/fish-audio-open-sources-s2/">
        <img alt="Fish Audio Blog" src="https://img.shields.io/badge/Blog-Fish_Audio_S2-1f7a8c?style=flat-square&logo=readme&logoColor=white"/>
    </a>
    <a target="_blank" href="https://github.com/fishaudio/fish-speech/blob/main/FishAudioS2TecReport.pdf">
        <img alt="Paper | Technical Report" src="https://img.shields.io/badge/Paper-Technical_Report-b31b1b?style=flat-square"/>
    </a>
</div>

> [!IMPORTANT]
> **Aviso de Licencia**
> Este código y sus pesos de modelo asociados se publican bajo **[FISH AUDIO RESEARCH LICENSE](LICENSE)**. Por favor, consulta [LICENSE](LICENSE) para más detalles. Tomaremos acciones contra cualquier violación de la licencia.

> [!WARNING]
> **Aviso Legal**
> No asumimos ninguna responsabilidad por el uso ilegal de este código. Por favor, consulta las leyes locales sobre DMCA y otras leyes relacionadas.

## Inicio Rápido

### Para Humanos

Aquí están los documentos oficiales de Fish Audio S2, sigue las instrucciones para comenzar fácilmente.

* [Instalación](https://speech.fish.audio/install/)
* [Inferencia por Línea de Comandos](https://speech.fish.audio/inference/#command-line-inference)
* [Inferencia con WebUI](https://speech.fish.audio/inference/#webui-inference)
* [Inferencia en Servidor](https://speech.fish.audio/server/)
* [Configuración con Docker](https://speech.fish.audio/install/#docker-setup)

> [!IMPORTANT]
> **Para el servidor SGLang, por favor lee [SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md).**

### Para Agente LLM

```
Instala y configura Fish-Audio S2 siguiendo las instrucciones aquí: https://speech.fish.audio/install/
```

## Fish Audio S2

**Mejor sistema de texto a voz entre código abierto y cerrado**

Fish Audio S2 es el modelo más reciente desarrollado por [Fish Audio](https://fish.audio/). Entrenado con más de 10 millones de horas de audio en aproximadamente 50 idiomas, S2 combina alineación mediante aprendizaje por refuerzo con una arquitectura Dual-Autoregresiva para generar voz natural, realista y emocionalmente rica.

S2 soporta control detallado en línea de prosodia y emoción usando etiquetas en lenguaje natural como `[laugh]`, `[whispers]` y `[super happy]`, así como generación nativa multi-hablante y multi-turno.

Visita el [sitio web de Fish Audio](https://fish.audio/) para el playground en vivo. Lee el [blog](https://fish.audio/blog/fish-audio-open-sources-s2/) y el [reporte técnico](https://github.com/fishaudio/fish-speech/blob/main/FishAudioS2TecReport.pdf) para más detalles.

### Variantes del Modelo

| Modelo | Tamaño        | Disponibilidad                                         | Descripción                                                                 |
| ------ | ------------- | ------------------------------------------------------ | --------------------------------------------------------------------------- |
| S2-Pro | 4B parámetros | [HuggingFace](https://huggingface.co/fishaudio/s2-pro) | Modelo insignia con todas las funcionalidades, máxima calidad y estabilidad |

Más detalles del modelo pueden encontrarse en el [reporte técnico](https://arxiv.org/abs/2411.01156).

## Resultados de Benchmark

| Benchmark                                   | Fish Audio S2                |
| ------------------------------------------- | ---------------------------- |
| Seed-TTS Eval — WER (Chino)                 | **0.54%** (mejor global)     |
| Seed-TTS Eval — WER (Inglés)                | **0.99%** (mejor global)     |
| Test de Turing de Audio (con instrucción)   | **0.515** media posterior    |
| EmergentTTS-Eval — Win Rate                 | **81.88%** (más alto global) |
| Fish Instruction Benchmark — TAR            | **93.3%**                    |
| Fish Instruction Benchmark — Calidad        | **4.51 / 5.0**               |
| Multilenguaje (MiniMax Testset) — Mejor WER | **11 de 24** idiomas         |
| Multilenguaje (MiniMax Testset) — Mejor SIM | **17 de 24** idiomas         |

En Seed-TTS Eval, S2 logra el menor WER entre todos los modelos evaluados incluyendo sistemas cerrados: Qwen3-TTS (0.77/1.24), MiniMax Speech-02 (0.99/1.90), Seed-TTS (1.12/2.25). En el Test de Turing de Audio, 0.515 supera a Seed-TTS (0.417) en un 24% y a MiniMax-Speech (0.387) en un 33%. En EmergentTTS-Eval, S2 obtiene resultados particularmente fuertes en paralingüística (91.61% win rate), preguntas (84.41%) y complejidad sintáctica (83.39%).

## Características Destacadas

<img src="./docs/assets/totalability.png" width=200%>

### Control Fino en Línea mediante Lenguaje Natural

S2 permite control localizado sobre la generación de voz incrustando instrucciones en lenguaje natural directamente en posiciones específicas del texto. En lugar de depender de un conjunto fijo de etiquetas, S2 acepta descripciones libres como `[whisper in small voice]`, `[professional broadcast tone]` o `[pitch up]`, permitiendo control expresivo abierto a nivel de palabra.

### Arquitectura Dual-Autoregresiva

S2 se basa en un transformer solo-decoder combinado con un códec de audio basado en RVQ (10 codebooks, ~21 Hz). La arquitectura Dual-AR divide la generación en dos etapas:

* **AR Lento** opera en el eje temporal y predice el codebook semántico principal.
* **AR Rápido** genera los 9 codebooks residuales restantes en cada paso temporal, reconstruyendo detalles acústicos finos.

Este diseño asimétrico — 4B parámetros en el eje temporal y 400M en el eje de profundidad — mantiene la inferencia eficiente sin perder fidelidad.

### Alineación con Aprendizaje por Refuerzo

S2 utiliza Group Relative Policy Optimization (GRPO) para alineación post-entrenamiento. Los mismos modelos usados para filtrar y anotar datos se reutilizan como modelos de recompensa durante RL, eliminando desajustes de distribución. La señal de recompensa combina precisión semántica, cumplimiento de instrucciones, preferencia acústica y similitud de timbre.

### Streaming en Producción con SGLang

Gracias a que la arquitectura Dual-AR es estructuralmente similar a LLMs autoregresivos, S2 hereda optimizaciones como batching continuo, caché KV paginado, CUDA graph replay y RadixAttention.

En una sola GPU NVIDIA H200:

* **RTF:** 0.195
* **Tiempo hasta primer audio:** ~100 ms
* **Throughput:** 3,000+ tokens acústicos/s manteniendo RTF < 0.5

### Soporte Multilenguaje

S2 soporta texto a voz multilenguaje sin necesidad de fonemas o preprocesamiento específico.

**Inglés, Chino, Japonés, Coreano, Árabe, Alemán, Francés...**

**¡Y MÁS!**

La lista sigue creciendo, revisa [Fish Audio](https://fish.audio/) para novedades.

### Generación Multi-Hablante Nativa

<img src="./docs/assets/chattemplate.png" width=200%>

Fish Audio S2 permite subir audio de referencia con múltiples hablantes, el modelo gestiona cada uno mediante tokens `<|speaker:i|>`. Así puedes generar múltiples voces en una sola ejecución sin subir audios separados.

### Generación Multi-Turno

Gracias al contexto ampliado, el modelo puede usar información previa para mejorar la expresividad del contenido generado, aumentando la naturalidad.

### Clonación de Voz Rápida

S2 permite clonar voces con una muestra corta (10–30 segundos), capturando timbre, estilo y emoción sin necesidad de fine-tuning adicional.
Consulta [SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md) para usar el servidor SGLang.

---

## Créditos

* [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
* [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
* [GPT VITS](https://github.com/innnky/gpt-vits)
* [MQTTS](https://github.com/b04901014/MQTTS)
* [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
* [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
* [Qwen3](https://github.com/QwenLM/Qwen3)

## Reporte Técnico

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
