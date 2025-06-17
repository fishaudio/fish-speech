<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | [简体中文](docs/README.zh.md) | [Portuguese](docs/README.pt-BR.md) | [日本語](docs/README.ja.md) | [한국어](docs/README.ko.md) | **Español** <br>

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
    <a target="_blank" href="https://huggingface.co/fishaudio/openaudio-s1-mini">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
</div>

> [!IMPORTANT]
> **Aviso de Licencia**
> Este repositorio se publica bajo la **Licencia Apache**, mientras que todos los pesos de los modelos se distribuyen bajo la **Licencia CC-BY-NC-SA-4.0**. Consulta [LICENSE](../LICENSE) para más detalles.

> [!WARNING]
> **Aviso legal**
> No asumimos responsabilidad alguna por el uso ilegal de este repositorio. Consulta la legislación local sobre DMCA y leyes relacionadas.

---

## 🎉 Anuncio

Nos complace anunciar que hemos pasado a llamarnos **OpenAudio**, presentando una nueva serie revolucionaria de modelos avanzados de texto a voz (TTS) basada en Fish-Speech.

Estamos orgullosos de lanzar **OpenAudio-S1** como el primer modelo de esta serie, que ofrece mejoras significativas en calidad, rendimiento y capacidades.

OpenAudio-S1 está disponible en dos versiones: **OpenAudio-S1** y **OpenAudio-S1-mini**. Ambos modelos están disponibles en [Fish Audio Playground](https://fish.audio) (**OpenAudio-S1**) y en [Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini) (**OpenAudio-S1-mini**).

Visita el [sitio web de OpenAudio](https://openaudio.com/blogs/s1) para acceder a blogs y reportes técnicos.

## ✨ Puntos destacados

### **Calidad TTS sobresaliente**

Usamos métricas de evaluación Seed TTS para medir el rendimiento de los modelos. OpenAudio S1 logra un **0.008 WER** (Tasa de Error de Palabras) y un **0.004 CER** (Tasa de Error de Caracteres) en texto en inglés, superando ampliamente a modelos anteriores.

| Modelo | Tasa de Error de Palabras (WER) | Tasa de Error de Caracteres (CER) | Distancia de locutor |
|-------|----------------------|---------------------------|------------------|
| **S1** | **0.008**  | **0.004**  | **0.332** |
| **S1-mini** | **0.011** | **0.005** | **0.380** |

### **Mejor modelo en TTS-Arena2** 🏆

OpenAudio S1 ocupa el **puesto #1** en [TTS-Arena2](https://arena.speechcolab.org/), el referente de evaluación de TTS.

<div align="center">
    <img src="assets/Elo.jpg" alt="TTS-Arena2 Ranking" style="width: 75%;" />
</div>

### **Control de voz**

OpenAudio S1 **admite una amplia gama de emociones, tonos y efectos especiales** para enriquecer la síntesis de voz.

- **Emociones básicas:**
```
(enojado) (triste) (emocionado) (sorprendido) (satisfecho) (encantado)  
(asustado) (preocupado) (molesto) (nervioso) (frustrado) (deprimido)  
(empático) (avergonzado) (disgustado) (conmovido) (orgulloso) (relajado)  
(agradecido) (seguro) (interesado) (curioso) (confundido) (alegre)
```

- **Emociones avanzadas:**  
```
(desdeñoso) (infeliz) (ansioso) (histérico) (indiferente)  
(impaciente) (culpable) (despectivo) (en pánico) (furioso) (reacio)  
(entusiasta) (desaprobador) (negativo) (negando) (asombrado) (serio)  
(sarcástico) (conciliador) (consolador) (sincero) (burlón)  
(dudando) (cediendo) (dolorido) (incómodo) (divertido)
```

- **Marcadores de tono:**  
```
(tono apresurado) (gritando) (aullando) (susurrando) (tono suave)
```

- **Efectos de audio especiales:**  
```
(riendo) (riendo por lo bajo) (sollozando) (llorando fuerte) (suspirando) (jadeando)  
(gimiendo) (risa de público) (risas de fondo) (audiencia riendo)
```

También puedes usar *Ha,ha,ha* para controlar, hay muchos otros casos esperando a ser explorados por ti mismo.

(Actualmente se admite en inglés, chino y japonés. ¡Próximamente más idiomas!)

### **Dos tipos de modelos**

| Modelo | Tamaño | Disponibilidad | Características |
|--------|--------|-----------------|-----------------|
| **S1** | 4B parámetros | Disponible en [fish.audio](https://fish.audio) | Modelo insignia con funciones completas |
| **S1-mini** | 0.5B parámetros | Disponible en huggingface [hf space](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini) | Versión destilada con funciones principales |

Tanto S1 como S1-mini incluyen Aprendizaje por Refuerzo con Retroalimentación Humana (RLHF).

## **Características**

1. **TTS zero-shot y few-shot:** Ingresa una muestra de voz de 10 a 30 segundos para generar salida TTS de alta calidad. Consulta [Buenas prácticas para clonación de voz](https://docs.fish.audio/text-to-speech/voice-clone-best-practices).

2. **Soporte multilingüe y entre idiomas:** Pega texto en varios idiomas en la caja de entrada sin preocuparte por el idioma. Actualmente admite inglés, japonés, coreano, chino, francés, alemán, árabe y español.

3. **Sin dependencia de fonemas:** El modelo generaliza bien sin necesidad de fonemas, manejando texto en cualquier escritura.

4. **Alta precisión:** CER (Tasa de Error de Caracteres) ~0.4% y WER (Tasa de Error de Palabras) ~0.8% en evaluaciones Seed-TTS.

5. **Rápido:** Acelerado con *torch compile*, con un factor de tiempo real de ~1:7 en una GPU Nvidia RTX 4090.

6. **Inferencia WebUI:** Interfaz web Gradio fácil de usar, compatible con Chrome, Firefox, Edge y otros navegadores.

7. **Inferencia GUI:** Interfaz gráfica PyQt6 que funciona con el servidor API. Compatible con Linux, Windows y macOS ([ver GUI](https://github.com/AnyaCoder/fish-speech-gui)).

8. **Amigable para despliegue:** Servidor de inferencia fácil de instalar, soporte nativo para Linux y Windows (pronto en macOS) con mínima pérdida de velocidad.

## Medios & Demos

<div align="center">

### Redes sociales:
<a href="https://x.com/FishAudio/status/1929915992299450398" target="_blank">
    <img src="https://img.shields.io/badge/𝕏-Latest_Demo-black?style=for-the-badge&logo=x&logoColor=white" alt="Latest Demo on X" />
</a>

### Demos interactivas:
<a href="https://fish.audio" target="_blank">
    <img src="https://img.shields.io/badge/Fish_Audio-Try_OpenAudio_S1-blue?style=for-the-badge" alt="Try OpenAudio S1" />
</a>
<a href="https://huggingface.co/spaces/fishaudio/openaudio-s1-mini" target="_blank">
    <img src="https://img.shields.io/badge/Hugging_Face-Try_S1_Mini-yellow?style=for-the-badge" alt="Try S1 Mini" />
</a>

### Videos:

<a href="https://www.youtube.com/watch?v=SYuPvd7m06A" target="_blank">
    <img src="docs/assets/Thumbnail.jpg" alt="OpenAudio S1 Video" style="width: 50%;" />
</a>

### Muestras de audio:
<div style="margin: 20px 0;">
    <em> Próximamente en alta calidad, demostrando capacidades multilingües y emocionales.</em>
</div>

</div>

---

## Documentación

- [Configurar entorno](docs/en/install.md)
- [Inferencia](docs/en/inference.md)

## Créditos

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

## Informe técnico (V1.4)
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