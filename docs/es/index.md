# OpenAudio (anteriormente Fish-Speech)

<div align="center">

<div align="center">

<img src="../assets/openaudio.jpg" alt="OpenAudio" style="display: block; margin: 0 auto; width: 35%;"/>

</div>

<strong>Serie avanzada de modelos de Texto a Voz (TTS)</strong>

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

<strong>Pruébalo ahora:</strong> <a href="https://fish.audio">Fish Audio Playground</a> | <strong>Más información:</strong> <a href="https://openaudio.com">Sitio web de OpenAudio</a>

</div>

---

!!! note "Aviso de Licencia"
    Este repositorio se publica bajo la**Licencia Apache** y todos los pesos de los modelos se distribuyen bajo la Licencia **CC-BY-NC-SA-4.0**. Consulta [LICENSE](../LICENSE) para más detalles.

!!! warning "Aviso legal"
    No asumimos responsabilidad alguna por el uso ilegal de este repositorio. Consulta la legislación local sobre DMCA y otras leyes relacionadas.

## **Introducción**

Nos complace anunciar que ahora somos **OpenAudio**, una nueva serie de modelos avanzados de texto a voz, basada en Fish-Speech y mejorada con nuevas capacidades.

**Openaudio-S1-mini**: [Blog](https://openaudio.com/blogs/s1); [Video](https://www.youtube.com/watch?v=SYuPvd7m06A); [Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini);

**Fish-Speech v1.5**: [Video](https://www.bilibili.com/video/BV1EKiDYBE4o/); [Hugging Face](https://huggingface.co/fishaudio/fish-speech-1.5);

## **Puntos destacados**

### **Calidad TTS sobresaliente**

Usamos métricas de evaluación Seed TTS para medir el rendimiento del modelo. OpenAudio S1 logra **0.008 WER** y **0.004 CER** en texto en inglés, superando notablemente a versiones anteriores. (Inglés, evaluación automática basada en OpenAI gpt-4o-transcribe, distancia de locutor usando Revai/pyannote-wespeaker-voxceleb-resnet34-LM)

| Modelo | Tasa de Error de Palabra (WER) | Tasa de Error de Caracter (CER) | Distancia de Locutor
 |
|:-----:|:--------------------:|:-------------------------:|:----------------:|
| **S1** | **0.008** | **0.004** | **0.332** |
| **S1-mini** | **0.011** | **0.005** | **0.380** |

### **Mejor Modelo en TTS-Arena2**

OpenAudio S1 ocupa el **puesto #1** en [TTS-Arena2](https://arena.speechcolab.org/), el referente para evaluación de TTS:

<div align="center">
    <img src="../assets/Elo.jpg" alt="TTS-Arena2 Ranking" style="width: 75%;" />
</div>

### **Control de Voz**
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

(Soporte disponible para inglés, chino y japonés; próximamente más idiomas)

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

### **Dos tipos de modelos**

Ofrecemos dos variantes para distintas necesidades:

- **OpenAudio S1 (4B parámetros)**: Modelo insignia con funciones completas, disponible en [fish.audio](https://fish.audio), proporciona la mejor calidad de síntesis con todas las funciones avanzadas.

- **OpenAudio S1-mini (0.5B parámetros)**: Versión destilada con capacidades clave, disponible en [Hugging Face Space](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini), optimizada para inferencia más rápida manteniendo excelente calidad.

Ambos modelos incluyen Aprendizaje por Refuerzo con Retroalimentación Humana (RLHF).

## **Características**

1. **TTS zero-shot y few-shot:** Ingresa una muestra de voz de 10 a 30 segundos para generar salida TTS de alta calidad. **Consulta [Buenas prácticas para clonación de voz](https://docs.fish.audio/text-to-speech/voice-clone-best-practices)**.

2. **Soporte multilingüe y entre idiomas:** Pega texto en varios idiomas en la caja de entrada sin preocuparte por el idioma. Actualmente admite inglés, japonés, coreano, chino, francés, alemán, árabe y español.

3. **Sin dependencia de fonemas:** El modelo generaliza bien sin necesidad de fonemas, manejando texto en cualquier escritura.

4. **Alta precisión:** CER (Tasa de Error de Caracteres) ~0.4% y WER (Tasa de Error de Palabras) ~0.8% en evaluaciones Seed-TTS.

5. **Rápido:** Acelerado con *torch compile*, con un factor de tiempo real de ~1:7 en una GPU Nvidia RTX 4090.

6. **Inferencia WebUI:** Interfaz web Gradio fácil de usar, compatible con Chrome, Firefox, Edge y otros navegadores.

7. **Inferencia GUI:** Interfaz gráfica PyQt6 que funciona con el servidor API. Compatible con Linux, Windows y macOS ([ver GUI](https://github.com/AnyaCoder/fish-speech-gui)).

8. **Amigable para despliegue:** Servidor de inferencia fácil de instalar, soporte nativo para Linux y Windows (pronto en macOS) con mínima pérdida de velocidad.

## **Media & Demos**

<!-- <div align="center"> -->

<h3><strong>Redes Sociales</strong></h3>
<a href="https://x.com/FishAudio/status/1929915992299450398" target="_blank">
    <img src="https://img.shields.io/badge/𝕏-Latest_Demo-black?style=for-the-badge&logo=x&logoColor=white" alt="Último demo en X" />
</a>

<h3><strong>Demos Interactivas</strong></h3>

<a href="https://fish.audio" target="_blank">
    <img src="https://img.shields.io/badge/Fish_Audio-Try_OpenAudio_S1-blue?style=for-the-badge" alt="Pruébalo en Fish Audio" />
</a>
<a href="https://huggingface.co/spaces/fishaudio/openaudio-s1-mini" target="_blank">
    <img src="https://img.shields.io/badge/Hugging_Face-Try_S1_Mini-yellow?style=for-the-badge" alt="Pruébalo en Hugging Face" />
</a>

<h3><strong>Demostración en Video</strong></h3>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/SYuPvd7m06A" title="Video de OpenAudio S1" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## **Documentación**

### Inicio rápido
- [Configurar entorno](install.md) - Prepara tu entorno de desarrollo
- [Guía de inferencia](inference.md) - Ejecuta el modelo y genera voz


## **Comunidad & Soporte**

- **Discord:** Únete a nuestra [comunidad en Discord](https://discord.gg/Es5qTB9BcN)
- **Sitio web:** Visita [OpenAudio.com](https://openaudio.com) para noticias y actualizaciones.
- **Prueba online:** [Fish Audio Playground](https://fish.audio)
