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
    <a target="_blank" href="https://huggingface.co/spaces/fishaudio/fish-speech-1">
        <img alt="Huggingface" src="https://img.shields.io/badge/🤗%20-space%20demo-yellow"/>
    </a>
    <a target="_blank" href="https://huggingface.co/fishaudio/openaudio-s1-mini">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
</div>

> [!IMPORTANT]
> **Aviso de Licença**  
> Esta base de código é lançada sob a **Licença Apache** e todos os pesos dos modelos são lançados sob a **Licença CC-BY-NC-SA-4.0**. Consulte [LICENSE](../LICENSE) para mais detalhes.

> [!WARNING]
> **Isenção de Responsabilidade Legal**  
> Não assumimos qualquer responsabilidade pelo uso ilegal da base de código. Consulte as leis locais sobre DMCA e outras leis relacionadas.

## Comece Aqui

Aqui estão os documentos oficiais do Fish Speech, siga as instruções para começar facilmente.

- [Instalação](https://speech.fish.audio/install/)
- [Fine-tune](https://speech.fish.audio/finetune/)
- [Inferência](https://speech.fish.audio/inference/)
- [Amostras](https://speech.fish.audio/examples)

## 🎉 Anúncio

Estamos animados em anunciar que mudamos nossa marca para **OpenAudio** — introduzindo uma nova série revolucionária de modelos avançados de Text-to-Speech que se baseia na fundação do Fish-Speech.

Temos o orgulho de lançar o **OpenAudio-S1** como o primeiro modelo desta série, oferecendo melhorias significativas em qualidade, desempenho e capacidades.

O OpenAudio-S1 vem em duas versões: **OpenAudio-S1** e **OpenAudio-S1-mini**. Ambos os modelos estão agora disponíveis no [Fish Audio Playground](https://fish.audio) (para **OpenAudio-S1**) e [Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini) (para **OpenAudio-S1-mini**).

Visite o [site OpenAudio](https://openaudio.com/blogs/s1) para blog e relatório técnico.

## Destaques ✨

### **Excelente qualidade TTS**

Usamos as métricas de avaliação Seed TTS para avaliar o desempenho do modelo, e os resultados mostram que o OpenAudio S1 alcança **0.008 WER** e **0.004 CER** em texto em inglês, que é significativamente melhor que modelos anteriores. (Inglês, avaliação automática, baseada no OpenAI gpt-4o-transcribe, distância do locutor usando Revai/pyannote-wespeaker-voxceleb-resnet34-LM)

| Modelo | Taxa de Erro de Palavra (WER) | Taxa de Erro de Caractere (CER) | Distância do Locutor |
|-------|----------------------|---------------------------|------------------|
| **S1** | **0.008**  | **0.004**  | **0.332** |
| **S1-mini** | **0.011** | **0.005** | **0.380** |

### **Melhor Modelo no TTS-Arena2** 🏆

O OpenAudio S1 alcançou a **classificação #1** no [TTS-Arena2](https://arena.speechcolab.org/), o benchmark para avaliação de text-to-speech:

<div align="center">
    <img src="assets/Elo.jpg" alt="Classificação TTS-Arena2" style="width: 75%;" />
</div>

### **Controle de Fala**

O OpenAudio S1 **suporta uma variedade de marcadores emocionais, de tom e especiais** para aprimorar a síntese de fala:

- **Emoções básicas**:
```
(raivoso) (triste) (animado) (surpreso) (satisfeito) (encantado) 
(assustado) (preocupado) (chateado) (nervoso) (frustrado) (deprimido)
(empático) (envergonhado) (enojado) (emocionado) (orgulhoso) (relaxado)
(grato) (confiante) (interessado) (curioso) (confuso) (alegre)
```

- **Emoções avançadas**:
```
(desdenhoso) (infeliz) (ansioso) (histérico) (indiferente) 
(impaciente) (culpado) (desprezível) (em pânico) (furioso) (relutante)
(entusiasmado) (desaprovador) (negativo) (negando) (espantado) (sério)
(sarcástico) (conciliador) (consolador) (sincero) (escarnecedor)
(hesitante) (cedendo) (doloroso) (constrangido) (divertido)
```

- **Marcadores de tom**:
```
(tom apressado) (gritando) (gritando alto) (sussurrando) (tom suave)
```

- **Efeitos de áudio especiais**:
```
(rindo) (dando risinhos) (soluçando) (chorando alto) (suspirando) (ofegando)
(gemendo) (risos da multidão) (risos de fundo) (risos da audiência)
```

Você também pode usar Ha,ha,ha para controlar, há muitos outros casos esperando para serem explorados por você mesmo.

(Suporte para inglês, chinês e japonês agora, e mais idiomas em breve!)

### **Dois Tipos de Modelos**

| Modelo | Tamanho | Disponibilidade | Recursos |
|-------|------|--------------|----------|
| **S1** | 4B parâmetros | Disponível em [fish.audio](https://fish.audio/) | Modelo flagship com recursos completos |
| **S1-mini** | 0.5B parâmetros | Disponível no Hugging Face [hf space](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini) | Versão destilada com capacidades principais |

Tanto S1 quanto S1-mini incorporam Aprendizado por Reforço online com Feedback Humano (RLHF).

## **Recursos**

1. **TTS Zero-shot e Few-shot:** Insira uma amostra vocal de 10 a 30 segundos para gerar saída TTS de alta qualidade. **Para diretrizes detalhadas, veja [Melhores Práticas de Clonagem de Voz](https://docs.fish.audio/resources/best-practices/voice-cloning).**

2. **Suporte Multilíngue e Cross-lingual:** Simplesmente copie e cole texto multilíngue na caixa de entrada—não precisa se preocupar com o idioma. Atualmente suporta inglês, japonês, coreano, chinês, francês, alemão, árabe e espanhol.

3. **Sem Dependência de Fonema:** O modelo tem fortes capacidades de generalização e não depende de fonemas para TTS. Pode lidar com texto em qualquer script de idioma.

4. **Altamente Preciso:** Alcança um baixo CER (Taxa de Erro de Caractere) de cerca de 0.4% e WER (Taxa de Erro de Palavra) de cerca de 0.8% para Seed-TTS Eval.

5. **Rápido:** Com aceleração por torch compile, o fator de tempo real é aproximadamente 1:7 em uma GPU Nvidia RTX 4090.

6. **Inferência via WebUI:** Apresenta uma interface de usuário baseada em Gradio, fácil de usar e compatível com Chrome, Firefox, Edge e outros navegadores.

7. **Amigável para Implantação:** Configure facilmente um servidor de inferência com suporte nativo para Linux e Windows (suporte para macOS em breve), minimizando a perda de desempenho.

## **Mídia e Demos**

<div align="center">

### **Mídia Social**
<a href="https://x.com/FishAudio/status/1929915992299450398" target="_blank">
    <img src="https://img.shields.io/badge/𝕏-Latest_Demo-black?style=for-the-badge&logo=x&logoColor=white" alt="Demo Mais Recente no X" />
</a>

### **Demos Interativos**
<a href="https://fish.audio" target="_blank">
    <img src="https://img.shields.io/badge/Fish_Audio-Try_OpenAudio_S1-blue?style=for-the-badge" alt="Experimente OpenAudio S1" />
</a>
<a href="https://huggingface.co/spaces/fishaudio/openaudio-s1-mini" target="_blank">
    <img src="https://img.shields.io/badge/Hugging_Face-Try_S1_Mini-yellow?style=for-the-badge" alt="Experimente S1 Mini" />
</a>

### **Vitrines de Vídeo**

<a href="https://www.youtube.com/watch?v=SYuPvd7m06A" target="_blank">
    <img src="assets/Thumbnail.jpg" alt="OpenAudio S1 Video" style="width: 50%;" />
</a>

</div>

---

## Créditos

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

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
