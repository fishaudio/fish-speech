# OpenAudio (anteriormente Fish-Speech)

<div align="center">

<div align="center">

<img src="../assets/openaudio.jpg" alt="OpenAudio" style="display: block; margin: 0 auto; width: 35%;"/>

</div>

<strong>Série Avançada de Modelos Text-to-Speech</strong>

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

<strong>Experimente agora:</strong> <a href="https://fish.audio">Fish Audio Playground</a> | <strong>Saiba mais:</strong> <a href="https://openaudio.com">Site OpenAudio</a>

</div>

---

!!! note "Aviso de Licença"
    Esta base de código é lançada sob a **Licença Apache** e todos os pesos dos modelos são lançados sob a **Licença CC-BY-NC-SA-4.0**. Consulte a [LICENÇA DO CÓDIGO](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) e a [LICENÇA DO MODELO](https://spdx.org/licenses/CC-BY-NC-SA-4.0) para mais detalhes.

!!! warning "Aviso Legal"
    Não assumimos nenhuma responsabilidade pelo uso ilegal da base de código. Consulte as leis locais sobre DMCA e outras leis relevantes.

## **Introdução**

Estamos empolgados em anunciar que mudamos nossa marca para **OpenAudio** - introduzindo uma nova série de modelos avançados de Text-to-Speech que se baseia na fundação do Fish-Speech com melhorias significativas e novas capacidades.

**OpenAudio-S1-mini**: [Blog](https://openaudio.com/blogs/s1); [Vídeo](https://www.youtube.com/watch?v=SYuPvd7m06A); [Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini);

**Fish-Speech v1.5**: [Vídeo](https://www.bilibili.com/video/BV1EKiDYBE4o/); [Hugging Face](https://huggingface.co/fishaudio/fish-speech-1.5);

## **Destaques**

### **Qualidade TTS Excelente**

Utilizamos as métricas Seed TTS Eval para avaliar o desempenho do modelo, e os resultados mostram que o OpenAudio S1 alcança **0.008 WER** e **0.004 CER** em texto inglês, que é significativamente melhor que modelos anteriores. (Inglês, avaliação automática, baseada na transcrição OpenAI gpt-4o, distância do falante usando Revai/pyannote-wespeaker-voxceleb-resnet34-LM)

| Modelo | Taxa de Erro de Palavras (WER) | Taxa de Erro de Caracteres (CER) | Distância do Falante |
|:-----:|:--------------------:|:-------------------------:|:----------------:|
| **S1** | **0.008** | **0.004** | **0.332** |
| **S1-mini** | **0.011** | **0.005** | **0.380** |

### **Melhor Modelo no TTS-Arena2**

OpenAudio S1 alcançou a **classificação #1** no [TTS-Arena2](https://arena.speechcolab.org/), o benchmark para avaliação de text-to-speech:

<div align="center">
    <img src="assets/Elo.jpg" alt="TTS-Arena2 Ranking" style="width: 75%;" />
</div>

### **Controle de Fala**
OpenAudio S1 **suporta uma variedade de marcadores emocionais, de tom e especiais** para aprimorar a síntese de fala:

- **Emoções básicas**:
```
(raivoso) (triste) (animado) (surpreso) (satisfeito) (encantado) 
(com medo) (preocupado) (chateado) (nervoso) (frustrado) (deprimido)
(empático) (envergonhado) (nojento) (comovido) (orgulhoso) (relaxado)
(grato) (confiante) (interessado) (curioso) (confuso) (alegre)
```

- **Emoções avançadas**:
```
(desdenhoso) (infeliz) (ansioso) (histérico) (indiferente) 
(impaciente) (culpado) (desprezível) (em pânico) (furioso) (relutante)
(entusiasmado) (desaprovador) (negativo) (negando) (espantado) (sério)
(sarcástico) (conciliador) (consolador) (sincero) (zombeteiro)
(hesitante) (cedendo) (doloroso) (constrangido) (divertido)
```

(Suporte para inglês, chinês e japonês agora, e mais idiomas em breve!)

- **Marcadores de tom**:
```
(em tom de pressa) (gritando) (berrando) (sussurrando) (tom suave)
```

- **Efeitos sonoros especiais**:
```
(rindo) (gargalhando) (soluçando) (chorando alto) (suspirando) (ofegante)
(gemendo) (risada da multidão) (risada de fundo) (risada da plateia)
```

Você também pode usar Ha,ha,ha para controlar, há muitos outros casos esperando para serem explorados por você mesmo.

### **Dois Tipos de Modelos**

Oferecemos duas variantes de modelo para atender diferentes necessidades:

- **OpenAudio S1 (4B parâmetros)**: Nosso modelo principal com todas as funcionalidades disponível em [fish.audio](https://fish.audio), oferecendo a mais alta qualidade de síntese de fala com todas as características avançadas.

- **OpenAudio S1-mini (0.5B parâmetros)**: Uma versão destilada com capacidades principais, disponível no [Hugging Face Space](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini), otimizada para inferência mais rápida mantendo excelente qualidade.

Tanto o S1 quanto o S1-mini incorporam Aprendizado por Reforço Online com Feedback Humano (RLHF).

## **Características**

1. **TTS Zero-shot e Few-shot:** Insira uma amostra vocal de 10 a 30 segundos para gerar saída TTS de alta qualidade. **Para diretrizes detalhadas, veja [Melhores Práticas de Clonagem de Voz](https://docs.fish.audio/text-to-speech/voice-clone-best-practices).**

2. **Suporte Multilíngue e Cross-lingual:** Simplesmente copie e cole texto multilíngue na caixa de entrada—não precisa se preocupar com o idioma. Atualmente suporta inglês, japonês, coreano, chinês, francês, alemão, árabe e espanhol.

3. **Sem Dependência de Fonemas:** O modelo tem fortes capacidades de generalização e não depende de fonemas para TTS. Pode lidar com texto em qualquer script de idioma.

4. **Altamente Preciso:** Alcança uma baixa Taxa de Erro de Caracteres (CER) de cerca de 0,4% e Taxa de Erro de Palavras (WER) de cerca de 0,8% para Seed-TTS Eval.

5. **Rápido:** Com aceleração fish-tech, o fator de tempo real é aproximadamente 1:5 em um laptop Nvidia RTX 4060 e 1:15 em um Nvidia RTX 4090.

6. **Inferência WebUI:** Apresenta uma interface web fácil de usar baseada em Gradio, compatível com Chrome, Firefox, Edge e outros navegadores.

7. **Inferência GUI:** Oferece uma interface gráfica PyQt6 que funciona perfeitamente com o servidor API. Suporta Linux, Windows e macOS. [Ver GUI](https://github.com/AnyaCoder/fish-speech-gui).

8. **Amigável para Deploy:** Configure facilmente um servidor de inferência com suporte nativo para Linux, Windows (MacOS em breve), minimizando a perda de velocidade.

## **Mídia e Demos**

<!-- <div align="center"> -->

<h3><strong>Mídia Social</strong></h3>
<a href="https://x.com/FishAudio/status/1929915992299450398" target="_blank">
    <img src="https://img.shields.io/badge/𝕏-Demo_Mais_Recente-black?style=for-the-badge&logo=x&logoColor=white" alt="Latest Demo on X" />
</a>

<h3><strong>Demos Interativos</strong></h3>

<a href="https://fish.audio" target="_blank">
    <img src="https://img.shields.io/badge/Fish_Audio-Experimente_OpenAudio_S1-blue?style=for-the-badge" alt="Try OpenAudio S1" />
</a>
<a href="https://huggingface.co/spaces/fishaudio/openaudio-s1-mini" target="_blank">
    <img src="https://img.shields.io/badge/Hugging_Face-Experimente_S1_Mini-yellow?style=for-the-badge" alt="Try S1 Mini" />
</a>

<h3><strong>Showcases em Vídeo</strong></h3>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/SYuPvd7m06A" title="OpenAudio S1 Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## **Documentação**

### Início Rápido
- [Configurar Ambiente](install.md) - Configure seu ambiente de desenvolvimento
- [Guia de Inferência](inference.md) - Execute o modelo e gere fala

## **Comunidade e Suporte**

- **Discord:** Junte-se à nossa [comunidade Discord](https://discord.gg/Es5qTB9BcN)
- **Site:** Visite [OpenAudio.com](https://openaudio.com) para as últimas atualizações
- **Experimente Online:** [Fish Audio Playground](https://fish.audio)

## Modelos

O OpenAudio S1 é o primeiro modelo da série OpenAudio. É um vocoder VQ-GAN de descodificador duplo que pode reconstruir áudio a partir de códigos VQ.
