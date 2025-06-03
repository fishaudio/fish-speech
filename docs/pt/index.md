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

!!! warning "Aviso Legal"
    Não assumimos nenhuma responsabilidade pelo uso ilegal da base de código. Consulte as leis locais sobre DMCA (Digital Millennium Copyright Act) e outras leis relevantes em sua área.
    
    **Licença:** Esta base de código é lançada sob a licença Apache 2.0 e todos os modelos são lançados sob a licença CC-BY-NC-SA-4.0.

## **Introdução**

Estamos empolgados em anunciar que mudamos nossa marca para **OpenAudio** - introduzindo uma nova série de modelos avançados de Text-to-Speech que se baseia na fundação do Fish-Speech com melhorias significativas e novas capacidades.

**Openaudio-S1-mini**: [Vídeo](A ser carregado); [Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini);

**Fish-Speech v1.5**: [Vídeo](https://www.bilibili.com/video/BV1EKiDYBE4o/); [Hugging Face](https://huggingface.co/fishaudio/fish-speech-1.5);

## **Destaques** ✨

### **Controle Emocional**
O OpenAudio S1 **suporta uma variedade de marcadores emocionais, de tom e especiais** para aprimorar a síntese de fala:

- **Emoções básicas**:
```
(angry) (sad) (excited) (surprised) (satisfied) (delighted)
(scared) (worried) (upset) (nervous) (frustrated) (depressed)
(empathetic) (embarrassed) (disgusted) (moved) (proud) (relaxed)
(grateful) (confident) (interested) (curious) (confused) (joyful)
```

- **Emoções avançadas**:
```
(disdainful) (unhappy) (anxious) (hysterical) (indifferent) 
(impatient) (guilty) (scornful) (panicked) (furious) (reluctant)
(keen) (disapproving) (negative) (denying) (astonished) (serious)
(sarcastic) (conciliative) (comforting) (sincere) (sneering)
(hesitating) (yielding) (painful) (awkward) (amused)
```

- **Marcadores de tom**:
```
(in a hurry tone) (shouting) (screaming) (whispering) (soft tone)
```

- **Efeitos sonoros especiais**:
```
(laughing) (chuckling) (sobbing) (crying loudly) (sighing) (panting)
(groaning) (crowd laughing) (background laughter) (audience laughing)
```

Você também pode usar Ha,ha,ha para controlar, há muitos outros casos esperando para serem explorados por você mesmo.

### **Qualidade TTS Excelente**

Utilizamos as métricas Seed TTS Eval para avaliar o desempenho do modelo, e os resultados mostram que o OpenAudio S1 alcança **0.008 WER** e **0.004 CER** em texto inglês, que é significativamente melhor que modelos anteriores. (Inglês, avaliação automática, baseada na transcrição OpenAI gpt-4o, distância do falante usando Revai/pyannote-wespeaker-voxceleb-resnet34-LM)

| Modelo | Taxa de Erro de Palavras (WER) | Taxa de Erro de Caracteres (CER) | Distância do Falante |
|-------|----------------------|---------------------------|------------------|
| **S1** | **0.008**  | **0.004**  | **0.332** |
| **S1-mini** | **0.011** | **0.005** | **0.380** |

### **Dois Tipos de Modelos**

| Modelo | Tamanho | Disponibilidade | Características |
|-------|------|--------------|----------|
| **S1** | 4B parâmetros | Disponível em [fish.audio](fish.audio) | Modelo principal com todas as funcionalidades |
| **S1-mini** | 0.5B parâmetros | Disponível no huggingface [hf space](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini) | Versão destilada com capacidades principais |

Tanto o S1 quanto o S1-mini incorporam Aprendizado por Reforço Online com Feedback Humano (RLHF).

## **Características**

1. **TTS Zero-shot e Few-shot:** Insira uma amostra vocal de 10 a 30 segundos para gerar saída TTS de alta qualidade. **Para diretrizes detalhadas, veja [Melhores Práticas de Clonagem de Voz](https://docs.fish.audio/text-to-speech/voice-clone-best-practices).**

2. **Suporte Multilíngue e Cross-lingual:** Simplesmente copie e cole texto multilíngue na caixa de entrada—não precisa se preocupar com o idioma. Atualmente suporta inglês, japonês, coreano, chinês, francês, alemão, árabe e espanhol.

3. **Sem Dependência de Fonemas:** O modelo tem fortes capacidades de generalização e não depende de fonemas para TTS. Pode lidar com texto em qualquer script de idioma.

4. **Altamente Preciso:** Alcança uma baixa Taxa de Erro de Caracteres (CER) de cerca de 0,4% e Taxa de Erro de Palavras (WER) de cerca de 0,8% para Seed-TTS Eval.

5. **Rápido:** Com aceleração fish-tech, o fator de tempo real é aproximadamente 1:5 em um laptop Nvidia RTX 4060 e 1:15 em um Nvidia RTX 4090.

6. **Inferência WebUI:** Apresenta uma interface web fácil de usar baseada em Gradio, compatível com Chrome, Firefox, Edge e outros navegadores.

7. **Inferência GUI:** Oferece uma interface gráfica PyQt6 que funciona perfeitamente com o servidor API. Suporta Linux, Windows e macOS. [Ver GUI](https://github.com/AnyaCoder/fish-speech-gui).

8. **Amigável para Deploy:** Configure facilmente um servidor de inferência com suporte nativo para Linux, Windows e MacOS, minimizando a perda de velocidade.

## **Isenção de Responsabilidade**

Não assumimos nenhuma responsabilidade pelo uso ilegal da base de código. Consulte suas leis locais sobre DMCA e outras leis relacionadas.

## **Mídia e Demos**

#### 🚧 Em Breve
Demonstrações em vídeo e tutoriais estão atualmente em desenvolvimento.

## **Documentação**

### Início Rápido
- [Configurar Ambiente](install.md) - Configure seu ambiente de desenvolvimento
- [Guia de Inferência](inference.md) - Execute o modelo e gere fala

## **Comunidade e Suporte**

- **Discord:** Junte-se à nossa [comunidade Discord](https://discord.gg/Es5qTB9BcN)
- **Site:** Visite [OpenAudio.com](https://openaudio.com) para as últimas atualizações
- **Experimente Online:** [Fish Audio Playground](https://fish.audio)
