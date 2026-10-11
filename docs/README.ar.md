<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | [简体中文](README.zh.md) | [Portuguese](README.pt-BR.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | **العربية** | [Español](README.es.md)  <br>

<a href="https://www.producthunt.com/products/fish-speech?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-fish&#0045;audio&#0045;s1" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1023740&theme=light&period=daily&t=1761164814710" alt="Fish&#0032;Audio&#0032;S1 - Expressive&#0032;Voice&#0032;Cloning&#0032;and&#0032;Text&#0045;to&#0045;Speech | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
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
    <a target="_blank" href="https://huggingface.co/fishaudio/drama-3">
        <img alt="HuggingFace Drama 3" src="https://img.shields.io/badge/🤗%20-Drama%203-orange"/>
    </a>
    <a target="_blank" href="https://huggingface.co/fishaudio/s2-pro">
        <img alt="HuggingFace S2-Pro" src="https://img.shields.io/badge/🤗%20-S2--Pro-orange"/>
    </a>
    <a target="_blank" href="https://fish.audio/blog/fish-audio-open-sources-s2/">
        <img alt="Fish Audio Blog" src="https://img.shields.io/badge/Blog-Fish_Audio_S2-1f7a8c?style=flat-square&logo=readme&logoColor=white"/>
    </a>
    <a target="_blank" href="https://arxiv.org/abs/2603.08823">
        <img alt="Paper | Technical Report" src="https://img.shields.io/badge/Paper-Technical_Report-b31b1b?style=flat-square"/>
    </a>
</div>

> [!IMPORTANT]
> **إشعار الترخيص**
> يتم إصدار قاعدة الأكواد هذه وأوزان النماذج المرتبطة بها تحت **[FISH AUDIO RESEARCH LICENSE](../LICENSE)**. يرجى الرجوع إلى ملف [LICENSE](../LICENSE) لمزيد من التفاصيل.


> [!WARNING]
> **إخلاء المسؤولية القانونية**
> نحن لا نتحمل أي مسؤولية عن أي استخدام غير قانوني لقاعدة الأكواد. يرجى الرجوع إلى القوانين المحلية المتعلقة بـ DMCA والقوانين الأخرى ذات الصلة.

## البداية السريعة

### الوثائق

- [التثبيت](https://speech.fish.audio/ar/install/): إعداد البيئة، تحميل الأوزان، النشر باستخدام Docker
- [الاستنتاج](https://speech.fish.audio/ar/inference/): سطر الأوامر، خادم API، واجهة WebUI

> [!IMPORTANT]
> **نوصي بشدة من يبحث عن الكفاءة باستخدام SGLang-Omni؛ راجع [SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md).**

### لوكيل LLM

```
اقرأ https://speech.fish.audio/ar/install/ أولاً، واتبع الوثائق لتثبيت وإعداد Fish Audio Drama 3 (أو S2-Pro).
```

## النماذج

| النموذج | الحجم | الأوزان | الوصف |
|------|------|------|------|
| **Drama 3** | 35B-A3B (يُنشَّط منها 3B لكل رمز) | [HuggingFace](https://huggingface.co/fishaudio/drama-3) | النموذج الرائد من الجيل الجديد، مخصص للمشاهد متعددة الشخصيات والمحتوى الطويل، مع المراجعة متعددة الأدوار والتحرير الموضعي |
| S2-Pro | 4B معامل | [HuggingFace](https://huggingface.co/fishaudio/s2-pro) | النموذج الرائد من الجيل السابق، باستنتاج أسرع |

يُستنتج نوع النموذج من ملف `config.json` في مجلد الأوزان، ويتشارك النموذجان في نفس سطر الأوامر وخادم API.

---

## Fish Audio Drama 3

**نموذج صوتي من الجيل الجديد، يرتقي عن S2-Pro من جميع الجوانب.**

يدعم Drama 3، كما في سلسلة S2، تعدد المتحدثين واستنساخ الصوت والتحكم باللغة الطبيعية، لكنه يحقق قفزة واضحة في التعامل مع الصوت الطويل وفي طريقة التفاعل وقابلية التحرير.

### المميزات الرئيسية

- **المراجعة متعددة الأدوار (إخراج)**: وجّه اللقطة السابقة مباشرة ("أكثر هدوءً، وأكثر حزنًا قليلًا")، فيعيد النموذج التوليد مع أخذ هذا السياق في الاعتبار دون البدء من الصفر.
- **الإقحام (Infilling) والتحرير الدقيق**: حرّر الصوت كما تحرّر النص، فتبدّل كلمة واحدة (من "الأربعاء" إلى "الخميس") مع بقاء لون الصوت والانتقالات قبلها وبعدها طبيعية.
- **مشاهد متعددة الشخصيات**: من وصف النص المسرحي، يمكنك تسجيل مشهد كامل يضم عدة شخصيات دفعة واحدة، مع دعم التدفق للنصوص الطويلة.
- **حوارات طويلة وصوت طويل**: صُمم للتسجيلات الطويلة والجلسات الممتدة، مع الحفاظ على ثبات الشخصيات ولون الصوت في المحتوى الطويل.
- **تعبير أقوى**: الضحك والبكاء والهمس والمشاعر الدرامية لم تعد تُسطّح بسبب المرجع، فصار الأداء أقرب إلى أداء بشري حقيقي.
- **استنساخ صوتي متعدد اللغات**: يحقق مستوى جديدًا من SOTA في معدلات الخطأ (CER / WER) لاستنساخ لون الصوت عبر اللغات.
- **التحكم بوسوم اللغة الطبيعية**: وسوم داخلية مثل `[whisper]` و `[angry]` و `[laughing]` تتحكم في المشاعر والنغم.

### ما تغيّر مقارنةً بـ S2-Pro

| القدرة | S2-Pro | Drama 3 |
| --- | --- | --- |
| تعدد المتحدثين | مدعوم | مدعوم، كمشاهد متعددة الشخصيات |
| استنساخ الصوت | مدعوم | مدعوم، بمعدلات خطأ أقل عبر اللغات |
| تعليمات اللغة الطبيعية الداخلية | مدعومة | مدعومة |
| التحرير الموضعي / الإقحام | غير مدعوم | مدعوم |
| المراجعة متعددة الأدوار | غير مدعومة | مدعومة |
| المحتوى الطويل | توليد على مقاطع | حوارات طويلة وصوت طويل، مع التدفق |
| المشاعر القوية (الضحك، البكاء، الهمس) | متوسطة | أقوى |

### النتائج

يحقق Drama 3 قفزة واضحة على S2 في المعايير العامة: الاستنساخ الصوتي متعدد اللغات (MiniMax Multilingual Testset)، وتحرير الكلام بالشكل الحر (Ming-Freeform)، والمراجعة العاطفية متعددة الأدوار (Step-Audio-Edit)، حيث يتجاوز الجيل السابق في دقة التحكم ومعدلات الخطأ متعددة اللغات (CER / WER) ودقة الأسلوب.

---

## Fish Audio S2-Pro

نموذجنا الرائد من الجيل السابق: كلام قابل للتحكم باللغة الطبيعية وغني بالمشاعر، يدعم نحو مئة لغة، وبحجم يمكن نشره على بطاقة GPU للمستهلكين.

- **تعليمات اللغة الطبيعية الداخلية**: وسوم مثل `[pause]` و `[excited]` و `[whisper]` تتحكم في النغم والمشاعر في أي موضع من النص.
- **توليد أصلي متعدد المتحدثين**: تفصل معرّفات المتحدثين بين الشخصيات داخل نص واحد، فلا حاجة إلى توليد كل متحدث ثم وصل المقاطع.
- **استنساخ صوتي سريع**: يكفي 10–30 ثانية من الصوت المرجعي لإعادة إنتاج لون الصوت.
- **توليد بدفعة واحدة**: يمكن توليد صوت طويل متدفقًا دفعة واحدة.
- **تدفق بمستوى الإنتاج**: RTF 0.195 على بطاقة H200 واحدة، وزمن نحو 100 مللي ثانية حتى أول صوت.

جرّب العرض المباشر على [موقع Fish Audio](https://fish.audio/)، واقرأ [تدوينة المدونة](https://fish.audio/blog/fish-audio-open-sources-s2/) و[التقرير التقني](https://arxiv.org/abs/2603.08823) لمزيد من التفاصيل.

<details>
<summary><b>نتائج قياس S2-Pro (انقر للتوسيع)</b></summary>

| المعيار | Fish Audio S2 |
|------|------|
| Seed-TTS Eval — WER (الصينية) | **0.54%** (الأفضل إجمالًا) |
| Seed-TTS Eval — WER (الإنجليزية) | **0.99%** (الأفضل إجمالًا) |
| Audio Turing Test (مع تعليمات) | **0.515** متوسط لاحق |
| EmergentTTS-Eval — معدل الفوز | **81.88%** (الأعلى إجمالًا) |
| Fish Instruction Benchmark — TAR | **93.3%** |
| Fish Instruction Benchmark — الجودة | **4.51 / 5.0** |
| متعدد اللغات (MiniMax Testset) — أفضل WER | **11 من 24** لغة |
| متعدد اللغات (MiniMax Testset) — أفضل SIM | **17 من 24** لغة |

في Seed-TTS Eval، حقق S2 أقل WER بين جميع النماذج المقيَّمة بما فيها الأنظمة مغلقة المصدر: Qwen3-TTS (0.77/1.24)، MiniMax Speech-02 (0.99/1.90)، Seed-TTS (1.12/2.25). وفي Audio Turing Test، يتفوق 0.515 على Seed-TTS (0.417) بنسبة 24% وعلى MiniMax-Speech (0.387) بنسبة 33%. وفي EmergentTTS-Eval، يبرز S2 خصوصًا في علم اللغة الموازي (معدل فوز 91.61%) والأسئلة (84.41%) والتعقيد النحوي (83.39%).

<img src="./assets/totalability.png" width=200%>

</details>

## شكر وتقدير

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

## التقرير الفني

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
