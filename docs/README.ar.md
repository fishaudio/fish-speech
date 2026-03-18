<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | [简体中文](README.zh.md) | [Portuguese](README.pt-BR.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | **العربية** <br>

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
    <a target="_blank" href="https://huggingface.co/fishaudio/s2">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
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

### روابط التوثيق

هذا هو التوثيق الرسمي لـ Fish Audio S2، يرجى اتباع التعليمات للبدء بسهولة.

- [التثبيت](https://speech.fish.audio/ar/install/)
- [الاستدلال عبر خط الأوامر](https://speech.fish.audio/ar/inference/)
- [الاستدلال عبر واجهة الويب](https://speech.fish.audio/ar/inference/)
- [استدلال الخادم](https://speech.fish.audio/ar/server/)
- [نشر Docker](https://speech.fish.audio/ar/install/)

> [!IMPORTANT]
> **إذا كنت ترغب في استخدام خادم SGLang، فيرجى الرجوع إلى [SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md).**

### دليل وكيل LLM

```
يرجى قراءة https://speech.fish.audio/ar/install/ أولاً، وتثبيت وتكوين Fish Audio S2 وفقاً للوثائق.
```

## Fish Audio S2 Pro
**نظام تحويل النص إلى كلام (TTS) متعدد اللغات الرائد في الصناعة، والذي يعيد تعريف حدود توليد الصوت.**

Fish Audio S2 Pro هو أحدث طراز متعدد الوسائط تم تطويره بواسطة [Fish Audio](https://fish.audio/). تم تدريبه على أكثر من **10 ملايين ساعة** من البيانات الصوتية الهائلة، التي تغطي أكثر من **80 لغة** حول العالم. من خلال بنية **ثنائية الانحدار الذاتي (Dual-AR)** المبتكرة وتقنية توافق التعلم التعزيزي (RL)، يمكن لـ S2 Pro توليد كلام يتمتع بإحساس طبيعي وواقعي وعمق عاطفي كبير، مما يجعله رائداً في المنافسة بين الأنظمة المفتوحة والمغلقة المصدر.

تكمن القوة الضاربة لـ S2 Pro في دعمه للتحكم الدقيق للغاية في النبرة والعاطفة على مستوى **ما دون الكلمة (Sub-word Level)** من خلال وسوم اللغة الطبيعية (مثل `[whisper]` و `[excited]` و `[angry]`) ، مع دعم أصلي لتوليد متحدثين متعددين وحوارات متعددة الجولات بسياق طويل جداً.

تفضل بزيارة [موقع Fish Audio الرسمي](https://fish.audio/) الآن لتجربة العرض المباشر، أو اقرأ [تقريرنا الفني](https://arxiv.org/abs/2603.08823) و[مقال المدونة](https://fish.audio/blog/fish-audio-open-sources-s2/) للتعرف على المزيد.

### متغيرات النموذج

| النموذج | الحجم | التوفر | الوصف |
|------|------|-------------|-------------|
| S2-Pro | 4 مليار معلمة | [HuggingFace](https://huggingface.co/fishaudio/s2-pro) | النموذج الرائد كامل الميزات، مع أعلى جودة واستقرار |

لمزيد من التفاصيل حول النماذج، يرجى مراجعة [التقرير الفني](https://arxiv.org/abs/2411.01156).

## نتائج الاختبارات المرجعية (Benchmarks)

| الاختبار | Fish Audio S2 |
|------|------|
| Seed-TTS Eval — WER (الصينية) | **0.54%** (الأفضل إجمالاً) |
| Seed-TTS Eval — WER (الإنجليزية) | **0.99%** (الأفضل إجمالاً) |
| Audio Turing Test (مع التعليمات) | **0.515** متوسط خلفي (Posterior mean) |
| EmergentTTS-Eval — معدل الفوز | **81.88%** (الأعلى إجمالاً) |
| Fish Instruction Benchmark — TAR | **93.3%** |
| Fish Instruction Benchmark — الجودة | **4.51 / 5.0** |
| متعدد اللغات (MiniMax Testset) — أفضل WER | **11** لغة من أصل **24** |
| متعدد اللغات (MiniMax Testset) — أفضل SIM | **17** لغة من أصل **24** |

في تقييم Seed-TTS، حقق S2 أقل معدل خطأ في الكلمات (WER) بين جميع النماذج التي تم تقييمها (بما في ذلك الأنظمة مغلقة المصدر): Qwen3-TTS (0.77/1.24)، و MiniMax Speech-02 (0.99/1.90)، و Seed-TTS (1.12/2.25). وفي اختبار Audio Turing Test، سجل S2 قيمة 0.515 بزيادة قدرها 24% مقارنة بـ Seed-TTS (0.417) و 33% مقارنة بـ MiniMax-Speech (0.387). وفي EmergentTTS-Eval، تميز S2 بشكل خاص في أبعاد مثل اللغويات المصاحبة (معدل فوز 91.61%)، والجمل الاستفهامية (84.41%)، والتعقيد النحوي (83.39%).

## أبرز المميزات

<img src="./assets/totalability.png" width=200%>

### تحكم دقيق للغاية عبر اللغة الطبيعية

يمنح S2 Pro الصوت "روحاً" لا مثيل لها. من خلال صيغة `[tag]` البسيطة، يمكنك تضمين تعليمات عاطفية بدقة في أي موضع من النص.
- **دعم أكثر من 15,000 وسم فريد**: لا يقتصر على الإعدادات المسبقة الثابتة، بل يدعم **أوصاف النص الحر**. يمكنك تجربة `[whisper in small voice]` (همس بصوت منخفض)، أو `[professional broadcast tone]` (نبرة إذاعية احترافية)، أو `[pitch up]` (رفع طبقة الصوت).
- **مكتبة عواطف غنية**:
  `[pause]` `[emphasis]` `[laughing]` `[inhale]` `[chuckle]` `[tsk]` `[singing]` `[excited]` `[laughing tone]` `[interrupting]` `[chuckling]` `[excited tone]` `[volume up]` `[echo]` `[angry]` `[low volume]` `[sigh]` `[low voice]` `[whisper]` `[screaming]` `[shouting]` `[loud]` `[surprised]` `[short pause]` `[exhale]` `[delight]` `[panting]` `[audience laughter]` `[with strong accent]` `[volume down]` `[clearing throat]` `[sad]` `[moaning]` `[shocked]`

### بنية مبتكرة ثنائية الانحدار الذاتي (Dual-Autoregressive)

يعتمد S2 Pro بنية Dual-AR بنظام "رئيسي-تابع"، تتكون من Decoder-only Transformer وترميز صوتي RVQ (10 قواميس أكواد، بمعدل إطارات يبلغ حوالي 21 هرتز):

- **Slow AR (4 مليار معلمة)**: يعمل على طول المحور الزمني، ويتنبأ بقاموس الأكواد الدلالي الأساسي.
- **Fast AR (400 مليون معلمة)**: يولد الـ 9 قواميس المتبقية في كل خطوة زمنية، لاستعادة أدق التفاصيل الصوتية ببراعة.

يحقق هذا التصغير غير المتماثل أقصى درجات الدقة الصوتية مع زيادة سرعة الاستدلال بشكل كبير.

### توافق التعلم التعزيزي (RL Alignment)

يستخدم S2 Pro تقنية **Group Relative Policy Optimization (GRPO)** للتوافق بعد التدريب. نستخدم نفس مجموعة النماذج المستخدمة في تنظيف البيانات وتصنيفها مباشرة كنماذج مكافأة (Reward Model)، مما يحل بشكل مثالي مشكلة عدم التطابق بين توزيع بيانات ما قبل التدريب وأهداف ما بعد التدريب.
- **إشارات مكافأة متعددة الأبعاد**: تقييم شامل للدقة الدلالية، والقدرة على اتباع التعليمات، وتسجيل التفضيل الصوتي، وتماثل نبرة الصوت، لضمان أن كل ثانية من الكلام المولد تتوافق مع الحدس البشري.

### أداء استدلال تدفقي فائق (يعتمد على SGLang)

نظراً لأن بنية Dual-AR تتماثل هيكلياً مع بنية LLM القياسية، فإن S2 Pro يدعم أصلاً جميع ميزات تسريع الاستدلال في SGLang، بما في ذلك الدفعات المستمرة (Continuous Batching)، و Paged KV Cache، و CUDA Graph، والتخزين المؤقت للبادئة القائم على RadixAttention.

**أداء وحدة معالجة رسومات NVIDIA H200 واحدة:**
- **عامل الوقت الحقيقي (RTF)**: 0.195
- **تأخر الصوت الأول (TTFA)**: حوالي 100 مللي ثانية
- **إنتاجية فائقة السرعة**: تصل إلى 3000+ وسم صوتي/ثانية مع الحفاظ على RTF < 0.5

### دعم قوي للغات المتعددة

يدعم S2 Pro أكثر من 80 لغة، مما يتيح تركيباً عالياً الجودة دون الحاجة إلى وحدات صوتية (phonemes) أو معالجة محددة لكل لغة:

- **المستوى الأول (Tier 1)**: اليابانية (ja)، الإنجليزية (en)، الصينية (zh)
- **المستوى الثاني (Tier 2)**: الكورية (ko)، الإسبانية (es)، البرتغالية (pt)، العربية (ar)، الروسية (ru)، الفرنسية (fr)، الألمانية (de)
- **تغطية عالمية**: sv, it, tr, no, nl, cy, eu, ca, da, gl, ta, hu, fi, pl, et, hi, la, ur, th, vi, jw, bn, yo, xsl, cs, sw, nn, he, ms, uk, id, kk, bg, lv, my, tl, sk, ne, fa, af, el, bo, hr, ro, sn, mi, yi, am, be, km, is, az, sd, br, sq, ps, mn, ht, ml, sr, sa, te, ka, bs, pa, lt, kn, si, hy, mr, as, gu, fo والمزيد.

### توليد متحدثين متعددين أصلي

<img src="./assets/chattemplate.png" width=200%>

يسمح Fish Audio S2 للمستخدمين بتحميل عينة مرجعية تحتوي على متحدثين متعددين، وسيقوم النموذج بمعالجة ميزات كل متحدث عبر وسم `<|speaker:i|>`. بعد ذلك، يمكنك التحكم في أداء النموذج عبر وسم معرف المتحدث، مما يتيح لتوليد واحد أن يتضمن متحدثين متعددين. لم تعد هناك حاجة لتحميل عينة مرجعية منفصلة وتوليد صوت لكل متحدث على حدة كما كان في السابق.

### توليد حوارات متعددة الجولات

بفضل توسيع سياق النموذج، يمكن لنموذجنا الآن الاستفادة من المعلومات السابقة لتحسين التعبير في المحتوى المولد لاحقاً، مما يعزز من طبيعية المحتوى.

### استنساخ الصوت السريع

يدعم Fish Audio S2 استنساخاً دقيقاً للصوت باستخدام عينات مرجعية قصيرة (عادةً 10-30 ثانية). يلتقط النموذج نبرة الصوت وأسلوب الكلام والميول العاطفية، مما يولد أصواتاً مستنسخة واقعية ومتسقة دون الحاجة إلى ضبط دقيق إضافي.
لاستخدام خادم SGLang، يرجى الرجوع إلى [SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md).

---

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
