# 예제

## 감정 제어 (*새 기능)

#### 기쁨
- 입력 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/zh/happy_refer.wav" />
- 합성 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/zh/happy.wav" />
- 프롬프트: (happy)嘿嘿...博士，悄悄告诉你一件事——我重新开始练小提琴了。

#### 혐오
- 입력 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/ja/ref.wav" />
- 합성 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/ja/disgusted.wav" />
- 프롬프트: (digusted)あなたは、本当に気持ち悪い、嫌い…(disgusted)それでも、慰めを求めますの？

#### 분노
- 직접 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/en/angry.wav" />
- 프롬프트: (angry)I want you to go out immediately! I don't want to see you again, or I will try to kill you!
- 입력 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/zh/作战中4.wav" />
- 합성 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/zh/angry.wav" />
- 프롬프트: (angry)我让你快滚，你是耳聋吗？！...(angry)信不信我揍你！

#### 놀람
- 입력 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/zh/ref1.wav" />
- 합성 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/zh/surprised.wav" />
- 프롬프트: (surprised)今天你过生日？既然这样的话，我就勉为其难祝你生日快乐吧。(surprised)要不要看看你的桌子底下？

#### 슬픔
- 입력 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/ja/ref2.wav" />
- 합성 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/ja/sad.wav" />
- 프롬프트: (sad)他の小隊長と比べて、私はまだ多くのことを学ばなくてはなりません......(sad)皆さんのペースに追いつけるよう精一杯努力いたしますわ。

## 부언어 효과 (*새 기능)

### 웃음소리

- **예제 1**
  - **프롬프트**: 大家好啊，(笑声)哈哈，我是从来不带节奏的血狼破军，今天来点大家想看的东西。
  - **오디오**: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/zh/laugh1.wav" />

- **예제 2**
  - **프롬프트**: (笑声)哈哈(笑声)，虽然说"三角洲行动"的策划说他们没有暗改(笑声)哈哈(笑声)，但是我相信，大家心里都有数。对不起，实在是太搞笑了，忍不住笑了出来。(笑声)哈哈(笑声)
  - **오디오**: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/zh/laugh2.wav" />

- **예제 3 (영어)**
  - **프롬프트**: (laughing)haha(laughing), though many people say that homeless cats need our help, (laughing)haha(laughing), but seldom do they really do something that is useful to the cats, (laughing)haha(laughing) sorry, but this is very interesting.
  - **오디오**: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/en/laugh.wav" />

### 외침 (shouting)

- **예제 (영어)**
  - **프롬프트**: (shouting)oh my god !!!(shouting)(shouting)(shouting), baby(shouting)you (shouting)are (shouting)a piece of sweet, soft(shouting), delicious cake!!!
  - **오디오**: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/en/shout.wav" />

## 장문 안정성 테스트

### 중국어 문장
```
你们这个是什么群啊，你们这是害人不浅啊你们这个群！谁是群主，出来！真的太过分了。你们搞这个群干什么？
我儿子每一科的成绩都不过那个平均分呐，他现在初二，你叫我儿子怎么办啊？他现在还不到高中啊？
你们害死我儿子了！快点出来你这个群主！再这样我去报警了啊！我跟你们说你们这一帮人啊，一天到晚啊，
搞这些什么游戏啊，动漫啊，会害死你们的，你们没有前途我跟你说。你们这九百多个人，好好学习不好吗？
一天到晚在上网。有什么意思啊？麻烦你重视一下你们的生活的目标啊？有一点学习目标行不行？一天到晚上网是不是人啊？
```

#### 시 (아크나이츠)
- 입력 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/zh/ref1.wav" />
- 합성 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/zh/audio.wav" />

#### 랜덤 화자 (볼륨 주의)
- 입력 오디오: 없음
- 합성 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/zh/audio2.wav" />

### 영어 문장

```
In the realm of advanced technology, the evolution of artificial intelligence stands as a 
monumental achievement. This dynamic field, constantly pushing the boundaries of what 
machines can do, has seen rapid growth and innovation. From deciphering complex data 
patterns to driving cars autonomously, AI's applications are vast and diverse.
```

#### 랜덤 화자 1
- 입력 오디오: 없음
- 합성 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/en/audio.wav" />

#### 랜덤 화자 2
- 입력 오디오: 없음
- 합성 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/en/audio2.wav" />

### 일본어 문장

```
宇宙に始まりはあるが、終わりはない。無限。
星にもまた始まりはあるが、自らの力をもって滅び逝く。有限。
英知を持つ者こそ、最も愚かであること。歴史からも読み取れる。
海に生ける魚は、陸の世界を知らない。彼らが英知を持てば、それもまた滅び逝く。
人間が光の速さを超えるのは、魚たちが陸で生活を始めるよりも滑稽。
これは抗える者たちに対する、神からの最後通告とも言えよう。
```

#### 도요가와 사키코
- 입력 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/ja/ref.wav" />
- 합성 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/ja/audio.wav" />

#### 랜덤 화자 2
- 입력 오디오: 없음
- 합성 오디오: <audio controls preload="auto" src="https://demo-r2.speech.fish.audio/s1-20250920/ja/audio2.wav" />
