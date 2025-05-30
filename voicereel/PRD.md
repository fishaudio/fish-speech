# VoiceReel Studio – 다중 화자 텍스트‑투‑스피치 API  PRD (Ver. 1.0)

---

## 1. 문서 정보

| 구분       | 내용                          |
| -------- | --------------------------- |
| 작성일      | 2025‑05‑22                  |
| 작성자      | ChatGPT (o3)                |
| 검토자      | Warlord G, Claude‑duck Oops |
| 최신 버전 위치 | TBD (Confluence)            |
| 변경 이력    | v1.0 – 최초 작성                |

---

## 2. Executive Summary

VoiceReel Studio는 **fish‑speech 1.5** 엔진을 기반으로, 다중 화자 스크립트를 입력하면 한 번의 호출로 시간 동기화된 오디오와 자막(JSON/VTT)을 반환하는 **B2B REST API**다. 레퍼런스 음성을 업로드해 화자를 등록하고, 스크립트만 주면 Netflix 품질의 TTS 콘텐츠를 수 초 내 생성한다.

---

## 3. 문제 정의

1. 다중 화자 오디오 제작은 녹음·편집·싱크 작업 때문에 **비용·시간이 과다** 소요.
2. 현존 TTS API는 **단일 화자** 중심이라 대사별 화자 지정, 자막 매핑 자동화가 어렵다.
3. **브랜드 톤 유지**를 위한 맞춤 화자 관리 기능이 부족.

---

## 4. 목표 & 성공 지표

| 목표         | KPI                            | 목표치     |
| ---------- | ------------------------------ | ------- |
| TTS 렌더링 속도 | 30 sec audio ≤ 8 s p95         | ✅ 8 초   |
| 음성 자연도     | MOS (Mean Opinion Score) ≥ 4.3 | ✅ ≥ 4.3 |
| 화자 등록 성공률  | 성공/시도 ≥ 99 %                   | ✅ 99 %  |
| 월간 유료 호출   | MRR 달성 지표                      | TBD     |

---

## 5. 범위

### 5.1 In‑Scope

* 화자 등록(음성+스크립트) API
* 스크립트→오디오(다중 화자) API
* 자막(JSON, WebVTT) 자동 생성
* 동기·비동기 작업 관리
* 대시보드(기본) & API 키 관리

### 5.2 Out‑of‑Scope

* 웹 UI 기반 **실시간** 음성 편집기(후속 버전)
* 얼굴 애니메이션/비디오 합성 기능

---

## 6. 사용자 페르소나 & 주요 시나리오

| 페르소나                  | 필요                  | 시나리오 예                   |
| --------------------- | ------------------- | ------------------------ |
| **Indie Game Dev 유진** | 게임 NPC 대사 20만 자를 더빙 | 화자 12개 등록 → 배치 생성        |
| **LX Studio PD 소라**   | 기업 교육 영상            | 성우 2명 등록→스크립트 업로드→오디오 수령 |

---

## 7. 기능 요구사항

| ID    | 기능             | 우선도    | 설명                                  |
| ----- | -------------- | ------ | ----------------------------------- |
| FR‑01 | 화자 등록          | High   | 레퍼런스 음성(≥ 30 초)과 매칭 스크립트로 음색 프로필 생성 |
| FR‑02 | 화자 목록 조회       | Medium | 사용자 계정 내 등록된 화자 메타데이터 반환            |
| FR‑03 | 다중 화자 합성       | High   | JSON 스크립트 입력→싱글 WAV or MP3 반환       |
| FR‑04 | 자막 export      | High   | WebVTT, SRT, JSON 선택 가능             |
| FR‑05 | 작업 상태 조회       | High   | 비동기 Job ID 상태 API                   |
| FR‑06 | Usage Metering | Medium | 월간 요청 수, 합성 길이 통계                   |

---

## 8. 비기능 요구사항

* **Latency**: 1 분 오디오 ≤ 20 s(p95)
* **Scalability**: 동시 500 req/s 처리
* **Availability**: 99.9 % / 월
* **Security**: TLS 1.3, JWT or HMAC API 키, GDPR & PIPL 준수
* **Compliance**: SOC 2 Type II, ISO 27001 로드맵
* **Localization**: ISO 639‑1 언어코드 지원(ko, en, ja 우선)

---

## 9. 시스템 아키텍처 (개념)

```text
Client → API Gateway (Flask + Gunicorn) → Task Queue (Celery + Redis) → Inference Workers (GPU)
                                  ↘︎ Object Storage (S3) ↙︎
                            Metadata DB (PostgreSQL)
```

---

## 10. API 설계

### 10.1 인증

* API‑Key: `X‑VR‑APIKEY` 헤더
* 권한: per‑project quota, HMAC‑SHA256 서명 옵션

### 10.2 엔드포인트 목록

| Method & Path               | 설명       | 동기/비동기 | 주요 파라미터                                                                  |
| --------------------------- | -------- | ------ | ------------------------------------------------------------------------ |
| **POST** `/v1/speakers`     | 화자 등록    | Async  | `name`, `lang`, `reference_audio`(multipart), `reference_script`(string) |
| **GET** `/v1/speakers/{id}` | 화자 상세    | Sync   |  —                                                                       |
| **GET** `/v1/speakers`      | 화자 목록    | Sync   | `page`, `page_size`                                                      |
| **POST** `/v1/synthesize`   | 다중 화자 합성 | Async  | `script`(JSON), `output_format`, `sample_rate`                           |
| **GET** `/v1/jobs/{id}`     | 작업 상태/결과 | Sync   |  —                                                                       |
| **DELETE** `/v1/jobs/{id}`  | 작업 삭제    | Sync   |  —                                                                       |

#### 10.2.1 화자 등록 – 예시

```
POST /v1/speakers HTTP/1.1
Content‑Type: multipart/form‑data; boundary=---
X‑VR‑APIKEY: <key>

---boundary
Content‑Disposition: form‑data; name="name"

Min‑ho Korean Male
---boundary
Content‑Disposition: form‑data; name="lang"

ko
---boundary
Content‑Disposition: form‑data; name="reference_audio"; filename="minho.wav"
Content‑Type: audio/wav

<binary>
---boundary
Content‑Disposition: form‑data; name="reference_script"

안녕하세요, 저는 민호입니다.
---boundary--
```

**Response 202 Accepted**

```json
{
  "job_id": "job_123456",
  "speaker_temp_id": "spk_tmp_789"
}
```

#### 10.2.2 합성 요청 – 예시

```json
POST /v1/synthesize
{
  "script": [
    { "speaker_id": "spk_1", "text": "안녕?" },
    { "speaker_id": "spk_2", "text": "오늘 일정 공유해줄래?" }
  ],
  "output_format": "wav",
  "sample_rate": 48000,
  "caption_format": "json"
}
```

**Response 202**

```json
{ "job_id": "job_987" }
```

#### 10.2.3 작업 결과 – 예시

```json
GET /v1/jobs/job_987 → 200
{
  "status": "succeeded",
  "audio_url": "https://cdn.voicereel.ai/jobs/job_987/output.wav",
  "captions": [
    { "start": 0.0, "end": 0.95, "speaker": "spk_1", "text": "안녕?" },
    { "start": 1.0, "end": 2.80, "speaker": "spk_2", "text": "오늘 일정 공유해줄래?" }
  ]
}
```

### 10.3 오류 코드

| Code                    | 의미       | 설명                     |
| ----------------------- | -------- | ---------------------- |
| 400 `INVALID_INPUT`     | 잘못된 파라미터 | 필수 필드 누락, 포맷 오류        |
| 401 `UNAUTHORIZED`      | 인증 실패    | API 키 불일치              |
| 404 `NOT_FOUND`         | 리소스 없음   | speaker_id/job_id 잘못 |
| 413 `PAYLOAD_TOO_LARGE` | 업로드 초과   | 오디오 > 30 MB            |
| 422 `INSUFFICIENT_REF`  | 레퍼런스 짧음  | ≥ 30 초 요구              |
| 500 `INTERNAL_ERROR`    | 서버 오류    | 예기치 못한 예외              |

---

## 11. 데이터 모델 (요약)

```typescript
type Speaker {
  id: string;
  name: string;
  lang: string;  // ISO‑639‑1
  sample_rate: number;
  created_at: ISODateString;
}

type CaptionUnit {
  start: number; // sec
  end: number;
  speaker: string; // speaker_id or name
  text: string;
}
```

---

## 12. 보안 & 프라이버시

* 전 구간 TLS 1.3
* S3 presigned URL 15 min 유효
* 모든 오디오 48 시간 후 자동 삭제(옵션)
* 개인정보 분리 보관, 한국 PIPA & EU GDPR 준수

---

## 13. 성능/용량 계획

| 항목          | 기준                          | 비고                           |
| ----------- | --------------------------- | ---------------------------- |
| 합성 동시 요청    | 500 req/s                   | K8s HPA auto-scale GPU nodes |
| 합성 속도       | audio_len × 0.2 ≤ wall sec | GPU: RTX A6000 x N           |
| 레퍼런스 등록 처리량 | 5 req/s/node                | CPU node 처리                  |

---

## 14. 의존 & 리스크

* fish‑speech 업스트림 업데이트 지연 가능성
* GPU 서버 비용 급등 위험
* 배포 리전: AWS ap‑northeast‑2 (서울) 우선, DR in Tokyo

---

## 15. 로드맵 & 마일스톤

| 단계             | 일정    | 산출물               |
| -------------- | ----- | ----------------- |
| Tech Spike     | 6월 1주 | POC 성능 리포트        |
| Alpha API v0.1 | 6월 3주 | 화자 등록 + 단일 화자 합성  |
| Beta v0.9      | 7월 4주 | 다중 화자, 캡션, 대시보드   |
| GA v1.0        | 8월 4주 | SLA 99.9 %, 결제 연동 |

---

## 16. 미해결 이슈

1. 화자 톤 정책(윤리·저작권) 가이드 확정
2. 무료 티어 쿼터 범위?

---

## 17. 부록

* fish‑speech 1.5 연구 노트 링크
* MOS 측정 방법론(JND 퀴즈)

---

*End of Document*
