# VoiceReel TODO

우선순위와 의존성을 고려하여 VoiceReel Studio PRD를 구현하기 위한 작업 목록이다.

1. **API 서버 기반 구축**
   - [x] Flask+Gunicorn 스켈레톤 작성 *(`flask_app.py` 추가)*
   - [x] Celery/Redis로 비동기 작업 큐 구성 *(`task_queue.py`에서 메모리 큐 구현)*
   - [x] Metadata DB(PostgreSQL) 초기 스키마 설계 *(`db.init_db`에서 테이블 생성)*

2. **화자 등록 기능 (FR-01)**
   - [x] `/v1/speakers` POST 엔드포인트 구현
   - [x] 업로드된 음성 검증(≥30초) 및 스크립트 매칭 로직 작성
   - [x] 등록 처리 Job 생성 → 작업 큐 전달

3. **화자 조회 기능 (FR-02)**
   - [x] `/v1/speakers` GET 목록 및 `/v1/speakers/{id}` 세부 조회 구현 *(완료)*
   - [x] Pagination 파라미터 처리

4. **다중 화자 합성 API (FR-03)**
   - [x] `/v1/synthesize` POST 엔드포인트 구현
   - [x] 입력 JSON 스크립트 파싱 및 음성 합성 Job 생성
   - [x] 결과 WAV/MP3 저장 후 presigned URL 반환

5. **자막 Export (FR-04)**
   - [x] 합성 결과에 대해 WebVTT/SRT/JSON 포맷 변환 모듈 작성
   - [x] 클라이언트에서 원하는 포맷 선택 가능하도록 옵션 처리

6. **작업 상태 관리 (FR-05)**
   - [x] Job metadata 테이블 설계
   - [x] `/v1/jobs/{id}` GET/DELETE 구현
   - [x] 비동기 완료 시 상태 업데이트 로직 작성

7. **Usage Metering (FR-06)**
   - [x] 호출 수/합성 길이 통계 수집 모듈 작성
   - [x] 월간 리포트용 쿼리 함수 제공

8. **클라이언트 개선 및 예제**
   - [x] `voicereel.client`에 목록 조회, 작업 삭제 등 기능 보완
   - [x] CLI 서브커맨드 추가 예제 작성

9. **테스트 및 문서화**
- [x] unit test와 통합 test 작성
- [x] mkdocs 기반 API 사용 가이드 작성
 - [x] 오류 코드 표준화 및 문서화(PRD 10.3)
 - [x] E2E 테스트 구현 및 문서화
   - [x] `test_voicereel_e2e.py` - 실제 서버와 DB 연동 테스트
   - [x] `voicereel/e2e.md` - E2E 테스트 실행 가이드
   - [x] PRD 섹션 16 - E2E 테스트 전략 문서화

각 단계는 앞선 기능이 선행되어야 다음 단계 진행이 원활하다. 우선 API 서버와 핵심 합성 기능을 완성한 뒤, 모니터링/통계 및 문서화를 진행한다.

10. **SaaS 배포 준비**
   - [x] 환경 변수 기반 설정(`VR_DSN`, `VR_REDIS_URL`, `VR_API_KEY` 등)
   - [x] API-Key 인증 미들웨어 및 HMAC 서명 옵션
 - [x] PostgreSQL 실제 연동(`init_db`에서 psycopg2 사용)
 - [ ] Celery/Redis 연동
 - [ ] 합성 결과를 S3에 업로드하는 모듈
 - [ ] Docker 이미지/Helm 차트 작성 및 K8s 배포 예시
 - [ ] Prometheus/Loki 등 모니터링 스택 연동
 - [ ] GitHub Actions 기반 CI/CD 파이프라인
  - [x] S3 presigned URL 15분 제한 및 48시간 후 자동 삭제

11. **보안 및 로컬라이제이션**
  - [ ] 전 구간 TLS 1.3 적용
  - [x] ISO 639-1 언어코드 기반 로케일 지원(ko, en, ja)
  - [ ] 기본 대시보드 및 API 키 관리 화면 구현
  - [ ] SOC 2 Type II / ISO 27001 컴플라이언스 로드맵

12. **성능 및 안정성 향상**
  - [ ] 30초 오디오를 8초 이내로 렌더링하는 최적화
  - [ ] 동시 500 req/s 처리를 위한 스케일링 전략 수립
 - [ ] 월간 가용성 99.9% 모니터링 및 알림 구성

13. **미해결 이슈**
  - [ ] 화자 톤 정책(윤리·저작권) 가이드 확정
  - [ ] 무료 티어 쿼터 범위 결정

