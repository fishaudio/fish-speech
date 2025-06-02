# VoiceReel TODO

## 🔴 Critical Priority (MVP 필수)

### 1. Celery/Redis 작업 큐 구현
- [ ] Redis 서버 연동 설정 추가
- [ ] Celery worker 구현
- [ ] 기존 인메모리 큐를 Celery 태스크로 마이그레이션
- [ ] 작업 상태 Redis에 저장
- [ ] 작업 재시도 로직 구현

### 2. Fish-Speech 엔진 통합
- [ ] fish_speech 모델 로더 구현
- [ ] 화자 등록 시 실제 음성 특징 추출
- [ ] 다중 화자 합성 실제 구현
- [ ] GPU 워커 풀 설정
- [ ] 성능 최적화 (30초 오디오 → 8초 이내)

### 3. S3 스토리지 연동
- [ ] boto3 클라이언트 설정
- [ ] 오디오 파일 S3 업로드 구현
- [ ] Presigned URL 생성 로직
- [ ] 48시간 자동 삭제 Lambda 함수
- [ ] 로컬 파일시스템 폴백 옵션

## 🟡 High Priority (프로덕션 필수)

### 4. 보안 강화
- [ ] TLS 1.3 인증서 설정
- [ ] CORS 정책 구현
- [ ] Rate limiting 미들웨어
- [ ] SQL injection 방어 검증
- [ ] 입력 데이터 검증 강화

### 5. Docker 컨테이너화
- [ ] VoiceReel API Dockerfile 작성
- [ ] docker-compose.yml 구성 (API + Redis + PostgreSQL)
- [ ] GPU 지원 Docker 이미지 빌드
- [ ] 환경별 설정 분리 (.env 파일)
- [ ] Health check 엔드포인트 추가

### 6. 에러 처리 및 로깅
- [ ] 구조화된 로깅 시스템 (JSON 로그)
- [ ] Sentry 에러 추적 연동
- [ ] API 에러 응답 표준화
- [ ] 디버그 모드 설정
- [ ] 요청/응답 로깅 미들웨어

## 🟢 Medium Priority (안정성/운영)

### 7. 모니터링 시스템
- [ ] Prometheus 메트릭 엑스포터
- [ ] Grafana 대시보드 템플릿
- [ ] 알림 규칙 설정 (PagerDuty/Slack)
- [ ] APM 도구 연동 (DataDog/NewRelic)
- [ ] 사용량 통계 대시보드

### 8. CI/CD 파이프라인
- [ ] GitHub Actions 워크플로우
- [ ] 자동화된 테스트 실행
- [ ] Docker 이미지 자동 빌드/푸시
- [ ] 스테이징 환경 자동 배포
- [ ] Blue-Green 배포 전략

### 9. 성능 최적화
- [ ] 데이터베이스 인덱스 최적화
- [ ] 연결 풀링 구현
- [ ] 캐싱 전략 (Redis 캐시)
- [ ] 배치 처리 최적화
- [ ] 동시성 제한 설정

## 🔵 Low Priority (향후 개선)

### 10. 관리자 대시보드
- [ ] React/Vue 기반 웹 UI
- [ ] 화자 관리 인터페이스
- [ ] 작업 모니터링 실시간 뷰
- [ ] 사용량 통계 시각화
- [ ] API 키 관리 UI

### 11. 고급 기능
- [ ] 음성 속도/피치 조절
- [ ] 감정 표현 파라미터
- [ ] 배경음악 합성
- [ ] 실시간 스트리밍 API
- [ ] 음성 품질 A/B 테스트

### 12. 문서화 및 지원
- [ ] API 문서 자동 생성 (OpenAPI/Swagger)
- [ ] 통합 테스트 시나리오 문서
- [ ] 운영 가이드 작성
- [ ] 트러블슈팅 가이드
- [ ] 고객 지원 티켓 시스템

## 📊 진행 상태

**전체 진행률: 15% (18/120 작업 완료)**

### 완료된 작업 ✅
- [x] Flask 애플리케이션 구조
- [x] HTTP 서버 구현
- [x] 인메모리 작업 큐
- [x] 데이터베이스 스키마
- [x] 화자 등록 API
- [x] 화자 조회 API
- [x] 다중 화자 합성 API (더미)
- [x] 자막 생성 모듈
- [x] 작업 상태 관리 API
- [x] 클라이언트 SDK
- [x] CLI 인터페이스
- [x] 단위 테스트
- [x] 통합 테스트
- [x] E2E 테스트 프레임워크
- [x] API 키 인증
- [x] HMAC 서명 검증
- [x] 사용량 통계 수집
- [x] Presigned URL (로컬)

### 다음 스프린트 목표 (2주)
1. Celery/Redis 통합 완료
2. Fish-Speech 엔진 기본 연동
3. Docker 컨테이너 구성
4. 기본 에러 처리 구현

---

*마지막 업데이트: 2025-02-06*
