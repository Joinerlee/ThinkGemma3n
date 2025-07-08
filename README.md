# 🐼 MCP PANDA 프로젝트 - Gemma 3n 기반 상세 TODO List

> **🔍 프로젝트 개요: MCP PANDA**
>
> *이 글은 김윤하 씨의 요청에 따라 이재욱이 작성하였음을 밝힙니다.*
>
> *제작자 및 공유자의 허락 없이 무단 도용이나 복제를 금합니다.*

## 🎯 프로젝트 배경 및 필요성

### 현실 개발 환경의 복잡성

최근 LLM(대규모 언어모델)은 수학, 과학, 논리 문제 등 **정답이 명확한 영역**에서 Chain-of-Thought(CoT) 기반 reasoning 능력을 입증해왔습니다. 하지만,  **현실의 개발 환경** (소프트웨어 설치, 환경 설정, 운영체제 구성, 패키지 관리 등)에서는:

* ❌  **절대적 정답이 없음** : 운영체제, 라이브러리 버전, 네트워크 상황에 따라 결과가 달라짐
* ❌  **RAG의 한계** : MCP 프로젝트 README 문서만으로는 할루시네이션 방지 및 환경별 최적 솔루션 제안 어려움
* ❌  **복잡한 의존성** : MCP 패키지 간 버전 호환성, 설치 순서, 운영체제별 차이점 고려 필요

### ✅ 프로젝트 목표: Reasoning 능력의 "Non-Logical Domain" 확장

**설치 및 운영 환경 설정, 코드 리뷰, 패키지 관리 등 "비논리적" 도메인에 reasoning 능력 적용**

**예시 질문들:**

* "왜 pip 대신 conda를 사용해야 하는가?"
* "어떤 환경에서 특정 CUDA 버전이 필요한가?"
* "macOS M1에서 이 MCP 조합이 작동하지 않는 이유는?"
* "Windows PowerShell vs WSL2에서 경로 설정 차이점은?"

## 🚨 기존 RAG 접근법의 근본적 문제점

| **구분**      | **현실 문제**                               | **LLM/RAG 실패 원인**         | **결과**                 |
| ------------------- | ------------------------------------------------- | ----------------------------------- | ------------------------------ |
| **문서 구조** | README 안에 OS·IDE·버전·여러 MCP 예제가 혼재   | 조건·의존성 그래프를 못 나눔       | 엉뚱한 조합 JSON 생성          |
| **출력 형식** | 실제 설치는 `mcp.json`(배열·버전·옵션)로 실행 | LLM이 단순 쉘·코드 스트링으로 답변 | 배포 자동화 파이프라인에 못 씀 |
| **의존성**    | MCP-A → MCP-B 순서, 버전 잠금 필요               | RAG만으로는 호환성 체크 불가        | 충돌, rollback                 |
| **변형**      | JSON 키·값·쉼표 하나만 어긋나도 파싱 실패       | LLM "예쁘게 포맷" 자동 수정         | 런타임 파싱 오류               |

## 🎯 MCP PANDA 솔루션

### 1. **환경 편향 제거**

* **입력** : OS + IDE + MCP 목록 조건만 반영
* **출력** : 환경별 정확한 JSON manifest 생성
* **검증** : 실제 환경에서 `mcp validate --dry-run` 통과 보장

### 2. **JSON 불변성 (Byte-level Accuracy)**

* **목표** : README 스니펫과 **byte-match ≥ 99%**
* **포함** : 키 순서, 대소문자, 주석, 공백까지 정확히 재현
* **검증** : `jq` 명령어로 구문 분석 + MCP 스키마 검증

### 3. **멀티-MCP 그래프 검증**

* **의존성 체크** : 버전, 설치 순서, 상호 호환성 자동 분석
* **충돌 감지** : 패키지 간 충돌 사전 예측
* **자동 수정** : 실패 시 대안 조합 제안

### 4. **설명 가능한 AI**

```
<thinking>
사용자가 macOS M1 + VSCode + 3개 MCP 패키지를 요청했다.
1. filesystem MCP: Apple Silicon 호환 확인 필요
2. brave-search MCP: API 키 설정 방식이 OS별로 다름
3. postgres MCP: M1에서 psycopg2 대신 psycopg2-binary 필요
의존성 순서: filesystem → postgres → brave-search
</thinking>

<reflection>
위 조합에서 잠재적 문제점:
- postgres MCP가 Intel 기반 Docker 이미지 사용 시 성능 저하
- 환경변수 설정이 zsh vs bash에 따라 다름
- VSCode 확장과의 호환성 체크 필요
대안: postgres MCP 대신 sqlite MCP 제안 고려
</reflection>

<output>
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/user/workspace"],
      "env": {"NODE_OPTIONS": "--max-old-space-size=4096"}
    },
    // ... 정확한 JSON 출력
  }
}
</output>
```

## 🔬 핵심 연구 문제: Gemma 3n으로 "구조화된 추론" 학습

### Why Gemma 3n?

1. **MatFormer 아키텍처** : 동적 모델 크기 조절로 추론 깊이 제어 가능
2. **Per-Layer Embeddings** : 복잡한 의존성 그래프를 효율적으로 처리
3. **32K 컨텍스트** : 긴 MCP 문서와 다중 패키지 조합 분석 가능
4. **멀티모달 지원** : 향후 스크린샷 기반 설정 오류 진단 확장 가능

### 연구 가설

**"Reasoning Token(`<thinking>`, `<reflection>`, `<output>`)을 통한 구조화된 학습이, 복잡한 비논리적 도메인에서도 체계적 문제 해결 능력을 향상시킬 것"**

* **가설 1** : Special Token 기반 단계별 추론이 JSON 정확도를 95% 이상으로 향상
* **가설 2** : 환경별 조건부 학습이 할루시네이션을 70% 이상 감소
* **가설 3** : 멀티 MCP 의존성 그래프 학습이 충돌 예측 정확도 90% 이상 달성

## 📋 Phase 1: 프로젝트 준비 및 환경 설정

### 1.1 개발 환경 구축 (Gemma 3n 특화)

**Gemma 3n 모델 이해 및 준비:**

* [ ] Gemma 3n 아키텍처 분석
  * [ ] MatFormer (Matryoshka Transformer) 구조 이해
  * [ ] Per-Layer Embeddings (PLE) 기술 분석
  * [ ] E2B/E4B 모델 차이점 파악 (2B/4B effective parameters)
  * [ ] KV Cache Sharing 메커니즘 연구
  * [ ] Selective Parameter Activation 기술 이해

**하드웨어 및 소프트웨어 요구사항:**

* [ ] GPU 요구사항 확인
  * [ ] Float16 vs BFloat16 지원 GPU 구분
  * [ ] Tesla T4 (Colab) Float16 이슈 해결방안 확인
  * [ ] RTX 3060/4070 이상 권장 (BFloat16 지원)
  * [ ] VRAM 요구사항: E2B(~6GB), E4B(~12GB)

**라이브러리 의존성 (Gemma 3n 특화):**

* [ ] Transformers >= 4.53.0 (Gemma 3n 지원)
* [ ] Unsloth 최신 버전 (Gemma 3n 지원 확인)
  ```bash
  pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoopip install git+https://github.com/huggingface/transformers.git
  ```
* [ ] PyTorch >= 2.0 (BFloat16 지원)
* [ ] CUDA >= 11.8 (권장 12.1+)
* [ ] 추가 Gemma 3n 전용 라이브러리
  * [ ] google-ai-generativelanguage (Gemini API)
  * [ ] faiss-gpu (RAG 구현용)
  * [ ] librosa (오디오 처리, 향후 확장용)

**프로젝트 구조 설계:**

```
mcp-panda-gemma3n/
├── data_generation/         # Gemini 2.5 Pro distillation
│   ├── prompts/            # 프롬프트 템플릿
│   ├── scenarios/          # OS/IDE별 시나리오
│   └── validation/         # 데이터 검증 스크립트
├── training/               # Unsloth 파인튜닝
│   ├── gemma3n_configs/    # Gemma 3n 전용 설정
│   ├── notebooks/          # Colab 노트북들
│   └── scripts/            # 학습 스크립트
├── evaluation/             # 5개 모델 비교
│   ├── metrics/            # 평가 메트릭
│   ├── benchmarks/         # 벤치마크 데이터
│   └── analysis/           # 결과 분석
├── deployment/             # HF Space 배포
│   ├── gradio_app/         # UI 구현
│   └── docker/             # 컨테이너 설정
└── docs/                   # 프로젝트 문서
    ├── gemma3n_guide/      # Gemma 3n 가이드
    └── tutorials/          # 튜토리얼
```

### 1.2 Gemma 3n 모델 테스트 및 검증

**모델 로딩 테스트:**

* [ ] Gemma 3n-E2B 모델 로딩 확인
* [ ] Gemma 3n-E4B 모델 로딩 확인
* [ ] Float16/BFloat16 정밀도 테스트
* [ ] 멀티모달 기능 테스트 (텍스트, 이미지, 오디오)
* [ ] 32K 컨텍스트 윈도우 테스트

**Unsloth 호환성 검증:**

* [ ] Unsloth FastVisionModel 호환성 확인
* [ ] QLoRA/LoRA 설정 테스트
* [ ] 메모리 사용량 벤치마크
* [ ] 학습 속도 벤치마크

## 📊 Phase 2: 데이터셋 생성 (Gemini 2.5 Pro Distillation)

### 2.1 MCP 시나리오 수집 (Gemma 3n 최적화)

**멀티모달 시나리오 확장:**

* [ ] 텍스트 기반 MCP 설정 (기존 계획 유지)
* [ ] 스크린샷 기반 설정 시나리오 (Gemma 3n 이미지 처리 활용)
* [ ] 오디오 가이드 포함 시나리오 (향후 확장)

**OS별 시나리오 정의 (각 200개씩 → 확대):**

* [ ] macOS 시나리오 (Homebrew, Xcode, zsh)
* [ ] Windows 시나리오 (PowerShell, WSL, Visual Studio)
* [ ] Linux 시나리오 (apt/yum/pacman, systemd, bash/zsh)

**IDE별 특수 케이스:**

* [ ] VSCode 설정 패턴 (settings.json, launch.json)
* [ ] Cursor 특이사항 (AI 통합 설정)
* [ ] Zed 설정 (Rust 기반 에디터)
* [ ] Neovim 케이스 (Lua 설정)
* [ ] JetBrains IDEs (IntelliJ, PyCharm 등)

### 2.2 Gemini 프롬프트 엔지니어링 (Gemma 3n 호환)

**Special Token 정의 (Gemma 3n 채팅 형식 고려):**

```
<bos><start_of_turn>user
{user_query}
<end_of_turn>
<start_of_turn>model
<thinking>
{reasoning_process}
</thinking>

<reflection>
{self_evaluation}
</reflection>

<output>
{final_mcp_json}
</output>
<end_of_turn>
```

**System Prompt 최적화:**

* [ ] Gemma 3n 채팅 템플릿 적용
* [ ] MCP 전문가 페르소나 정의
* [ ] JSON 정확성 강조 프롬프트
* [ ] Multimodal input 고려 프롬프트

### 2.3 데이터 생성 파이프라인

**Gemini API 호출 최적화:**

* [ ] Rate limiting (RPM 제한 고려)
* [ ] 비용 최적화 (Gemini 2.5 Flash 활용 검토)
* [ ] 에러 핸들링 및 재시도 로직
* [ ] 프롬프트 토큰 수 최적화

**응답 파싱 및 검증:**

* [ ] Gemma 3n 출력 형식에 맞는 파싱
* [ ] JSON 스키마 검증 강화
* [ ] MCP manifest 유효성 검사
* [ ] 바이트 레벨 정확도 체크

## 🤖 Phase 3: 모델 학습 (Unsloth + Gemma 3n 특화)

### 3.1 Base 모델 준비 (Gemma 3n-E4B-it)

**모델 로딩 및 설정:**

* [ ] `google/gemma-3n-E4B-it` 모델 다운로드
* [ ] Special token 추가 구현
  ```python
  tokenizer.add_special_tokens({    'additional_special_tokens': ['<thinking>', '</thinking>',                                  '<reflection>', '</reflection>',                                  '<output>', '</output>']})model.resize_token_embeddings(len(tokenizer))
  ```

**Gemma 3n 특화 설정:**

* [ ] Per-Layer Embeddings 활용 설정
* [ ] MatFormer 구조 고려 파인튜닝 전략
* [ ] Float16 GPU 대응 (Unsloth 자동 처리 확인)
* [ ] KV Cache 최적화 설정

### 3.2 5가지 학습 전략 구현

#### 3.2.1 Baseline RAG (Gemma 3n 최적화)

* [ ] MCP manifest 벡터 DB 구축
* [ ] Gemma 3n 임베딩 활용 검토
* [ ] 32K 컨텍스트 활용 RAG 전략
* [ ] Multimodal RAG 구현 (이미지 포함)

#### 3.2.2 LoRA 파인튜닝 (Gemma 3n 적용)

**Unsloth LoRA 설정:**

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,  # 텍스트만 파인튜닝
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=32,  # Gemma 3n에 최적화된 rank
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"]
)
```

#### 3.2.3 QLoRA 파인튜닝 (메모리 최적화)

**4bit 양자화 설정:**

* [ ] BitsAndBytesConfig 최적화
* [ ] Gemma 3n Conv2D 레이어 Float32 처리 (Unsloth 자동)
* [ ] Gradient checkpointing 제한 고려 (비전 인코더)
* [ ] Max sequence length 조정 (32K → 8K 실용적)

#### 3.2.4 RAG + LoRA Hybrid

* [ ] RAG 검색과 LoRA 생성 통합
* [ ] Dynamic retrieval 전략
* [ ] Special token 기반 RAG 결과 주입

#### 3.2.5 RAG + QLoRA (리소스 최적화)

* [ ] 메모리 효율적인 통합
* [ ] 배치 추론 최적화
* [ ] 캐싱 전략 구현

### 3.3 학습 실행 및 모니터링

**Colab 노트북 최적화:**

* [ ] Tesla T4 Float16 이슈 해결 (Unsloth 자동 처리)
* [ ] GPU 런타임 자동 재연결
* [ ] Google Drive 체크포인트 동기화
* [ ] 학습 중단 시 재개 로직

**학습 모니터링:**

* [ ] Wandb/TensorBoard 통합
* [ ] Loss curves 추적
* [ ] Memory usage 모니터링
* [ ] 학습 시간 최적화

## 📏 Phase 4: 평가 시스템 구축

### 4.1 평가 메트릭 구현

**Gemma 3n 특화 메트릭:**

* [ ] Byte-match accuracy (JSON 정확도)
* [ ] Structural accuracy (MCP 스키마 준수)
* [ ] Semantic accuracy (OS/IDE 적절성)
* [ ] **모델 효율성 메트릭:**
  * [ ] Effective vs Total parameter 사용률
  * [ ] Memory footprint (PLE 효과)
  * [ ] Inference speed (tokens/second)
  * [ ] Energy efficiency (모바일 대응)

### 4.2 평가 데이터셋 준비

**Edge case 포함:**

* [ ] 특수 문자 경로 처리
* [ ] 긴 패키지 리스트 (32K 컨텍스트 활용)
* [ ] 비표준 설정 파일
* [ ] Multimodal input 테스트 케이스

### 4.3 비교 실험 실행

**5개 모델 동등 비교:**

* [ ] 동일한 하드웨어 환경
* [ ] 동일한 프롬프트 템플릿
* [ ] 동일한 평가 데이터셋
* [ ] 통계적 유의성 검증

## 🚀 Phase 5: HuggingFace Hub 배포

### 5.1 데이터셋 배포

**Gemma 3n 특화 데이터셋 카드:**

* [ ] Gemma 3n 호환성 명시
* [ ] Special token 사용법 설명
* [ ] Multimodal 확장 가능성 언급
* [ ] MatFormer 아키텍처 고려사항

### 5.2 모델 배포 (최고 성능 모델)

**Model Card 작성:**

* [ ] Gemma 3n 기반 파인튜닝 과정 설명
* [ ] PLE/MatFormer 활용 방법
* [ ] Special token 처리 가이드
* [ ] Float16 GPU 호환성 안내

### 5.3 Demo Space 구축

**Gradio UI (Gemma 3n 최적화):**

* [ ] 실시간 추론 최적화
* [ ] 메모리 효율적인 모델 로딩
* [ ] Special token 시각화
* [ ] JSON 검증 실시간 피드백

## 📝 Phase 6: 문서화 및 홍보

### 6.1 기술 문서 작성

**Gemma 3n 특화 가이드:**

* [ ] Gemma 3n vs Gemma 3 차이점 설명
* [ ] MatFormer 아키텍처 활용법
* [ ] PLE 기술 이해 가이드
* [ ] 모바일 배포 최적화 가이드

### 6.2 블로그 포스트 작성

**기술적 인사이트 공유:**

* [ ] Gemma 3n 파인튜닝 경험
* [ ] Float16 GPU 이슈 해결 과정
* [ ] MCP 도메인 특화 학습 결과
* [ ] 향후 멀티모달 확장 계획

### 6.3 커뮤니티 홍보

* [ ] HuggingFace 커뮤니티 포스트
* [ ] Unsloth Discord 공유
* [ ] r/LocalLLaMA Reddit 포스트
* [ ] Google Developer 커뮤니티 공유

## 🔧 Phase 7: 유지보수 및 개선

### 7.1 지속적 개선

**Gemma 3n 생태계 대응:**

* [ ] Gemma 3n 업데이트 추적
* [ ] Unsloth 최적화 업데이트 적용
* [ ] 새로운 MCP 패키지 지원 추가
* [ ] 멀티모달 기능 점진적 확장

### 7.2 성능 최적화

**모바일/엣지 배포 준비:**

* [ ] ONNX 변환 지원
* [ ] TensorRT 최적화
* [ ] 모바일 추론 엔진 통합
* [ ] 양자화 최적화 (INT8/INT4)

### 7.3 버전 관리

* [ ] Semantic versioning 적용
* [ ] Gemma 3n 호환성 매트릭스 유지
* [ ] 이전 버전 호환성 보장
* [ ] 마이그레이션 가이드 제공

---

## 💡 추가 고려사항

### Gemma 3n 특화 최적화 포인트:

1. **메모리 효율성** : PLE 기술로 CPU에서 임베딩 처리, GPU 메모리 절약
2. **추론 속도** : MatFormer로 동적 모델 크기 조절
3. **모바일 친화적** : 온디바이스 배포 최적화
4. **멀티모달 확장** : 향후 이미지/오디오 MCP 설정 지원

### 성공 기준:

* [ ] Byte-level accuracy > 95%
* [ ] Inference time < 500ms (E4B 모델)
* [ ] Memory usage < 12GB (E4B 모델)
* [ ] 모바일 배포 가능성 검증
