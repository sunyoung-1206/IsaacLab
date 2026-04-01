# MuJoCo Sim-to-Sim Pipeline for Unitree Go2

Isaac Lab에서 학습한 보행 정책을 MuJoCo에서 재현하고 두 시뮬레이터 간의 동작을 비교하는 파이프라인입니다.

## 개요

```
IsaacLab 학습 (DCMotor / ElectricMotor)
        ↓
   정책 내보내기 (.pt JIT 파일)
        ↓
MuJoCo 재현 (play_mujoco.py)    ←──→    IsaacLab 재생 (play.py --log_data)
        ↓                                          ↓
   mujoco_data.npz                         isaac_data.npz
        └──────────────────┬────────────────────────┘
                           ↓
              비교 플롯 (compare_sims.py)
```

## 사전 요구사항

```bash
pip install mujoco torch numpy matplotlib scipy
```

MuJoCo 모델 파일 (Unitree Go2 Menagerie):
```bash
git clone https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie
```

---

## 스크립트 목록

| 스크립트 | 목적 |
|----------|------|
| `play_mujoco.py` | Isaac Lab 정책을 MuJoCo에서 재현 (PD / 전기모터 모드) |
| `test_actuators.py` | 액추에이터 단위 테스트 (뷰어 없이 동작 확인) |
| `view_assembly.py` | Go2 조립 상태 뷰어 (body별 켜기/끄기) |
| `view_parts.py` | Go2 개별 부품(OBJ) 뷰어 |
| `compare_sims.py` | IsaacLab ↔ MuJoCo 데이터 비교 플롯 생성 |

---

## 1. MuJoCo 정책 재현 (`play_mujoco.py`)

Isaac Lab에서 학습된 `.pt` JIT 정책 파일을 MuJoCo 환경에서 실행합니다.

### 액추에이터 모드

| 모드 | 설명 |
|------|------|
| `pd` (기본) | 순수 PD 토크 제어 (Isaac Lab `DCMotor` 재현) |
| `electric` | 전기모터 ODE 적분 (`dI/dt = (V_cmd - R·I - Ke·ω) / L`) |

### 기본 실행

```bash
# PD 모드 (기본)
python scripts/mujoco/play_mujoco.py \
    --policy logs/rsl_rl/Go2_Rough/exported/policy.pt \
    --scene ~/mujoco_menagerie/unitree_go2/go2_scene.xml

# 전기모터 모드
python scripts/mujoco/play_mujoco.py \
    --policy logs/rsl_rl/Go2_Rough/exported/policy.pt \
    --scene ~/mujoco_menagerie/unitree_go2/go2_scene.xml \
    --actuator electric
```

### 속도 명령 설정

```bash
python scripts/mujoco/play_mujoco.py \
    --policy policy.pt --scene go2_scene.xml \
    --cmd_vx 1.0 --cmd_vy 0.0 --cmd_yaw 0.0
```

### 고장 시뮬레이션 (electric 모드 전용)

특정 관절의 저항값을 점진적으로 증가시켜 전기모터 고장을 시뮬레이션합니다.

```bash
# RL_calf 관절(인덱스 10)에 t=5s부터 5초에 걸쳐 저항 0.3Ω → 3.0Ω으로 증가
python scripts/mujoco/play_mujoco.py \
    --policy policy.pt --scene go2_scene.xml \
    --actuator electric \
    --fault_joint 10 --fault_R 3.0 --fault_start 5.0 --fault_dur 5.0
```

**관절 인덱스 (policy 순서):**
```
 0: FL_hip    1: FR_hip    2: RL_hip    3: RR_hip
 4: FL_thigh  5: FR_thigh  6: RL_thigh  7: RR_thigh
 8: FL_calf   9: FR_calf  10: RL_calf  11: RR_calf
```

### 데이터 로깅

```bash
python scripts/mujoco/play_mujoco.py \
    --policy policy.pt --scene go2_scene.xml \
    --log_data logs/mujoco_data.npz \
    --duration 30.0
```

저장되는 데이터:
- `time`, `tau` (토크), `q` (관절각), `qdot` (관절속도)
- `lin_vel_b` (body frame 선속도), `base_height`
- `foot_contact` (바이너리), `foot_force` (접촉력 크기)

### 전체 옵션

```
--policy        정책 파일 경로 (.pt JIT) [필수]
--scene         MuJoCo XML 씬 파일 경로 [필수]
--actuator      액추에이터 모드: pd | electric (기본: pd)
--cmd_vx        전진 속도 명령 [m/s] (기본: 1.0)
--cmd_vy        측면 속도 명령 [m/s] (기본: 0.0)
--cmd_yaw       요 각속도 명령 [rad/s] (기본: 0.0)
--duration      시뮬레이션 시간 [s] (기본: 30.0)
--log_data      데이터 저장 경로 .npz
--fault_joint   고장 관절 인덱스 (electric 전용)
--fault_R       고장 시 저항 [Ω] (기본: 3.0)
--fault_start   고장 시작 시각 [s] (기본: 5.0)
--fault_dur     고장 전이 시간 [s] (기본: 5.0)
--keep_dof_params  XML의 damping/armature/frictionloss 유지 (기본: 0으로 초기화)
```

---

## 2. Isaac Lab 측 데이터 수집 (`play.py`)

Sim-to-Sim 비교를 위해 Isaac Lab에서도 동일한 조건으로 데이터를 수집합니다.

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --num_envs 1 \
    --cmd_vx 1.0 --cmd_vy 0.0 --cmd_yaw 0.0 \
    --sim2sim_init_z 0.355 \
    --log_data logs/isaac_data.npz
```

**`--sim2sim_init_z`**: MuJoCo `play_mujoco.py` 실행 시 출력되는
`[INFO] initial base z = X.XXX m` 값을 그대로 입력합니다.

Sim-to-Sim 모드 활성화 시 자동으로:
- 속도 명령을 고정값으로 설정
- Domain Randomization (마찰, 질량, 모터 강도) 비활성화
- 초기 위치 고정 (yaw=0, x=0, y=0)

---

## 3. 시뮬레이터 비교 (`compare_sims.py`)

두 시뮬레이터의 `.npz` 데이터를 로드하여 비교 플롯을 생성합니다.

### 기본 사용법

```bash
python scripts/mujoco/compare_sims.py \
    --isaac  logs/isaac_data.npz \
    --mujoco logs/mujoco_data.npz \
    --out    logs/comparison
```

### 플롯 종류 선택 (`--plots`)

| 방법론 | 출력 내용 |
|--------|-----------|
| `basic` | IsaacLab / MuJoCo 시계열 + Residual (I-M) (기본) |
| `density` | 2D 히스토그램 (x=Isaac, y=MuJoCo, 대각=완벽일치) |
| `envelope` | rolling mean ± std + 통계 요약 |
| `psd` | Welch Power Spectral Density |
| `contact` | 발 접촉 바이너리 + 접촉력 |

```bash
# 여러 방법론 동시 생성
python scripts/mujoco/compare_sims.py \
    --isaac logs/isaac_data.npz \
    --mujoco logs/mujoco_data.npz \
    --plots basic psd contact \
    --t_start 2.0 \
    --out logs/comparison
```

### 전체 옵션

```
--isaac     IsaacLab .npz 데이터 [필수]
--mujoco    MuJoCo .npz 데이터 [필수]
--plots     출력 방법론 선택 (기본: basic)
--t_start   초반 transient 제거 시작 시각 [s] (기본: 2.0)
--t_end     비교 종료 시각 [s]
--y_pct     basic 플롯 y축 백분위수 줌 (기본: 99)
--fs        샘플링 주파수 [Hz] (기본: 50)
--out       출력 파일 prefix (기본: logs/comparison)
```

출력 파일 예시:
```
logs/comparison_basic_00_base.png
logs/comparison_basic_01_FL_hip.png
...
logs/comparison_psd_00_base.png
logs/comparison_contact.png
```

---

## 4. 액추에이터 단위 테스트 (`test_actuators.py`)

MuJoCo 환경에서 액추에이터만 독립적으로 테스트합니다.

```bash
# PD 제어로 default 자세 유지 (안정성 확인)
python scripts/mujoco/test_actuators.py \
    --scene ~/mujoco_menagerie/unitree_go2/go2_scene.xml \
    --mode pd_hold

# 전기모터 ODE로 default 자세 유지
python scripts/mujoco/test_actuators.py \
    --scene ~/mujoco_menagerie/unitree_go2/go2_scene.xml \
    --mode electric_hold

# 한 관절에 step 입력 후 추종 성능 측정
python scripts/mujoco/test_actuators.py \
    --scene ~/mujoco_menagerie/unitree_go2/go2_scene.xml \
    --mode pd_step --step_joint 4 --step_angle 0.3
```

---

## 5. Go2 조립 뷰어 (`view_assembly.py`)

Go2 로봇의 body(링크)를 선택적으로 켜고 끄면서 조립 상태를 확인합니다.

> **주의**: `SCENE` 경로가 스크립트 내에 하드코딩되어 있습니다.
> 실행 전 `view_assembly.py` 상단의 `SCENE` 변수를 본인 환경에 맞게 수정하세요.

```bash
python scripts/mujoco/view_assembly.py
```

**터미널 명령어:**

| 명령 | 동작 |
|------|------|
| `l` | 전체 body 목록 및 현재 상태 출력 |
| `0`~`N` | 해당 번호의 body 토글 |
| `all` | 전체 켜기 |
| `none` | 전체 끄기 |
| `FL` / `FR` / `RL` / `RR` | 해당 다리 전체 토글 |
| `q` | 종료 |

---

## 6. Go2 부품 뷰어 (`view_parts.py`)

Go2의 개별 OBJ 메시 파일을 하나씩 확인합니다.

> **주의**: `ASSETS_DIR` 경로가 스크립트 내에 하드코딩되어 있습니다.
> `~/mujoco_menagerie/unitree_go2/assets` 디렉토리를 가리키도록 설정하세요.

```bash
python scripts/mujoco/view_parts.py
```

번호를 입력하면 해당 부품의 3D 뷰어가 열립니다. 창을 닫으면 다음 부품을 선택할 수 있습니다.

---

## 전체 워크플로우 예시

```bash
# 1. MuJoCo 실행 및 데이터 수집
python scripts/mujoco/play_mujoco.py \
    --policy logs/rsl_rl/Go2_Rough/exported/policy.pt \
    --scene ~/mujoco_menagerie/unitree_go2/go2_scene.xml \
    --cmd_vx 1.0 --duration 30.0 \
    --log_data logs/mujoco_data.npz
# → "[INFO] initial base z = 0.355 m" 값을 메모

# 2. Isaac Lab 실행 및 데이터 수집
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --num_envs 1 \
    --cmd_vx 1.0 \
    --sim2sim_init_z 0.355 \
    --log_data logs/isaac_data.npz

# 3. 비교 플롯 생성
python scripts/mujoco/compare_sims.py \
    --isaac  logs/isaac_data.npz \
    --mujoco logs/mujoco_data.npz \
    --plots  basic psd contact \
    --out    logs/comparison
```

---

## Isaac Lab 측 주요 변경사항

### `velocity_env_cfg.py`
- **마찰 범위 확장**: `(0.8, 0.8) → (0.4, 1.5)` (MuJoCo 범위 포함)
- **모터 강도 DR 추가**: 스타트업 시 stiffness / damping ±20% 랜덤화

### `ElectricMotorCfg` 파라미터
- 인덕턴스 `L`: `7.5e-4 → 1e-4` H
- 관성 모멘트 `J`: `0.05 → 1e-4` kg·m²

### `UNITREE_GO2_CFG` (unitree.py)
- 현재 학습 설정: `ElectricMotorCfg → DCMotorCfg` (학습은 DCMotor 기준)
  ElectricMotor로 전환하려면 `unitree.py`에서 `DCMotorCfg`를 `ElectricMotorCfg`로 변경
