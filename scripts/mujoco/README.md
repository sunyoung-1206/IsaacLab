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

MuJoCo 모델 파일 (Unitree Go2 Menagerie) — 이 프로젝트와 함께 fork된 저장소 사용:
```bash
git clone https://github.com/sunyoung-1206/mujoco_menagerie ~/mujoco_menagerie
```

스크립트들은 `~/mujoco_menagerie`를 기본 경로로 사용합니다.
다른 위치에 clone 했다면 `--scene` / `--assets_dir` 인자로 경로를 지정하세요.

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

Isaac Lab `DCMotorCfg`와 동일한 PD 토크 제어를 재현합니다:
- `τ = Kp·(q_target − q) + Kd·(−qdot)`
- velocity saturation 및 effort_limit 클리핑 포함

### 기본 실행

```bash
python scripts/mujoco/play_mujoco.py \
    --policy logs/rsl_rl/Go2_Rough/exported/policy.pt \
    --scene ~/mujoco_menagerie/unitree_go2/go2_scene.xml
```

### 속도 명령 설정

```bash
python scripts/mujoco/play_mujoco.py \
    --policy policy.pt --scene go2_scene.xml \
    --cmd_vx 1.0 --cmd_vy 0.0 --cmd_yaw 0.0
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
--policy          정책 파일 경로 (.pt JIT) [필수]
--scene           MuJoCo XML 씬 파일 경로 [필수]
--cmd_vx          전진 속도 명령 [m/s] (기본: 1.0)
--cmd_vy          측면 속도 명령 [m/s] (기본: 0.0)
--cmd_yaw         요 각속도 명령 [rad/s] (기본: 0.0)
--duration        시뮬레이션 시간 [s] (기본: 30.0)
--log_data        데이터 저장 경로 .npz
--keep_dof_params XML의 damping/armature/frictionloss 유지 (기본: 0으로 초기화)
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

# 한 관절에 step 입력 후 추종 성능 측정
python scripts/mujoco/test_actuators.py \
    --scene ~/mujoco_menagerie/unitree_go2/go2_scene.xml \
    --mode pd_step --step_joint 4 --step_angle 0.3
```

---

## 5. Go2 조립 뷰어 (`view_assembly.py`)

Go2 로봇의 body(링크)를 선택적으로 켜고 끄면서 조립 상태를 확인합니다.

기본 경로: `~/mujoco_menagerie/unitree_go2/go2_white_bg.xml`

```bash
# 기본 경로 사용 (mujoco_menagerie를 홈 디렉토리에 clone한 경우)
python scripts/mujoco/view_assembly.py

# 경로 직접 지정
python scripts/mujoco/view_assembly.py \
    --scene ~/mujoco_menagerie/unitree_go2/go2_white_bg.xml
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

기본 경로: `~/mujoco_menagerie/unitree_go2/assets`

```bash
# 기본 경로 사용
python scripts/mujoco/view_parts.py

# 경로 직접 지정
python scripts/mujoco/view_parts.py \
    --assets_dir ~/mujoco_menagerie/unitree_go2/assets
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

### `UNITREE_GO2_CFG` (unitree.py)
- 현재 학습 설정: `DCMotorCfg` 사용 (ElectricMotor 시도 후 복귀)

---

## ElectricMotor Isaac Lab 구현 시행착오

`source/isaaclab_assets/isaaclab_assets/actuators/electric_motor.py`에 구현이 남아 있다.
제대로 동작하지 않아 현재 학습에는 사용하지 않지만, 시행착오 기록으로 보존한다.

### 구현 목표

PhysX(Isaac Sim)의 PD 제어 대신 전기-기계 연립 ODE를 직접 적분해 더 물리적으로 정확한 액추에이터를 구현하는 것.

```
dI/dt     = (V_cmd - R·I - Ke·ω) / L       # 전기 방정식
dω/dt     = (Kt·I·gr - B·ω - τ_load) / J   # 기계 방정식
τ_joint   = Kt · I · gr
```

### 핵심 문제: ω 이중 적분

가장 근본적인 문제는 **ω(관절 속도)를 ODE와 PhysX가 동시에 결정**한다는 점이다.

- ODE는 `dω/dt`를 적분해 ω를 계산
- PhysX도 같은 ω를 rigid body dynamics로 독립적으로 계산
- 매 스텝 ODE 결과 ω를 버리고 PhysX 값으로 덮어써야 함 → `dω/dt` 적분이 무의미해짐

```python
# 코드에서의 처리 — ODE로 구한 omega를 결국 PhysX 값으로 덮어씀
self.omega = omega_physx.clone()  # ODE 결과 무시
self.I     = I_ode.clone()        # 전류만 ODE 결과 사용
```

결론적으로 ω까지 ODE에서 풀 이유가 없고, 전기 방정식(`dI/dt`)만 적분하면 충분하다.
MuJoCo 버전(`play_mujoco.py`)이 이 방식으로 구현되어 있었으나, 커플링 없는 적분은 물리적 의미가 제한적이어서 MuJoCo 쪽도 최종적으로 제거했다.

### τ_load 지연 문제

ODE의 기계 방정식에는 `τ_load`(부하 토크)가 필요한데, PhysX에서 읽어오는 값이 **항상 이전 스텝 값**이다.

```python
# get_dof_projected_joint_forces()는 이전 physics step의 값
self.tau_load = self._get_tau_load()  # 1 step 지연
```

200Hz 시뮬레이션에서 5ms 지연이지만, 전기 시상수(`τ_e = L/R ≈ 0.33ms`)보다 훨씬 커서
`τ_load` 피드백이 부정확하다.

### PhysX view 외부 주입 문제

`get_dof_projected_joint_forces()`를 호출하려면 `root_physx_view`가 필요한데,
Isaac Lab의 액추에이터 초기화 시점에는 이 view가 아직 준비되어 있지 않다.
따라서 play 스크립트에서 환경 초기화 후 수동으로 주입해야 했다.

```python
# play.py에서 매번 수동 호출 필요
for actuator in robot.actuators.values():
    if hasattr(actuator, 'inject_physx_view'):
        actuator.inject_physx_view(robot.root_physx_view)
```

이는 Isaac Lab의 표준 인터페이스를 벗어나며, 학습 루프와 통합이 복잡해진다.

### 파라미터 튜닝 시행착오

| 파라미터 | 초기값 | 최종값 | 이유 |
|----------|--------|--------|------|
| `L` (인덕턴스) | `7.5e-4` H | `1e-4` H | `τ_e = L/R`이 너무 커서 ODE 적분 스텝 수 과다, 불안정 |
| `J` (관성 모멘트) | `0.05` kg·m² | `1e-4` kg·m² | J가 크면 ODE의 ω 동역학이 PhysX보다 너무 느려 매 스텝 덮어쓰기 오차 누적 |

### V_cmd 계산 방식 변경

처음에는 전압 공간에서 직접 PD를 구성했으나 (주석으로 남아 있음):

```python
# 1차 시도 (폐기): 전압 공간 PD
# Kp_v = stiffness * R / (Kt * gr)
# V_cmd = Kp_v * e_q + Kd_v * e_qdot + Ke * omega
```

게인 스케일링이 직관적이지 않아, 최종적으로는 **역산 방식**으로 변경했다:

```python
# 최종: 토크 역산 → 전류 → 전압
tau_des = Kp * e_q + Kd * e_qdot          # 토크 공간 PD (기존과 동일)
I_des   = tau_des / (Kt * gr)              # 목표 전류 역산
V_cmd   = I_des * R + Ke * omega          # steady-state 전압 역산
```

이 방식은 기존 DCMotor의 토크 게인(Kp=25, Kd=0.6)을 그대로 재사용할 수 있어
기존 학습된 정책과 호환성이 높다.

### 결론 및 향후 방향

Isaac Lab(PhysX) 환경에서 전기모터 ODE를 제대로 적분하려면:

1. **ω를 ODE에서 빼고 전기 방정식만 적분** — PhysX가 ω를 담당하므로 `dI/dt`만 풀면 됨
2. **τ_load 불필요** — `dω/dt` 항 자체를 제거하면 τ_load 지연 문제도 사라짐
3. **PhysX view 주입 제거 가능** — projected joint forces를 읽을 필요가 없어짐

즉, 올바른 구조는 다음과 같다:

```python
dI/dt   = (V_cmd - R·I - Ke·ω_physx) / L   # ω는 PhysX에서 읽어옴
τ_joint = Kt · I · gr
```

이는 MuJoCo 버전(`play_mujoco.py`)이 처음에 구현했던 방식이다.
그러나 이렇게 하면 `dω/dt` 없이 `dI/dt`만 적분하는 단순 1차 ODE가 되며,
이는 물리적으로 **커플링된 시스템을 정확히 모델링하지 못한다**.
결국 전기-기계 연립 ODE의 의미가 사라지므로, 현재는 단순 DCMotor(PD 제어)를 사용한다.
