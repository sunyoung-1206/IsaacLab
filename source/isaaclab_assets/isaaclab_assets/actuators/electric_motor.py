from __future__ import annotations

from dataclasses import dataclass
import torch
from torchdiffeq import odeint

from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.utils.types import ArticulationActions
from isaaclab.sim import SimulationContext


class ElectricMotor(DCMotor):
    """
    전기-기계 연립 ODE를 dopri5로 적분하는 전기 모터 액추에이터.

    State variables:
        I     : 전류 [A]
        omega : 모터 각속도 [rad/s]  ← 매 스텝 PhysX로 동기화

    ODE:
        dI/dt     = (V_cmd - R*I - Ke*omega) / L
        domega/dt = (Kt*I*gr - B*omega - tau_load) / J

    tau_load:
        root_physx_view.get_dof_projected_joint_forces() 로 읽어옴.
        외부에서 inject_physx_view()로 주입 필요.
        주입 전에는 Kt*I*gr 근사값 사용 (fallback).

    스텝 처리 순서:
        1. V_cmd 계산 (전압 공간 PD)
        2. tau_load 읽기 (PhysX projected joint forces, 이전 스텝 값)
        3. ODE 적분 (이전 스텝 보정 I, 동기화된 omega, tau_load)
        4. omega 동기화 (PhysX), I 보정 (새 omega 기준 steady-state)
        5. 토크 계산 (ODE 결과 I 기준) -> PhysX

    외부 주입 방법 (train.py / play 스크립트):
        robot = env.unwrapped.scene["robot"]
        for actuator in robot.actuators.values():
            if hasattr(actuator, 'inject_physx_view'):
                actuator.inject_physx_view(robot.root_physx_view)

    고장 모드:
        faulty=True 시 fault_start_time부터 fault_duration에 걸쳐
        지정 관절의 R을 R_fault까지 선형 증가.
        V_cmd는 고장을 모르므로 R 증가 -> I 감소 -> tau 감소.

    권장 세팅:
        sim.dt     = 0.005   (200Hz)
        decimation = 4       (policy 50Hz)
    """

    cfg: ElectricMotorCfg

    def __init__(self, cfg: ElectricMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        num_envs   = self._num_envs
        num_joints = self.num_joints

        # 전기-기계 state
        self.I          = torch.zeros(num_envs, num_joints, dtype=torch.float32, device=self._device)
        self.omega      = torch.zeros(num_envs, num_joints, dtype=torch.float32, device=self._device)
        self.omega_prev = torch.zeros(num_envs, num_joints, dtype=torch.float32, device=self._device)
        self.tau_load   = torch.zeros(num_envs, num_joints, dtype=torch.float32, device=self._device)

        # 관절별 저항 (고장 시 변화)
        self._R = torch.full(
            (num_envs, num_joints), cfg.R,
            dtype=torch.float32, device=self._device
        )

        # 로깅용
        self.current = torch.zeros_like(self.I)
        self.voltage  = torch.zeros_like(self.I)

        # PhysX view (외부 주입)
        self._root_physx_view = None
        # print(f"velocity_limit: {self.velocity_limit}")

    def inject_physx_view(self, root_physx_view):
        """외부에서 root_physx_view 주입. env 초기화 후 호출."""
        self._root_physx_view = root_physx_view

    def reset(self, env_ids):
        super().reset(env_ids)
        self.I[env_ids]          = 0.0
        self.omega[env_ids]      = 0.0
        self.omega_prev[env_ids] = 0.0
        self.tau_load[env_ids]   = 0.0
        self._R[env_ids]         = self.cfg.R

    def _get_tau_load(self) -> torch.Tensor:
        """
        PhysX에서 projected joint forces 읽어서 tau_load 반환.
        주입 전이면 Kt*I*gr 근사값 반환 (fallback).

        get_dof_projected_joint_forces() shape: (num_envs, num_joints_total)
        """
        if self._root_physx_view is None:
            # fallback: 모터 토크 = 부하 토크 근사
            return self.cfg.Kt * self.I * self.cfg.gear_ratio

        try:
            forces = self._root_physx_view.get_dof_projected_joint_forces()
            if not isinstance(forces, torch.Tensor):
                forces = torch.tensor(forces, dtype=torch.float32, device=self._device)
            else:
                forces = forces.to(self._device)
            # 우리 관절만 슬라이싱
            return forces[:, self._joint_indices]
        except Exception:
            return self.cfg.Kt * self.I * self.cfg.gear_ratio

    def _update_fault_R(self, sim_time: float):
        """고장 시 저항값 선형 증가."""
        if not self.cfg.faulty:
            return

        t0 = self.cfg.fault_start_time
        t1 = t0 + self.cfg.fault_duration
        R0 = self.cfg.R
        R1 = self.cfg.R_fault

        if sim_time < t0:
            r_val = R0
        elif sim_time >= t1:
            r_val = R1
        else:
            alpha = (sim_time - t0) / self.cfg.fault_duration
            r_val = R0 + alpha * (R1 - R0)

        for j_idx in self.cfg.fault_joint_indices:
            if j_idx < self.num_joints:
                self._R[:, j_idx] = r_val

    def _make_ode(self, V_cmd: torch.Tensor, tau_load: torch.Tensor):
        """
        ODE 함수 생성. V_cmd, tau_load는 적분 구간 동안 상수 (ZOH).

        state shape: (num_envs, num_joints, 2)
            [..., 0] = I
            [..., 1] = omega
        """
        R  = self._R.clone()
        L  = self.cfg.L
        Ke = self.cfg.Ke
        Kt = self.cfg.Kt
        B  = self.cfg.B
        J  = self.cfg.J
        gr = self.cfg.gear_ratio
        vel_lim = self.velocity_limit * gr

        def ode_func(t, state):
            I     = state[..., 0]
            omega = torch.clamp(state[..., 1], -vel_lim, vel_lim)
            dI     = (V_cmd - R * I - Ke * omega) / L
            domega = (Kt * I * gr - B * omega - tau_load) / J
            return torch.stack([dI, domega], dim=-1)

        return ode_func

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:

        gr       = self.cfg.gear_ratio
        dt       = SimulationContext.instance().get_physics_dt()
        sim_time = SimulationContext.instance().current_time

        # ------------------------------------------------------------------
        # 1. V_cmd: 전압 공간 PD
        # ------------------------------------------------------------------
        # Kp_v = self.cfg.stiffness * self.cfg.R / (self.cfg.Kt * gr)
        # Kd_v = self.cfg.damping   * self.cfg.R / (self.cfg.Kt * gr)

        e_q    = control_action.joint_positions  - joint_pos
        e_qdot = control_action.joint_velocities - joint_vel

        omega_physx = joint_vel * gr

        # V_cmd = Kp_v * e_q + Kd_v * e_qdot + self.cfg.Ke * omega_physx
        # V_cmd = torch.clamp(V_cmd, -self.cfg.V_max, self.cfg.V_max)
        # 목표 토크 (ImplicitActuator와 동일한 스케일)
        tau_des = self.cfg.stiffness * e_q + self.cfg.damping * e_qdot
        tau_des = torch.clamp(tau_des, -self.cfg.effort_limit, self.cfg.effort_limit)

        # 목표 전류
        I_des = tau_des / (self.cfg.Kt * gr)

        # 목표 전압 (steady-state 역산)
        V_cmd = I_des * self._R + self.cfg.Ke * omega_physx
        V_cmd = torch.clamp(V_cmd, -self.cfg.V_max, self.cfg.V_max)

        # ------------------------------------------------------------------
        # 2. 고장 R 업데이트
        # ------------------------------------------------------------------
        self._update_fault_R(sim_time)

        # ------------------------------------------------------------------
        # 3. tau_load 읽기 (PhysX, 이전 스텝 값)
        # ------------------------------------------------------------------
        self.tau_load = self._get_tau_load()

        # ------------------------------------------------------------------
        # 4. ODE 적분
        # ------------------------------------------------------------------
        state0 = torch.stack([self.I, self.omega], dim=-1)

        tau_e    = self.cfg.L / self.cfg.R  # electrical time constant ~0.33ms
        n_points = max(3, int(dt / (tau_e * 0.5)))
        t_span   = torch.linspace(0.0, dt, n_points, dtype=torch.float32, device=self._device)

        ode_func = self._make_ode(V_cmd, self.tau_load)
        state1   = odeint(ode_func, state0, t_span, method='dopri5', rtol=1e-4, atol=1e-6)[-1]

        I_ode = state1[..., 0]
        I_max = self.cfg.V_max / self._R
        I_ode = torch.clamp(I_ode, -I_max, I_max)

        # ------------------------------------------------------------------
        # 5. omega 동기화 (PhysX), I 보정 (다음 스텝 초기조건)
        # ------------------------------------------------------------------
        self.omega_prev = omega_physx.clone()
        self.omega      = omega_physx.clone()
        # self.I = torch.clamp(                         
        #     (V_cmd - self.cfg.Ke * self.omega) / self._R,
        #     -I_max, I_max
        # )
        self.I = I_ode.clone()
        # ------------------------------------------------------------------
        # 6. 토크 계산 -> PhysX
        # ------------------------------------------------------------------
        tau_joint = self.cfg.Kt * I_ode * gr
        tau_joint = torch.clamp(tau_joint, -self.cfg.effort_limit, self.cfg.effort_limit)

        # ------------------------------------------------------------------
        # 7. 로깅용 버퍼
        # ------------------------------------------------------------------
        self.current = I_ode.clone()
        self.voltage  = V_cmd.clone()

        self.computed_effort = tau_joint.clone()
        self.applied_effort  = tau_joint

        control_action.joint_efforts    = tau_joint
        control_action.joint_positions  = None
        control_action.joint_velocities = None
        # if sim_time < 0.1:
        #     print(f"tau_load mean: {self.tau_load.abs().mean():.3f}, max: {self.tau_load.abs().max():.3f}")
        #     print(f"I_ode mean: {I_ode.abs().mean():.3f}, max: {I_ode.abs().max():.3f}")
        #     print(f"tau_joint mean: {tau_joint.abs().mean():.3f}")
        #     max_idx = I_ode.abs().argmax()
        #     env_idx = max_idx // self.num_joints
        #     joint_idx = max_idx % self.num_joints
        #     print(f"I_ode max env={env_idx}, joint={joint_idx}, val={I_ode.abs().max():.1f}")
        #     print(f"V_cmd at max: {V_cmd[env_idx, joint_idx]:.3f}")
        #     print(f"omega at max: {self.omega[env_idx, joint_idx]:.3f}")
        #     print(f"tau_load at max: {self.tau_load[env_idx, joint_idx]:.3f}")
        #     print(f"self.I min: {self.I.min():.3f}, max: {self.I.max():.3f}")

        return control_action


@dataclass
class ElectricMotorCfg(DCMotorCfg):
    """ElectricMotor 설정."""

    class_type: type = ElectricMotor

    # 기어비
    gear_ratio: float = 6.33

    # 전기 파라미터
    Kt: float    = 0.128   # 토크 상수 [N·m/A]
    Ke: float    = 0.128   # 역기전력 상수 [V·s/rad]
    R: float     = 0.3     # 권선 저항 [Ω]
    L: float     = 7.5*1e-4    # 인덕턴스 [H] 원래는 7.5 안주고 줬었음
    V_max: float = 24.0    # 최대 공급 전압 [V]

    # 기계 파라미터
    J: float = 0.05   # 모터 관성 모멘트 [kg·m²]
    B: float = 1e-3   # 점성 마찰 계수 [N·m·s/rad]

    # 고장 파라미터
    faulty: bool            = False
    fault_start_time: float = 5.0
    fault_duration: float   = 5.0
    R_fault: float          = 3.0
    fault_joint_indices: list = None

    def __post_init__(self):
        if self.fault_joint_indices is None:
            self.fault_joint_indices = [10]  # 기본값: RL_calf