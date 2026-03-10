"""
Electric motor actuator with thermal model for PHM research.
Go2 quadruped robot - 12 joints total.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.utils.types import ArticulationActions
from isaaclab.sim import SimulationContext


class ElectricMotor(DCMotor):
    """
    DC 모터 기반 커스텀 액추에이터.
    PD 제어 + 속도-토크 클리핑(DCMotor 상속)에 전기 모델과 열 모델을 추가.

    고장 모드:
        faulty=True일 때, fault_start_time(초)부터 fault_joint_indices에 해당하는 관절의
        R_thermal을 선형으로 증가시킴 (R_thermal_init → R_thermal_fault, fault_duration초에 걸쳐).
    """

    cfg: ElectricMotorCfg

    def __init__(self, cfg: ElectricMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.T_winding = torch.full(
            (self._num_envs, self.num_joints),
            cfg.T_ambient,
            dtype=torch.float32,
            device=self._device,
        )

        # 로깅용 버퍼
        self.current     = torch.zeros(self._num_envs, self.num_joints, dtype=torch.float32, device=self._device)
        self.voltage     = torch.zeros(self._num_envs, self.num_joints, dtype=torch.float32, device=self._device)
        self.temperature = torch.full(
            (self._num_envs, self.num_joints),
            cfg.T_ambient,
            dtype=torch.float32,
            device=self._device,
        )

        # 관절별 R_thermal 버퍼 (초기값으로 채움)
        self._R_thermal = torch.full(
            (self._num_envs, self.num_joints),
            cfg.R_thermal,
            dtype=torch.float32,
            device=self._device,
        )

    def reset(self, env_ids):
        super().reset(env_ids)
        self.T_winding[env_ids] = self.cfg.T_ambient
        self._R_thermal[env_ids] = self.cfg.R_thermal

    def _update_fault_R_thermal(self, sim_time: float):
        """
        fault_start_time 이후부터 fault_duration초에 걸쳐
        지정된 관절의 R_thermal을 선형으로 증가시킴.
        """
        if not self.cfg.faulty:
            return

        t0 = self.cfg.fault_start_time
        t1 = t0 + self.cfg.fault_duration
        R0 = self.cfg.R_thermal
        R1 = self.cfg.R_thermal_fault

        if sim_time < t0:
            r_val = R0
        elif sim_time >= t1:
            r_val = R1
        else:
            # 선형 보간
            alpha = (sim_time - t0) / self.cfg.fault_duration
            r_val = R0 + alpha * (R1 - R0)

        # 고장 관절 인덱스에만 적용
        for j_idx in self.cfg.fault_joint_indices:
            if j_idx < self.num_joints:
                self._R_thermal[:, j_idx] = r_val

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:

        # ------------------------------------------------------------------
        # 1. DCMotor: PD 토크 계산 + velocity-torque curve 클리핑
        # ------------------------------------------------------------------
        control_action = super().compute(control_action, joint_pos, joint_vel)
        tau_joint = self.applied_effort.clone()

        # ------------------------------------------------------------------
        # 2. 관절 → 모터 공간 변환
        # ------------------------------------------------------------------
        gr = self.cfg.gear_ratio
        tau_motor   = tau_joint  / gr
        omega_motor = joint_vel  * gr

        # ------------------------------------------------------------------
        # 3. 전류 역산
        # ------------------------------------------------------------------
        I = tau_motor / self.cfg.Kt

        # ------------------------------------------------------------------
        # 4. 전압 역산
        # ------------------------------------------------------------------
        V = I * self.cfg.R + self.cfg.Ke * omega_motor
        V = torch.clamp(V, -self.cfg.V_max, self.cfg.V_max)

        # ------------------------------------------------------------------
        # 5. 고장 모드: R_thermal 업데이트
        # ------------------------------------------------------------------
        sim_time = SimulationContext.instance().current_time
        self._update_fault_R_thermal(sim_time)

        # ------------------------------------------------------------------
        # 6. 열 모델 업데이트 (관절별 R_thermal 사용)
        # ------------------------------------------------------------------
        P_loss = I ** 2 * self.cfg.R
        dt = SimulationContext.instance().get_physics_dt()
        dT = (P_loss - (self.T_winding - self.cfg.T_ambient) / self._R_thermal) * dt / self.cfg.C_thermal
        self.T_winding = self.T_winding + dT

        # ------------------------------------------------------------------
        # 7. 온도에 따른 실제 저항
        # ------------------------------------------------------------------
        R_actual = self.cfg.R * (1.0 + self.cfg.alpha * (self.T_winding - self.cfg.T_ambient))

        # ------------------------------------------------------------------
        # 8. 실제 모터 토크 재계산
        # ------------------------------------------------------------------
        I_actual         = (V - self.cfg.Ke * omega_motor) / R_actual
        tau_motor_actual = I_actual * self.cfg.Kt

        # ------------------------------------------------------------------
        # 9. 모터 → 관절 공간 역변환
        # ------------------------------------------------------------------
        tau_joint_actual = tau_motor_actual * gr

        # 로깅
        self.current     = I_actual
        self.voltage     = V
        self.temperature = self.T_winding.clone()

        self.applied_effort                 = tau_joint_actual
        control_action.joint_efforts        = tau_joint_actual
        control_action.joint_positions      = None
        control_action.joint_velocities     = None

        return control_action


@dataclass
class ElectricMotorCfg(DCMotorCfg):
    """ElectricMotor 액추에이터 설정."""

    class_type: type = ElectricMotor

    # 기어비
    gear_ratio: float = 6.33

    # 전기 파라미터
    Kt: float = 0.128
    Ke: float = 0.128
    R: float = 0.3
    V_max: float = 24.0

    # 열 파라미터
    alpha: float = 0.00393
    T_ambient: float = 25.0
    R_thermal: float = 2.0
    C_thermal: float = 5.0

    # ── 고장 파라미터 ──────────────────────────────────────────
    faulty: bool = False
    fault_start_time: float = 5.0       # 고장 시작 시각 [s]
    fault_duration: float = 5.0         # R_thermal 증가에 걸리는 시간 [s]
    R_thermal_fault: float = 10.0       # 고장 시 최종 R_thermal 값
    # Go2 calf 관절 인덱스: FL_calf=2, FR_calf=5, RL_calf=8, RR_calf=11
    fault_joint_indices: list = field(default_factory=lambda: [10])