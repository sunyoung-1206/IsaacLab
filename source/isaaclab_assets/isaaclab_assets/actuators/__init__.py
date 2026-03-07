"""
isaaclab_assets/actuators 패키지.
커스텀 액추에이터 모델들을 모아두는 곳.
"""

from .electric_motor import ElectricMotor, ElectricMotorCfg

__all__ = ["ElectricMotor", "ElectricMotorCfg"]