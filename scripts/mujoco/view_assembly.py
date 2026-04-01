"""
Go2 조립 뷰어 - 부품별 켜기/끄기
사용법: python view_assembly.py [--scene <xml_path>]

터미널 명령:
  l              전체 부품 목록 (현재 상태 포함)
  0~13           해당 body 토글 (켜기/끄기)
  all            전체 켜기
  none           전체 끄기
  FL / FR / RL / RR   해당 다리 전체 토글
  q              종료
"""

import argparse
import threading
import time
import mujoco
import mujoco.viewer
import numpy as np

DEFAULT_SCENE = "~/mujoco_menagerie/unitree_go2/go2_white_bg.xml"


def build_body_geom_map(model):
    """body 이름 → [geom 인덱스 리스트] 매핑 (visual geom만)"""
    bmap = {}
    for gi in range(model.ngeom):
        bid = model.geom_bodyid[gi]
        bname = model.body(bid).name
        if bname not in bmap:
            bmap[bname] = []
        bmap[bname].append(gi)
    return bmap


def set_body_alpha(model, bmap, bname, alpha, original_rgba):
    for gi in bmap.get(bname, []):
        model.geom_rgba[gi][3] = alpha if alpha > 0 else 0.0
        if alpha > 0:
            model.geom_rgba[gi][:3] = original_rgba[gi][:3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE,
                        help=f"MuJoCo XML 씬 파일 경로 (기본: {DEFAULT_SCENE})")
    args = parser.parse_args()

    import os
    scene_path = os.path.expanduser(args.scene)
    model = mujoco.MjModel.from_xml_path(scene_path)
    data  = mujoco.MjData(model)

    # 원본 rgba 저장
    original_rgba = model.geom_rgba.copy()

    bmap = build_body_geom_map(model)

    # world body 제외한 body 목록
    bodies = [model.body(i).name for i in range(model.nbody)
              if model.body(i).name != "world"]

    # 현재 켜진 상태 추적
    visible = {b: True for b in bodies}

    def print_status():
        print("\n=== Go2 Body 목록 ===")
        for i, b in enumerate(bodies):
            state = "ON " if visible[b] else "OFF"
            print(f"  {i:2d}. [{state}] {b}")
        print()

    def toggle(bname):
        if bname not in visible:
            return
        visible[bname] = not visible[bname]
        alpha = 1.0 if visible[bname] else 0.0
        set_body_alpha(model, bmap, bname, alpha, original_rgba)
        state = "ON" if visible[bname] else "OFF"
        print(f"  → {bname}: {state}")

    def set_all(on: bool):
        for b in bodies:
            visible[b] = on
            alpha = 1.0 if on else 0.0
            set_body_alpha(model, bmap, b, alpha, original_rgba)
        print(f"  → 전체 {'ON' if on else 'OFF'}")

    def toggle_leg(prefix):
        leg_bodies = [b for b in bodies if b.startswith(prefix)]
        if not leg_bodies:
            print(f"  '{prefix}'로 시작하는 body 없음")
            return
        # 하나라도 켜져 있으면 전체 끄기, 모두 꺼져 있으면 켜기
        any_on = any(visible[b] for b in leg_bodies)
        for b in leg_bodies:
            visible[b] = not any_on
            alpha = 1.0 if visible[b] else 0.0
            set_body_alpha(model, bmap, b, alpha, original_rgba)
        state = "OFF" if any_on else "ON"
        print(f"  → {prefix}_* 전체 {state}: {leg_bodies}")

    print_status()
    print("명령어: 번호=토글, all=전체ON, none=전체OFF, FL/FR/RL/RR=다리토글, l=목록, q=종료")

    running = [True]

    def input_loop():
        while running[0]:
            try:
                cmd = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                running[0] = False
                break

            if cmd in ("q", "quit"):
                running[0] = False
            elif cmd == "l":
                print_status()
            elif cmd == "all":
                set_all(True)
            elif cmd in ("none", "off"):
                set_all(False)
            elif cmd in ("fl", "fr", "rl", "rr"):
                toggle_leg(cmd.upper())
            elif cmd.isdigit():
                i = int(cmd)
                if 0 <= i < len(bodies):
                    toggle(bodies[i])
                else:
                    print(f"  0~{len(bodies)-1} 범위로 입력하세요")
            elif cmd in [b.lower() for b in bodies]:
                match = next(b for b in bodies if b.lower() == cmd)
                toggle(match)
            else:
                print("  번호, body이름, all, none, FL/FR/RL/RR, l, q 중 입력하세요")

    t = threading.Thread(target=input_loop, daemon=True)
    t.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.2
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        while viewer.is_running() and running[0]:
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.02)

    running[0] = False


if __name__ == "__main__":
    main()
