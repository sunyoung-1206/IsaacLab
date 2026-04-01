"""
Go2 부품 개별 뷰어
사용법: python view_parts.py [--assets_dir <path>]
  번호 입력 → 해당 부품 뷰어 열기 → 창 닫으면 다음 선택
"""

import argparse
import os
import mujoco
import mujoco.viewer

DEFAULT_ASSETS_DIR = "~/mujoco_menagerie/unitree_go2/assets"

ASSETS_DIR = os.path.expanduser(DEFAULT_ASSETS_DIR)
parts: list = []


def show_part(obj_file):
    obj_path = os.path.join(ASSETS_DIR, obj_file)
    xml = f"""
<mujoco>
  <asset>
    <mesh name="part" file="{obj_path}"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="-0.3 -0.3 -1" directional="true"
           diffuse="0.9 0.9 0.9" specular="0.3 0.3 0.3"/>
    <light pos="2 2 2" directional="false" diffuse="0.4 0.4 0.4"/>
    <body>
      <geom type="mesh" mesh="part" rgba="0.75 0.78 0.85 1"/>
    </body>
  </worldbody>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)   # 창 닫을 때까지 블로킹


def print_parts():
    print("\n=== Go2 부품 목록 ===")
    for i, p in enumerate(parts):
        print(f"  {i:2d}. {p}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets_dir", default=DEFAULT_ASSETS_DIR,
                        help=f"Go2 OBJ assets 디렉토리 (기본: {DEFAULT_ASSETS_DIR})")
    args = parser.parse_args()

    global ASSETS_DIR, parts
    ASSETS_DIR = os.path.expanduser(args.assets_dir)
    parts = sorted([f for f in os.listdir(ASSETS_DIR) if f.endswith(".obj")])

    print_parts()
    while True:
        cmd = input("번호 입력 (l=목록, q=종료): ").strip()
        if cmd in ("q", "quit", "exit"):
            break
        elif cmd in ("l", "list"):
            print_parts()
        elif cmd.isdigit():
            i = int(cmd)
            if 0 <= i < len(parts):
                print(f"  → {parts[i]} 열기...")
                show_part(parts[i])
            else:
                print(f"  0~{len(parts)-1} 범위로 입력하세요")
        else:
            print("  숫자, l, q 중 입력하세요")


if __name__ == "__main__":
    main()
