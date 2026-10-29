from pathlib import Path
import mujoco
import numpy as np

xml = Path("models/ur3e_camera/scene_pick.xml")
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print("\n========== ACTUATOR / STATE ANALYSIS ==========")
print("Timestep:", model.opt.timestep, "s")
print("Integrator:", model.opt.integrator)
print("Robot joints:", 6)
print("Robot position states:", 6)
print("Robot velocity states:", 6)

print("\n--- ACTUATOR CONFIGURATION ---")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    trnid = model.actuator_trnid[i]
    joint_id = int(trnid[0])
    joint_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
    )
    print(f"\n{i}: {name}")
    print("  Joint:", joint_name)
    print("  Transmission type:", int(model.actuator_trntype[i]))
    print("  Dynamics type:", int(model.actuator_dyntype[i]))
    print("  Gain type:", int(model.actuator_gaintype[i]))
    print("  Bias type:", int(model.actuator_biastype[i]))
    print("  Gain parameters:", model.actuator_gainprm[i, :3])
    print("  Bias parameters:", model.actuator_biasprm[i, :3])
    print("  Control range:", model.actuator_ctrlrange[i])
    print("  Control limited:", bool(model.actuator_ctrllimited[i]))

print("\n--- KEYFRAME JOINT POSITIONS ---")
for i in range(6):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    qadr = int(model.jnt_qposadr[i])
    dadr = int(model.jnt_dofadr[i])
    print(
        f"{name}: q={data.qpos[qadr]:.5f} rad, "
        f"qd={data.qvel[dadr]:.5f} rad/s"
    )

print("\n--- JOINT LIMITS ---")
for i in range(6):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(name, model.jnt_range[i], 
          "limited =", bool(model.jnt_limited[i]))

print("\n--- ACTUAL CONTROL INPUTS AT KEYFRAME ---")
print("ctrl =", np.round(data.ctrl, 5))
print("===============================================\n")
