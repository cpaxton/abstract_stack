"""Browser teleoperation host for MABEL in the shelf-stocking simulation.

The host follows the Quest Teleop v1 envelope and state machine
(``docs/quest_teleop/README.md``) with a web-specific ``web_teleop_v1`` profile:
absolute end-effector, grip, head, lift and torso intents instead of tracked
human skeletons. See ``docs/web_teleop/README.md``.
"""
