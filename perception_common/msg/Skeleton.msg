Header header
int32 user_id
int32 state # 0: new, 1: calibrating, 2: tracking, 4: out_of_sight, 5: lost
# Joint Indices
# 0: head
# 1: neck
# 2: left_shoulder
# 3: right_shoulder
# 4: left_elbow
# 5: right_elbow
# 6: left_hand
# 7: right_hand
# 8: torso
# 9: left_hip
# 10: right_hip
# 11: left_knee
# 12: right_knee
# 13: left_foot
# 14: right_foot
string[] joint_names
geometry_msgs/Point[] joint_positions_world
geometry_msgs/Point[] joint_positions_projective
geometry_msgs/Quaternion[] joint_orientations
float32[] confidences
int32 gesture # 0: none, 1: point-left, 2: point-right, 3: point-forward, 4: ok
