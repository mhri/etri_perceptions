int32 index
sensor_msgs/RegionOfInterest face_roi
sensor_msgs/RegionOfInterest left_eye_roi
sensor_msgs/RegionOfInterest right_eye_roi
sensor_msgs/RegionOfInterest body_roi
# gender = 0: Unknown | 1: Male | 2: Female
int32 gender
# age
int32 age
# eye_glasses = 0: Unknown | 1: Yes | 2: No
int32 eye_glasses
int32[] stasm_landmarks
string cloth_color
# hair_length = 'long' | 'short'
string hair_length
# eye_size = 0: small | 1: big
int32 eye_size
# face_direction = 0: front, 1: right, 2: left
int32 face_direction
# face_learned = 1 if this face is a new one!
int8 face_learned
int8 face_appeared
int8 learned
string image_path
# face id: face enrollment id
string face_id
# face recognition results
string person_id
# face tracking result
# face_id == prev_face_id
# the real name of the identified person
string name
string session_face_id
# confidence
float32 session_confidence
float32 person_confidence
# cognitive status
string cognitive_status
# face position in 3d
geometry_msgs/Point32 face_pos3d
string frame_id
sensor_msgs/RegionOfInterest face_pos3droi
# gesture
int32 gesture # 0: none, 1: point-left, 2: point-right, 3: point-forward, 4: ok
# saliency
float32 saliency
# face detected?
int32 face_detected
# emotion from facial expression
int32 emotion # 0: neutral, 1: smile/happy
float32 emotion_prob # emotion recognition probability
# lips opened or closed
int8 mouth_opened # 0: false, 1: true
# tracking information
string trk_id
float32 trk_pos_x
float32 trk_pos_y
float32 trk_pos_z
float32 trk_height
float32 trk_confidence
float32 trk_age
int32 trk_bbox_x
int32 trk_bbox_y
int32 trk_bbox_width
int32 trk_bbox_height
