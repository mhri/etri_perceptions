# social_perception_core #

지각 수준(perception_core)의 인식 결과를 받아 사회적 의미를 지니는 인식 결과를 제공하는 기능을 모은 패키지이다.

## person_identity_cognition.py ##

### 기능 개요 ###

사람의 존재와 특성을 판단한다. 상호작용할 사람이 존재한다고 판단하면 얼굴 영역의 중심 좌표를 제공하고, 상의 색상, 성별 등의 개인 특성 정보를 함께 제공한다.

### 참고 ###

face_detector 패키지의 social_face_detector.py 노드를 수정하여 제작하였다. 노드 이름을 수정하고, 메모리 템플릿을 변경했으며, 인식 결과들을 추가했다.

### Subscribed Topics ###

* /mhri/perception_core/cloth_color_detector/persons (mhri_common/PersonPerceptArray)

### 이벤트 구조 ###

person_identity_detector가 메모리에 기록하는 템플릿은 person_identified이다. person_identified의 구조는 memory_monitor/config/memory_template.yaml에 정의되어 있다. 각 필드의 타입과 의미는 다음과 같다.

* identified (type=boolean): 사람 존재 여부. true이면 사람이 등장하여 존재하는 상황이고, false이면 있던 사람이 사라져버린 상황이다.
* face_pos (type=list of lists): 얼굴 영역의 중심 좌표 목록이다. face_pos[i]는 [x,y]와 같이 x 좌표와 y 좌표값의 목록이다.
* name (type=list of string): name[i]는 i번째 사람의 얼굴인식 결과로서 사람의 이름 또는 아이디이다. 인식 결과가 없는 경우 공백 문자열이다.
* confidence (type=list of floats): confidence[i]는 i번째 사람의 얼굴인식 신뢰도이다. 인식 결과가 없는 경우 0이다.
* gender (type=list of ints): gender[i]는 i번째 사람의 성별 인식 결과이다. 값이 0이면 여성, 1이면 남성이다.
* cloth_color (type=list of strings): cloth_color[i]는 i번째 사람의 상의 색상 이름이다. 인식 결과가 없는 경우 공백 문자열이다.
* eyeglasses (type=list of booleans): eyeglasses[i]는 i번째 사람의 안경 착용 여부이다. 안경 착용 시 값은 True이다. 반대의 경우 False이다.
* height (type=list of floats): height[i]는 i번째 사람의 신장이다. 인식 결과가 없는 경우 값은 0이다.
* hair_style (type=list of strings): hair_style[i]는 i번째 사람의 머리카락 스타일이다. 인식 결과가 없으면 공백 문자열, 긴머리는 'long', 짧은 머리는 'short'이다.

## voice_activity_cognition.py ##

### 기능 개요 ###

사용자의 음성 활동 여부를 판단한다. 사용자가 발화를 시작할 때와 종료했을 때 memory_monitor를 통해 이벤트를 발생한다.

### Subscribed Topics ###

* /mhri/perception_core/speech_detector_for_nao/speech (mhri_common/SpeechPercept)

### 이벤트 구조 ###
person_identity_detector가 메모리에 기록하는 템플릿은 voice_activity_detected이다. voice_activity_detected의 구조는 memory_monitor/config/memory_template.yaml에 정의되어 있다. 각 필드의 타입과 의미는 다음과 같다.

* activity (type=integer): 음성 활동 유무. 값이 0이면 음성 활동이 없는 상태이고 1이면 음성 활동이 존재하는 상태이다.
* phrase (type=string): 음성 인식된 결과가 존재하는 경우 문장이 저장된다. 인식 결과가 없으면 공백 문자열이다.
* confidence (type=float): 음성 인식 결과의 신뢰도 값이다. (default = 0)