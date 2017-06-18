# perception_face

얼굴인식 등 얼굴 관련 영상 처리를 수행하는 패키지이다.

## Requirements

### openface (https://cmusatyalab.github.io/openface/)
얼굴인식과 초구면 구분을 위해 활용하는 오픈소스 프로젝트이다.
설치 절차는 https://cmusatyalab.github.io/openface/setup 을 참고.

### dlib
얼굴 검출, 얼굴특징점 검출에 활용한다.

### opencv
다양한 영상처리에 활용한다.

### keras (https://keras.io/)
표정인식을 위한 딥러닝 프레임워크이다. keras를 설치하면 theano가 함께 설치된다. 표정인식 모델은 theano를 기반으로 훈련되어 있으므로 theano가 필요하다.

### numpy
다양한 수치 계산에 활용한다.

## 설정
keras 설치 후  ~/.keras/keras.json 문서의 내용 중 image_dim_ordering의 값은 "th", backend의 값은 "theano"로 변경해야 한다.