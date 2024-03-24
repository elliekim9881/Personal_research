# realsense sdk to macOS(삽질)
> macbook pro intel & m1 (sonoma 14.4)<br>
> intel depth camera 435, 455<br>
> [pyrealsense2 ver2.56.2](https://pypi.org/project/pyrealsense2/#files)

## case
> case1. m1 & pyrealsense2 ver2.56.2 / **fail**<br>
> case2. intel & pyrealsense2 ver2.56.2/ **fail**<br>
> case3. intel & pyrealsense2 ver2.56.2(re-build)/ **fail**<br>
> case4. m1 & pyrealsense2 ver2.56.2(re-build)/ **fail**<br>
> case5. m1(vrBox, ubuntu)& pyrealsense2 ver2.56.2/ **fail**<br>
> case6. intel(vrBox, ubuntu)& pyrealsense2 ver2.56.2 / **Success**<br>

## case1 & 2 & 3 & 4
- segmentation falut<br>
    - 프로그램이 메모리를 잘못 액세스했을 때 발생하는 심각한 오류입니다. 이는 일반적으로 프로그램이 잘못된 **메모리 위치**를 읽거나 쓰려고 시도할 때, 운영 체제에 의해 강제 종료됨을 의미<br>
    = [build Realsense for macOS](https://lightbuzz.com/realsense-macos/)
        - problem 1. macos, realsense SDK 버전 호환 문제<br>
            = 서치 결과 SDK rebuild 해서 설치함.<br>
            result. fail

        - problem 2. m1, realsense SDK 버전 호환 문제<br>
            = m1에서 소프트웨어 호환 문제 아직 존재. intel로 교체, SDK rebuild<br>
            result. fail

## case 5 & 6
- m1 에서 virtualbox & ubuntu 설치 시도. <br>
 result. fail(해당 페이지 공지로는 m1에서도 동작한다고 하나, 여전히 실행 불가)

-  intel 로 교체 후 맥 virtualbox & ubuntu 설치<br>
 result. success <br>
 ![(Personal_research\sensor_fusion\realsense_01.jpeg)](https://github.com/elliekim9881/Personal_research/blob/main/sensor_fusion/realsense_01.jpeg)
    - problem<br>
    virtualbox ubuntu 에서 기기 인식 불가<br>
    = sudo virtualbox 실행. 인식 성공.<br>
    ![(Personal_research\sensor_fusion\realsense_02.jpeg)](https://github.com/elliekim9881/Personal_research/blob/main/sensor_fusion/realsense_02.jpeg)