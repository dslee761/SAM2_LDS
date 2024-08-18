### Setup Enviroment (SAM2 환경 구축)

```jsx
conda create --name sam2 python=3.12
conda activate sam2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e ".[demo]"
```

- `pip install -e ".[demo]"` : [setup.py](http://setup.py) 의 설정값을 읽고, 해당 설정값을 설정.
    - `requirement.txt` 를 쓰던 옛날 방식에서 바뀐것으로 추측. (리서치 필요.)
    - 환경이 불안정한지, 여러 값들이 git 내부 안에 있는것을 확인.
        - 환경 구축을 위해서는 git을 clone이 필수. → 그 후에 코드 따오는것은 같음.
- SAM 모델 리스트
    - 메인 주소 :  `https://dl.fbaipublicfiles.com/segment_anything_2/072824/`
    - sam2_hiera_tiny.pt
    - sam2_hiera_small.pt
    - sam2_hiera_base_plus.pt
    - sam2_hiera_large.pt
    
    ```jsx
    wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
    ```
    
- 주요 변경사항
    - SAM, SAM2 주요 골자는 비슷한 것으로 확인.
    - import 하는 class 자체가 SAM2 인 것으로 확인.
    - 그러면, predict 하고 나서, 그 후에 어떻게 처리되는지 지연, delay를 확인할 필요가 있음.
- 주요 확인 사항
    - SAM1 에서 사용한 코드를 import만 바꿔도 사용이 가능할까? → 테스트 필요.
- 귀찮은 사항.
    - 이놈들 config.yaml이랑 model name이랑 자동 연동 안해놨다. 귀찮게.