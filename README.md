### Setup Enviroment (SAM2 환경 구축)

```jsx
conda create --name sam2 python=3.12
conda activate sam2
pip install -e ".[demo]"
```

- `pip install -e ".[demo]"` : [setup.py](http://setup.py) 의 설정값을 읽고, 해당 설정값을 설정.
    - `requirement.txt` 를 쓰던 옛날 방식에서 바뀐것으로 추측. (리서치 필요.)