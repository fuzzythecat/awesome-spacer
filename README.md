# awesome-spacer

awesome-spacer is a project for automatic Korean word spacing, using **TensorFlow 2** + **Keras**.

## Requirements

- `Python 3.6`
- `TensorFlow 2.0`
- `NumPy`
- `tqdm`
- `scikit-learn`

## Getting Started
* [awesome-spacer-train-colab.ipynb](notebooks/awesome-spacer-train-colab.ipynb): You can use this to train your own model on Sejong corpus with Google Colab. To train on custom datasets, try using CLI instead.  

* [awesome-spacer-test-colab.ipynb](notebooks/awesome-spacer-test-colab.ipynb): You can use this to test pre-trained models trained on Sejong corpus. Weight links and corresponding model configurations are included in the notebook.

## Train

To train the model, you should provide the path to your dataset.
You can import this module and train in Jupyter Notebook(see notebooks for example), 
or train from CLI. 

Currently training with CPU is **not** supported.
```
# Train a new model from scratch. 
python train.py --data_path path/to/dataset --gpu_list 0

# Continue training from pre-trained model.
python train.py --data_path path/to/dataset --gpu_list 0 --trained_model path/to/weights.h5

# Pass GPU ids for multi-GPU training.
python train.py --data_path path/to/dataset --gpu_list 0,1,2,3
```

## Examples

See [Jupyter Notebook examples](notebooks/awesome-spacer-test-colab.ipynb) for usage. 
* Before
```
내가그린기린그림은긴기린그린그림이고,네가그린기린그림은길지않은기린그린그림이다.
```
```
영국의철학자인화이트헤드는"서양의2000년철학은모두플라톤의각주에불과하다"라고말했으며,
시인에머슨은"철학은플라톤이고,플라톤은철학"이라평하였는데,플라톤은소크라테스의수제자이다. 
플라톤이20대인시절,스승소크라테스가민주주의에의해끝내사형당하는것을보고크게분개했으며, 
이는그의귀족주의"철인정치"지지의큰계기가되었다.
```
* After
```
내가 그린 기린 그림은 긴 기린 그린 그림이고, 네가 그린 기린 그림은 길지 않은 기린 그린 그림이다.
```
```
영국의 철학자인 화이트헤드는 "서양의 2000년 철학은 모두 플라톤의 각주에 불과하다"라고 말했으며,
시인 에머슨은 "철학은 플라톤이고, 플라톤은 철학"이라 평하였는데, 플라톤은 소크라테스의 수제자이다. 
플라톤이 20대인 시절, 스승 소크라테스가 민주주의에 의해 끝내 사형당하는 것을 보고 크게 분개했으며, 
이는 그의 귀족주의 "철인정치"지지의 큰 계기가 되었다.
```

## Workshop Materials
* Hands-on Workshop @ [2019 Global AI Bootcamp - Seoul](https://festa.io/events/772)
