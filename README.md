# Rede Neural para Classificacao de Digitos

Este projeto apresenta a construcao de uma rede neural multicamadas (MLP) utilizando PyTorch para classificar imagens de digitos manuscritos do conjunto de dados MNIST.

## Objetivo

Treinar um modelo capaz de identificar qual digito, de 0 a 9, aparece em uma imagem de tamanho 28 x 28 pixels.

## Etapas realizadas

1. Importacao das bibliotecas `PyTorch`, `torchvision`, `NumPy` e `Matplotlib`.
2. Download e carregamento do conjunto MNIST para treinamento e validacao.
3. Conversao das imagens para tensores com `transforms.ToTensor()`.
4. Criacao dos `DataLoader`s com lotes de 64 imagens e embaralhamento dos dados.
5. Visualizacao de uma imagem de treinamento e verificacao do formato dos tensores.
6. Definicao da rede neural com a seguinte arquitetura:
   - Entrada: 784 valores, correspondentes aos pixels de uma imagem 28 x 28.
   - Primeira camada oculta: 128 neuronios.
   - Segunda camada oculta: 64 neuronios.
   - Saida: 10 neuronios, um para cada classe de digito.
7. Aplicacao da funcao de ativacao ReLU nas camadas ocultas.
8. Aplicacao de `log_softmax` na camada de saida para gerar os valores usados pela funcao de perda.
9. Treinamento com:
   - Funcao de perda `NLLLoss`.
   - Otimizador SGD.
   - Taxa de aprendizado `0.01`.
   - Momentum `0.5`.
   - 10 epocas.
10. Validacao do modelo por meio da comparacao entre o digito previsto e o rotulo correto.

## Fluxo do modelo

```text
Imagem MNIST (28 x 28)
        |
Conversao para vetor (784 valores)
        |
Camada linear: 784 -> 128 + ReLU
        |
Camada linear: 128 -> 64 + ReLU
        |
Camada linear: 64 -> 10
        |
Probabilidades das classes 0 a 9
```

## Tecnologias utilizadas

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Jupyter Notebook

## Como executar

Instale as dependencias:

```bash
pip install torch torchvision numpy matplotlib jupyter
```

Abra o notebook:

```bash
jupyter notebook rede_neural.ipynb
```

Depois, execute as celulas em ordem. O dispositivo de processamento e selecionado automaticamente: CUDA, quando disponivel, ou CPU como alternativa.

## Observacao sobre o estado atual

A funcao `validacao` esta implementada e calcula a precisao do modelo no conjunto de teste. Para exibir esse resultado ao final do notebook, execute o treinamento e chame a funcao com:

```python
treino(modelo, trainloader, device)
validacao(modelo, valloader, device)
```

Assim, o notebook apresentara a quantidade de imagens avaliadas e a precisao obtida pelo modelo.
