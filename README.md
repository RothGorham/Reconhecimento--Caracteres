# Reconhecimento de Dígitos ✍️

CNN treinada no MNIST + interface Streamlit pra reconhecer dígitos escritos à mão. Você sobe uma foto ou desenha direto no Site, e o modelo te diz qual número é.


## A ideia

Projeto de faculdade, com a proposta de treinar uma rede neural para reconhecimento de dígitos manuscritos e ir além do básico.


## Como executar no Google Colab

Para rodar o projeto no Google Colab, basta fazer o upload do arquivo **Reconhecimento-Caracteres.ipynb** e executá-lo no ambiente do Colab. Todas as etapas de treinamento e teste podem ser realizadas diretamente pelo notebook.

Caso prefira não executar o código e apenas testar o modelo pronto, utilize a opção abaixo.

## Como testar o modelo

Para testar sem precisar rodar o código, basta baixar o arquivo **melhor_modelo.pth** e acessar o site:

https://reconhecimento--caracteres-gvjazx7bmmcbhlyw8qhgyu.streamlit.app/

No site, é possível fazer o upload de uma imagem com um dígito manuscrito ou desenhar diretamente no canvas o número desejado.

Após isso, o modelo carregado realizará a predição automaticamente.


## Tecnologias utilizadas

- **Python 3.10+**
- **PyTorch** — modelo e inferência
- **Streamlit** — interface
- **Pillow + NumPy** — manipulação de imagem
- **SciPy** — filtro gaussiano, `center_of_mass` e `binary_closing` no pré-processamento


## Rodando localmente

```bash
git clone https://github.com/RothGorham/Reconhecimento--Caracteres.git
cd Reconhecimento--Caracteres

pip install -r requirements.txt

streamlit run app.py
```

## Aviso sobre as imagens

Ao testar o projeto, é importante considerar que o desempenho do modelo depende diretamente da qualidade da imagem utilizada.

Imagens com fundo irregular, sombras, baixa iluminação, baixo contraste ou dígitos mal centralizados podem prejudicar a leitura. A etapa de pré-processamento tenta corrigir esses problemas, mas não garante resultados perfeitos em todos os casos.

A estratégia de detecção de fundo, por exemplo, funciona bem na maioria das situações, porém pode falhar em cenários com gradientes complexos ou variações intensas de iluminação, o que pode levar a erros na predição.

Para obter melhores resultados, recomenda-se utilizar imagens com bom contraste, fundo uniforme e o dígito bem centralizado.

**Projeto de estudo**
