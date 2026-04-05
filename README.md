# ugpt

A tiny C++ transformer ("GPT") model. Performs tokenization, embedding,
multi-head attention, multilayer perceptron, and decoding, alongside a custom
autograd engine.

Based on Karpathy's [microgpt](https://karpathy.github.io/2026/02/12/microgpt/)

## Usage

1. `make get-data`
2. `make`
3. `./ugpt`

## Example Output

`./ugpt` trains a model and then runs inference to generate new names.

```
Loss: 3.20334   0 (jahleel)
Loss: 3.44826   1 (carmen)
Loss: 3.43033   2 (rennata)
Loss: 3.26269   3 (rozalia)
...
Loss: 2.86803   997 (diego)
Loss: 2.68182   998 (dedrick)
Loss: 2.23179   999 (zaavan)
 --- Inference ---
salele
aranl
manna
darana
jadon
konay
morah
araya
enita
jaty
joman
elin
janan
taner
chiiy
dyna
sened
borani
brita
```
