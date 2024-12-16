
<center>

![](https://kirigaya.cn/files/images/bird.png)


[[torchood](https://github.com/LSTM-Kirigaya/torchood)]: pytorch implementation for universal out of distribution methods

```bash
pip install torchood
```

</center>

## Support Methods

| Support Methods | Source |
|:---|:---|
| Robust Classification with Convolutional Prototype Learning (Prototype Networks) | [CVPR 2018: Rbust classification with convolutional prototype learning](https://arxiv.org/pdf/1805.03438) |
| Predictive Uncertainty Estimation via Prior Networks(Prior Networks) | [NeurIPS 2018: Predictive Uncertainty Estimation via Prior Networks]() |
| Evidential Deep Learning to Quantify Classification Uncertainty (EDL) | [NeurIPS 2018: Evidential Deep Learning to Quantify Classification Uncertainty](https://papers.nips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html) |
| Posterior Network (PostNet) | [NeurIPS 2020: Posterior Network: Uncertainty Estimation without OOD Samples via Density-Based Pseudo-Counts]() |
| Evidential Neural Network (ENN) | [NeurIPS 2018: Evidential Deep Learning to Quantify Classification Uncertainty](https://arxiv.org/pdf/1806.01768) |
| Evidence Reconciled Neural Network(ERNN) | [MICCAI 2023: Evidence Reconciled Neural Network for Out-of-Distribution Detection in Medical Images](https://conferences.miccai.org/2023/papers/249-Paper2401.html) |
| Redundancy Removing Evidential Neural Network(R2ENN) | - |

---

## Usage

```bash
pip install torchood
```

```python
import torchood

class UserDefineModel(nn.Module):
    ...

classifier = UserDefineModel(...)
classifier = torchood.EvidenceNeuralNetwork(classifier)
```

### Train

```python
for data, label in train_loader:
    # transform data & label to correct device
    # ...

    _, evidence, _ = classifier(data)
    loss = classifier.criterion(evidence, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Infer

```python
prob, uncertainty = classifier.predict(inputs)
```

![](./images/robot.png)