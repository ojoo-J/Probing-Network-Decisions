# Probing Network Decisions: Capturing Uncertainties and Unveiling Vulnerabilities without Label Information
**ICPRAI 2024 (Oral, Honorable Mentions✨)** [[Paper](https://link.springer.com/chapter/10.1007/978-981-97-8702-9_21)]

## Summary
With the rapid advancements in deep learning, Deep Neural Networks (DNNs) are being actively utilized in critical domains such as healthcare and autonomous driving. In these applications, the decisions made by neural networks can directly impact human safety and lives, making the reliability and interpretability of models essential. Specifically, understanding the causes of misclassification plays a crucial role in enhancing the transparency and trustworthiness of models. Furthermore, if a model could autonomously identify and correct the causes of its misclassifications, it would provide an even higher level of stability and reliability.

Currently, attribution-based methods are widely used to interpret neural network decisions at the instance level. By specifying a target class, users can visually identify which parts of an input image contributed to the model’s classification of that class. However, when there are multiple candidate classes, explanations must be generated for all classes, and humans must manually compare these explanations to identify the reasons for misclassification. This process is prone to human bias and requires human supervision, posing significant limitations.

**We aim to go beyond answering “why the model classified the image as a specific class” and focus on “why the model failed to classify it as the true label.” In other words, we are interested in identifying the aspects of an image that the model found confusing. Our ultimate goal is to detect the features to which the model is particularly vulnerable and, eventually, enable the model to correct these features autonomously. We define our research question as follows:**

1. Can neural networks recognize their own misclassifications?
2. Can neural networks identify the vulnerable features that cause confusion?
3. Can neural networks automatically correct the features to which they are particularly vulnerable?
  



## Methods
<div align="center">
    <img src="https://github.com/user-attachments/assets/fb8db5b3-c203-4649-93a7-cfb6db4b4baa" alt="image" width="600">
</div>

In this paper, we present a novel framework to uncover the weakness of the classifier via counterfactual examples. A prober is introduced to learn the correctness of the classifier’s decision in terms of binary code - _hit_ or _miss_. It enables the creation of the counterfactual example concerning the prober’s decision. We test the performance of our prober’s misclassification detection and verify its effectiveness on the image classification benchmark datasets. Furthermore, by generating counterfactuals that penetrate the prober, we demonstrate that our framework effectively identifies vulnerabilities in the target classifier without relying on label information on the low-resolution dataset. For more details, please refer to the paper.



## Results
<div align="center">
    <img src="https://github.com/user-attachments/assets/b86962df-a82f-4539-89aa-02d295c9ee2b" alt="image" width="600">
</div>
<div align="center">
    <img src="https://github.com/user-attachments/assets/cac0ad85-aa74-49be-88e9-58e351a19d96" alt="image" width="600">
</div>

The prober in our framework effectively encodes hit-miss outcomes by detecting uncertainty from the hidden representations of a deep classifier.

<div align="center">
    <img src="https://github.com/user-attachments/assets/591be8ca-c79d-44e6-985c-b913ecad6622" alt="image" width="600">
</div>

By using the prober to generate counterfactuals that reduce the uncertainty of the neural network, we can observe the correction of the parts($\delta$) where the model is confused. In other words, we are able to detect the features to which the neural network responds most vulnerable.

<div align="center">
    <img src="https://github.com/user-attachments/assets/06bbadfa-9bb4-4db6-976b-c566dfe7495e" alt="image" width="600">
</div>

When the misclassified (True miss) images are edited and fed back into the neural network, we observe an approximately 87% improvement in performance. This demonstrates the potential of the neural network’s auto-correction capability.

## run
Given a trained classifier and generative network, follow these steps to: (1) evaluate the representation, (2) train a prober to model the implicit probability, and (3) generate counterfactual examples. You can refer to `scripts/run_mnist.sh` for example usage.

```
# Evaluate the representation
python experiments/eval_clf.py
```
```
# Train the Prober
python experiments/train_prober.py
```
```
# Generate the counterfactuals (corrected samples)
python experiments/generate_counterfactual.py
```
