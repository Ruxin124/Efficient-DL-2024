Personal Repository for the course "Efficient Deep Learning" at IMT Atlantique
--

Course organisation / Syllabus
--

### Session 1

- constructed Resnet18 from scratch
    - test_lab1_earlystop.py
- apply data augmentation and use learning rate scheduler
    - try the step scheduler first
        - test_lab2_1_data augmentaion.py
            - 70 epochs, accuracy 89%
            - test_resnet18_cifar10_5.pth
    - Change to Cosine LR adjuster
        - test_lab2_CosineLR.py
            - 100 epochs, accuracy 91%
            - Session1_lab2_Resnet18_cifar10.pth
            - Record_lab1.ipynb

### Session 2

- post quantitation
    - from 32 bits to 16 bits
        - test_lab3_quantinition_16bit.py
            - lab3_Resnet18_cifar10_Cosine_16bits.pth
    - 32-bit to binary
        - complete the BC class to perform binary quantitation
        - test_lab3_2_binary_quantination.ipynb
            - Session2_lab1_Resnet18_cifar10_binnary.pth
            - test the binary model performance on the test set
                - accuracy 62.4%
    - Fine-tuning by simple training loop 100 epochs
        - test_lab3_2_binary_quantination.ipynb
            - 50 epochs - Session2_lab1_Resnet18_cifar10_binnary_tunning_1.pth
                - accuracy 62.4%
            - 100 epochs - Session2_lab1_Resnet18_cifar10_binnary_tunning_2.pth
                - accuracy 88.51%

### Session 3

- test on different pruning methods
    - Globale pruning 0.2
        - test_lab4_global_pruning.ipynb
        - globale_pruned_0.2.pth  —  Test Accuracy: 79.02%
    - Globale pruning 0.3
        - globale_pruned_0.3.pth — Test Accuracy: 52.21%
    - Globale pruning 0.5
        - globale_pruned_0.5.pth — Test Accuracy: 16.2%
