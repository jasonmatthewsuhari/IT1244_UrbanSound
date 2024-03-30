# Dataset Description:
- This is an audio classification dataset containing 10 classes in total
    - classes: {0: 'air_conditioner', 1: 'car_horn', 2: 'children_playing', 3: 'dog_bark', 4: 'drilling', 5:'engine_idling',
          6: 'gun_shot', 7:'jackhammer', 8: 'siren', 9: 'street_music'}
- Each audio is of variable length
- Train and Test dataset have around 1700 and 800 audio samples, respectively
- All classes are not equally represented
- Each audio file in both folders has their class name mentioned

# Task Expectation:
- The classes in this dataset are challenging
- You need to find out which class samples are being misclassified more and try to extract appropriate feature to mitigate such misclassification
- Also the audio length can be an important factor for accurate or inaccurate prediction: you need to handle this as well
- Some of the samples of the training set don't really sound at all as their mentioned class
    - you should ideally find out such outlier samples and remove them during training
    - note that you cannot perform this outlier detection manually
