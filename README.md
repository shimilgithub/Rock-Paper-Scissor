# HW5 - Rock-Paper-Scissor Game Using Teachable Machine

## Organization
- `model` directory includes `keras_model.h5` and `labels.txt`
- `samples` directory include four testing images for each classes.
- Three program files.
    1. `rock-paper-scissors.py`
        - This file has the implementation of rock paper scissors game.
        - This code is taken creating dataset and training the model using Google Teachable Machine
        - Code is refactored into functions for better readability and folllow best practices
        - This program can be executed by running (example)
            ``` 
            python rock-paper-scissors.py --image-path "samples\paper.jpg"
            ```
        - NOTE  :Test image path has to be passed as the command line argument
        - Output display the test image and the  `class name` and `confidence score` will be displayed in the terminal.
    2. `teachable.ipynb`
        - This is the jupyter notebook version of rock-paper-scissor game. 
        - This program is used to perform testing with different images across different classes.
        - Once you execute the testing code, the test image,class labe and confidence score will be displayed.
        - This file has the testing examples of all the four classes.
    3. `rock-paper-scissors-live.py`
        - This program allows us the take the live video recording of rock-paper-scissor game trained using `Google Teachable Machine'.
        - To execute this program run,
            ``` 
            python rock-paper-scissors-live.py
            ```
        - Webcam will be opened and we can see the live prediction that includes the  class label.
        - press `q` to quit
        - Once the video end, it will be saved to `output.avi` file which is uploaded in youtube.



    4. **Click [here](https://www.youtube.com/watch?v=S5aeCC6bK08) to see the recorded video in youtube channel.**