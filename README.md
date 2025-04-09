Yo.


Not finished.

But here it is.


Its a Letter Recognizer,which just loads in the Letter and predicts which one it is.
GUI with Drawing Canvas never finished.



To get it started you need Python version 3.12.7 on a venv to minimize Conflict errors with other python Modules
you'll also need these modules (some on the specific libraries because of different versions)

OS
NUMPY
CV2
Tensorflow (Version 2.18.0)
scikit-learn
matplotlib.pyplot

So Basically,this Code has EVERYTHING inside ready to use, so it prepares a Folder to use it as train and test fodder,by downsizing the Pictures to 28x28 and making them grayscale
Then it creates that model using the edited Pictures.
That model is saved and then used to predict five pictures randomly picked from the Folder.
Das Dat.

There was a whole assignment which also included adding a Drawing GUI to the Code,so you can Predict self-made Letters quickly,but i just couldn't finish it in Time.
Last time that i caught up with the teacher was when predicting the 5 Random Pictures.




Some Pics of it in Action:

![image](https://github.com/user-attachments/assets/e13a0a3c-1647-453f-b7f6-67854421a11a)

Thats the Letter Edited into a Size which the AI Likes to use and also turned into Greyscale

![image](https://github.com/user-attachments/assets/984c71b2-1d98-462f-b0b9-5218ed101a9a)

Here is the Letter the AI is predicting


Some Stupidity FAQ: 

Why did I use Cv2,then Matplotlib? Why not sticking to one of them when opening pictures?

I wanted to continue on Matplotlib since the Drawing canvas wouldn't be that much more work(It still was)
since that was Code that i wrote together with melvin,and i couldnt bother using it further on since i didn't really know the Module

I guess I could convert the code to only use Matplotlib or cv2 since i dont even have the Canvas drawing part in this code
i guess....

But yea. Das Dat.


