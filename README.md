# Face recognition
Machine Learning project to identify the face and print their name by using python and OpenCV.

<h2>Dependencies</h2>
<ul>
    <li>Python 3.x</li>
    <li>OpenCV</li>
    <li>Numpy</li>
    <li><b>Extras:</b></li>
    <ul type="circle">
        <li>Local Binary Patterns Histograms (LBPH) Face Recognizer - <code>cv2.face.createLBPHFaceRecognizer()</code></li><br>
        <pre>pip install opencv-contrib-python</pre>
        <li>haarcascade_frontalface_default.xml-pre-trained face detector, provided by the developers and maintainers of the OpenCV library.</li>
    </ul>
</ul>
<br>
<hr>

<h2>Steps</h2>

step 1: First create a directory for your new project and navigate into it.

<pre>
mkdir Face_recognition
cd Face_recognition
</pre>


step 2: Copy the above code and paste it in that directory.


step 3: Run <code>dataSetGenerator.py</code> to create the dataset, it ask your "your_name" and captures the image and store it in <code>image_data/"your_name"</code>.

<pre>
python dataSetGenerator.py
</pre>


step 4: Run <code>faceTrainer.py</code> to train the model, it will create the <code>labels.pickle</code> file and <code>trainner.yml</code> [explained in code].

<pre>
python faceTrainer.py
</pre>


step 5: Run <code>faceRecognise.py</code> to identify your face.

<pre>
python faceRecognise.py
</pre>


<hr>

<h2>Result</h2>

<img src="screenshots/Screenshot.jpg" alt="Loading....">

<hr>
<h2>References</h2>
https://www.mygreatlearning.com/blog/face-recognition/

https://github.com/Nabajyotighosh/Facelocking-Door-Using-Python-and-Arduino-Programing
