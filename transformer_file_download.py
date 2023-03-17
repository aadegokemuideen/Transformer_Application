## download Compressed file
import pathlib

import tensorflow as tf

#download dataset provided by Anki: https://www.manythings.org/anki/
text_file = tf.keras.utils.get_file(fname="fra-eng.zip",
	        origin="http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip",
	         extract=True )
#show where the file is located now
text_file = pathlib.Path(text_file).parent / "fra.txt"
print(text_file)
