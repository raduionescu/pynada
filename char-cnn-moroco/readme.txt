Install required packages:
numpy
tensorflow
sklearn
statsmodels

Configure:
The architecture is defined in the config.json file inside the key `char_cnn_zhang`. 
The definition for the convolutional_layers is as follows:
	* the convolutional layers are defined inside the key `convolutional_layers`
	* each convolutional layer contains a list of 5 parameters [number of filters, kernel size, activation type, pool size for the max pooling layer, SE block ratio]; pool size and SE block ration can be `-1`, and means that block is deactivated

Run:
$ python CharCNNZhang_main.py
