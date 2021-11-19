# coding=utf-8
from splits import *
from loadDataSet import *
from CharCNNZhang import CharCNNZhang
import json

trainIDs, trainSamples, trainDialectLabels, trainCategoryLabels = loadMOROCODataSamples("train")
print("Loaded %d training samples..." % len(trainSamples))

validationIDs, validationSamples, validationDialectLabels, validationCategoryLabels = loadMOROCODataSamples("validation")
print("Loaded %d validation samples..." % len(validationSamples))

testIDs, testSamples, testDialectLabels, testCategoryLabels = loadMOROCODataSamples("test")
print("Loaded %d test samples..." % len(testSamples))

config = json.load(open("config.json"))

alphabet = config["data"]["alphabet"]
input_size = config["data"]["input_size"]
number_of_classes = config["data"]["num_of_classes"]

######### RO vs MD start ########
dataTrain    = Data(list(zip(trainSamples     , trainDialectLabels))     , alphabet, input_size, number_of_classes)
dataValidate = Data(list(zip(validationSamples, validationDialectLabels)), alphabet, input_size, number_of_classes)
dataTest     = Data(list(zip(testSamples      , testDialectLabels))      , alphabet, input_size, number_of_classes)
######### RO vs MD end ########

train_data   , train_labels    = dataTrain.convert_data()
validate_data, validate_labels = dataValidate.convert_data()
test_data    , test_labels     = dataTest.convert_data()

model = CharCNNZhang(input_size=config["data"]["input_size"],
                             alphabet_size=config["data"]["alphabet_size"],
                             embedding_size=config["char_cnn_zhang"]["embedding_size"],
                             conv_layers=config["char_cnn_zhang"]["conv_layers"],
                             fully_connected_layers=config["char_cnn_zhang"]["fully_connected_layers"],
                             num_of_classes=config["data"]["num_of_classes"],
                             threshold=config["char_cnn_zhang"]["threshold"],
                             dropout_p=config["char_cnn_zhang"]["dropout_p"],
                             optimizer=config["char_cnn_zhang"]["optimizer"],
                             loss=config["char_cnn_zhang"]["loss"])

model.train(training_inputs=train_data, training_labels=train_labels, validation_inputs=validate_data, validation_labels=validate_labels, epochs=config["training"]["epochs"], batch_size=config["training"]["batch_size"], checkpoint_every=config["training"]["checkpoint_every"])

# model_txt = "weights-improvement-25-acc-0.94-loss-0.21.hdf5"
model.model.load_weights("checkpoints/%s" % model_txt)

# results = model.test(testing_inputs=validate_data, testing_labels=validate_labels, batch_size=128)
np.savetxt("test_labels.txt", np.argmax(test_labels,axis=1))
results = model.test(testing_inputs=test_data, testing_labels=test_labels, batch_size=128, output_txt = "pred-%s.txt" % model_txt)

print("----> Results on Test:\n")
print(results)
