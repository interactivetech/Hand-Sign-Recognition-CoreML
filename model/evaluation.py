import numpy as np
import tensorflow as tf
def evaluate_test(model,test_inputs):
    optimizer = tf.train.AdamOptimizer()

    # # We will now compile and print out a summary of our model
    model.compile(loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])
    acc = []
    pred=[]
    labels = []
    for eval_image,eval_labels in test_inputs:
        loss,accuracy = model.evaluate(eval_image,eval_labels)

        acc.append(accuracy)
    accuracy = np.mean(np.array(acc))
    return accuracy


def evaluate(model,eval_inputs):
    acc = []
    pred=[]
    labels = []
    for eval_image,eval_labels in eval_inputs:
        loss,accuracy = model.evaluate(eval_image,eval_labels)
        # output = np.argmax(model.predict(eval_image))
        # print(output,np.argmax(eval_labels))
        # pred.append(output)
        # labels.append(eval_labels)
        # if output!=eval_labels:
            # plot incorrect labels
            # tf.summary.image('incorrectly_labeled_{}'.format(eval_labels),eval_image)
        acc.append(accuracy)
    accuracy = np.mean(np.array(acc))
    return accuracy

def get_missclassified_images(model,eval_inputs):

    for eval_image,eval_labels in eval_inputs:
        loss,accuracy = model.evaluate(eval_image,eval_labels)
        output = np.argmax(model.predict(eval_image))
        print(output,np.argmax(eval_labels))
        # pred.append(output)
        # labels.append(eval_labels)
        if int(output)!=int(np.argmax(eval_labels)):
            # plot incorrect labels
            print(eval_image.shape)
            tf.contrib.summary.image(name='incorrectly_labeled_{}'.format(str(output)),tensor=eval_image)
        # acc.append(accuracy)