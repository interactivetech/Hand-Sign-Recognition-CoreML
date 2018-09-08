## Test of tf.keras and eager execution

'''
tf.data api provides tools for working with data (common functionality, shuffling, batching)
as well as high performance utilities: parallel reads, prefetching to GPUs, etc

1. using model.train_on_batch to iterate over dataset during training
2. create Tensorflow optimizer to pass to Keras Model
3. Enable eager execution
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.enable_eager_execution()# Eager exec enabling
(train_images, train_labels),(test_images,test_labels)= tf.keras.datasets.fashion_mnist.load_data()
# plt.imshow(train_images[0])
# plt.title(train_labels[0].__str__())
# plt.show()

TRAIN_SIZE = len(train_images)
TEST_SIZE = len(test_images)

# Reshape from (N,28,28) to (N,28*28), == (N,784)
train_images = np.reshape(train_images,(TRAIN_SIZE,784))
test_images = np.reshape(test_images,(TEST_SIZE,784))

# Convert array to float32, data currently uint8
train_images = train_images.astype(np.float32)
test_images = test_images.astype(np.float32)

# Convert pixel values from integers between 0 and 255 to floats between 0 and 1
train_images /=255
test_images /=255

NUM_CAT = 10

print("Before", train_labels[0])

train_labels = tf.keras.utils.to_categorical(train_labels,NUM_CAT)
print("After: ",train_labels[0]) #the format of labels after converting
test_labels = tf.keras.utils.to_categorical(test_labels,NUM_CAT)

# Cast the labels to floats, needed later
train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

def model_fn():
    input = tf.keras.layers.Input(shape=(None,784))
    x = tf.keras.layers.Dense(512,activation=tf.nn.relu)(input)
    x = tf.keras.layers.Dense(NUM_CAT,activation=tf.nn.softmax)(x)
    model = tf.keras.models.Model(inputs=input,outputs=x)
    return model
# Build model
model = model_fn()

# Create a Tensorflow optimizer, rather than using Keras version
# This is currently necessary when working in eager mode
optimizer = tf.train.AdamOptimizer()

# We will now compile and print out a summary of our model
model.compile(loss='categorical_crossentropy',
optimizer=optimizer,
metrics=['accuracy'])

model.summary()

# Create tf.data Dataset

'''
we'll use the tf.data.Dataset API to convert Numpy arrays into Tensorflow dataset.

Next, we will create a simple for loop that will serve as intro into creating custom
training loops. Although this essentially does the same thing as model.fit
it allows us to get creative and customize the overall training process
(should you like to) and collect diff metrics throughout the process
'''

BATCH_SIZE=128

'''Because tf.data works with potentially **large** collections of data
we dont shuffle the entire dataset by default
Instead, we maintain a buffer of SHUFFLE_SIZE elements
and sample from there
'''
compute_stats=False
if compute_stats:
    SHUFFLE_SIZE=10000

    acc = []
    dom = [int(i) for i in np.logspace(1,4.6,num=30)]
    for i in dom:
        print(i)
        model = model_fn()
        model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        # Create the dataset
        dataset = tf.data.Dataset.from_tensor_slices((train_images[:i],train_labels[:i]))
        dataset=dataset.shuffle(SHUFFLE_SIZE)
        dataset=dataset.batch(BATCH_SIZE)

        # Iterate over the dataset

        '''Here we'll iterate over the dataset, and train our model using
        model.train_on_batch. To learn more about elements returned from dataset,
        you can print them out and try the .numpy() method
        '''

        EPOCHS=5
        for epoch in range(EPOCHS):
            for images, labels in dataset:
                train_loss,train_accuracy=model.train_on_batch(images,labels)

            # Here you can gather metrics or adjust your training parameters
            print('Epoch: {}\tLoss: {}\tAccuracy: {}'.format(epoch+1,train_loss,train_accuracy))

        loss,accuracy = model.evaluate(test_images,test_labels)
        acc.append(accuracy)
        print('Test accuracy: {}'.format(accuracy))


    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    log_dom = np.log(np.array(dom))
    m.fit(log_dom.reshape(-1,1)
            ,np.array(acc))

    plt.scatter(log_dom,acc)

    plt.plot(log_dom,m.predict(log_dom.reshape(-1,1)
            ),c='r')
    plt.show()
    print("Accuracy {}".format(m.score(log_dom.reshape(-1,1)
            ,np.array(acc))))
    print("To get if you had{} amount of data, you would get {}% acc".format(np.array([50000]),
            m.predict(np.log(np.array([50000]).reshape(-1,1))) ))
    print(np.polyfit(log_dom,np.array(acc),1))
    B,A = np.polyfit(log_dom,np.array(acc),1)
    print("y={}log(x)+{}".format(B,A))
    print("If had {}, would get {} acc".format(50000,B*np.log(50000)+A ) )
else:
# print(i)
    SHUFFLE_SIZE=10000
    model = model_fn()
    model.compile(loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
    # Create the dataset
    dataset = tf.data.Dataset.from_tensor_slices((train_images,train_labels))
    dataset=dataset.shuffle(SHUFFLE_SIZE)
    dataset=dataset.batch(BATCH_SIZE)

    # Iterate over the dataset

    '''Here we'll iterate over the dataset, and train our model using
    model.train_on_batch. To learn more about elements returned from dataset,
    you can print them out and try the .numpy() method
    '''
    # t = tf.keras.callbacks.TensorBoard(log_dir='.', histogram_freq=0, write_graph=True)
    # t.set_model(model)
    # t.on_batch_end()
    global_step = tf.train.get_or_create_global_step()
    summary_writer = tf.contrib.summary.create_file_writer(
        'logs/', flush_millis=10000)

    '''python
    global_step = tf.train.get_or_create_global_step()
    summary_writer = tf.contrib.summary.create_file_writer(
        train_dir, flush_millis=10000)
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
      # model code goes here
      # and in it call
      tf.contrib.summary.scalar("loss", my_loss)
      # In this case every call to tf.contrib.summary.scalar will generate a record
      # ...
    '''
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
      # model code goes here
      # and in it call
      # In this case every call to tf.contrib.summary.scalar will generate a record
      # ...
        EPOCHS=5
        step=0
        for epoch in range(EPOCHS):
            for images, labels in dataset:
                train_loss,train_accuracy=model.train_on_batch(images,labels)
                if step%500==0:
                    # ToDo(Andrew): How to write gradient activations
                    tf.contrib.summary.scalar("train_loss", train_loss,step=step)
                    tf.contrib.summary.scalar("train_accuracy", train_accuracy,step=step)
                    for layer in model.layers:
                        for weight in layer.weights:
                            mapped_weight_name = weight.name.replace(':', '_')
                            tf.contrib.summary.histogram(mapped_weight_name, weight,step=step
                            
                            )
                            '''
                            RuntimeError: tf.gradients not supported when eager execution is enabled. 
                            Use tf.contrib.eager.GradientTape instead.
                            '''
                            # grads = model.optimizer.get_gradients(model.total_loss,
                            #                                 weight)

                            # def is_indexed_slices(grad):
                            #     return type(grad).__name__ == 'IndexedSlices'
                            # grads = [
                            #     grad.values if is_indexed_slices(grad) else grad
                            #     for grad in grads]
                            # tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads)

                        
                step+=1
            # Here you can gather metrics or adjust your training parameters
            print('Epoch: {}\tLoss: {}\tAccuracy: {}'.format(epoch+1,train_loss,train_accuracy))

        loss,accuracy = model.evaluate(test_images,test_labels)
        # acc.append(accuracy)
        print('Test accuracy: {}'.format(accuracy))
