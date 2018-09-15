import tensorflow as tf
from model.evaluation import evaluate
import os
import numpy as np
def train_sess(model,images,labels,step):
        # print(labels)
        train_loss,train_accuracy=model.train_on_batch(images,labels)
        if step%10==0:
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
        return (train_loss,train_accuracy)

def train_and_evaluate(model,train_inputs,eval_inputs,args):
        best_eval_acc = 0.0
        EPOCHS=20
        step=0
        for epoch in range(EPOCHS):
            train_acc = []
            for images, labels in train_inputs:
                print(images.shape)
                # a = model(images)
                # print(a.shape)
                train_loss,train_accuracy=train_sess(model,images,labels,step)
                print("BATCH_ACC={}".format(train_accuracy))
                train_acc.append(train_accuracy)
                step+=1
            train_accuracy = np.mean(np.array(train_acc))
            # run evaluation after epoch
            accuracy= evaluate(model,eval_inputs)
            if accuracy > best_eval_acc:
                best_eval_acc=accuracy
                # Save weights
                best_save_path = os.path.join(args.model_dir,'best_weights','after-epoch-{}'.format(epoch))
                
                if not os.path.exists(best_save_path):
                    os.makedirs(best_save_path)
                model.save(os.path.join(best_save_path,"model_acc_"+str(best_eval_acc)+".h5"))
                print(" - Found new best accuracy, saving in {}".format(best_save_path))
            # acc.append(accuracy)
            
            print('Test accuracy: {}'.format(accuracy))
            # Here you can gather metrics or adjust your training parameters
            print('Epoch: {}\tLoss: {}\tTrain Accuracy: {}\tEval Acc:{}'.format(epoch+1,train_loss,train_accuracy,accuracy))
            tf.contrib.summary.scalar("eval_accuracy", accuracy,step=step)

# def train_and_evaluate():
#     """Train the model and evaluate every epoch

#     Args:
#         train_model_spec: (dict) contains the graph operations or nodes needed for training
#         eval_mode_spec: (dict) contains the graph operations or nodes needed for evaluation
#         model_dir: (string) direcory containing config, weights and log
#         params: (Params) contains hyperparameters of the model
#             Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
#         restore_from: (string) directory or file containing weights to restore the graph

#     """

#     return
