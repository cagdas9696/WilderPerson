
"""
Created on Wed May 27 14:48:39 2020

@author: cagda
"""
import numpy as np
from fonk import get_random_data
from model import preprocess_true_boxes, network, network_loss
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Lambda
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping





def _main():
    annotation_val_path='annotation_val.txt'
    annotation_path = 'annotation.txt'
    log_dir = 'logs/000/'
    class_names=['pedestrians','riders','partially-visible persons','ignore regions','crowd']
    num_classes=len(class_names)
    anchors=[[  8 , 13],
             [ 13 , 32],
             [ 15 , 20],
             [ 18 , 45],
             [ 24 , 59],
             [ 27 , 30],
             [ 28 , 104],
             [ 32 , 76],
             [ 43 , 105],
             [ 62 , 152],
             [ 90 ,  44],
             [107 , 252]]
    anchors=np.array(anchors, dtype=float).reshape(-1, 2)

    input_shape = (640,640) # multiple of 32, hw

   
    model = create_model(input_shape, anchors, num_classes, load_pretrained=False)
    
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    #val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    #num_val = int(len(lines)*val_split)
    num_train = len(lines) 
    
    with open(annotation_val_path) as f:
        lines2 = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines2)
    np.random.seed(None)        
    num_val=len(lines2)

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'network_loss': lambda y_true, y_pred: y_pred})

        batch_size = 10
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history = model.fit_generator(data_generator_wrapper(lines, batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines2, batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=100,
                initial_epoch=0,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    # if True:
    #     for i in range(len(model.layers)):
    #         model.layers[i].trainable = True
    #     model.compile(optimizer=Adam(lr=1e-4), loss={'network_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
    #     print('Unfreeze all of the layers.')

    #     batch_size = 20 # note that more GPU memory is required after unfreezing the body
    #     print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    #     model.fit_generator(data_generator_wrapper(lines, batch_size, input_shape, anchors, num_classes),
    #         steps_per_epoch=max(1, num_train//batch_size),
    #         validation_data=data_generator_wrapper(lines2, batch_size, input_shape, anchors, num_classes),
    #         validation_steps=max(1, num_val//batch_size),
    #         epochs=100,
    #         initial_epoch=50,
    #         callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    #     model.save_weights(log_dir + 'trained_weights_final.h5')
    return history




def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)



def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:64, 1:32, 2:16, 3:8}[l], w//{0:64, 1:32, 2:16, 3:8}[l], \
        num_anchors//4, num_classes+5)) for l in range(4)]

    model_body = network(image_input, num_anchors//4, num_classes)
    print('Create model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze  body or freeze all but 4 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    

    model_loss = Lambda(network_loss, output_shape=(1,), name='network_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


if __name__ == '__main__':
    history = _main()


