import keras,sys
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout
from keras.utils import to_categorical
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def array_to_color(array, cmap="Oranges"):
  s_m = plt.cm.ScalarMappable(cmap=cmap)
  return s_m.to_rgba(array)[:,:-1]


def process_image(im):
    im = cv2.resize(im, (55,55))
    im = im.transpose()
    return im

# Class dictionary
label_dict = {'fire': 0, 'nofire': 1}

def rgb_data_transform(data):
  data_t = []
  data_t.append(array_to_color(data.reshape(55, 55, 3)).reshape(18, 11, 20, 3))
  return np.asarray(data_t)



def process_data(im):
    image = process_image(im)
    image = rgb_data_transform(image)
    return image


# load json and create model


if __name__ == '__main__':


    # construct and display model
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model/model.h5")
    print("Loaded model from disk")


    #imageshape should be 1,3,55,55
    model.summary()


    # display and loop settings

    windowName = "Live Fire Detection";
    keepProcessing = True;


    if len(sys.argv) == 2:

        # load video file from first command line argument

        video = cv2.VideoCapture(sys.argv[1])
        print("Loaded video ...")

        # create window

        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);

        # get video properties

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_time = round(1000/fps);

        while (keepProcessing):

            # start a timer (to see how long processing and display takes)

            start_t = cv2.getTickCount();

            # get video frame from file, handle end of file

            ret, frame = video.read()
            if not ret:
                print("... end of video file reached");
                break;

            # re-size image to network input size and perform prediction

            small_frame = process_data(frame)
            output = model.predict([small_frame])

            # label image based on prediction

            if round(output[0][0]) == 1:
                cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 50)
                cv2.putText(frame,'FIRE',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
            else:
                cv2.rectangle(frame, (0,0), (width,height), (0,255,0), 50)
                cv2.putText(frame,'CLEAR',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);

            # stop the timer and convert to ms. (to see how long processing and display takes)

            stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;

            # image display and key handling

            cv2.imshow(windowName, frame);

            # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)

            key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF;
            if (key == ord('x') or cv2.getWindowProperty('Live Fire Detection',1) == -1):
                keepProcessing = False;
            elif (key == ord('f')):
                cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
    else:
        video = cv2.VideoCapture(0)
        print("Loading Camera ...")

        # create window

        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);

        # get video properties

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_time = round(1000/fps);

        while (keepProcessing):

            # start a timer (to see how long processing and display takes)

            start_t = cv2.getTickCount();

            # get video frame from file, handle end of file

            ret, frame = video.read()
            

            # re-size image to network input size and perform prediction

            small_frame = process_data(frame)
            output = model.predict([small_frame])

            # label image based on prediction

            if round(output[0][0]) == 1:
                cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 50)
                cv2.putText(frame,'FIRE',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
            else:
                cv2.rectangle(frame, (0,0), (width,height), (0,255,0), 50)
                cv2.putText(frame,'CLEAR',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);

            # stop the timer and convert to ms. (to see how long processing and display takes)

            stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;

            # image display and key handling

            cv2.imshow(windowName, frame);

            # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)

            key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF;
            if (key == ord('x') or cv2.getWindowProperty('Live Fire Detection',1) == -1):
                keepProcessing = False;
            elif (key == ord('f')):
                cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
    
