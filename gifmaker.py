import imageio
import os

directory =  os.path.join('data','video')
filenames = os.listdir(directory)

with imageio.get_writer('./data/video/det.gif', mode='I') as writer:
    for filename in filenames:
        name = filename.replace(".jpg", '')
        if int(name) >= 100 and int(name) <= 200:
            path = os.path.join(directory, filename)
            image = imageio.imread(path)
            writer.append_data(image)



        # path = './Pong/frames/'
        # image = imageio.imread(path+filename)
        # writer.append_data(image)