import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
import numpy as np
import torch
import random
RIGHT = 4
LEFT = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def preprocess_single(image, bkg_color=np.array([144, 72, 17])):
    # preprocess a single frame
    # crop image and downsample to 80x80
    # stack two frames together as input
    img = np.mean(image[34:-16:2, ::2] - bkg_color, axis=-1) / 255.
    return img


def preprocess_batch(images, bkg_color=np.array([144, 72, 17])):
    # convert outputs of parallelEnv to inputs to pytorch neural net
    # this is useful for batch processing especially on the GPU
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5:
        list_of_images = np.expand_dims(list_of_images, 1)
    # subtract bkg and crop

    list_of_images_prepro = np.mean(list_of_images[:, :, 34:-16:2, ::2] - bkg_color,
                                    axis=-1) / 255.
    batch_input = np.swapaxes(list_of_images_prepro, 0, 1)
    return torch.from_numpy(batch_input).float().to(device)


# function to animate a list of frames
def animate_frames(frames):
    plt.axis('off')

    # color option for plotting
    # use Greys for greyscale
    cmap = None if len(frames[0].shape) == 3 else 'Greys'
    patch = plt.imshow(frames[0], cmap=cmap)

    fanim = animation.FuncAnimation(plt.gcf(), \
                                    lambda x: patch.set_data(frames[x]), frames=len(frames), interval=30)

    display(display_animation(fanim, default_mode='once'))


def evaluate_model(env, model, time=2000, preprocess=None, nrand=5, no_op_action: int = 0, num_stacked_frames: int = 2):
    # play a game and display the animation
    # nrand = number of random steps before using the policy
    env.reset()

    # star game
    env.step(1)

    # # perform nrand random steps in the beginning
    # for _ in range(nrand):
    #     frame1, reward1, is_done, _ = env.step(np.random.choice([RIGHT, LEFT]))
    #     frame2, reward2, is_done, _ = env.step(0)
    for _ in range(nrand):
        fr1, _, _, _ = env.step(np.random.choice([RIGHT, LEFT]))
        next_frames = [env.step(no_op_action)[0] for _ in range(num_stacked_frames - 1)]
        
    anim_frames = []

    for _ in range(time):

        frame_input = preprocess_batch([fr1] + next_frames)
        prob = model(frame_input)

        action = RIGHT if random.random() < prob else LEFT
        frame1, _, is_done, _ = env.step(action)
        # frame2, _, is_done, _ = env.step(0)

        if preprocess is None:
            anim_frames.append(frame1)
        else:
            anim_frames.append(preprocess(frame1))

        if is_done:
            break

    env.close()

    animate_frames(anim_frames)
    return
