import imageio
import numpy as np
import os
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
def main():
    loaded_array = np.load("resume_from_here.npy")
    testsavedir = ""
    imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(loaded_array), fps=30, quality=8)

if __name__ == "__main__":
    main()