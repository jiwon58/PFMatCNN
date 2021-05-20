import os, sys, numpy as np, argparse, shutil
from PIL import Image
from skimage.measure import regionprops, label
from skimage import filters
from tqdm import tqdm
import matplotlib.pyplot as plt

def sample_particle(args):

    master = './data/' + args.dataset
    newmaster = master + '_dis' + str(int(args.distance))

    if not os.path.isdir(master):
        print('There is no dataset!')
        sys.exit(1)

    if not os.path.exists(newmaster):
        os.mkdir(newmaster)
    folder_list = ['/test','/train', '/validate']
    # Copy test images, labels, labels_npy files
    for subfolders in folder_list:
        if not os.path.exists(newmaster+subfolders):
            os.mkdir(newmaster+subfolders)
        if subfolders == '/test':
            for content_folder in ['/images/', '/images_npy/', '/labels/']:
                if not os.path.exists(newmaster+subfolders+content_folder):
                    os.mkdir(newmaster+subfolders+content_folder)
                for file in [f for f in os.listdir(master+'/test'+content_folder) if not f[0] == '.']:
                    shutil.copy(master+'/test'+content_folder+file,newmaster+'/test'+content_folder)
            print('Test folder is copied.')
        else:
            for content_folder in ['/images', '/labels']:
                if not os.path.exists(newmaster+subfolders+content_folder):
                    os.mkdir(newmaster+subfolders+content_folder)
            print('>',subfolders[1:])
            for file in tqdm(sorted([f for f in os.listdir(master+subfolders+'/images') if not f[0] == '.'])):
                img = Image.open(master+subfolders+'/images/'+file).convert('L')
                np_img = np.asarray(img)
                N = np_img.shape[0]
                val = filters.threshold_otsu(np_img)
                background = np.unique(np_img[np_img > val])[-1]
                label_bg = label((np_img < background)*255)
                rp = regionprops(label_bg)
                backboard = np.tile(background, (N, N))

                # Sampling particles
                rp = [r for r in rp if np.sqrt((r.centroid[0]-N/2)**2+(r.centroid[1]-N/2)**2) <= property]

                for region in rp:
                    for coord in region.coords:
                        backboard[coord[0], coord[1]] = np_img[coord[0], coord[1]]
                plt.imsave('{0}{1}/images/{2}.png'.format(newmaster, subfolders, file[:6]),backboard,cmap='gray',vmin=0, vmax=255)
                plt.imsave('{0}{1}/labels/{2}_L.png'.format(newmaster, subfolders, file[:6]),backboard < val,cmap='gray')
    #print(newmaster[6:])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove particles. If there is no flags, they will copy the original dataset.', usage = 'python randomregion.py [dataset] [flags] [property] [property_distance]' )
    parser.add_argument('dataset', help='name of the dataset folder')

    parser.add_argument('-d','--distance', action='store_true', help='choose particles by distance from center')
    #parser.add_argument('-p','--property', help='detail property for options, applied for r (0 ~ 100), e (0 ~ 1), a(0 ~ largest particle size), d (0 ~ image size / 2)', type=float)

    args = parser.parse_args()
    sample_particle(args)
