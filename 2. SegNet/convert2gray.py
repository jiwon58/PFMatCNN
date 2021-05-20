import os, sys, shutil, numpy as np, argparse, time
from matplotlib import pyplot as plt
from skimage import filters
from tqdm import tqdm
from PIL import Image

def mean_nonzero(array):
    return np.sum(array)/np.count_nonzero(array)

def avg_list(list):
    return sum(list)/len(list)

def convert_gray(args):
    master = './data/' + args.dataset
    content_folders = ['/images/', '/labels/', '/labels_npy/']
    target_dend_bri = 90.69
    target_dend_cont = 8.8
    target_back_bri = 133.39
    target_back_cont = 4.88
    if not os.path.isdir(master):
        print('There is no dataset!')
        sys.exit(1)
    if args.noise:
        newmaster = master + '_g_n'
    else:
        newmaster = master + '_g'
    if not os.path.exists(newmaster):
        os.makedirs(newmaster)

    folder_list = ['/train', '/validate']
    dendrites_bri, dendrites_cont, backgrounds_bri, backgrounds_cont = [], [], [], []

    for subfolders in folder_list:
        if not os.path.exists(newmaster+subfolders):
            os.makedirs(newmaster+subfolders)
        for content_folder in content_folders:
            if not os.path.exists(newmaster+subfolders+content_folder):
                os.makedirs(newmaster+subfolders+content_folder)
        for file in tqdm([f for f in os.listdir(master+subfolders+'/images') if not f[0] == '.']):
            try:
                # Load files from numpy format
                img = np.load(master+subfolders+'/images_npy/'+file[:-3]+'npy')
                img = img + abs(np.min(img))
                img = img * 255 / np.max(img)
                img = round(img, 1).astype(np.uint8)
            except:
                img = Image.open(master+subfolders+'/images/'+file).convert('L')
                img = np.asarray(img) # Dendrites are white in images
        
            N = img.shape[0]
            gt = Image.open(master+subfolders+'/labels/'+file[:-4]+'_L.png').convert('L')
            gt = np.asarray(gt)
            dendrites_avg = mean_nonzero(img[gt != 0])
            backgorunds_avg = mean_nonzero(img[gt == 0])
            #brightest = np.max(img)
            #darkest = np.min(img)
            
            if args.noise:
                # "target_back_bri+2" is adjustment for pixel values for tomography images
                image = np.random.normal(target_back_bri+2, target_back_cont, (N, N)) - img * (target_back_bri-target_dend_bri)/(dendrites_avg-backgorunds_avg)
            else:
                # "target_back_cont/3" is adjustment for pixel values for tomography images
                image = np.tile(target_back_bri+target_back_cont/3 + 0.4,(N, N)) - (img * (target_back_bri-target_dend_bri)/(dendrites_avg-backgorunds_avg))
            
            # Calculate the stat
            dendrites_bri.append(mean_nonzero(image[gt !=0]))
            dendrites_cont.append(np.std(image[gt != 0]))
            backgrounds_bri.append(mean_nonzero(image[gt ==0]))
            backgrounds_cont.append(np.std(image[gt == 0]))
            plt.imsave('{0}{1}/images/{2}.png'.format(newmaster, subfolders,file[:6]), image, cmap='gray', vmin=0, vmax=255)
            shutil.copy(master+subfolders+'/labels/'+file[:-4]+'_L.png',newmaster+subfolders+'/labels/'+file[:-4]+'_L.png')
            shutil.copy(master+subfolders+'/labels_npy/'+file[:-4]+'_L.npy',newmaster+subfolders+'/labels_npy/'+file[:-4]+'_L.npy')
        print('{} folder is done!\n'.format(subfolders[1:]))

    print('Dendrites Average Brightness: {:.3f}'.format(avg_list(dendrites_bri)))
    print('Dendrites Average Contrast: {:.3f}'.format(avg_list(dendrites_cont)))
    print('Backgrounds Average Brightness: {:.3f}'.format(avg_list(backgrounds_bri)))
    print('Backgrounds Average Contrast: {:.3f}'.format(avg_list(backgrounds_cont)))
    print()
    print('Target Dendrites Brightness: {}'.format(target_dend_bri))
    print('Target Dendrites Contrast: {}'.format(target_dend_cont))
    print('Target Backgrounds Brightness: {}'.format(target_back_bri))
    print('Target Backgrounds Contrast: {}'.format(target_back_cont))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert B/W images into grayscale images')
    parser.add_argument('dataset', help='Name of Dataset in which Conversion will Take Place')
    parser.add_argument('-n', '--noise', action='store_true', help='noise using normal distribution of numpy')
    args = parser.parse_args()
    convert_gray(args)
