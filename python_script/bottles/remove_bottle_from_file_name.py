import glob, os

def rename(dir, pattern):
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        print title[7:] + '.png'
        #os.rename(pathAndFilename, 
        #          os.path.join(dir, titlePattern % title + ext))
        os.rename(pathAndFilename, os.path.join(dir, title[7:] +'.png'))
        
                                                
				  
rename('C:\\Users\\pcban\\Downloads\\object_images\\bottles_only', r'*.png')
