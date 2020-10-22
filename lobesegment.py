import  torch,os
from .model.PLS3D import Pls_Net_3D_cp
from .args import  checkpoint_path,batch_size,lobe_predict_output_path
import  numpy as np
import SimpleITK as sitk

def read_dicom(filepath):
    
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(filepath)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    lung = sitk.GetArrayFromImage(image) # z, y, x
    lung_Origin = image.GetOrigin() # x, y, z
    lung_Spacing = image.GetSpacing() # x, y, z
    
    return lung,lung_Origin,lung_Spacing

def genarate_mha(img,spacing,origin,path):
    output_img = sitk.GetImageFromArray(img.astype(np.ubyte), isVector=False)
    output_img.SetSpacing(spacing)
    output_img.SetOrigin(origin)
    sitk.WriteImage(output_img, path, True)

class LobeSegmentor(object):

    def __init__(self, model_path=''):

        
        self.net = Pls_Net_3D_cp(n_channels=1, n_classes=6,g = 12)
        self.checkpoint = torch.load(model_path)
        self.batch_size = batch_size

        filtered = {k: v for k, v in self.checkpoint['model'].items() if 'num_batches_tracked' not in k}
        
        self.net.load_state_dict(filtered, strict=True)
       

    def __call__(self, img,lung,spacing,origin,save_path):

        
        #self.read_file()
        #print("start lobe segmentation")
        lobe_mask = self.seg(img)
        #postpreprocessing
        lobe_mask[np.where(lung == 0) ] = 0 
        #print(" lobe segmentation done")
        lobe1 = (lobe_mask == 1)
        lobe2 = (lobe_mask == 2)
        lobe3 = (lobe_mask == 3)
        lobe4 = (lobe_mask == 4)
        lobe5 = (lobe_mask == 5)
        # save to direct diretory
        self.genarate_mha(lobe_mask,spacing,origin,os.path.join(save_path,'segmentation_lobe.mha'))
        self.genarate_mha(lobe1,spacing,origin,os.path.join(save_path,'segmentation_lobe_1.mha'))
        self.genarate_mha(lobe2,spacing,origin,os.path.join(save_path,'segmentation_lobe_2.mha'))
        self.genarate_mha(lobe3,spacing,origin,os.path.join(save_path,'segmentation_lobe_3.mha'))
        self.genarate_mha(lobe4,spacing,origin,os.path.join(save_path,'segmentation_lobe_4.mha'))
        self.genarate_mha(lobe5,spacing,origin,os.path.join(save_path,'segmentation_lobe_5.mha'))
        
        #print(" save lobe mask done")
        return lobe_mask



    def read_file(self):
        
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(self.filepath)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        self.lung = sitk.GetArrayFromImage(image) # z, y, x
        self.lung_Origin = image.GetOrigin() # x, y, z
        self.lung_Spacing = image.GetSpacing() # x, y, z
        
    def genarate_mha(self,img,spacing,origin,path):
        output_img = sitk.GetImageFromArray(img.astype(np.ubyte), isVector=False)
        output_img.SetSpacing(spacing)
        output_img.SetOrigin(origin)
        sitk.WriteImage(output_img, path, True)
        
        

    def scaler(self,lung):

        div = 3071 + 1024

        assert div != 0 , 'scaler error : 0 value error'

        return (lung + 1024)/div



    def seg(self,img):



        self.net.eval()
        predict = np.zeros(img.shape)
        data = np.copy(img)
        data[data<-1024] = -1024
        data[data>3071] = 3071
        data = self.scaler(data)

        x2_begin = int((data.shape[1] - 400) / 2)
        
        if data.shape[0] > 200:
            crop_volumes = np.copy(data[20:data.shape[0] - 20, x2_begin:x2_begin + 400, x2_begin:x2_begin + 400])
        else:
            crop_volumes = np.copy(data)
        crop_volumes = crop_volumes.reshape((1, 1, crop_volumes.shape[0], crop_volumes.shape[1], crop_volumes.shape[2]))
        crop_volumes = torch.from_numpy(crop_volumes)
        crop_volumes = crop_volumes.type(torch.FloatTensor)

        if torch.cuda.is_available():
            crop_volumes = crop_volumes.cuda()
            self.net.cuda()
        else:
            raise Exception("cuda is not available")

        with torch.no_grad():
            predict_i = self.net(crop_volumes)

        predict_i = predict_i.cpu().numpy()
        predict_i = np.argmax(predict_i.reshape((6, predict_i.shape[2], predict_i.shape[3], predict_i.shape[4])),
                              axis=0)

        #print(predict_i.shape)
        if data.shape[0] > 200:
            predict[20:data.shape[0] - 20, x2_begin:x2_begin + 400, x2_begin:x2_begin + 400] = predict_i
        else:
            predict = predict_i
        del crop_volumes,predict_i
        
        return predict
